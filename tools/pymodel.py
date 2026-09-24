#!/usr/bin/env python3
"""Independent Python port of the Swift reference (Qwen4Exp*.swift) for
cross-checking the Rust engine. Computes the golden-prompt prefill and
prints diagnostics. Weights are memory-mapped shard by shard."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MODEL = Path.home() / ".cache/lisa-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
GOLDEN = Path("reference/correctness_prompts/public_longcopy_gate_english_1024_256.json")

# ---------------------------------------------------------------- weights
index = json.load(open(MODEL / "model.safetensors.index.json"))["weight_map"]
_shards = {}


def W(name):
    sh = index[name]
    if sh not in _shards:
        _shards[sh] = mx.load(str(MODEL / sh))
    return _shards[sh][name]


# ---------------------------------------------------------------- modules
EPS = 1e-6


def rms_mod(x, weight, group=None, eps=EPS):
    scale = weight  # offset baked (rms_norm_weight_offset = 0)
    if group is None:
        return mx.fast.rms_norm(x, scale, eps)
    shape = x.shape
    grouped = x.reshape(*shape[:-1], shape[-1] // group, group)
    normed = mx.fast.rms_norm(grouped, None, eps).reshape(shape)
    return normed * scale


def qmm(x, name):
    return mx.quantized_matmul(
        x,
        W(f"language_model.{name}.weight"),
        W(f"language_model.{name}.scales"),
        W(f"language_model.{name}.biases"),
        True,
        32,
        4,
    )


def embed(ids, name="model.embed_tokens"):
    wt = W(f"language_model.{name}.weight")[ids]
    sc = W(f"language_model.{name}.scales")[ids]
    bi = W(f"language_model.{name}.biases")[ids]
    return mx.dequantize(wt, sc, bi, 32, 4)


def silu(x):
    return nn.silu(x)


# ---------------------------------------------------------------- rotary
ROTARY_DIMS = 64
ROPE_BASE = 1e7


def cos_sin(positions):
    even = mx.arange(0, ROTARY_DIMS, 2).astype(mx.float32)
    inv_freq = mx.exp(even * (-float(mx.log(mx.array(ROPE_BASE))) / ROTARY_DIMS))
    freqs = positions.astype(mx.float32)[..., None] * inv_freq
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def rope_partial(x, cos, sin):
    d = cos.shape[-1]
    c = cos.astype(x.dtype)
    s = sin.astype(x.dtype)
    rotated = x[..., :d]
    half = d // 2
    x1 = rotated[..., :half]
    x2 = rotated[..., half:]
    swapped = mx.concatenate([-x2, x1], axis=-1)
    out = rotated * c + swapped * s
    if x.shape[-1] == d:
        return out
    return mx.concatenate([out, x[..., d:]], axis=-1)


# ---------------------------------------------------------------- hyper
HC, H = 4, 2560


def gated_residual_mix(normed, prefix, use_inject):
    lo = qmm(normed, prefix + ".input_mix_weight_down")
    w = silu(lo / HC)
    w = mx.sigmoid(qmm(w, prefix + ".input_mix_weight_up"))
    lead = w.shape[:-1]
    w4 = w.reshape(*lead, HC, H)
    n4 = normed.reshape(*lead, HC, H)
    mixed = (w4 * n4).mean(axis=-2)
    if use_inject:
        inject = 2 * mx.sigmoid(qmm(normed, prefix + ".block_inject_weight") / HC)
        return mixed, inject
    return mixed, None


def hc_norm(x, prefix):
    return rms_mod(x, W(f"language_model.{prefix}.hc_norm.weight"), group=H)


def inject(residual, output, inject_w):
    spread = output[..., None, :] * inject_w[..., None]
    lead = output.shape[:-1]
    return residual + spread.reshape(*lead, -1)


# ---------------------------------------------------------------- attention
def attention(x, prefix, layer_cache, offset):
    B, S = x.shape[0], x.shape[1]
    projected = qmm(x, prefix + ".q_proj").reshape(B, S, 24, -1)
    queries, gate = mx.split(projected, 2, axis=-1)
    gate = gate.reshape(B, S, -1)
    queries = rms_mod(queries, W(f"language_model.{prefix}.q_norm.weight")).transpose(0, 2, 1, 3)
    keys = qmm(x, prefix + ".k_proj").reshape(B, S, 2, -1)
    keys = rms_mod(keys, W(f"language_model.{prefix}.k_norm.weight")).transpose(0, 2, 1, 3)
    values = qmm(x, prefix + ".v_proj").reshape(B, S, 2, -1).transpose(0, 2, 1, 3)

    pos = mx.arange(offset, offset + S)[None]
    cos, sin = cos_sin(pos)
    cos, sin = cos[:, None], sin[:, None]
    queries = rope_partial(queries, cos, sin)
    keys = rope_partial(keys, cos, sin)

    # indexer tape update (raw keys)
    qk = qmm(x, prefix + ".indexer.index_qk_proj")
    raw_k = qk[..., 512:].reshape(B, S, 128)
    layer_cache["tape"] = raw_k if layer_cache["tape"] is None else mx.concatenate([layer_cache["tape"], raw_k], 1)

    kv_len = offset + S
    cache_k = keys if layer_cache["k"] is None else mx.concatenate([layer_cache["k"], keys], 2)
    cache_v = values if layer_cache["v"] is None else mx.concatenate([layer_cache["v"], values], 2)
    layer_cache["k"] = cache_k
    layer_cache["v"] = cache_v
    out = mx.fast.scaled_dot_product_attentions(
        queries, cache_k, cache_v, 256 ** -0.5, None
    ) if False else mx.fast.scaled_dot_product_attention(
        queries, cache_k, cache_v, scale=256 ** -0.5
    )
    out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
    return qmm(out * mx.sigmoid(gate), prefix + ".o_proj")


# ---------------------------------------------------------------- GDN
def gated_deltanet(x, prefix, layer_cache):
    B, S = x.shape[0], x.shape[1]
    mixed_qkv = qmm(x, prefix + ".in_proj_qkv")
    z = qmm(x, prefix + ".in_proj_z").reshape(B, S, 48, 128)
    b = qmm(x, prefix + ".in_proj_b")
    a = qmm(x, prefix + ".in_proj_a")

    conv_state = layer_cache["conv"]
    if conv_state is None:
        conv_state = mx.zeros((B, 3, mixed_qkv.shape[-1]), dtype=mixed_qkv.dtype)
    conv_input = mx.concatenate([conv_state, mixed_qkv], 1)
    layer_cache["conv"] = conv_input[:, -3:, :]
    conv_w = W(f"language_model.{prefix}.conv1d.weight")
    if conv_w.shape[1] == 1 and conv_w.shape[2] > 1:
        conv_w = conv_w.transpose(0, 2, 1)
    conv_out = silu(mx.conv1d(conv_input, conv_w, stride=1, padding=0, dilation=1, groups=10240))
    q, k, v = mx.split(conv_out, [2048, 4096], axis=-1)
    q = q.reshape(B, S, 16, 128)
    k = k.reshape(B, S, 16, 128)
    v = v.reshape(B, S, 48, 128)
    inv_scale = 128 ** -0.5
    q = (inv_scale * inv_scale) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    beta = mx.sigmoid(b.astype(mx.float32))
    a_log = W(f"language_model.{prefix}.A_log").astype(mx.float32)
    dt_bias = W(f"language_model.{prefix}.dt_bias").astype(mx.float32)
    beta_ = mx.sigmoid(b.astype(mx.float32))
    g_ = mx.exp(-mx.exp(a_log) * nn.softplus(a.astype(mx.float32) + dt_bias))
    out, state = gated_delta_ops(q, k, v, g_, beta_, state=layer_cache["ssm"])
    layer_cache["ssm"] = state

    normed = mx.fast.rms_norm(out, W(f"language_model.{prefix}.norm.weight"), EPS)
    g = mx.sigmoid(z.astype(mx.float32))
    normed = (g * normed.astype(mx.float32)).astype(x.dtype)
    return qmm(normed.reshape(B, S, -1), prefix + ".out_proj")


def gated_delta_ops(q, k, v, g, beta, state=None):
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2], v.shape[3]
    rep = Hv // Hk
    if rep > 1:
        q = mx.repeat(q, rep, axis=-2)
        k = mx.repeat(k, rep, axis=-2)
    st = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32) if state is None else state
    ys = []
    for t in range(T):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        gt = g[:, t].reshape(B, Hv, 1, 1)
        bt = beta[:, t].reshape(B, Hv, 1)
        new = st * gt
        kv_mem = (new * kt.reshape(B, Hv, 1, Dk)).sum(-1)
        delta = (vt - kv_mem) * bt
        new = new + kt.reshape(B, Hv, 1, Dk) * delta.reshape(B, Hv, Dv, 1)
        y = (new * qt.reshape(B, Hv, 1, Dk)).sum(-1)
        ys.append(y.astype(q.dtype))
        st = new
    return mx.stack(ys, axis=1), st


# ---------------------------------------------------------------- MoE
def moe(x, prefix="model.layers.0.mlp"):
    logits = qmm(x.astype(mx.float32), prefix + ".gate") if False else (
        x.astype(mx.float32) @ W(f"language_model.{prefix}.gate.weight").T
    )
    indices = mx.argpartition(-logits, kth=9, axis=-1)[..., :10]
    weights = mx.softmax(mx.take_along_axis(logits, indices, axis=-1), axis=-1, precise=True)

    B, S, Hh = x.shape
    rows = B * S
    x_rep = mx.repeat(x.reshape(rows, Hh), 10, axis=0)
    idx_flat = indices.reshape(rows * 10)

    def gqmm(inp, key):
        return mx.gather_qmm(
            inp.reshape(rows * 10, 1, -1),
            W(f"language_model.{prefix}.switch_mlp.{key}.weight"),
            W(f"language_model.{prefix}.switch_mlp.{key}.scales"),
            W(f"language_model.{prefix}.switch_mlp.{key}.biases"),
            rhs_indices=idx_flat,
            transpose=True,
            group_size=32,
            bits=4,
        ).reshape(rows * 10, -1)

    gate_out = gqmm(x_rep, "gate_proj")
    up_out = gqmm(x_rep, "up_proj")
    act = silu(gate_out) * up_out
    down_out = gqmm(act, "down_proj")
    down = down_out.reshape(B, S, 10, Hh)
    routed = (down * weights[..., None]).sum(axis=-2).astype(x.dtype)

    sg = qmm(x, prefix + ".shared_expert_gate")
    sh = silu(qmm(x, prefix + ".shared_expert.gate_proj")) * qmm(x, prefix + ".shared_expert.up_proj")
    sh = qmm(sh, prefix + ".shared_expert.down_proj")
    return routed + mx.sigmoid(sg) * sh


# ---------------------------------------------------------------- PLE
def ngram_row_ids(ids, previous_context, eos=248044):
    # host hash — mirrors the Swift hostRowIds
    import numpy as np

    heads_per_ngram = 8
    ngram_size = 3
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    # sizes: 16 primes after 19,999,999
    def is_prime(v):
        if v < 2:
            return False
        if v % 2 == 0:
            return v == 2
        d = 3
        while d * d <= v:
            if v % d == 0:
                return False
            d += 2
        return True

    sizes, offsets, total = [], [], 0
    p = 19_999_999
    for head in range(ngram_heads):
        cnt = head + 1
        while cnt > 0:
            p += 1
            while not is_prime(p):
                p += 1
            cnt -= 1
        sizes.append(p)
        offsets.append(total)
        total += p

    gamma = 0x9E3779B97F4A7C15
    half = max(1, ((2**63 - 1) // 248320) // 2)
    base_seed = 0
    mults = []
    for i in range(ngram_size):
        v = (base_seed + gamma * (i + 1)) & 0xFFFFFFFFFFFFFFFF
        v = (v + gamma) & 0xFFFFFFFFFFFFFFFF
        v ^= v >> 30
        v = (v * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        v ^= v >> 27
        v = (v * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        v ^= v >> 31
        mults.append((2 * (v % half) + 1) & 0xFFFFFFFFFFFFFFFF)

    out = []
    B = ids.shape[0]
    for b in range(B):
        history = [int(t) for t in previous_context[b]] + [int(t) for t in ids[b]]
        T = len(history)
        prev = [-1] * T
        last = -1
        for t in range(T):
            prev[t] = last
            if history[t] == eos:
                last = t

        def shifted(s, t):
            if s == 0:
                return history[t]
            in_segment = t - (prev[t] + 1)
            source = t - s
            return history[source] if (in_segment >= s and source >= 0) else eos

        for t in range(max(0, T - ids.shape[1]), T):
            for ngram in range(2, ngram_size + 1):
                mixed = (shifted(0, t) * mults[0]) & 0xFFFFFFFFFFFFFFFF
                for pp in range(1, ngram):
                    mixed ^= (shifted(pp, t) * mults[pp]) & 0xFFFFFFFFFFFFFFFF
                mixed = mixed - 2**64 if mixed >= 2**63 else mixed
                low = (ngram - 2) * heads_per_ngram
                for head in range(low, low + heads_per_ngram):
                    r = mixed % sizes[head]
                    if r != 0 and (r < 0) != (sizes[head] < 0):
                        r += sizes[head]
                    out.append(int(r + offsets[head]))
    return out


def ple(hidden, ids, previous_context, layer_cache, prefix="model.layers.1.ple"):
    table = layer_cache["table"]
    rows = ngram_row_ids(ids, previous_context)
    pk = table["weight"][rows]
    sc = table["scales"][rows]
    bi = table["biases"][rows]
    embedded = mx.dequantize(pk, sc, bi, 32, 4).reshape(*ids.shape, -1).astype(hidden.dtype)

    key = rms_mod(qmm(embedded, prefix + ".key_proj"), W(f"language_model.{prefix}.norm_key.weight"), group=H)
    key = key.reshape(*key.shape[:-1], HC, H)
    value = qmm(embedded, prefix + ".value_proj")
    query = rms_mod(hidden, W(f"language_model.{prefix}.norm_query.weight"), group=H)
    query = query.reshape(*query.shape[:-1], HC, H)

    gate = (key * query).sum(axis=-1, keepdims=True) / (H ** 0.5)
    gate = mx.sqrt(mx.maximum(mx.abs(gate), mx.array(1e-6, gate.dtype))) * mx.sign(gate)
    gated = mx.sigmoid(gate) * value[..., None, :]
    gated = gated.reshape(*gated.shape[:-2], -1)
    normed = rms_mod(gated, W(f"language_model.{prefix}.norm_conv.weight"), group=H)

    n = 9
    state = layer_cache["ple_conv"]
    if state is None:
        state = mx.zeros((normed.shape[0], n, normed.shape[-1]), dtype=normed.dtype)
    full = mx.concatenate([state, normed], 1)
    layer_cache["ple_conv"] = full[:, -n:, :]
    conv_w = W(f"language_model.{prefix}.conv1d.weight")
    if conv_w.shape[1] == 1 and conv_w.shape[2] > 1:
        conv_w = conv_w.transpose(0, 2, 1)
    conv = silu(mx.conv1d(full, conv_w, stride=1, padding=0, dilation=3, groups=10240))
    return gated + conv


# ---------------------------------------------------------------- forward
def forward(ids, caches, use_ple=True, dump_prefix=None):
    B, S = ids.shape
    hidden = embed(ids)
    ctx = caches["history"]
    hidden = mx.tile(hidden, [1, 1, HC])
    for i in range(48):
        lp = f"model.layers.{i}"
        dump = dump_prefix is not None and i == dump_prefix
        if use_ple and i == 1:
            stream = hidden + ple(hidden, ids, ctx, caches)
        else:
            stream = hidden
        if dump:
            mx.eval(stream)
            np.save(f"/tmp/py_layer{i}_in.npy", np.array(stream.astype(mx.float32)))
        normed = hc_norm(stream, lp + ".attn_hyper_connection")
        mixed, inj = gated_residual_mix(normed, lp + ".attn_hyper_connection", True)
        if f"{i}" in caches["full"]:
            att = attention(mixed, lp + ".self_attn", caches["full"][f"{i}"], caches["offset"])
        else:
            att = gated_deltanet(mixed, lp + ".linear_attn", caches["lin"][f"{i}"])
        hidden = inject(stream, att, inj)
        normed = hc_norm(hidden, lp + ".mlp_hyper_connection")
        mixed2, inj2 = gated_residual_mix(normed, lp + ".mlp_hyper_connection", True)
        out = moe(mixed2, lp + '.mlp')
        hidden = inject(hidden, out, inj2)
        if dump:
            mx.eval(hidden)
            np.save(f"/tmp/py_layer{i}_out.npy", np.array(hidden.astype(mx.float32)))
        if "--probe" in sys.argv:
            mx.eval(hidden)
            nan_ct = int(np.isnan(np.array(hidden.astype(mx.float32))).sum())
            print(f"layer {i}: nan={nan_ct} max={float(np.abs(np.array(hidden.astype(mx.float32))).max()):.3e}")
    caches["history"] = [ctx[b] + [int(t) for t in ids[b]] for b in range(B)]
    caches["history"] = [h[-2:] for h in caches["history"]]
    caches["offset"] += S
    normed_final = hc_norm(hidden, "model.hyper_connection_mixer")
    mixed_out, _ = gated_residual_mix(normed_final, "model.hyper_connection_mixer", False)
    return mixed_out


def main():
    use_ple = "--no-ple" not in sys.argv
    g = json.load(open(GOLDEN))
    prompt = g["cases"][0]["prompt_tokens"]
    expected = g["cases"][0]["expected_tokens"]

    table = NgramMmap(MODEL, index)
    caches = {
        "full": {str(i): {"k": None, "v": None, "tape": None} for i in range(3, 48, 4)},
        "lin": {str(i): {"conv": None, "ssm": None} for i in range(48) if i % 4 != 3},
        "history": [[248044, 248044]],
        "offset": 0,
        "table": table,
        "ple_conv": None,
    }

    ids = mx.array([prompt[:400]], dtype=mx.int32)
    caches["history"] = [[248044, 248044]]
    mixed = forward(ids, caches, use_ple=use_ple)
    logits = qmm(mixed[:, -1:, :], "lm_head")
    mx.eval(logits)
    neg = -logits
    top = mx.argpartition(neg, kth=7, axis=-1)[..., :8]
    print("PY top8 @400:", np.array(top).flatten().tolist())

    # mid-passage self-copy agreement
    ids2 = mx.array([prompt[:430]], dtype=mx.int32)
    caches2 = {
        "full": {str(i): {"k": None, "v": None, "tape": None} for i in range(3, 48, 4)},
        "lin": {str(i): {"conv": None, "ssm": None} for i in range(48) if i % 4 != 3},
        "history": [[248044, 248044]],
        "offset": 0,
        "table": table,
        "ple_conv": None,
    }
    mixed2 = forward(ids2, caches2, use_ple=use_ple)
    logits2 = qmm(mixed2, "lm_head")
    mx.eval(logits2)
    top2 = mx.argpartition(-logits2, kth=0, axis=-1)[..., :1]
    flat = np.array(top2).flatten()
    agree = sum(1 for p in range(400, 429) if flat[p] == prompt[p + 1])
    print(f"PY mid-passage agreement (400..429): {agree}/29")
    print("PY sample:", [(int(flat[p]), prompt[p + 1]) for p in range(400, 406)])


if __name__ == "__main__":
    main()
