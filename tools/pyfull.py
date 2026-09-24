#!/usr/bin/env python3
"""Full 48-layer Python reference at small scale: dumps per-layer hidden,
final mixed and logits for parity comparison against the Rust engine.

Uses the plain ops-fallback GDN (pair with LISA_GDN_OPS=1 on the Rust side)
and a proper np.memmap n-gram gather with shard splitting."""

import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MODEL = Path.home() / ".cache/lisa-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
GOLDEN = Path("reference/correctness_prompts/public_longcopy_gate_english_1024_256.json")
OUT = Path("/tmp/lisa-full")
N_TOKENS = 48

index = json.load(open(MODEL / "model.safetensors.index.json"))["weight_map"]
_shards = {}


def W(name):
    sh = index[name]
    if sh not in _shards:
        _shards[sh] = mx.load(str(MODEL / sh))
    return _shards[sh][name]


# ------------------------------------------------------------- n-gram table
ROWS_PER_SHARD = 2_500_012
NGRAM_PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"


class NgramMmap:
    """Gather rows by global id with proper shard splitting (like the Rust)."""

    def __init__(self, model_dir, index_map):
        self.maps = {}
        for shard_idx in range(128):
            prefix = f"{NGRAM_PREFIX}.shard_{shard_idx}"
            entries = {}
            for suffix in ("weight", "scales", "biases"):
                raw = f"{prefix}.{suffix}"
                if raw not in index_map:
                    raise RuntimeError(f"missing {raw}")
                f = model_dir / index_map[raw]
                with open(f, "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    header = json.loads(fh.read(n))
                info = header[raw]
                off = 8 + n + info["data_offsets"][0]
                dtype = {"U32": np.uint32, "BF16": np.uint16}[info["dtype"]]
                shape = tuple(info["shape"])
                mm = np.memmap(f, dtype=dtype, mode="r", offset=off, shape=shape)
                entries[suffix] = (mm, shape)
            self.maps[shard_idx] = entries

    def gather(self, global_ids):
        n = len(global_ids)
        packed = np.empty((n, 20), dtype=np.uint32)
        scales = np.empty((n, 5), dtype=np.uint16)
        biases = np.empty((n, 5), dtype=np.uint16)
        for i, gid in enumerate(global_ids):
            shard = gid // ROWS_PER_SHARD
            row = gid % ROWS_PER_SHARD
            e = self.maps[shard]
            packed[i] = e["weight"][0][row]
            scales[i] = e["scales"][0][row].view(np.uint16)
            biases[i] = e["biases"][0][row].view(np.uint16)
        pk = mx.array(np.ascontiguousarray(packed))
        sc = mx.array(np.ascontiguousarray(scales)).view(mx.bfloat16)
        bi = mx.array(np.ascontiguousarray(biases)).view(mx.bfloat16)
        return pk, sc, bi


# ------------------------------------------------------------- modules
EPS = 1e-6


def rms_mod(x, weight, group=None, eps=EPS):
    scale = weight  # offset baked
    if group is None:
        return mx.fast.rms_norm(x, scale, eps)
    shape = x.shape
    grouped = x.reshape(*shape[:-1], shape[-1] // group, group)
    normed = mx.fast.rms_norm(grouped, None, eps).reshape(shape)
    return normed * scale


def qmm(x, name):
    return mx.quantized_matmul(
        x, W(f"language_model.{name}.weight"), W(f"language_model.{name}.scales"),
        W(f"language_model.{name}.biases"), True, 32, 4)


def embed(ids, name="model.embed_tokens"):
    wt = W(f"language_model.{name}.weight")[ids]
    sc = W(f"language_model.{name}.scales")[ids]
    bi = W(f"language_model.{name}.biases")[ids]
    return mx.dequantize(wt, sc, bi, 32, 4)


def cos_sin(positions):
    even = mx.arange(0, 64, 2).astype(mx.float32)
    inv_freq = mx.exp(even * (-float(np.log(1e7)) / 64))
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


def gated_residual_mix(normed, prefix, use_inject):
    lo = qmm(normed, prefix + ".input_mix_weight_down")
    w = nn.silu(lo / 4)
    w = mx.sigmoid(qmm(w, prefix + ".input_mix_weight_up"))
    lead = w.shape[:-1]
    mixed = (w.reshape(*lead, 4, 2560) * normed.reshape(*lead, 4, 2560)).mean(axis=-2)
    if use_inject:
        inject = 2 * mx.sigmoid(qmm(normed, prefix + ".block_inject_weight") / 4)
        return mixed, inject
    return mixed, None


def inject(residual, output, inject_w):
    spread = output[..., None, :] * inject_w[..., None]
    lead = output.shape[:-1]
    return residual + spread.reshape(*lead, -1)


def attention(x, prefix, cache, offset):
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

    # indexer tape (raw keys) — no mask below budget
    qk = qmm(x, prefix + ".indexer.index_qk_proj")
    raw_k = qk[..., 512:].reshape(B, S, 128)
    cache["tape"] = raw_k if cache["tape"] is None else mx.concatenate([cache["tape"], raw_k], 1)

    cache["k"] = keys if cache["k"] is None else mx.concatenate([cache["k"], keys], 2)
    cache["v"] = values if cache["v"] is None else mx.concatenate([cache["v"], values], 2)
    out = mx.fast.scaled_dot_product_attention(
        queries, cache["k"], cache["v"], scale=256 ** -0.5)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
    return qmm(out * mx.sigmoid(gate), prefix + ".o_proj")


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


def gated_deltanet(x, prefix, cache):
    B, S = x.shape[0], x.shape[1]
    mixed_qkv = qmm(x, prefix + ".in_proj_qkv")
    z = qmm(x, prefix + ".in_proj_z").reshape(B, S, 48, 128)
    b = qmm(x, prefix + ".in_proj_b")
    a = qmm(x, prefix + ".in_proj_a")

    st = cache["conv"]
    if st is None:
        st = mx.zeros((B, 3, mixed_qkv.shape[-1]), dtype=mixed_qkv.dtype)
    conv_input = mx.concatenate([st, mixed_qkv], 1)
    cache["conv"] = conv_input[:, -3:, :]
    cw = W(f"language_model.{prefix}.conv1d.weight")
    if cw.shape[1] == 1 and cw.shape[2] > 1:
        cw = cw.transpose(0, 2, 1)
    conv_out = nn.silu(mx.conv1d(conv_input, cw, stride=1, padding=0, dilation=1, groups=10240))
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
    g = mx.exp(-mx.exp(a_log) * nn.softplus(a.astype(mx.float32) + dt_bias))
    out, state = gated_delta_ops(q, k, v, g, beta, state=cache["ssm"])
    cache["ssm"] = state

    normed_out = mx.fast.rms_norm(out, W(f"language_model.{prefix}.norm.weight"), EPS)
    gg = mx.sigmoid(z.astype(mx.float32))
    normed_out = (gg * normed_out.astype(mx.float32)).astype(x.dtype)
    return qmm(normed_out.reshape(B, S, -1), prefix + ".out_proj")


def moe(x, prefix):
    logits = x.astype(mx.float32) @ W(f"language_model.{prefix}.gate.weight").T
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
            rhs_indices=idx_flat, transpose=True, group_size=32, bits=4,
        ).reshape(rows * 10, -1)

    gate_out = gqmm(x_rep, "gate_proj")
    up_out = gqmm(x_rep, "up_proj")
    act = nn.silu(gate_out) * up_out
    down_out = gqmm(act, "down_proj")
    down = down_out.reshape(B, S, 10, Hh)
    routed = (down * weights[..., None]).sum(axis=-2).astype(x.dtype)

    sg = qmm(x, prefix + ".shared_expert_gate")
    sh = nn.silu(qmm(x, prefix + ".shared_expert.gate_proj")) * qmm(x, prefix + ".shared_expert.up_proj")
    sh = qmm(sh, prefix + ".shared_expert.down_proj")
    return routed + mx.sigmoid(sg) * sh


# host hash (verified identical to the Rust)
def ngram_row_ids(ids, previous_context, eos=248044):
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

    sizes, offsets = [], []
    total = 0
    for head in range(16):
        p = 19_999_999
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
    half = max(1, ((2 ** 63 - 1) // 248320) // 2)
    mults = []
    for i in range(3):
        v = (gamma * (i + 1)) & 0xFFFFFFFFFFFFFFFF
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
            in_seg = t - (prev[t] + 1)
            src = t - s
            return history[src] if (in_seg >= s and src >= 0) else eos

        for t in range(max(0, T - ids.shape[1]), T):
            for ngram in range(2, 4):
                mixed = (shifted(0, t) * mults[0]) & 0xFFFFFFFFFFFFFFFF
                for pp in range(1, ngram):
                    mixed ^= (shifted(pp, t) * mults[pp]) & 0xFFFFFFFFFFFFFFFF
                mixed = mixed - 2 ** 64 if mixed >= 2 ** 63 else mixed
                low = (ngram - 2) * 8
                for head in range(low, low + 8):
                    r = mixed % sizes[head]
                    if r != 0 and (r < 0) != (sizes[head] < 0):
                        r += sizes[head]
                    out.append(int(r + offsets[head]))
    return out


def ple(hidden, ids, previous_context, cache, prefix="model.layers.1.ple"):
    rows = ngram_row_ids(ids, previous_context)
    pk, sc, bi = cache["table"].gather(rows)
    embedded = mx.dequantize(pk, sc, bi, 32, 4).reshape(*ids.shape, -1).astype(hidden.dtype)
    dump("ple_embed", embedded)

    key = rms_mod(qmm(embedded, prefix + ".key_proj"), W(f"language_model.{prefix}.norm_key.weight"), group=2560)
    key = key.reshape(*key.shape[:-1], 4, 2560)
    value = qmm(embedded, prefix + ".value_proj")
    query = rms_mod(hidden, W(f"language_model.{prefix}.norm_query.weight"), group=2560)
    query = query.reshape(*query.shape[:-1], 4, 2560)

    gate = (key * query).sum(axis=-1, keepdims=True) / 2560 ** 0.5
    gate = mx.sqrt(mx.maximum(mx.abs(gate), mx.array(1e-6, gate.dtype))) * mx.sign(gate)
    gated = mx.sigmoid(gate) * value[..., None, :]
    gated = gated.reshape(*gated.shape[:-2], -1)
    dump("ple_gated", gated)

    normed_c = rms_mod(gated, W(f"language_model.{prefix}.norm_conv.weight"), group=2560)
    n = 9
    st = cache["ple_conv"]
    if st is None:
        st = mx.zeros((hidden.shape[0], n, hidden.shape[-1]), dtype=hidden.dtype)
    full = mx.concatenate([st, normed_c], 1)
    cache["ple_conv"] = full[:, -n:, :]
    cw = W(f"language_model.{prefix}.conv1d.weight")
    if cw.shape[1] == 1 and cw.shape[2] > 1:
        cw = cw.transpose(0, 2, 1)
    conv = nn.silu(mx.conv1d(full, cw, stride=1, padding=0, dilation=3, groups=10240))
    return gated + conv


def dump(name, arr):
    a = np.array(arr.astype(mx.float32))
    a.tofile(OUT / f"{name}.bin")


def main():
    OUT.mkdir(exist_ok=True)
    g = json.load(open(GOLDEN))
    prompt = g["cases"][0]["prompt_tokens"][:N_TOKENS]
    ids = mx.array([prompt], dtype=mx.int32)

    table = NgramMmap(MODEL, index)
    caches = {
        "full": {str(i): {"k": None, "v": None, "tape": None} for i in range(3, 48, 4)},
        "lin": {str(i): {"conv": None, "ssm": None} for i in range(48) if i % 4 != 3},
        "history": [[248044, 248044]],
        "offset": 0,
        "table": table,
        "ple_conv": None,
    }

    hidden = embed(ids)
    hidden = mx.tile(hidden, [1, 1, 4])
    for i in range(48):
        lp = f"model.layers.{i}"
        if i == 1:
            hidden = hidden + ple(hidden, ids, caches["history"], caches)
        normed = rms_mod(hidden, W(f"language_model.{lp}.attn_hyper_connection.hc_norm.weight"), group=2560)
        mixed, inj = gated_residual_mix(normed, lp + ".attn_hyper_connection", True)
        if str(i) in caches["full"]:
            att = attention(mixed, lp + ".self_attn", caches["full"][str(i)], caches["offset"])
        else:
            att = gated_deltanet(mixed, lp + ".linear_attn", caches["lin"][str(i)])
        hidden = inject(hidden, att, inj)
        normed2 = rms_mod(hidden, W(f"language_model.{lp}.mlp_hyper_connection.hc_norm.weight"), group=2560)
        mixed2, inj2 = gated_residual_mix(normed2, lp + ".mlp_hyper_connection", True)
        out = moe(mixed2, lp + ".mlp")
        hidden = inject(hidden, out, inj2)
        dump(f"layer_{i:02d}", hidden)

    caches["history"] = [(h + [int(t) for t in ids[0]])[-2:] for h in caches["history"]]
    caches["offset"] += ids.shape[1]

    normed_final = rms_mod(hidden, W("language_model.model.hyper_connection_mixer.hc_norm.weight"), group=2560)
    mixed_out, _ = gated_residual_mix(normed_final, "model.hyper_connection_mixer", False)
    dump("mixed", mixed_out)
    logits = qmm(mixed_out, "lm_head")
    dump("logits", logits)
    mx.eval(logits)
    top = mx.argpartition(-logits[:, -1, :], kth=7, axis=-1)[..., :8]
    print("PY top8 last:", np.array(top).flatten().tolist())
    print("done")


if __name__ == "__main__":
    main()
