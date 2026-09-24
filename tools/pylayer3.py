#!/usr/bin/env python3
"""Dump intermediates for a tiny forward (first 2 layers) as raw f32 bins.
Rust compares against these to find the diverging op."""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MODEL = Path.home() / ".cache/lisa-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
OUT = Path("/tmp/lisa-diff")
OUT.mkdir(exist_ok=True)

index = json.load(open(MODEL / "model.safetensors.index.json"))["weight_map"]
_shards = {}


def W(name):
    sh = index[name]
    if sh not in _shards:
        _shards[sh] = mx.load(str(MODEL / sh))
    return _shards[sh][name]


def rms_mod(x, weight, group=None, eps=1e-6):
    scale = weight
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


def save(name, arr):
    a = np.array(arr.astype(mx.float32))
    a.tofile(OUT / f"{name}.bin")
    print(name, a.shape, float(a.reshape(-1)[:4].sum()))


def dump(tag, arr):
    mx.eval(arr)
    save(tag, arr)


import json as _json
_g = _json.load(open("reference/correctness_prompts/public_longcopy_gate_english_1024_256.json"))
ids = mx.array([_g["cases"][0]["prompt_tokens"][:48]], dtype=mx.int32)

# 1. embedding
hidden = mx.dequantize(
    W("language_model.model.embed_tokens.weight")[ids],
    W("language_model.model.embed_tokens.scales")[ids],
    W("language_model.model.embed_tokens.biases")[ids],
    32, 4)
dump("01_embed", hidden)
hidden = mx.tile(hidden, [1, 1, 4])
dump("02_tile", hidden)

L = "model.layers.0"
# 2. attn hc
normed = rms_mod(hidden, W(f"language_model.{L}.attn_hyper_connection.hc_norm.weight"), group=2560)
dump("03_hc_norm", normed)
lo = qmm(normed, f"{L}.attn_hyper_connection.input_mix_weight_down")
dump("04_down", lo)
w = nn.silu(lo / 4)
w = mx.sigmoid(qmm(w, f"{L}.attn_hyper_connection.input_mix_weight_up"))
dump("05_up", w)
lead = w.shape[:-1]
mixed = (w.reshape(*lead, 4, 2560) * normed.reshape(*lead, 4, 2560)).mean(axis=-2)
dump("06_mixed", mixed)
inj = 2 * mx.sigmoid(qmm(normed, f"{L}.attn_hyper_connection.block_inject_weight") / 4)
dump("07_inject_w", inj)

# 3. GDN
B, S = mixed.shape[0], mixed.shape[1]
mixed_qkv = qmm(mixed, f"{L}.linear_attn.in_proj_qkv")
dump("08_qkv", mixed_qkv)
z = qmm(mixed, f"{L}.linear_attn.in_proj_z").reshape(B, S, 48, 128)
b = qmm(mixed, f"{L}.linear_attn.in_proj_b")
a = qmm(mixed, f"{L}.linear_attn.in_proj_a")
conv_state = mx.zeros((B, 3, mixed_qkv.shape[-1]), dtype=mixed_qkv.dtype)
conv_input = mx.concatenate([conv_state, mixed_qkv], 1)
conv_w = W(f"language_model.{L}.linear_attn.conv1d.weight")
print("conv1d weight shape:", conv_w.shape)
if conv_w.shape[1] == 1 and conv_w.shape[2] > 1:
    conv_w = conv_w.transpose(0, 2, 1)
conv_out = nn.silu(mx.conv1d(conv_input, conv_w, stride=1, padding=0, dilation=1, groups=10240))
dump("09_conv_out", conv_out)
q, k, v = mx.split(conv_out, [2048, 4096], axis=-1)
q = q.reshape(B, S, 16, 128)
k = k.reshape(B, S, 16, 128)
v = v.reshape(B, S, 48, 128)
inv_scale = 128 ** -0.5
q = (inv_scale * inv_scale) * mx.fast.rms_norm(q, None, 1e-6)
k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
dump("10_q", q)
dump("11_k", k)
dump("12_v", v)

beta = mx.sigmoid(b.astype(mx.float32))
a_log = W(f"language_model.{L}.linear_attn.A_log").astype(mx.float32)
dt_bias = W(f"language_model.{L}.linear_attn.dt_bias").astype(mx.float32)
dump("13_g_in", (a.astype(mx.float32) + dt_bias))
# gated_delta_update inlined (matches mlx-lm gated_delta.py)
beta_ = mx.sigmoid(b.astype(mx.float32))
g_ = mx.exp(-mx.exp(a_log) * nn.softplus(a.astype(mx.float32) + dt_bias))

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

out, state = gated_delta_ops(q, k, v, g_, beta_, state=None)
dump("14_gdn_out", out)
dump("15_gdn_state", state)

normed_out = mx.fast.rms_norm(out, W(f"language_model.{L}.linear_attn.norm.weight"), 1e-6)
g = mx.sigmoid(z.astype(mx.float32))
normed_out = (g * normed_out.astype(mx.float32)).astype(mixed.dtype)
dump("16_gated", normed_out.reshape(B, S, -1))
attended = qmm(normed_out.reshape(B, S, -1), f"{L}.linear_attn.out_proj")
dump("17_attended", attended)

spread = attended[..., None, :] * inj[..., None]
stream = hidden + spread.reshape(B, S, -1)
dump("18_after_attn_inject", stream)

# 4. mlp hc + moe
normed2 = rms_mod(stream, W(f"language_model.{L}.mlp_hyper_connection.hc_norm.weight"), group=2560)
lo2 = qmm(normed2, f"{L}.mlp_hyper_connection.input_mix_weight_down")
w2 = nn.silu(lo2 / 4)
w2 = mx.sigmoid(qmm(w2, f"{L}.mlp_hyper_connection.input_mix_weight_up"))
mixed2 = (w2.reshape(*lead, 4, 2560) * normed2.reshape(*lead, 4, 2560)).mean(axis=-2)
dump("19_moe_in", mixed2)
inj2 = 2 * mx.sigmoid(qmm(normed2, f"{L}.mlp_hyper_connection.block_inject_weight") / 4)

logits_r = mixed2.astype(mx.float32) @ W(f"language_model.{L}.mlp.gate.weight").T
indices = mx.argpartition(-logits_r, kth=9, axis=-1)[..., :10]
weights = mx.softmax(mx.take_along_axis(logits_r, indices, axis=-1), axis=-1)
dump("20_router_w", weights)
dump("21_router_idx", indices.astype(mx.float32))

rows = B * S
x_rep = mx.repeat(mixed2.reshape(rows, 2560), 10, axis=0)
idx_flat = indices.reshape(rows * 10)


def gqmm(inp, key):
    return mx.gather_qmm(
        inp.reshape(rows * 10, 1, -1),
        W(f"language_model.{L}.mlp.switch_mlp.{key}.weight"),
        W(f"language_model.{L}.mlp.switch_mlp.{key}.scales"),
        W(f"language_model.{L}.mlp.switch_mlp.{key}.biases"),
        rhs_indices=idx_flat, transpose=True, group_size=32, bits=4,
    ).reshape(rows * 10, -1)


gate_out = gqmm(x_rep, "gate_proj")
up_out = gqmm(x_rep, "up_proj")
act = nn.silu(gate_out) * up_out
down_out = gqmm(act, "down_proj")
down = down_out.reshape(B, S, 10, 2560)
routed = (down * weights[..., None]).sum(axis=-2).astype(mixed2.dtype)
dump("22_routed", routed)

sg = qmm(mixed2, f"{L}.mlp.shared_expert_gate")
sh = nn.silu(qmm(mixed2, f"{L}.mlp.shared_expert.gate_proj")) * qmm(mixed2, f"{L}.mlp.shared_expert.up_proj")
sh = qmm(sh, f"{L}.mlp.shared_expert.down_proj")
moe_out = routed + mx.sigmoid(sg) * sh
dump("23_moe_out", moe_out)

spread2 = moe_out[..., None, :] * inj2[..., None]
final = stream + spread2.reshape(B, S, -1)
dump("24_layer0_out", final)

lin_cache = {"conv": None, "ssm": None}
history = [248044, 248044]
ple_conv = None

def ngram_rows(ids):
    def is_prime(v):
        if v < 2: return False
        if v % 2 == 0: return v == 2
        d = 3
        while d*d <= v:
            if v % d == 0: return False
            d += 2
        return True
    sizes, offsets = [], []
    total = 0
    for head in range(16):
        p = 19_999_999
        cnt = head + 1
        while cnt > 0:
            p += 1
            while not is_prime(p): p += 1
            cnt -= 1
        sizes.append(p); offsets.append(total); total += p
    gamma = 0x9E3779B97F4A7C15
    half = max(1, ((2**63 - 1) // 248320) // 2)
    mults = []
    for i in range(3):
        v = (gamma * (i + 1)) & 0xFFFFFFFFFFFFFFFF
        v = (v + gamma) & 0xFFFFFFFFFFFFFFFF
        v ^= v >> 30; v = (v * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        v ^= v >> 27; v = (v * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        v ^= v >> 31
        mults.append((2 * (v % half) + 1) & 0xFFFFFFFFFFFFFFFF)
    print("PY mults:", mults, "sizes[:4]:", sizes[:4], "half:", half)
    eos = 248044
    out = []
    hist = [248044, 248044] + [int(t) for t in ids[0]]
    T = len(hist)
    prev = [-1]*T; last = -1
    for t in range(T):
        prev[t] = last
        if hist[t] == eos: last = t
    def shifted(s, t):
        if s == 0: return hist[t]
        in_seg = t - (prev[t] + 1)
        src2 = t - s
        return hist[src2] if (in_seg >= s and src2 >= 0) else eos
    for t in range(2, T):
        for ngram in range(2, 4):
            mx_ = (shifted(0, t) * mults[0]) & 0xFFFFFFFFFFFFFFFF
            for pp in range(1, ngram):
                mx_ ^= (shifted(pp, t) * mults[pp]) & 0xFFFFFFFFFFFFFFFF
            mx_ = mx_ - 2**64 if mx_ >= 2**63 else mx_
            low = (ngram - 2) * 8
            for head in range(low, low + 8):
                r = mx_ % sizes[head]
                if r != 0 and (r < 0) != (sizes[head] < 0): r += sizes[head]
                out.append(int(r + offsets[head]))
    return mx.array(out, dtype=mx.int32)


def run_linear_layer(i, hidden, ids):
    global history, ple_conv
    if i == 1:
        pk_rows = ngram_rows(ids)
        print("PY row ids[:32]:", [int(x) for x in pk_rows[:32]])
        tp = W("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
        ts = W("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.scales")
        tb = W("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.biases")
        embedded = mx.dequantize(tp[pk_rows], ts[pk_rows], tb[pk_rows], 32, 4)
        embedded = embedded.reshape(*ids.shape, -1).astype(hidden.dtype)
        dump("30_ple_embed", embedded)
        lp = f"model.layers.{i}"
        key = rms_mod(qmm(embedded, lp + ".ple.key_proj"), W(f"language_model.{lp}.ple.norm_key.weight"), group=2560)
        key = key.reshape(*key.shape[:-1], 4, 2560)
        value = qmm(embedded, lp + ".ple.value_proj")
        query = rms_mod(hidden, W(f"language_model.{lp}.ple.norm_query.weight"), group=2560)
        query = query.reshape(*query.shape[:-1], 4, 2560)
        gate = (key * query).sum(axis=-1, keepdims=True) / 2560 ** 0.5
        gate = mx.sqrt(mx.maximum(mx.abs(gate), mx.array(1e-6, gate.dtype))) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*gated.shape[:-2], -1)
        dump("31_ple_gated", gated)
        normed_c = rms_mod(gated, W(f"language_model.{lp}.ple.norm_conv.weight"), group=2560)
        st = ple_conv
        if st is None:
            st = mx.zeros((hidden.shape[0], 9, hidden.shape[-1]), dtype=hidden.dtype)
        full_c = mx.concatenate([st, normed_c], 1)
        ple_conv = full_c[:, -9:, :]
        cw = W(f"language_model.{lp}.ple.conv1d.weight")
        if cw.shape[1] == 1 and cw.shape[2] > 1:
            cw = cw.transpose(0, 2, 1)
        conv = nn.silu(mx.conv1d(full_c, cw, stride=1, padding=0, dilation=3, groups=10240))
        hidden = hidden + gated + conv
        history = (history + [int(t) for t in ids[0]])[-2:]
    lp = f"model.layers.{i}"
    lead = hidden.shape[:-1]
    normed = rms_mod(hidden, W(f"language_model.{lp}.attn_hyper_connection.hc_norm.weight"), group=2560)
    lo = qmm(normed, lp + ".attn_hyper_connection.input_mix_weight_down")
    wmix = mx.sigmoid(qmm(nn.silu(lo / 4), lp + ".attn_hyper_connection.input_mix_weight_up"))
    mixed = (wmix.reshape(*lead, 4, 2560) * normed.reshape(*lead, 4, 2560)).mean(axis=-2)
    inj = 2 * mx.sigmoid(qmm(normed, lp + ".attn_hyper_connection.block_inject_weight") / 4)
    att = gated_deltanet(mixed, lp + ".linear_attn", lin_cache)
    hidden = hidden + (att[..., None, :] * inj[..., None]).reshape(*lead, -1)
    normed2 = rms_mod(hidden, W(f"language_model.{lp}.mlp_hyper_connection.hc_norm.weight"), group=2560)
    lo2 = qmm(normed2, lp + ".mlp_hyper_connection.input_mix_weight_down")
    w2m = mx.sigmoid(qmm(nn.silu(lo2 / 4), lp + ".mlp_hyper_connection.input_mix_weight_up"))
    mixed2 = (w2m.reshape(*lead, 4, 2560) * normed2.reshape(*lead, 4, 2560)).mean(axis=-2)
    inj2 = 2 * mx.sigmoid(qmm(normed2, lp + ".mlp_hyper_connection.block_inject_weight") / 4)
    out = moe(mixed2, lp + ".mlp")
    return hidden + (out[..., None, :] * inj2[..., None]).reshape(*lead, -1)


def gated_deltanet(x, lp, cache):
    B, S = x.shape[0], x.shape[1]
    mixed_qkv = qmm(x, lp + ".in_proj_qkv")
    z = qmm(x, lp + ".in_proj_z").reshape(B, S, 48, 128)
    b = qmm(x, lp + ".in_proj_b")
    a = qmm(x, lp + ".in_proj_a")
    st = cache["conv"]
    if st is None:
        st = mx.zeros((B, 3, mixed_qkv.shape[-1]), dtype=mixed_qkv.dtype)
    conv_input = mx.concatenate([st, mixed_qkv], 1)
    cache["conv"] = conv_input[:, -3:, :]
    cw = W(f"language_model.{lp}.conv1d.weight")
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
    a_log = W(f"language_model.{lp}.A_log").astype(mx.float32)
    dt_bias = W(f"language_model.{lp}.dt_bias").astype(mx.float32)
    g_ = mx.exp(-mx.exp(a_log) * nn.softplus(a.astype(mx.float32) + dt_bias))
    out, state = gated_delta_ops(q, k, v, g_, beta, state=cache["ssm"])
    cache["ssm"] = state
    normed_out = mx.fast.rms_norm(out, W(f"language_model.{lp}.norm.weight"), 1e-6)
    g = mx.sigmoid(z.astype(mx.float32))
    normed_out = (g * normed_out.astype(mx.float32)).astype(x.dtype)
    return qmm(normed_out.reshape(B, S, -1), lp + ".out_proj")


def moe(x, lp):
    logits_r = x.astype(mx.float32) @ W(f"language_model.{lp}.gate.weight").T
    indices = mx.argpartition(-logits_r, kth=9, axis=-1)[..., :10]
    weights = mx.softmax(mx.take_along_axis(logits_r, indices, axis=-1), axis=-1, precise=True)
    B, S, Hh = x.shape
    rows = B * S
    x_rep = mx.repeat(x.reshape(rows, Hh), 10, axis=0)
    idx_flat = indices.reshape(rows * 10)
    def gqmm(inp, key):
        return mx.gather_qmm(
            inp.reshape(rows * 10, 1, -1),
            W(f"language_model.{lp}.switch_mlp.{key}.weight"),
            W(f"language_model.{lp}.switch_mlp.{key}.scales"),
            W(f"language_model.{lp}.switch_mlp.{key}.biases"),
            rhs_indices=idx_flat, transpose=True, group_size=32, bits=4,
        ).reshape(rows * 10, -1)
    gate_out = gqmm(x_rep, "gate_proj")
    up_out = gqmm(x_rep, "up_proj")
    act = nn.silu(gate_out) * up_out
    down_out = gqmm(act, "down_proj")
    down = down_out.reshape(B, S, 10, Hh)
    routed = (down * weights[..., None]).sum(axis=-2).astype(x.dtype)
    sg = qmm(x, lp + ".shared_expert_gate")
    sh = nn.silu(qmm(x, lp + ".shared_expert.gate_proj")) * qmm(x, lp + ".shared_expert.up_proj")
    sh = qmm(sh, lp + ".shared_expert.down_proj")
    return routed + mx.sigmoid(sg) * sh


hidden = final
for i in (1, 2):
    hidden = run_linear_layer(i, hidden, ids)
    if i == 1:
        dump("39_layer1_out", hidden)
dump("40_layer2_out", hidden)

lp = "model.layers.3"
lead = hidden.shape[:-1]
normed = rms_mod(hidden, W(f"language_model.{lp}.attn_hyper_connection.hc_norm.weight"), group=2560)
lo = qmm(normed, lp + ".attn_hyper_connection.input_mix_weight_down")
wmix = mx.sigmoid(qmm(nn.silu(lo / 4), lp + ".attn_hyper_connection.input_mix_weight_up"))
mixed3 = (wmix.reshape(*lead, 4, 2560) * normed.reshape(*lead, 4, 2560)).mean(axis=-2)
inj3 = 2 * mx.sigmoid(qmm(normed, lp + ".attn_hyper_connection.block_inject_weight") / 4)
dump("41_layer3_mixed", mixed3)

B3, S3 = mixed3.shape[0], mixed3.shape[1]
projected = qmm(mixed3, lp + ".self_attn.q_proj").reshape(B3, S3, 24, -1)
queries, gate = mx.split(projected, 2, axis=-1)
dump("42_q_proj", queries)
gate = gate.reshape(B3, S3, -1)
queries = rms_mod(queries, W(f"language_model.{lp}.self_attn.q_norm.weight"))
dump("43_q_norm", queries)
keys = qmm(mixed3, lp + ".self_attn.k_proj").reshape(B3, S3, 2, -1)
keys = rms_mod(keys, W(f"language_model.{lp}.self_attn.k_norm.weight"))
dump("44_k_norm", keys)
values = qmm(mixed3, lp + ".self_attn.v_proj").reshape(B3, S3, 2, -1)
dump("45_v", values)
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


def cos_sin_(positions):
    even = mx.arange(0, 64, 2).astype(mx.float32)
    inv_freq = mx.exp(even * (-float(mx.log(mx.array(1e7))) / 64))
    freqs = positions.astype(mx.float32)[..., None] * inv_freq
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)

pos = mx.arange(0, S3)[None]
cos, sin = cos_sin_(pos)
cos4, sin4 = cos[:, None], sin[:, None]
q4 = queries.transpose(0, 2, 1, 3)
k4 = keys.transpose(0, 2, 1, 3)
dump("46_cos", cos)
q4 = rope_partial(q4, cos4, sin4)
k4 = rope_partial(k4, cos4, sin4)
dump("47_q_rope", q4)
dump("48_k_rope", k4)
out3 = mx.fast.scaled_dot_product_attention(q4, k4, values.transpose(0, 2, 1, 3), scale=256**-0.5)
dump("49_sdpa", out3)
out3 = out3.transpose(0, 2, 1, 3).reshape(B3, S3, -1)
gated3 = out3 * mx.sigmoid(gate)
dump("50_gated", gated3)
att3 = qmm(gated3, lp + ".self_attn.o_proj")
dump("51_o_proj", att3)

# final mixer + head from the CURRENT hidden (layers 3-47 skipped: we test the
# mixer itself on this hidden)
def gated_residual_mix2(normed, prefix, use_inject):
    lo = qmm(normed, prefix + ".input_mix_weight_down")
    w = nn.silu(lo / 4)
    w = mx.sigmoid(qmm(w, prefix + ".input_mix_weight_up"))
    lead = w.shape[:-1]
    mixed = (w.reshape(*lead, 4, 2560) * normed.reshape(*lead, 4, 2560)).mean(axis=-2)
    return mixed

nf = rms_mod(hidden, W("language_model.model.hyper_connection_mixer.hc_norm.weight"), group=2560)
dump("60_mixer_normed", nf)
mx_out = gated_residual_mix2(nf, "model.hyper_connection_mixer", False)
dump("61_mixed", mx_out)
logits = qmm(mx_out, "lm_head")
dump("62_logits", logits)
print("done3+mixer")
