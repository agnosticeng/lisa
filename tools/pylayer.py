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


ids = mx.array([[11, 72, 11762, 279, 20438, 1881, 279, 9212]], dtype=mx.int32)

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
print("done")
