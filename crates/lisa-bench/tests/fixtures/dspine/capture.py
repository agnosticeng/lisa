"""Capture the pinned Tencent MLX draft and refinement oracle on Apple Silicon.

Run in a venv with mlx==0.32.4, mlx-lm==0.32.2, numpy==2.4.6.
The upstream checkout and model are supplied explicitly; no weights are edited.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--upstream', type=pathlib.Path, required=True)
parser.add_argument('--model', type=pathlib.Path, required=True)
parser.add_argument('--out', type=pathlib.Path, required=True)
parser.add_argument('--four-bit', action='store_true', help='Independently quantize the 36 linear projections with MLX')
parser.add_argument('--block-size', type=int, choices=[7, 16], default=7)
args = parser.parse_args()
block = args.block_size
revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '321b0fc1388c4cdf03483ab73856975f7c0c0be7'
sys.path.insert(0, str(args.upstream / 'dspine_mlx'))
import mlx.core as mx
import numpy as np
from dspine_mlx.model import load_draft
path = args.model.resolve()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
subprocess.run(['sysctl','hw.memsize'],check=True)
subprocess.run(['memory_pressure'],check=True)
# Upstream MLX supports full-attention Qwen3 only. For this 8+7-token
# fixture, causal full attention is exactly the W8192 attention domain.
# Adapt metadata only; original weights and the cache remain untouched.
source_path=path
adapter=out/'short-context-adapter'
adapter.mkdir(exist_ok=True)
config=json.loads((path/'config.json').read_text())
assert config['sliding_window']==8192 and config['layer_types']==['sliding_attention']*5
config['sliding_window']=None
config['use_sliding_window']=False
config['layer_types']=['full_attention']*5
(adapter/'config.json').write_text(json.dumps(config))
link=adapter/'model.safetensors'
if not link.exists(): link.symlink_to(path/'model.safetensors')
model,cfg=load_draft(adapter)
if args.four_bit:
    import mlx.nn as nn
    nn.quantize(model, group_size=64, bits=4, mode='affine',
                class_predicate=lambda path, module: isinstance(module, nn.Linear)
                and (path == 'fc' or path.startswith('layers.')))
model.set_block_size(block)
def values(n, shape):
    return (((mx.arange(n,dtype=mx.int32)*37+11)%257-128).astype(mx.float32)/256).reshape(shape).astype(mx.bfloat16)
embedding=values(block*5120,(1,block,5120))
model._embed_tokens=lambda ids: embedding
hidden=values(8*5*5120,(1,8,5*5120))
cache=model.make_cache()
model.append_context(hidden,cache)
ids=mx.array([[42]+[cfg.mask_token_id]*(block-1)],dtype=mx.int32)
captures={}
for m in ([1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 16, 32, 36, 64] if args.four_bit else [1, 7, 16, 32, 64]):
    captures[f'fc_m{m}']=model.fc(values(m*25600,(1,m,25600)))
if args.four_bit:
    for i, layer in enumerate(model.layers):
        for m in [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 16, 32, 36, 64]:
            context_input = values(m*5120, (1,m,5120))
            captures[f'context_k{i}_m{m}'] = layer.self_attn.k_proj(context_input)
            captures[f'context_v{i}_m{m}'] = layer.self_attn.v_proj(context_input)
captures['fc']=model.fc(hidden)
ctx=model.hidden_norm(captures['fc']); captures['context']=ctx
x=model.layers[0].input_layernorm(embedding); residual=embedding; captures['input']=x
for i,layer in enumerate(model.layers):
    a=layer.self_attn
    q=a.q_proj(x); captures[f'qproj{i}']=q
    q=a.q_norm(q.reshape(1,block,32,128)).transpose(0,2,1,3); captures[f'qnorm{i}']=q
    q=model.rope(q,offset=8); captures[f'qrope{i}']=q
    ck,cv=a._context_kv(ctx,model.rope,0); captures[f'ck{i}']=ck; captures[f'cv{i}']=cv
    pk,pv=a._context_kv(x,model.rope,8); captures[f'pk{i}']=pk; captures[f'pv{i}']=pv
    av=mx.fast.scaled_dot_product_attention(q,mx.concatenate([ck,pk],2),mx.concatenate([cv,pv],2),scale=128**-0.5,mask='causal')
    captures[f'av{i}']=av
    attn=a.o_proj(av.transpose(0,2,1,3).reshape(1,block,-1)); captures[f'attn{i}']=attn
    h=residual+attn
    norm=layer.post_attention_layernorm(h); captures[f'norm{i}']=norm
    gate=layer.mlp.gate_proj(norm); up=layer.mlp.up_proj(norm)
    captures[f'gate{i}']=gate; captures[f'up{i}']=up
    from mlx_lm.models.activations import swiglu
    act=swiglu(gate,up); captures[f'act{i}']=act
    raw=h+layer.mlp.down_proj(act); captures[f'layer{i}']=raw
    next_norm=model.layers[i+1].input_layernorm if i<4 else model.norm
    stash={}
    x,residual=model.selected_prefix_bridge(raw,None,ids,i,next_norm,stash)
    captures[f'bridge{i}']=x; captures[f'residual{i}']=residual
output=x
# Confirm the staged diagnostic replay is the unchanged upstream forward.
reference_output, reference_stash = model(None, ids, cache)
assert bool(mx.all(output == reference_output).item())
assert all(bool(mx.all(stash[k] == reference_stash[k]).item()) for k in stash)
from dspine_mlx.bridge import build_serial_tables
from dspine_mlx.propose import RefinementTail
head_weight=values(512*5120,(512,5120))
class Head:
    def rows(self, ids): return head_weight[ids]
    def matmul(self, rows):
        return ((mx.arange(512)[None,:]*37+mx.arange(rows.shape[0])[:,None]*17)%257).astype(mx.float32)/256
bridge=model.selected_prefix_bridge
tg,tsv=build_serial_tables(head_rows=lambda s,e:head_weight[s:e],vocab_size=512,norm_w=model.norm.weight,write_w=bridge.feature_write_weights[-1],pv_w=mx.concatenate([bridge.predecessor_gate.weight,bridge.prev.weight]),codebook=bridge.codebook[:512])
tokens,candidates=RefinementTail(model,Head(),tg,tsv).greedy(output[0,1:],stash,42)
captures['candidates']=candidates
captures['proposals']=tokens
mx.eval(output,stash,captures)

for name,a in list(captures.items())+[('rows',output[:,1:]),('raw',stash['raw']),('packed',stash['packed']),('rms',stash['rms'])]:
    np.asarray(a.astype(mx.float32)).tofile(out/(name+'.f32'))
(out/'reference.json').write_text(json.dumps({'revision':'321b0fc1388c4cdf03483ab73856975f7c0c0be7','mlx':mx.__version__,'four_bit':args.four_bit,'draft':str(path),'block_size':block,'context_tokens':8,'input_rule':'((i*37+11)%257-128)/256, BF16','anchor':42,'adapter':f'full-attention metadata for 8+{block} tokens, below native W8192; weights unchanged'},indent=2))
print('saved',out)

inventory = {}
for item in sorted(out.glob('*.f32')):
    data = item.read_bytes()
    inventory[item.name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
(out/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
