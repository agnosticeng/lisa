"""Development-only reference capture; never used by LISA inference.

Requires PyTorch, Transformers, safetensors and the pinned public BFL source.
The owning Rust test requires these fixtures explicitly and fails if absent.
"""
import torch, json, math, gc, sys, types, importlib.util
from pathlib import Path
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding
from transformers import AutoTokenizer
import argparse
parser=argparse.ArgumentParser(description='Capture external FLUX 3 stage fixtures. Run alone on the GPU.')
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--reference',type=Path,required=True,help='BFL checkout at bfc8889a4497f846416f253a3972a1941ac698b9')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.model
out=args.output;out.mkdir(parents=True,exist_ok=True)
print('torch',torch.__version__,'MPS',torch.backends.mps.is_available(),flush=True)
torch.set_num_threads(8)
result={}
prompt='Studio photograph of a blue ceramic vase holding three yellow tulips on a pale stone table, soft window light, plain cream background.'
(out/'prompt.txt').write_text(prompt)
cfg=json.load(open(root/'text_encoder/config.json'))['text_config'];cfg['_attn_implementation']='eager'
with torch.device('meta'): model=Qwen3VLTextModel(Qwen3VLTextConfig(**cfg))
state={}
for file in (root/'text_encoder').glob('*.safetensors'):
 with safe_open(file,framework='pt',device='cpu') as f:
  for k in f.keys():
   if k.startswith('model.language_model.'): state[k.removeprefix('model.language_model.')]=f.get_tensor(k)
model.load_state_dict(state,assign=True);del state
model.rotary_emb=Qwen3VLTextRotaryEmbedding(model.config)
model=model.to('mps').eval()
tok=AutoTokenizer.from_pretrained(str(root/'text_encoder'))
txt=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
real=len(tok(txt)['input_ids']);n=min(math.ceil(real/80)*80,4096)
inputs=tok(txt,return_tensors='pt',padding='max_length',truncation=True,max_length=n).to('mps')
mask=inputs['attention_mask'];pos=(mask.long().cumsum(-1)-1).masked_fill(mask==0,1)
with torch.inference_mode():
 output=model(**inputs,position_ids=pos[None].expand(4,-1,-1),output_hidden_states=True,use_cache=False)
 enc=torch.cat([output.hidden_states[k] for k in [4,8,12,16,20,24,28,32]],dim=-1)
 result['text']=enc.float().cpu().contiguous()
print('text',enc.shape,flush=True)
del output,model,enc,inputs;gc.collect();torch.mps.empty_cache()

def module(name,file):
 spec=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
ref=args.reference/'src/flux3'
dit=module('flux3_oracle_dit',ref/'dit.py')
with torch.device('meta'): model=dit.DiT(flow_map=True)
model.load_state_dict(load_file(str(root/'model.safetensors')),assign=True)
model=model.to('mps').eval()
h,w=4,6;ni=h*w
x=torch.linspace(-1,1,ni*128).reshape(ni,128).to(torch.bfloat16).float()
ids=torch.tensor([[0,r*.5,c*.5,0] for r in range(h) for c in range(w)],dtype=torch.float32)
tids=torch.tensor([[0,0,0,i] for i in range(n)],dtype=torch.float32)
result['latent']=x[None].contiguous()
data={'image':x.to('mps'),'image_cond':torch.empty(0,128,device='mps',dtype=torch.bfloat16),'text':result['text'][0].to('mps',torch.bfloat16)}
packed=dit.Packed(data=data,ids={'image':ids.to('mps'),'image_cond':torch.empty(0,4,device='mps'),'text':tids.to('mps')},timesteps={'image':torch.full((ni,),.5,device='mps',dtype=torch.bfloat16),'image_cond':torch.empty(0,device='mps',dtype=torch.bfloat16),'text':torch.zeros(n,device='mps',dtype=torch.bfloat16)},mode_indices={'image':[(0,ni)],'image_cond':[(0,0)],'text':[(0,n)]})
# The source computes RoPE in float64; MPS does not support float64. Compute the
# identical positional tables on CPU, then copy to MPS. All learned ops stay BF16.
original_rope=dit.rope_embed
def cpu_rope(ids): return original_rope(ids.cpu()).to(ids.device)
dit.rope_embed=cpu_rope
with torch.inference_mode(),torch.autocast('mps',dtype=torch.bfloat16): y=model(packed,timesteps_ref=torch.tensor([.25],device='mps'))
result['velocity']=y[None].float().cpu().contiguous()
print('velocity',y.shape,flush=True)
del model,packed,data,y;gc.collect();torch.mps.empty_cache()
ae=module('flux3_oracle_ae',ref/'autoencoder.py')
with torch.device('meta'): model=ae.AutoEncoder()
model.load_state_dict(load_file(str(root/'ae.safetensors')),assign=True)
model=model.to('mps',torch.float32).eval()
with torch.inference_mode(): y=model.decode(x.to('mps'),ae.latent_ids(h,w,'mps'))
result['pixels']=y.permute(1,2,0)[None].cpu().contiguous()
print('pixels',y.shape,flush=True)
save_file(result,str(out/'stages.safetensors'))
print('DONE',flush=True)
