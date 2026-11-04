"""Development-only same-weight MFLUX reference capture. Requires mflux and safetensors."""
from pathlib import Path
import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.common.config import ModelConfig
import argparse
parser=argparse.ArgumentParser(description='Capture external Klein 9B stage fixtures. Run alone on the GPU.')
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.model
out=args.output;out.mkdir(parents=True,exist_ok=True)
prompt='Studio photograph of a blue ceramic vase holding three yellow tulips on a pale stone table, soft window light, plain cream background.'
(out/'prompt.txt').write_text(prompt)
model=Flux2Klein(model_path=str(root),quantize=4,model_config=ModelConfig.flux2_klein_9b())
enc,txt,_,_=model._encode_prompt_pair(prompt=prompt,negative_prompt=None,guidance=1.0)
mx.eval(enc);print('text',enc.shape,enc.dtype,flush=True)
h,w=4,6
latent=mx.array(np.linspace(-1,1,h*w*128,dtype=np.float32).reshape(1,h*w,128)).astype(mx.bfloat16)
ids=mx.array([[0,r,c,0] for r in range(h) for c in range(w)])[None]
y=model.transformer(hidden_states=latent,encoder_hidden_states=enc,timestep=mx.array([500.0]),img_ids=ids,txt_ids=txt,guidance=None)
mx.eval(y);print('velocity',y.shape,y.dtype,flush=True)
save_file({k:np.asarray(v.astype(mx.float32)) for k,v in {'latent':latent,'text':enc,'velocity':y}.items()},str(out/'stages.safetensors'))
print('DONE',flush=True)
