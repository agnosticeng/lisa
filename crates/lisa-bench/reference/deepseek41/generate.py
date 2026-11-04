"""Format-only adapters and mapping manifest; executed by build.py after pin checks."""
from pathlib import Path
import json, re, math, hashlib, subprocess
code=(src/'ds4.c').read_text()
def replace_body(code,sig,body):
 start=code.index(sig);begin=code.index('{',start);end=code.index('\n}',begin)+2
 return code[:begin]+'{\n'+body+'\n}'+code[end:]
# Only unavailable formats are adapted. DS4's native graph and other kernels stay original.
proto='''int adapter_mv(ds4_gpu_tensor *,const void *,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,const ds4_gpu_tensor *);
int adapter_expert(ds4_gpu_tensor *,const void *,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,const ds4_gpu_tensor *,const ds4_gpu_tensor *,uint32_t);
int adapter_activation(ds4_gpu_tensor *,ds4_gpu_tensor *,ds4_gpu_tensor *,ds4_gpu_tensor *);
int adapter_sum(ds4_gpu_tensor *,ds4_gpu_tensor *);
'''
pos=code.index('static bool ds41_matmul(');code=code[:pos]+proto+code[pos:]
for sig in ['static bool ds41_matmul(', 'static bool ds41_matmul_batch(']:
 begin=code.index('{',code.index(sig))+1
 extra='''\n    if (weight->type == 13 || weight->type == 30) {
        if (!adapter_mv(out,m->map,m->size,weight->abs_offset,weight->type,weight->dim[0],weight->dim[1],in)) return false;
        return !round || ds41_bf16(out,weight->dim[1]);
    }\n'''
 code=code[:begin]+extra+code[begin:]
code=replace_body(code,'static bool ds41_attention_low(','''
    if (g->tp_world != 1 || l->attn_output_a->type != 13) return false;
    if (!transformed && rope_il >= 0 && !ds4_gpu_dsv41_rope_bf16(g->heads,512,64,1,g->pos,
        ds4_layer_compress_ratio(rope_il)!=0,true)) return false;
    for (uint32_t i=0;i<8;i++) {
        ds4_gpu_tensor *x=ds4_gpu_tensor_view(g->heads,i*4096ull*4,4096ull*4);
        ds4_gpu_tensor *y=ds4_gpu_tensor_view(g->low,i*1024ull*4,1024ull*4);
        bool ok=x && y && ds41_matmul_rows(y,m,l->attn_output_a,x,i*1024,1024);
        ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);if(!ok)return false;
    }
    return true;''')
# Preserve the original router/shared expert/HC execution. Replace only the unsupported routed projection call.
start=code.index('    routed_ok = ds4_gpu_routed_moe_one_tensor',code.index('static bool ds41_moe_partial('))
end=code.index(';',start)+1
code=code[:start]+'''    routed_ok = adapter_expert(g->gate,m->map,m->size,l->ffn_gate_exps->abs_offset,l->ffn_gate_exps->type,5120,2304,g->norm,selected,1) &&
        adapter_expert(g->up,m->map,m->size,l->ffn_up_exps->abs_offset,l->ffn_up_exps->type,5120,2304,g->norm,selected,1) &&
        adapter_activation(g->mid,g->gate,g->up,(ds4_gpu_tensor *)weights) &&
        adapter_expert(g->experts,m->map,m->size,l->ffn_down_exps->abs_offset,l->ffn_down_exps->type,2304,5120,g->mid,selected,6) &&
        adapter_sum(routed,g->experts);'''+code[end:]
# The original row-byte helper excludes IQ1_M and IQ2_XS.
start=code.index('{',code.index('static DS4_MAYBE_UNUSED uint64_t routed_expert_row_bytes('))+1
code=code[:start]+'\n    if(t->type==17 || t->type==29) return t->dim[0]/256*(t->type==17?74:56);\n'+code[start:]
# DS4's built-in embedding assumes its own checkpoint's payload format.
# Decode exactly one original GGUF row with upstream GGML, then repeat it.
code=replace_body(code,'static bool ds41_embed(','''
    (void)g; (void)scratch; (void)pos;
    extern void dequantize_row_q8_0(const void *,float *,int64_t);
    extern void dequantize_row_q5_K(const void *,float *,int64_t);
    const ds4_tensor *t=w->token_embd;
    if(token<0 || (uint32_t)token>=t->dim[1]) return false;
    float values[20480]; const void *row=m->map+t->abs_offset+(uint64_t)token*t->bytes/t->dim[1];
    if(t->type==13) dequantize_row_q5_K(row,values,5120);
    else if(t->type==8) dequantize_row_q8_0(row,values,5120);
    else return false;
    for(unsigned i=1;i<4;i++)memcpy(values+i*5120,values,5120*4);
    return ds4_gpu_tensor_write(out,0,values,sizeof(values));''')
(out/'ds4-adapter.c').write_text(code)
# Generate a same-byte virtual mapping manifest; no model payload is rewritten.
page=16384;base=0;ts=[];shards=[]
for s in a[:-1]:
 size=(s['size']+page-1)//page*page;shards.append((s['path'],base,size));
 for t in s['tensors']:
  t=dict(t);t['offset']+=base;ts.append(t)
 base+=size
lines=['static const struct {const char *path;uint64_t base,size;} shards[]={']
lines+=['{'+json.dumps(p)+f',{b}ull,{n}ull'+'},' for p,b,n in shards];lines+=['};',f'static const uint64_t resident_bytes={base}ull;','static ds4_tensor tensors[]={']
block={0:(1,4),8:(32,34),13:(256,176),30:(1,2),29:(256,56),16:(256,66),17:(256,74)}
for t in ts:
 n=math.prod(t['dims']);q,b=block[t['kind']];dims=','.join(map(str,t['dims']))
 lines.append('{.name={'+json.dumps(t['name'])+f',{len(t["name"])}'+'},'+f'.ndim={len(t["dims"])},.dim={{{dims}}},.type={t["kind"]},.abs_offset={t["offset"]}ull,.elements={n}ull,.bytes={n//q*b}ull'+'},')
lines+=['};','static void bind_weights(ds4_model *m,ds4_weights *w){']
fields=set(re.findall(r'ds4_tensor \*(\w+);',code[code.index('typedef struct {\n    ds4_tensor *hc_attn_fn;'):code.index('} ds4_layer_weights;')]))
for t in ts:
 name=t['name']
 if name.startswith('blk.'):
  _,l,suffix=name.split('.',2); field=suffix.removesuffix('.weight').replace('.','_')
  field={'exp_probs_b_bias':'ffn_exp_probs_b','exp_probs_b_vl_bias':'ffn_exp_probs_vl'}.get(field,field)
  if field in fields:lines.append(f'w->layer[{l}].{field}=model_find_tensor(m,{json.dumps(name)});')
 elif name in ['token_embd.weight','output.weight','output_norm.weight']:lines.append(f'w->{name[:-7]}=model_find_tensor(m,{json.dumps(name)});')
lines+=['}'];meta=a[0]['metadata'];tm=meta['deepseek41.engram.token_map'];pr=meta['deepseek41.ple.head_vocab_sizes'];mult=meta['deepseek41.ple.layer_multipliers']
import struct
(out/'token_map.bin').write_bytes(struct.pack('<'+'I'*len(tm),*tm))
lines += ['static ds4_engram_layout engram={'+f'.vocab_size={len(tm)},.compressed_vocab_size={max(tm)+1},.pad_id={meta["deepseek41.engram.pad_token_id"]},'+'.rows={'+f'{sum(pr[:24])},{sum(pr[24:])}'+'},.primes={{'+','.join(map(str,pr[:24]))+'},{'+','.join(map(str,pr[24:]))+'}},.multipliers={{'+','.join(str(v)+'ull' for v in mult[:4])+'},{'+','.join(str(v)+'ull' for v in mult[4:])+'}}};']
lines += ['static const char *engram_path='+json.dumps(a[-1]['path'])+';',f'static const uint64_t engram_offset={a[-1]["tensors"][0]["offset"]}ull;']
# Restore scratch macro used verbatim by DS4's own graph allocation.
beg=code.index('#define DS41_SCRATCH(X)');end=code.index('\n\n',beg);lines.append(code[beg:end])
(out/'manifest.h').write_text('\n'.join(lines))
# Native GGML extension source: unchanged, already independently checked against the GGML C decoders.
(out/'packed.metal').write_bytes((repo/'crates/lisa-mlx/src/shaders/deepseek41/iq.metal').read_bytes())
print('Prepared',len(ts),'tensor mappings',base/2**30,'GiB')
