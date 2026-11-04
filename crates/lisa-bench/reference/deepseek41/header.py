"""Read GGUF metadata for the pinned reference package."""
import struct
def header(p):
 f=p.open('rb')
 def num(c):return struct.unpack('<'+c,f.read(struct.calcsize('<'+c)))[0]
 def string():return f.read(num('Q')).decode()
 def val(t):
  if t==8:return string()
  if t==9:
   k,n=num('I'),num('Q');return [val(k) for _ in range(n)]
  return num({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t])
 assert f.read(4)==b'GGUF' and num('I')==3
 nt,nm=num('Q'),num('Q');meta={}
 for _ in range(nm):key=string();meta[key]=val(num('I'))
 ts=[]
 for _ in range(nt):
  name=string();dims=[num('Q') for _ in range(num('I'))];kind=num('I');offset=num('Q');ts.append(dict(name=name,dims=dims,kind=kind,offset=offset))
 align=meta.get('general.alignment',32);base=(f.tell()+align-1)//align*align
 for t in ts:t['offset']+=base
 return dict(path=str(p),size=p.stat().st_size,metadata=meta,tensors=ts)
