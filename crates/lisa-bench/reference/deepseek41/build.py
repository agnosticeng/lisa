#!/usr/bin/env python3
"""Build the pinned same-weight DS4 reference; never rewrites model payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from header import header

DS4 = 'b3f50180ec2c7324892be1f50bc83555ccb8d5e7'
GGML = '7071ba378afa96dcd69c41e1ec2b4912364dbee4'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--ds4', type=Path, required=True)
p.add_argument('--ggml', type=Path, required=True, help='llama.cpp checkout')
p.add_argument('--model', type=Path, required=True, help='first GGUF shard')
p.add_argument('--out', type=Path, required=True, help='new build directory')
args = p.parse_args()
src, ggml, out = (x.resolve() for x in (args.ds4, args.ggml, args.out))
# Hub snapshot filenames are symlinks; preserve the package directory layout.
first = args.model.absolute()
here = Path(__file__).resolve().parent
repo = here.parents[3]
for tree, revision in [(src, DS4), (ggml, GGML)]:
    actual = subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise SystemExit(f'{tree}: expected {revision}, got {actual}')
    if subprocess.check_output(['git', '-C', str(tree), 'diff', 'HEAD', '--'], text=True):
        raise SystemExit(f'{tree}: upstream tracked source must be clean')
out.mkdir(parents=True, exist_ok=False)
files = sorted(first.parent.glob('*-0000[1-5]-of-00006.gguf'))
if len(files) != 5 or files[0] != first:
    raise SystemExit('expected the five transformer shards and split.no=0 entry')
files += [first.parent.parent / 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf']
a = [header(path) for path in files]
if a[0]['metadata'].get('general.architecture') != 'deepseek41':
    raise SystemExit('expected DeepSeek V4.1')
(out / 'package.json').write_text(json.dumps(a))
exec(compile((here / 'generate.py').read_text(), str(here / 'generate.py'), 'exec'))
for name in ['adapter.m', 'reference.c']:
    shutil.copyfile(here / name, out / name)
(out / 'reference_config.h').write_text(
    '#define REFERENCE_PACKED ' + json.dumps(str(out / 'packed.metal')) + '\n' +
    '#define REFERENCE_TOKEN_MAP ' + json.dumps(str(out / 'token_map.bin')) + '\n')

# Original GGML CPU decoders are independent of LISA's Rust decoders.
quants = (ggml / 'ggml/src/ggml-quants.c').read_text()
def function(signature):
    begin = quants.index(signature)
    end = quants.index('\n}', begin) + 2
    return quants[begin:end] + '\n'
preamble = '''#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#define GGML_RESTRICT restrict
static float fp16(uint16_t x) { _Float16 h; memcpy(&h,&x,2); return (float)h; }
#define GGML_FP16_TO_FP32 fp16
'''
(out / 'quant.c').write_text(preamble + function('static inline void get_scale_min_k4(') +
    ''.join(function('void dequantize_row_' + name + '(') for name in ['q8_0', 'q5_K', 'iq4_nl']))

# Preserve DS4's asynchronous readers; adapt only the payload width and decoder.
engram_h = (src / 'ds4_engram.h').read_text().replace('DS4_ENGRAM_ROW_BYTES = 264', 'DS4_ENGRAM_ROW_BYTES = 144')
(out / 'ds4_engram.h').write_text(engram_h)
engram_c = (src / 'ds4_engram.c').read_text()
engram_c = engram_c.replace('    pthread_once(&engram_bf16_once, init_engram_bf16);', '')
begin = engram_c.index('        for (int j = 0; j < DS4_ENGRAM_DIM; j++) {', engram_c.index('bool ds4_engram_read('))
end = engram_c.index('\n        }', begin) + len('\n        }')
engram_c = engram_c[:begin] + '        dequantize_row_iq4_nl(raw, out + i * DS4_ENGRAM_DIM, DS4_ENGRAM_DIM);' + engram_c[end:]
engram_c = engram_c.replace('#include "ds4_engram.h"', '#include "ds4_engram.h"\nextern void dequantize_row_iq4_nl(const void *,float *,int64_t);')
(out / 'engram.c').write_text(engram_c)

# Clang 21 was observed to crash at O3 on the full DS4 C translation unit.
# O1 affects host orchestration, not the JIT Metal source or its compile flags.
commands = []
def run(command):
    commands.append(command)
    subprocess.run(command, check=True)
common = ['clang', '-ffast-math', '-mcpu=apple-m4', '-I', str(out), '-I', str(src)]
run(common + ['-O1', '-Wno-unused-function', '-c', str(out / 'reference.c'), '-o', str(out / 'reference.o')])
run(common + ['-O3', '-fobjc-arc', '-c', str(out / 'adapter.m'), '-o', str(out / 'adapter.o')])
run(common + ['-O3', '-I', str(ggml / 'ggml/src'), '-c', str(out / 'quant.c'), '-o', str(out / 'quant.o')])
run(common + ['-O3', '-c', str(out / 'engram.c'), '-o', str(out / 'engram.o')])
objects = [out / f'{name}.o' for name in ['reference', 'adapter', 'quant', 'engram']]
for name in ['ds4_image', 'ds4_distributed', 'ds4_tp', 'ds4_ssd', 'ds4_layer_pack']:
    obj = out / f'{name}.o'
    run(common + ['-O3', '-c', str(src / f'{name}.c'), '-o', str(obj)])
    objects.append(obj)
run(['clang', '-o', str(out / 'reference')] + list(map(str, objects)) +
    ['-lm', '-pthread', '-framework', 'Foundation', '-framework', 'Metal'])
manifest = {
    'ds4_revision': DS4, 'ggml_revision': GGML, 'commands': commands,
    'templates': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in here.iterdir() if p.is_file()},
    'generated': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.suffix in ['.c', '.m', '.metal', '.h']},
    'weights': [{'path': str(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in files],
}
(out / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Built {out / "reference"}; run with working directory {src}')
