"""Create an independent MLX affine 4-bit DSpine checkpoint without overwrites.

Requires mlx==0.32.4. Quantizes fc and all attention/MLP linear weights;
bridge, codebook, normalization and metadata tensors retain original bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve()
output = args.output.expanduser().resolve()
if output.exists():
    raise SystemExit(f'Refusing to overwrite {output}')
config = json.loads((source / 'config.json').read_text())
assert config['hidden_size'] == 5120 and config['num_hidden_layers'] == 5
assert not config.get('quantization'), 'Source must be the original BF16 checkpoint'
subprocess.run(['sysctl', 'hw.memsize'], check=True)
subprocess.run(['memory_pressure'], check=True)
import mlx.core as mx
assert mx.__version__ == '0.32.4', mx.__version__
weights = {}
source_inventory = {}
for shard in sorted(source.glob('*.safetensors')):
    source_inventory[shard.name] = {
        'bytes': shard.stat().st_size,
        'sha256': hashlib.file_digest(shard.open('rb'), 'sha256').hexdigest(),
    }
    loaded = mx.load(str(shard))
    assert not weights.keys() & loaded.keys(), 'Duplicate tensor names'
    weights.update(loaded)
converted = []
original_bytes = sum(w.nbytes for w in weights.values())
for name in list(weights):
    clean = name.removeprefix('model.')
    if clean == 'fc.weight' or (
        clean.startswith('layers.') and clean.endswith('_proj.weight')
    ):
        w = weights[name]
        assert w.ndim == 2 and w.dtype == mx.bfloat16 and w.shape[-1] % 64 == 0
        packed, scales, biases = mx.quantize(w, group_size=64, bits=4, mode='affine')
        mx.eval(packed, scales, biases)
        weights[name] = packed
        weights[name.removesuffix('.weight') + '.scales'] = scales
        weights[name.removesuffix('.weight') + '.biases'] = biases
        converted.append(name)
        print('quantized', name, flush=True)
assert len(converted) == 36, f'Expected 36 projections, got {len(converted)}'
mx.eval(weights)
config['quantization'] = {'group_size': 64, 'bits': 4, 'mode': 'affine'}
output.mkdir(parents=True)
mx.save_safetensors(str(output / 'model.safetensors'), weights, metadata={'format': 'mlx'})
(output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
manifest = {
    'source': str(source), 'source_inventory': source_inventory,
    'mlx': mx.__version__, 'quantization': config['quantization'],
    'quantized_tensors': converted,
    'preserved': 'All other tensors, including bridge/codebook/norms, retain their original dtype and payload',
    'source_tensor_bytes': original_bytes,
    'quantized_tensor_bytes': sum(w.nbytes for w in weights.values()),
    'weight_file_bytes': (output / 'model.safetensors').stat().st_size,
    'weight_file_sha256': hashlib.file_digest((output / 'model.safetensors').open('rb'), 'sha256').hexdigest(),
}
(output / 'quantization.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
