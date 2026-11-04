#!/usr/bin/env python3
"""Strictly compare a LISA NPZ capture to an adapted DS4 capture directory."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('lisa', type=Path)
p.add_argument('reference', type=Path)
p.add_argument('--logits-only', action='store_true')
p.add_argument('--output', type=Path)
args = p.parse_args()
if args.output and args.output.exists():
    raise SystemExit('refusing to overwrite evidence')
a = np.load(args.lisa)
keys = [k for k in a if not args.logits_only or k.endswith('.logits')]
if not keys:
    raise SystemExit('no capture arrays')
missing, mismatch = [], []
for key in keys:
    path = args.reference / (key + '.bin')
    if not path.is_file():
        missing.append(key)
        continue
    actual = a[key].reshape(-1)
    expected = np.fromfile(path, dtype='<i4' if key.endswith('ffn.router') else '<f4').astype('f4')
    if actual.shape != expected.shape:
        mismatch.append({'key': key, 'shape': [actual.size, expected.size]})
        continue
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        mismatch.append({'key': key, 'error': 'nonfinite'})
        continue
    count = int(np.count_nonzero(actual.view('u4') != expected.view('u4')))
    if count:
        mismatch.append({'key': key, 'different': count, 'count': actual.size,
                         'max_abs': float(np.max(np.abs(actual - expected)))})
report = {'arrays': len(keys), 'matched': len(keys)-len(missing)-len(mismatch),
          'missing': missing, 'mismatches': mismatch}
text = json.dumps(report, indent=2) + '\n'
if args.output:
    with args.output.open('x') as f:
        f.write(text)
print(text)
raise SystemExit(bool(missing or mismatch))
