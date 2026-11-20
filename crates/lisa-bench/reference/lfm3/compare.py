"""Compare a native capture with immutable MLX tensor digests, without loading a model."""
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

CONV = {0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}

def digest(path):
    arrays = np.load(path)
    keys = []
    for key in arrays.files:
        if key == 'logits' or re.fullmatch(r'p\d+\.(embed|layer\d+)', key):
            keys.append(key)
        elif (m := re.fullmatch(r'p\d+\.layer(\d+)\.state0', key)) and int(m[1]) in CONV:
            keys.append(key)
    h = hashlib.sha256()
    for key in sorted(keys):
        a = np.asarray(arrays[key], dtype='<f4')
        h.update(key.encode() + b'\0')
        h.update(json.dumps(list(a.shape), separators=(',', ':')).encode() + b'\0')
        h.update(a.tobytes())
    return h.hexdigest(), len(keys)

if __name__ == '__main__':
    fixture = json.loads(Path(__file__).with_name('goldens.json').read_text())['cases'][sys.argv[1]]
    actual, count = digest(sys.argv[2])
    assert (actual, count) == (fixture['sha256'], fixture['tensors']), (actual, count, fixture)
    print(f'{sys.argv[1]}: {count} tensors match MLX exactly')
