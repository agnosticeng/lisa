"""Independent diagnostic oracle; usage: capture.py MODEL CHUNK[:TAIL] OUTPUT.npz [TOKENS.json]."""
import importlib.metadata
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
from mlx_vlm.utils import load_model
from transformers import AutoTokenizer

assert importlib.metadata.version('mlx') == '0.32.4'
assert importlib.metadata.version('mlx-vlm') == '0.7.11'
path, schedule, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
parts = [int(value) for value in schedule.split(':')]
chunk, tail = parts[0], parts[1] if len(parts) > 1 else 0
model = load_model(path).language_model
tokenizer = AutoTokenizer.from_pretrained(path)
if len(sys.argv) > 4:
    tokens = json.loads(Path(sys.argv[4]).read_text())
else:
    tokens = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': 'You are a helpful assistant.'},
         {'role': 'user', 'content': 'What is the capital of France?'}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_dict=False,
    )
assert chunk > 0 and 0 <= tail <= len(tokens) and len(parts) <= 2
Path(str(output) + '.tokens.json').write_text(json.dumps(tokens))
cache, result = model.make_cache(), {}


def save(name, tensor):
    result[name] = np.array(tensor.astype(mx.float32))


prefix = len(tokens) - tail
batches = [(i, min(chunk, prefix - i)) for i in range(0, prefix, chunk)]
batches += [(i, 1) for i in range(prefix, len(tokens))]
for offset, length in batches:
    hidden = model.model.embed_tokens(mx.array([tokens[offset:offset + length]]))
    save(f'p{offset}.embed', hidden)
    for index, (layer, state) in enumerate(zip(model.layers, cache)):
        window = model.config.sliding_window if layer.layer_type == 'sliding_attention' else None
        mask = (create_attention_mask(hidden, state, window_size=window)
                if layer.is_attention_layer else create_ssm_mask(hidden, state))
        hidden = layer(hidden, mask, state)
        save(f'p{offset}.layer{index}', hidden)
        if not layer.is_attention_layer:
            save(f'p{offset}.layer{index}.state0', state[0])
    hidden = model.model.embedding_norm(hidden)
logits = model.model.embed_tokens.as_linear(hidden[:, -1, :])
save('logits', logits)
np.savez(output, **result)
print('tokens', len(tokens), 'argmax', int(mx.argmax(logits)),
      tokenizer.decode([int(mx.argmax(logits))]), 'peak GiB', mx.get_peak_memory() / 2**30)
