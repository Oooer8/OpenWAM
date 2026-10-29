"""Verify cached vectors do not depend on the training lookup batch size."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from openwam.model.video_backbone.wan.text_cache import FrozenTextCache


cache = FrozenTextCache('work/cosmos3_text_cache')
encoder = torch.nn.Linear(1, 1).eval().requires_grad_(False)
prompts = sorted(cache.index, key=lambda p: cache.index[p][2])[-16:]
full, full_lengths = cache.encode(prompts, text_encoder=encoder, device='cpu')
checks = []
for batch in (4, 8, 16):
    values, lengths = [], []
    for start in range(0, 16, batch):
        value, length = cache.encode(prompts[start:start+batch], text_encoder=encoder, device='cpu', compact=True)
        values.append(F.pad(value, (0, 0, 0, 512-value.shape[1])))
        lengths.append(length)
    torch.testing.assert_close(torch.cat(values), full, rtol=0, atol=0)
    assert torch.equal(torch.cat(lengths), full_lengths)
    checks.append({'lookup_batch':batch, 'equality':'bitwise BF16'})
result = {'status':'passed', 'offline_t5_batch':cache.manifest['batch_size'], 'checks':checks}
Path('work/setup_logs/cache_lookup_batch_check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
