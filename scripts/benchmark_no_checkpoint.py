"""Normal timing followed by masked-padding-only longest-context memory stress.

The final 12 batches retain their original prompts and valid embeddings. Only
universally masked padding is extended to the dataset maximum context width.
This entry point is for benchmarking, never for the formal training launcher.
"""
import os
import runpy

import torch.nn.functional as F
from openwam.model.video_backbone.wan import encode


original = encode.compact_masked_text_context
calls = 0
normal_steps = int(os.environ.get('OPENWAM_MEMORY_STRESS_AFTER_STEP', '36'))


def compact_with_memory_stress(context, lengths):
    global calls
    calls += 1
    result = original(context, lengths)
    if calls > normal_steps:
        if int(lengths.max()) > 128 or result.shape[1] > 128:
            raise ValueError('Dataset maximum context length changed; update memory stress bound')
        result = F.pad(result, (0, 0, 0, 128 - result.shape[1]))
    return result


encode.compact_masked_text_context = compact_with_memory_stress
runpy.run_path('scripts/train.py', run_name='__main__')
