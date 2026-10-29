"""Optional operation-level activation retention within a layer checkpoint."""
from functools import partial

import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


_EXPENSIVE = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
              torch.ops.aten.bmm.default}


def matmul_policy(ctx, op, *args, **kwargs):
    # Keep expensive projection results; cheap pointwise operations may be
    # recomputed. This never caches parameters, random choices or module state.
    return (CheckpointPolicy.MUST_SAVE if op in _EXPENSIVE
            else CheckpointPolicy.PREFER_RECOMPUTE)


matmul_contexts = partial(create_selective_checkpoint_contexts, matmul_policy)
