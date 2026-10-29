"""VAV batches with the same epoch-tail coverage as the immutable mutual run."""
import math
from itertools import cycle, islice


class AlignedVAVBatchSampler:
    """Global256 full updates, final update padded only to baseline micro-global32.

    Accelerate shards these batches by rank with even_batches=False. Every
    update still supplies one batch to each of8 ranks; its final local batch
    can be smaller than32, matching baseline4x8/GA8 partial-epoch coverage.
    """
    batch_size = None
    drop_last = False

    def __init__(self, sampler, *, micro_batch=32, world_size=8, baseline_micro_batch=4):
        self.sampler = sampler
        self.micro_batch = micro_batch
        self.world_size = world_size
        self.baseline_micro_batch = baseline_micro_batch
        self.global_batch = micro_batch * world_size
        assert micro_batch * world_size == 256 and world_size in (8, 32) and baseline_micro_batch == 4

    def __len__(self):
        return math.ceil(len(self.sampler) / self.global_batch) * self.world_size

    def __iter__(self):
        indices = list(self.sampler)
        complete = len(indices) // self.global_batch * self.global_batch
        for start in range(0, complete, self.micro_batch):
            yield indices[start:start + self.micro_batch]
        tail = indices[complete:]
        if not tail:
            return
        baseline_global_micro = self.baseline_micro_batch * 8  # Preserve the original 32-window tail quantum.
        padded = math.ceil(len(tail) / baseline_global_micro) * baseline_global_micro
        tail += list(islice(cycle(indices), padded - len(tail)))
        local_size = padded // self.world_size
        for rank in range(self.world_size):
            yield tail[rank * local_size:(rank + 1) * local_size]
