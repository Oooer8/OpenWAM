"""Two-update functional probe: real full32 and epoch-tail8 on all eight ranks."""
import os

from openwam.dataloader.vav_aligned_batch_sampler import AlignedVAVBatchSampler
import profile_libero_vav_bounded as profile


original_iter = AlignedVAVBatchSampler.__iter__


def only_full_and_tail(self):
    batches = list(original_iter(self))
    assert len(batches) == 8552
    assert all(len(b) == 32 for b in batches[:8])
    assert all(len(b) == 8 for b in batches[-8:])
    yield from batches[:8]
    yield from batches[-8:]


if __name__ == '__main__':
    # Keep authoritative loader length/LR horizon; skip intermediate indices
    # before dataset IO. This probe is never a formal training entry point.
    assert os.environ['OPENWAM_VAV_PROFILE_STOP_STEP'] == '2'
    AlignedVAVBatchSampler.__iter__ = only_full_and_tail
    profile.main()
