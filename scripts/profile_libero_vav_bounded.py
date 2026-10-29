"""Bounded VAV performance check; full 10-epoch LR horizon, no model saves."""
import json
import os
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf

# scripts/train.py supplies the same dataset, Accelerator and trainer entry.
import train
from openwam.train.openwam_trainer import OpenWAMTrainer


class ProfileComplete(Exception):
    pass


def main():
    root = Path(os.environ['OPENWAM_VAV_PROFILE_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    import openwam
    assert Path(openwam.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1], openwam.__file__
    rank = int(os.environ['RANK'])
    stop_step = int(os.environ.get('OPENWAM_VAV_PROFILE_STOP_STEP', '12'))
    cfg = OmegaConf.load(os.environ['OPENWAM_VAV_PROFILE_CONFIG'])
    # This direct entry does not run Hydra's ${now:...} resolver setup.
    # Use an explicit trial path and validate every interpolation before loading.
    cfg.project.output_dir = str(root / 'artifacts')
    cfg.training.output_path = str(root / 'artifacts')
    cfg.project.wandb = None
    OmegaConf.to_container(cfg, resolve=True)
    assert cfg.model.architecture.variant == 'vav'
    assert cfg.training.batch_size == 32 and cfg.training.gradient_accumulation_steps == 1
    assert cfg.training.num_epochs == 10 and cfg.training.max_steps is None
    assert int(os.environ['WORLD_SIZE']) == 8
    original_loss, original_log = OpenWAMTrainer.compute_loss, OpenWAMTrainer.log_step
    state = {}

    def loss(self, batch):
        torch.cuda.synchronize()
        state['start'] = time.monotonic()
        state['actual_batch_size'] = len(batch)
        return original_loss(self, batch)

    def log(self, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.monotonic() - state['start']
        original_log(self, **kwargs)
        row = dict(rank=rank, micro_step=kwargs['global_step'], optimizer_step=kwargs['opt_step'],
            elapsed_seconds=elapsed, actual_batch_size=state['actual_batch_size'], metrics=kwargs['metrics'], lr=kwargs['lr'],
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, unix_time=time.time())
        with (root/f'rank{rank}.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        if kwargs['global_step'] >= stop_step:
            raise ProfileComplete()

    OpenWAMTrainer.compute_loss, OpenWAMTrainer.log_step = loss, log
    try:
        train._train_openwam(cfg)
        # The checkpoint drill returns normally after all eight shards are saved.
        validation_step = int(os.environ.get('OPENWAM_VALIDATE_CHECKPOINT_STEP', '0'))
        assert validation_step > 0, 'Unexpected early return from bounded training'
        torch.distributed.barrier()
        (root/f'rank{rank}_completed.json').write_text(json.dumps(dict(status='checkpoint_saved',steps=validation_step,rank=rank)))
        torch.distributed.destroy_process_group()
    except ProfileComplete:
        torch.distributed.barrier()
        (root/f'rank{rank}_completed.json').write_text(json.dumps(dict(status='completed',steps=stop_step,rank=rank)))
        torch.distributed.destroy_process_group()
    except BaseException as exc:
        (root/f'rank{rank}_failure.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)[-2000:])))
        raise


if __name__ == '__main__':
    main()
