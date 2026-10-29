"""Opt-in full-scale checkpoint drill; inactive during ordinary training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def _digest(value):
    h = hashlib.sha256()
    tensor_count = 0
    tensor_bytes = 0

    def visit(item):
        nonlocal tensor_count, tensor_bytes
        if torch.is_tensor(item):
            tensor_count += 1
            tensor_bytes += item.numel() * item.element_size()
            h.update(str((str(item.dtype), tuple(item.shape))).encode())
            raw = item.detach().contiguous().reshape(-1).view(torch.uint8)
            for start in range(0, raw.numel(), 32 * 1024 * 1024):
                h.update(raw[start:start + 32 * 1024 * 1024].cpu().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str((str(item.dtype), item.shape)).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            h.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        else:
            h.update((type(item).__name__ + ':' + repr(item)).encode())

    visit(value)
    return dict(sha256=h.hexdigest(), tensor_count=tensor_count, tensor_bytes=tensor_bytes)


def capture(accelerator):
    engine = accelerator._models[0]
    assert hasattr(engine, 'optimizer') and hasattr(engine, 'module')
    result = {
        'model': _digest(engine.module.state_dict()),
        'optimizer': _digest(engine.optimizer.state_dict()),
        'schedulers': _digest([s.state_dict() for s in accelerator._schedulers]),
        'learning_rates': [float(g['lr']) for g in engine.optimizer.param_groups],
        'engine_global_steps': int(engine.global_steps),
        'python_rng': _digest(random.getstate()),
        'numpy_rng': _digest(np.random.get_state()),
        'torch_cpu_rng': _digest(torch.get_rng_state()),
        'torch_cuda_rng': _digest(torch.cuda.get_rng_state_all()),
    }
    assert result['optimizer']['tensor_bytes'] > 0
    return result


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def save_validation_checkpoint(accelerator, architecture, output_path, global_step, opt_step, epoch):
    from openwam.train.utils.checkpointing import save_weights, save_full_state

    rank = accelerator.process_index
    receipt_dir = Path(output_path) / 'checkpoint_validation'
    if accelerator.is_main_process:
        receipt_dir.mkdir(exist_ok=False)
    accelerator.wait_for_everyone()
    started = time.time()
    if accelerator.is_main_process:
        print('[checkpoint-validation] Capturing model/optimizer/scheduler/RNG identity before full-state save.',flush=True)
    before = capture(accelerator)
    _atomic(receipt_dir / f'before_save_rank_{rank}.json', before)
    accelerator.wait_for_everyone()
    save_weights(accelerator, architecture, output_path, global_step, final=False)
    save_full_state(accelerator, output_path, global_step, opt_step, epoch)
    accelerator.wait_for_everyone()
    _atomic(receipt_dir / f'save_completed_rank_{rank}.json', {
        'rank':rank, 'global_step':global_step, 'opt_step':opt_step,
        'elapsed_seconds':time.time() - started,
        'completed_at_shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    })
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'[checkpoint-validation] Complete checkpoint at step {global_step}; all ranks exiting normally for actual restore test.', flush=True)


def verify_loaded_checkpoint(accelerator, state_dir):
    if os.environ.get('OPENWAM_VERIFY_CHECKPOINT_RESTORE') != '1':
        return
    rank = accelerator.process_index
    receipt_dir = Path(state_dir).parent / 'checkpoint_validation'
    before = json.loads((receipt_dir / f'before_save_rank_{rank}.json').read_text())
    if accelerator.is_main_process:
        print('[checkpoint-validation] Comparing restored model/optimizer/scheduler/LR/steps/RNG identity.',flush=True)
    after = capture(accelerator)
    matches = {key: before[key] == after[key] for key in before}
    _atomic(receipt_dir / f'after_restore_rank_{rank}.json', {
        'rank':rank, 'restored_from':str(state_dir), 'state':after,
        'matches':matches, 'all_match':all(matches.values()),
        'verified_at_shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    })
    assert all(matches.values()), f'Checkpoint restore mismatch on rank {rank}: {matches}'
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print('[checkpoint-validation] All ranks restored identical model, optimizer, scheduler, LR, engine steps and RNG states.', flush=True)
