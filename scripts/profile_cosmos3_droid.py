"""Diagnostic entrypoint: synchronized phase timing plus one rank-0 CUDA trace.

No production trainer changes. Timings include CPU work and communication
inside each phase; nested phases must not be summed twice.
"""
import functools
import json
import os
import time
from pathlib import Path

import torch
from deepspeed.runtime.engine import DeepSpeedEngine
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.video_backbone.wan import encode
from openwam.train import openwam_trainer as trainer_module

rank = int(os.environ.get('RANK', 0))
out = Path(os.environ['OPENWAM_PROFILE_DIR'])
out.mkdir(parents=True, exist_ok=True)
phase = {}
active = False
previous_end = None
step_begin = None
step_number = 0
profiler = None


def sync():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def timed(target, attr, label):
    original = getattr(target, attr)
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if not active:
            return original(*args, **kwargs)
        sync()
        begin = time.monotonic()
        result = original(*args, **kwargs)
        sync()
        phase[label] = phase.get(label, 0.0) + time.monotonic() - begin
        return result
    setattr(target, attr, wrapped)


for target, attr, label in [
    (BaseWAMArchitecture, 'prepare_inputs', 'prepare_inputs_seconds'),
    (BaseWAMArchitecture, 'compute_loss', 'model_forward_loss_seconds'),
    (encode, 'encode_text', 'text_encoding_seconds'),
    (encode, 'preprocess_video', 'video_pixel_preprocess_seconds'),
    (encode, 'preprocess_video_batch', 'video_pixel_preprocess_seconds'),
    (encode, 'encode_video', 'vae_encoding_seconds'),
    (DeepSpeedEngine, 'backward', 'backward_including_gradient_communication_seconds'),
    (DeepSpeedEngine, 'step', 'deepspeed_step_seconds'),
    (trainer_module, 'reduce_step_metrics', 'metric_reduce_seconds'),
]:
    timed(target, attr, label)

original_loss = trainer_module.OpenWAMTrainer.compute_loss
@functools.wraps(original_loss)
def loss(self, batch):
    global active, phase, previous_end, step_begin, step_number, profiler
    sync()
    now = time.monotonic()
    step_number += 1
    phase = {'data_wait_and_loop_seconds': None if previous_end is None else now - previous_end}
    step_begin = now
    if rank == 0 and step_number == 10:
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_stack=False)
        profiler.start()
    active = True
    return original_loss(self, batch)
trainer_module.OpenWAMTrainer.compute_loss = loss

original_log = trainer_module.OpenWAMTrainer.log_step
@functools.wraps(original_log)
def log(self, **kwargs):
    global active, previous_end, profiler
    sync()
    phase['compute_wall_seconds'] = time.monotonic() - step_begin
    phase.update(rank=rank, step=kwargs['global_step'], profiler_active=profiler is not None)
    original_log(self, **kwargs)
    active = False
    if profiler is not None:
        profiler.stop()
        profiler.export_chrome_trace(str(out / 'rank0_step10_trace.json'))
        kernels = {}
        for event in profiler.events():
            if str(event.device_type).endswith('CUDA'):
                us = float(event.device_time_total)
                kernels[event.name] = kernels.get(event.name, 0.0) + us
        summary = sorted(({'name': name, 'cuda_milliseconds': us/1000} for name, us in kernels.items()),
                         key=lambda x: -x['cuda_milliseconds'])
        (out / 'rank0_step10_kernels.json').write_text(json.dumps(summary, indent=2))
        profiler = None
    with (out / f'phases_rank{rank}.jsonl').open('a') as f:
        f.write(json.dumps(phase) + '\n')
    previous_end = time.monotonic()
trainer_module.OpenWAMTrainer.log_step = log

import train
train.main()
