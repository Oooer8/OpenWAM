"""Finite accumulation benchmark: timings grouped by real parameter update."""
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

rank = int(os.environ["RANK"])
accumulation = int(os.environ["OPENWAM_BENCH_ACCUMULATION"])
out = Path(os.environ["OPENWAM_PROFILE_DIR"])
out.mkdir(parents=True, exist_ok=True)
micro_step = 0
active = False
phase = {}
previous_end = None
begin = None


def sync():
    torch.cuda.synchronize()


def timed(target, attr, key):
    original = getattr(target, attr)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        if not active:
            return original(*args, **kwargs)
        sync()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        sync()
        phase[key] = phase.get(key, 0.) + time.perf_counter() - start
        return result

    setattr(target, attr, wrapper)


for target, attr, key in [
    (BaseWAMArchitecture, "prepare_inputs", "prepare_inputs_seconds"),
    (BaseWAMArchitecture, "compute_loss", "forward_loss_seconds"),
    (encode, "encode_video", "vae_encoding_seconds"),
    (DeepSpeedEngine, "backward", "backward_and_gradient_communication_seconds"),
    (DeepSpeedEngine, "step", "optimizer_and_parameter_sync_seconds"),
]:
    timed(target, attr, key)

original_model_step = DeepSpeedEngine._take_model_step


@functools.wraps(original_model_step)
def model_step(self, *args, **kwargs):
    result = original_model_step(self, *args, **kwargs)
    self._bench_actual_updates = getattr(self, "_bench_actual_updates", 0) + 1
    return result


DeepSpeedEngine._take_model_step = model_step
original_loss = trainer_module.OpenWAMTrainer.compute_loss


@functools.wraps(original_loss)
def loss(self, batch):
    global micro_step, active, phase, begin
    micro_step += 1
    update = (micro_step - 1) // accumulation + 1
    active = 21 <= update <= 24
    if active:
        sync()
    now = time.perf_counter()
    phase = dict(
        micro_step=micro_step, update_group=update,
        data_wait_and_loop_seconds=None if previous_end is None else now - previous_end,
        instrumented=active, unix_start=time.time(),
    )
    begin = now
    return original_loss(self, batch)


trainer_module.OpenWAMTrainer.compute_loss = loss
original_log = trainer_module.OpenWAMTrainer.log_step


@functools.wraps(original_log)
def log(self, **kwargs):
    global active, previous_end
    if active:
        sync()
    actual_updates = getattr(self.architecture, "_bench_actual_updates", 0)
    expected = micro_step // accumulation
    assert kwargs["opt_step"] == expected, (micro_step, kwargs["opt_step"], expected)
    assert actual_updates == expected, (micro_step, actual_updates, expected)
    assert self.architecture.gradient_accumulation_steps() == accumulation
    phase.update(
        rank=rank, optimizer_step=kwargs["opt_step"], actual_parameter_updates=actual_updates,
        compute_wall_seconds=time.perf_counter() - begin,
        trainer_micro_step_seconds=1. / kwargs["steps_per_sec"],
        unix_end=time.time(), peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    original_log(self, **kwargs)
    with (out / f"phases_rank{rank}.jsonl").open("a") as f:
        f.write(json.dumps(phase) + "\n")
    active = False
    previous_end = time.perf_counter()


trainer_module.OpenWAMTrainer.log_step = log

import train

train.main()
