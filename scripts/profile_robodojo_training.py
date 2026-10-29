"""Finite benchmark entrypoint with separate clean and instrumented step ranges.

Steps 1-20 have no added CUDA synchronization. Steps 21-32 measure synchronized
phases on all ranks. Step 34 captures a rank-zero trace and is excluded from
throughput estimates. Nested prepare_inputs timings must not be summed twice.
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

rank = int(os.environ.get("RANK", 0))
out = Path(os.environ["OPENWAM_PROFILE_DIR"])
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
        start = time.monotonic()
        with torch.profiler.record_function(label):
            result = original(*args, **kwargs)
        sync()
        phase[label] = phase.get(label, 0.0) + time.monotonic() - start
        return result

    setattr(target, attr, wrapped)


for target, attr, label in [
    (BaseWAMArchitecture, "prepare_inputs", "prepare_inputs_seconds"),
    (BaseWAMArchitecture, "compute_loss", "forward_loss_seconds"),
    (encode, "encode_text", "cached_text_seconds"),
    (encode, "preprocess_video", "video_preprocess_seconds"),
    (encode, "preprocess_video_batch", "video_preprocess_seconds"),
    (encode, "encode_video", "vae_encoding_seconds"),
    (DeepSpeedEngine, "backward", "backward_and_gradient_communication_seconds"),
    (DeepSpeedEngine, "step", "optimizer_and_parameter_sync_seconds"),
    (trainer_module, "reduce_step_metrics", "metric_reduce_seconds"),
]:
    timed(target, attr, label)

original_loss = trainer_module.OpenWAMTrainer.compute_loss


@functools.wraps(original_loss)
def loss(self, batch):
    global active, phase, step_begin, step_number, profiler
    step_number += 1
    active = 21 <= step_number <= 32 or step_number == 34
    if active:
        sync()
    now = time.monotonic()
    phase = {
        "data_wait_and_loop_seconds": None if previous_end is None else now - previous_end,
        "instrumented": active,
        "unix_start": time.time(),
    }
    step_begin = now
    if rank == 0 and step_number == 34:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        )
        profiler.start()
    return original_loss(self, batch)


trainer_module.OpenWAMTrainer.compute_loss = loss
original_log = trainer_module.OpenWAMTrainer.log_step


@functools.wraps(original_log)
def log(self, **kwargs):
    global active, previous_end, profiler
    if active:
        sync()
    phase.update(
        compute_wall_seconds=time.monotonic() - step_begin,
        rank=rank, step=kwargs["global_step"], profiler_active=profiler is not None,
        trainer_step_seconds=1.0 / kwargs["steps_per_sec"], unix_end=time.time(),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    original_log(self, **kwargs)
    active = False
    if profiler is not None:
        profiler.stop()
        trace_path = out / "rank0_step34_trace.json"
        profiler.export_chrome_trace(str(trace_path))
        kernels = {}
        for event in json.loads(trace_path.read_text())["traceEvents"]:
            if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and event.get("ph") == "X":
                name = event["name"]
                kernels[name] = kernels.get(name, 0.0) + event["dur"]
        summary = sorted(
            [{"name": name, "cuda_milliseconds": us / 1000} for name, us in kernels.items()],
            key=lambda x: -x["cuda_milliseconds"],
        )
        (out / "rank0_step34_kernels.json").write_text(json.dumps(summary, indent=2))
        profiler = None
    with (out / f"phases_rank{rank}.jsonl").open("a") as f:
        f.write(json.dumps(phase) + "\n")
    previous_end = time.monotonic()


trainer_module.OpenWAMTrainer.log_step = log

import train

train.main()
