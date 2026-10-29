"""Summarize finite benchmark records without mixing profiled/clean timings."""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def describe(values):
    return {
        "mean": statistics.mean(values), "median": statistics.median(values),
        "p95": float(np.quantile(values, .95)), "min": min(values), "max": max(values),
    }


def summarize(root):
    run = json.loads((root / "run.json").read_text())
    assert run["status"] == "completed", run
    phases = {rank: load_rows(root / f"phases_rank{rank}.jsonl") for rank in range(run["gpus"])}
    for rank, rows in phases.items():
        assert [r["step"] for r in rows] == list(range(1, 41)), (rank, len(rows))
    clean_steps = range(6, 21)
    clean = [max(phases[rank][s-1]["trainer_step_seconds"] for rank in phases) for s in clean_steps]
    metric_files = list((root / "training").rglob("metrics.jsonl"))
    assert len(metric_files) == 1
    metrics = load_rows(metric_files[0])
    assert len(metrics) == 40
    assert all(math.isfinite(r[k]) for r in metrics for k in
               ("loss_total", "loss_action", "loss_video", "grad_norm"))
    categories = [
        "data_wait_and_loop_seconds", "prepare_inputs_seconds", "forward_loss_seconds",
        "backward_and_gradient_communication_seconds", "optimizer_and_parameter_sync_seconds",
        "metric_reduce_seconds",
    ]
    instrumented = [phases[0][s-1] for s in range(21, 33)]
    # Phases on rank 0 include waits for other ranks in collective operations.
    phase_means = {name: statistics.mean(r.get(name, 0.) for r in instrumented) for name in categories}
    wall = statistics.mean(r["compute_wall_seconds"] + r["data_wait_and_loop_seconds"] for r in instrumented)
    phase_means["other_loop_seconds"] = wall - sum(phase_means.values())
    nested = {name: statistics.mean(r.get(name, 0.) for r in instrumented) for name in
              ("cached_text_seconds", "video_preprocess_seconds", "vae_encoding_seconds")}
    start, end = phases[0][5]["unix_start"], phases[0][19]["unix_end"]
    gpu_readings = [[] for _ in phases]
    gpu_ids = run.get("gpu_ids", list(phases))
    for reading in load_rows(root / "gpu_samples.jsonl"):
        if start <= reading["unix_time"] <= end:
            for row in reading.get("rows", []):
                vals = [float(v.strip()) for v in row.split(",")]
                gpu_readings[gpu_ids.index(int(vals[0]))].append(vals[1:])
    gpu = {}
    for rank, values in enumerate(gpu_readings):
        if values:
            a = np.asarray(values)
            gpu[rank] = dict(samples=len(a), gpu_util_mean=float(a[:, 0].mean()),
                             memory_util_mean=float(a[:, 1].mean()), memory_used_mib_max=float(a[:, 2].max()),
                             power_watts_mean=float(a[:, 3].mean()), sm_clock_mhz_mean=float(a[:, 4].mean()))
    # CUDA user annotations include inclusive device times: do not count them
    # again alongside real kernels (including NCCL's annotation + kernel).
    trace = json.loads((root / "rank0_step34_trace.json").read_text())
    kernel_totals = {}
    for event in trace["traceEvents"]:
        if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and event.get("ph") == "X":
            name = event["name"]
            kernel_totals[name] = kernel_totals.get(name, 0.) + event["dur"] / 1000
    kernels = sorted(
        [dict(name=name, cuda_milliseconds=ms) for name, ms in kernel_totals.items()],
        key=lambda x: -x["cuda_milliseconds"],
    )
    (root / "rank0_step34_kernels_verified.json").write_text(json.dumps(kernels, indent=2))
    groups = {}
    for kernel in kernels:
        name = kernel["name"].lower()
        if "nccl" in name:
            group = "nccl_communication"
        elif any(x in name for x in ("flash", "fmha", "attention")):
            group = "attention"
        elif any(x in name for x in ("conv", "cudnn")):
            group = "convolution"
        elif any(x in name for x in ("gemm", "cutlass", "matmul", "xmma")):
            group = "matrix_multiplication"
        elif "memcpy" in name or "memset" in name:
            group = "memory_copy_set"
        else:
            group = "other_elementwise_reduction"
        groups[group] = groups.get(group, 0.) + kernel["cuda_milliseconds"]
    mean_s = statistics.mean(clean)
    result = dict(
        status="passed", source_directory=str(root), run=run,
        steady_unprofiled_steps=[6, 20], steady_step_seconds=describe(clean),
        throughput_samples_per_second=run["global_batch"] / mean_s,
        peak_allocated_gib=max(r["peak_allocated_gib"] for rows in phases.values() for r in rows),
        peak_reserved_gib=max(r["peak_reserved_gib"] for rows in phases.values() for r in rows),
        initialization_to_first_batch_seconds=phases[0][0]["unix_start"] - run["unix_start"],
        instrumented_step_seconds=wall, phase_mean_seconds=phase_means,
        phase_percent={k: 100*v/wall for k, v in phase_means.items()},
        prepare_inputs_nested_mean_seconds=nested, gpu_during_clean_steps=gpu,
        kernel_group_accumulated_ms=groups, top_kernels=kernels[:30],
        metric_file=str(metric_files[0]),
        estimated_60000_steps_hours=60000 * mean_s / 3600,
        note="Steady timing excludes warmup and profiling. Phase timers synchronize CUDA and perturb overlap. "
             "Kernel times may overlap across streams. No checkpoint saving or evaluation included. "
             "Different checkpoint intervals are different memory/recompute tradeoffs.",
    )
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    print(json.dumps({k: v for k, v in result.items() if k != "top_kernels"}, indent=2))
