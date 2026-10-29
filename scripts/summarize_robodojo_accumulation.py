"""Summarize equal-work accumulation runs; never compare raw micro-step rates."""
import argparse
import copy
import json
import math
import statistics
from pathlib import Path

import yaml


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(path):
    run = json.loads((path/"run.json").read_text())
    assert run["status"] == "completed", run
    accum, updates = run["accumulation"], run["parameter_updates"]
    phases = [rows(path/f"phases_rank{rank}.jsonl") for rank in range(run["gpus"])]
    for rank_rows in phases:
        assert len(rank_rows) == updates * accum
        assert [r["micro_step"] for r in rank_rows] == list(range(1, updates*accum+1))
        assert all(r["actual_parameter_updates"] == r["micro_step"]//accum
                   and r["optimizer_step"] == r["micro_step"]//accum for r in rank_rows)
    clean = [
        max(sum(r["trainer_micro_step_seconds"] for r in rank_rows[(u-1)*accum:u*accum])
            for rank_rows in phases)
        for u in range(5, 21)
    ]
    metric_path = next((path/"training").rglob("metrics.jsonl"))
    metrics = rows(metric_path)
    assert len(metrics) == updates*accum
    assert all(math.isfinite(r[k]) for r in metrics for k in
               ("loss_total", "loss_video", "loss_action", "grad_norm"))
    assert all(r["optimizer_step"] == r["micro_step"]//accum for r in metrics)
    config = yaml.safe_load((metric_path.parent/"config.yaml").read_text())
    assert config["training"]["batch_size"] == run["batch_size_per_gpu"]
    assert config["training"]["gradient_accumulation_steps"] == accum
    assert config["training"]["use_gradient_checkpointing"] == run["block_checkpointing"]
    assert config["model"]["architecture"]["mot_checkpoint_mixed_attn"] == run["mixed_attention_checkpointing"]
    assert config["project"]["runtime_optimizations"]["exact_vae_norm"] == "divide_affine"
    assert config["project"]["runtime_optimizations"]["zero_overlap_comm"] == run["zero_overlap"]
    phase_names = ("vae_encoding_seconds", "prepare_inputs_seconds", "forward_loss_seconds",
                   "backward_and_gradient_communication_seconds", "optimizer_and_parameter_sync_seconds")
    phase_seconds = {k: statistics.mean(
        sum(r.get(k, 0.) for r in phases[0][(u-1)*accum:u*accum])
        for u in range(21, 25)) for k in phase_names}
    start, end = phases[0][4*accum]["unix_start"], phases[0][20*accum-1]["unix_end"]
    telemetry = {}
    for sample in rows(path/"gpu_samples.jsonl"):
        if start <= sample.get("unix_time", 0.) <= end:
            for row in sample["rows"]:
                parts = [float(x) for x in row.split(",")]
                telemetry.setdefault(str(int(parts[0])), []).append(parts[1:])
    gpu_summary = {
        gpu: dict(utilization_mean=statistics.mean(r[0] for r in readings),
                  memory_mib_max=max(r[1] for r in readings),
                  power_watts_mean=statistics.mean(r[2] for r in readings),
                  clock_mhz_mean=statistics.mean(r[3] for r in readings))
        for gpu, readings in telemetry.items()
    }
    result = dict(
        status="passed", source=str(path), run=run, metrics_path=str(metric_path),
        steady_updates=[5, 20],
        seconds_per_parameter_update=dict(mean=statistics.mean(clean), median=statistics.median(clean),
                                          min=min(clean), max=max(clean), values=clean),
        samples_per_second=run["effective_global_batch"]/statistics.mean(clean),
        peak_allocated_gib=max(r["peak_allocated_gib"] for rr in phases for r in rr),
        peak_reserved_gib=max(r["peak_reserved_gib"] for rr in phases for r in rr),
        synchronized_phase_seconds_per_update=phase_seconds,
        actual_update_counts_verified=True, all_losses_and_grad_norms_finite=True,
        gpu_telemetry=gpu_summary,
    )
    (path/"summary.json").write_text(json.dumps(result, indent=2))
    selected = copy.deepcopy({k: config[k] for k in ("model", "training", "dataloader")})
    for key in ("batch_size", "gradient_accumulation_steps", "max_steps", "output_path",
                "use_gradient_checkpointing"):
        selected["training"].pop(key)
    selected["model"]["architecture"].pop("mot_checkpoint_mixed_attn")
    selected["seed"] = config["project"]["seed"]
    return result, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--overlap-root", type=Path)
    args = parser.parse_args()
    small, config8 = summarize(args.directory/"b8_acc4_no_recompute")
    large, config32 = summarize(args.directory/"b32_acc1_current")
    assert config8 == config32, "Unexpected configuration drift"
    assert small["run"]["effective_global_batch"] == large["run"]["effective_global_batch"]
    assert small["run"]["gpu_ids"] == large["run"]["gpu_ids"]
    report = dict(
        status="passed", equal_work_verified=True, gpus=small["run"]["gpus"],
        effective_global_batch=small["run"]["effective_global_batch"],
        parameter_updates_per_case=24, total_samples_per_case=24*small["run"]["effective_global_batch"],
        batch8_accum4=small, batch32_accum1=large,
        batch8_throughput_change_percent=100*(small["samples_per_second"]/large["samples_per_second"]-1),
        batch8_update_time_change_percent=100*(small["seconds_per_parameter_update"]["mean"]/
                                              large["seconds_per_parameter_update"]["mean"]-1),
        winner="batch8_accum4" if small["samples_per_second"]>large["samples_per_second"] else "batch32_accum1",
        note="Same GPUs, data configuration, effective batch, LR and update count. Batch8 disables all activation recomputation; batch32 checkpoints 2/3 of blocks and mixed attention on retained blocks. Both use exact VAE fusion. Clean timings exclude initialization, warmup and synchronized phase profiling. Changing microbatch grouping/RNG/reduction order does not imply bitwise-identical training or prove policy-quality equivalence.",
    )
    if args.overlap_root:
        overlap, config_overlap = summarize(args.overlap_root/"b8_acc4_no_recompute_overlap")
        assert config_overlap == config32
        assert overlap["run"]["gpu_ids"] == large["run"]["gpu_ids"]
        assert overlap["run"]["effective_global_batch"] == large["run"]["effective_global_batch"]
        report["batch8_accum4_overlap"] = overlap
        report["overlap_batch8_throughput_change_vs_batch32_percent"] = (
            100*(overlap["samples_per_second"]/large["samples_per_second"]-1))
        candidates = {"batch8_accum4": small, "batch32_accum1": large,
                      "batch8_accum4_overlap": overlap}
        report["winner_including_overlap"] = max(candidates, key=lambda k: candidates[k]["samples_per_second"])
    (args.directory/"comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in
                     ("batch8_accum4","batch32_accum1","batch8_accum4_overlap")},indent=2))
    for name, result in (("batch8_accum4", small), ("batch32_accum1", large)):
        print(name, json.dumps({k:result[k] for k in (
            "seconds_per_parameter_update", "samples_per_second", "peak_allocated_gib",
            "synchronized_phase_seconds_per_update")}))


if __name__ == "__main__":
    main()
