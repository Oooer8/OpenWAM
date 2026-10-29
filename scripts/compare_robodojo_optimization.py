"""Compare two completed finite runs with the same data/model/training settings."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from summarize_robodojo_speed import summarize, load_rows


def effective_config(root):
    files = list((root / "training").rglob("config.yaml"))
    assert len(files) == 1
    cfg = yaml.safe_load(files[0].read_text())
    selected = copy.deepcopy({k: cfg[k] for k in ("model", "dataloader", "training")})
    selected["training"].pop("output_path", None)
    selected["seed"] = cfg["project"]["seed"]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert effective_config(args.baseline) == effective_config(args.optimized), "Training configuration drift"
    summaries = [summarize(p) for p in (args.baseline, args.optimized)]
    base, fast = summaries
    assert base["run"]["gpus"] == fast["run"]["gpus"]
    assert base["run"]["global_batch"] == fast["run"]["global_batch"]
    metrics = [load_rows(Path(r["metric_file"])) for r in summaries]
    numerical = {}
    for key in ("loss_total", "loss_video", "loss_action", "grad_norm"):
        x, y = (np.array([row[key] for row in rows], dtype=np.float64) for rows in metrics)
        numerical[key] = dict(
            exact_equal=bool(np.array_equal(x, y)),
            baseline_first=float(x[0]), optimized_first=float(y[0]),
            max_absolute_difference=float(np.abs(x-y).max()),
            relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x), 1e-30)),
            baseline_mean=float(x.mean()), optimized_mean=float(y.mean()),
        )
    def compact(r):
        return {key: r[key] for key in ("source_directory", "steady_step_seconds",
                "throughput_samples_per_second", "peak_allocated_gib", "peak_reserved_gib",
                "prepare_inputs_nested_mean_seconds", "phase_mean_seconds")}
    report = dict(
        status="completed", same_model_data_training_config=True,
        gpus=base["run"]["gpus"], batch_per_gpu=base["run"]["batch_size_per_gpu"],
        baseline=compact(base), optimized=compact(fast),
        throughput_improvement_percent=100*(fast["throughput_samples_per_second"]/base["throughput_samples_per_second"]-1),
        step_duration_reduction_percent=100*(1-fast["steady_step_seconds"]["mean"]/base["steady_step_seconds"]["mean"]),
        training_metrics_comparison=numerical,
        note="40-step finite runs; clean steady timings use steps 6–20. These are numerical/runtime checks, not a downstream policy evaluation. CUDA training retains the original attention/reduction determinism settings.",
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
