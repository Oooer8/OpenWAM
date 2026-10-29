"""Supervise finite preparation -> verification -> four optimizer-step smoke test."""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from prepare_robodojo_openwam import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/mnt/vepfs01/output/tiecheng.guo/RoboDojo_openwam_prepared_v1")
LOGS = ROOT / "work/robodojo_adapter"
SMOKE = ROOT / "outputs/robodojo_adapter_smoke"


def run(stage, command, env):
    atomic_json(OUT / "supervision.json", dict(status="running", stage=stage, unix_time=time.time()))
    print("START", stage, flush=True)
    with open(LOGS / (stage + ".log"), "w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("PASS", stage, flush=True)


def main():
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    for _ in range(240):  # bounded two-hour deadline, not an unattended recurring job
        required = ["videos_complete.json", "repairs_complete.json", "raw_reference_checks.json",
                    "text_cache/validation.json"]
        if all((OUT / x).exists() for x in required):
            break
        time.sleep(30)
    else:
        raise TimeoutError("Offline inputs did not complete within two hours")
    run("assembly", [sys.executable, "scripts/assemble_robodojo_cache.py", "--prepared", str(OUT), "--workers", "35"], env)
    run("validation", [sys.executable, "scripts/validate_robodojo_prepared.py", "--prepared", str(OUT)], env)
    assert json.loads((OUT / "validation.json").read_text())["status"] == "passed"
    assert json.loads((OUT / "manifest.json").read_text())["status"] == "ready"
    # Read-only resource check before using only idle GPUs. GPU 0 belongs to
    # another process and is intentionally excluded from this finite test.
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                      "--format=csv,noheader,nounits"], text=True)
    memory = {int(line.split(",")[0]): int(line.split(",")[1]) for line in result.splitlines()}
    if any(memory[i] > 1024 for i in (1, 2, 3, 4)):
        raise RuntimeError("Smoke-test GPUs are occupied; data validation is complete")
    if list(SMOKE.rglob("metrics.jsonl")):
        raise RuntimeError("Smoke output already exists; select a fresh run rather than mixing metrics")
    env.update(CUDA_VISIBLE_DEVICES="1,2,3,4", NPROC_PER_NODE="4", WANDB_MODE="disabled",
               NNODES="1", NODE_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29631")
    run("training_smoke", ["bash", "scripts/train_robodojo_prepared.sh",
                          "training.max_steps=4", "training.save_steps=0",
                          f"training.output_path={SMOKE}", "project.wandb.run_name=robodojo_adapter_smoke"], env)
    finalize_smoke()


def finalize_smoke():
    metric_files = list(SMOKE.rglob("metrics.jsonl"))
    assert len(metric_files) == 1, f"Expected exactly one smoke run, got {metric_files}"
    rows = [json.loads(line) for line in metric_files[0].read_text().splitlines()]
    assert len(rows) == 4 and rows[-1]["optimizer_step"] == 4
    for r in rows:
        assert all(math.isfinite(float(r[k])) for k in ("loss_total", "loss_action", "loss_video", "grad_norm"))
        assert r["grad_norm"] > 0
    result = dict(status="passed", gpus=[1, 2, 3, 4], optimizer_steps=4, batch_size_per_gpu=4,
                  max_steps_override=4, checkpoints_saved=False, metrics_path=str(metric_files[0]), metrics=rows)
    atomic_json(OUT / "training_smoke.json", result)
    atomic_json(OUT / "supervision.json", dict(status="passed", stage="complete", unix_time=time.time()))
    print("ALL_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        atomic_json(OUT / "supervision.json", dict(status="failed", error=repr(e), unix_time=time.time()))
        raise
