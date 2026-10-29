"""Two bounded runs on the same idle GPUs; equal samples and parameter updates."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin/python"
BASE = Path("/mnt/vepfs01/output/tiecheng.guo/robodojo_performance_20261020")
child = None


def stop(signum, _frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
    raise SystemExit(128 + signum)


def main():
    global child
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--case", choices=("pair", "b8_overlap"), default="pair")
    args = parser.parse_args()
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    assert gpu_ids and len(set(gpu_ids)) == len(gpu_ids)
    out = BASE / args.tag
    out.mkdir(parents=True, exist_ok=False)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    (out / "launcher.json").write_text(json.dumps(
        dict(pid=os.getpid(), gpus=gpu_ids, unix_start=time.time()), indent=2))
    cases = [
        ("b8_acc4_no_recompute", 8, 4, False, False, "", False),
        ("b32_acc1_current", 32, 1, True, True, "110", False),
    ]
    if args.case == "b8_overlap":
        cases = [("b8_acc4_no_recompute_overlap", 8, 4, False, False, "", True)]
    for name, batch, accum, block_gc, attn_gc, pattern, overlap in cases:
        current = subprocess.check_output(
            ["nvidia-smi", "--id=" + args.gpus, "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        assert len(current.splitlines()) == len(gpu_ids)
        assert all(int(line.split(",")[1]) < 1024 for line in current.splitlines()), current
        path = out / name
        path.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, NPROC_PER_NODE=str(len(gpu_ids)),
                   NNODES="1", NODE_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29633",
                   WANDB_MODE="disabled", PYTHONPATH=str(ROOT),
                   OPENWAM_TRAIN_ENTRY="scripts/profile_robodojo_accumulation.py",
                   OPENWAM_PROFILE_DIR=str(path), OPENWAM_BENCH_ACCUMULATION=str(accum),
                   OPENWAM_CHECKPOINT_EVERY_N_LAYER="1", OPENWAM_CHECKPOINT_PATTERN=pattern,
                   OPENWAM_EXACT_VAE_NORM="divide_affine", OPENWAM_ZERO_OVERLAP="1" if overlap else "0")
        command = [
            "bash", "scripts/train_robodojo_prepared.sh",
            f"training.batch_size={batch}", f"training.gradient_accumulation_steps={accum}",
            f"training.max_steps={24*accum}", "training.save_steps=0",
            "training.save_full_states_for_resume=false", "training.lr_scheduler=null",
            f"training.use_gradient_checkpointing={str(block_gc).lower()}",
            f"model.architecture.mot_checkpoint_mixed_attn={str(attn_gc).lower()}",
            f"training.output_path={path/'training'}",
            f"project.wandb.run_name=robodojo_accum_{args.tag}_{name}",
        ]
        receipt = dict(
            status="running", case=name, batch_size_per_gpu=batch, accumulation=accum,
            gpu_ids=gpu_ids, gpus=len(gpu_ids), effective_global_batch=batch*accum*len(gpu_ids),
            parameter_updates=24, micro_steps=24*accum, clean_updates=[5, 20], phase_updates=[21, 24],
            block_checkpointing=block_gc, mixed_attention_checkpointing=attn_gc,
            checkpoint_pattern=pattern, exact_vae_norm="divide_affine",
            zero_overlap=overlap, command=command, unix_start=time.time(),
        )
        (path/"run.json").write_text(json.dumps(receipt, indent=2))
        print("Starting", name, flush=True)
        with (path/"train.log").open("w") as log, (path/"gpu_samples.jsonl").open("w") as telemetry:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt["pid"] = child.pid
            (path/"run.json").write_text(json.dumps(receipt, indent=2))
            while child.poll() is None:
                try:
                    reading = subprocess.check_output(
                        ["nvidia-smi", "--id=" + args.gpus,
                         "--query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm",
                         "--format=csv,noheader,nounits"], text=True, timeout=10)
                    telemetry.write(json.dumps(dict(unix_time=time.time(), rows=reading.splitlines()))+"\n")
                    telemetry.flush()
                except (OSError, subprocess.SubprocessError) as error:
                    telemetry.write(json.dumps(dict(error=str(error)))+"\n")
                if time.time() - receipt["unix_start"] > 900:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    receipt["error"] = "15-minute per-case deadline"
                    break
                time.sleep(1)
        receipt.update(status="completed" if child.returncode == 0 else "failed",
                       exit_code=child.returncode, unix_end=time.time())
        (path/"run.json").write_text(json.dumps(receipt, indent=2))
        print(name, receipt["status"], flush=True)
        if child.returncode != 0:
            raise SystemExit(child.returncode)
        child = None


if __name__ == "__main__":
    main()
