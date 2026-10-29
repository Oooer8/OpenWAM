"""Launch a bounded throughput/phase benchmark on explicitly selected idle GPUs."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/mnt/vepfs01/output/tiecheng.guo/robodojo_performance_20261020")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--checkpoint-every", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = p.parse_args()
    out = BASE / args.tag
    out.mkdir(parents=True, exist_ok=False)
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    assert gpu_ids and len(gpu_ids) == len(set(gpu_ids)) and min(gpu_ids) >= 0
    gpu_state = subprocess.check_output(
        ["nvidia-smi", f"--id={args.gpus}", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    assert len(gpu_state.splitlines()) == len(gpu_ids)
    assert all(int(line.split(",")[1]) < 1024 for line in gpu_state.splitlines()), gpu_state
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=args.gpus,
        NPROC_PER_NODE=str(len(gpu_ids)), NNODES="1", NODE_RANK="0",
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29632", WANDB_MODE="disabled",
        OPENWAM_TRAIN_ENTRY="scripts/profile_robodojo_training.py",
        OPENWAM_PROFILE_DIR=str(out),
        OPENWAM_CHECKPOINT_EVERY_N_LAYER=str(args.checkpoint_every),
        PYTHONPATH=str(ROOT),
    )
    command = [
        "bash", "scripts/train_robodojo_prepared.sh",
        f"training.batch_size={args.batch}", "training.max_steps=40",
        "training.save_steps=0", "training.save_full_states_for_resume=false",
        "training.use_gradient_checkpointing=true", "training.lr_scheduler=null",
        f"training.output_path={out / 'training'}",
        f"project.wandb.run_name=robodojo_speed_{args.tag}",
    ]
    started = time.time()
    receipt = dict(status="running", batch_size_per_gpu=args.batch, gpus=len(gpu_ids),
                   gpu_ids=gpu_ids, global_batch=args.batch*len(gpu_ids),
                   gradient_checkpointing=True, checkpoint_every=args.checkpoint_every,
                   checkpoint_pattern=env.get("OPENWAM_CHECKPOINT_PATTERN", ""),
                   exact_vae_norm=env.get("OPENWAM_EXACT_VAE_NORM", "none"),
                   command=command, unix_start=started, baseline_steps=[6, 20],
                   phase_steps=[21, 32], trace_step=34)
    (out / "run.json").write_text(json.dumps(receipt, indent=2))
    with (out / "train.log").open("w") as log, (out / "gpu_samples.jsonl").open("w") as samples:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            try:
                reading = subprocess.check_output(
                    ["nvidia-smi", f"--id={args.gpus}", "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm",
                     "--format=csv,noheader,nounits"], text=True, timeout=10,
                )
                samples.write(json.dumps(dict(unix_time=time.time(), rows=reading.splitlines())) + "\n")
                samples.flush()
            except (subprocess.SubprocessError, OSError) as e:
                samples.write(json.dumps(dict(unix_time=time.time(), error=repr(e))) + "\n")
            if time.time() - started > 1200:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                receipt["error"] = "20-minute benchmark deadline exceeded"
                break
            time.sleep(1)
    receipt.update(status="completed" if process.returncode == 0 else "failed",
                   exit_code=process.returncode, unix_end=time.time())
    (out / "run.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
