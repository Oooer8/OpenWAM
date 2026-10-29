"""Validate strict equality on real augmented RoboDojo clips and benchmark."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors import safe_open

from openwam.dataloader.robodojo_prepared import PreparedRoboDojoDataset
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38
from openwam.model.video_backbone.wan.encode import preprocess_video_batch
from openwam.train.utils.robodojo_exact_optimizations import install_exact_vae_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    vae = WanVideoVAE38()
    path = "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors"
    prefix = "video_backbone.vae."
    with safe_open(path, framework="pt", device="cpu") as f:
        state = {key[len(prefix):]: f.get_tensor(key) for key in f.keys() if key.startswith(prefix)}
    vae.load_state_dict(state, strict=True, assign=True)
    del state
    vae.cuda().bfloat16().eval().requires_grad_(False)
    originals = [(m, m.forward) for m in vae.model.encoder.modules()]
    ds = PreparedRoboDojoDataset(yaml.safe_load(Path("configs/dataloader/robodojo_prepared.yaml").read_text()), split="train")
    indices = np.random.default_rng(514).integers(0, len(ds), 64).tolist()
    result = {"status": "running", "real_augmented_clips": 64, "batch_size": 32, "modes": {}}
    for batch in range(2):
        random.seed(91 + batch)
        np.random.seed(91 + batch)
        torch.manual_seed(91 + batch)
        samples = [ds[i]["video"] for i in indices[batch*32:(batch+1)*32]]
        pixels = preprocess_video_batch(samples, dtype=torch.bfloat16, device="cuda")
        reference = None
        for mode in ("baseline", "affine", "divide_affine", "baseline_recheck"):
            for module, original in originals:
                module.forward = original
            if mode in ("affine", "divide_affine"):
                count = install_exact_vae_norms(vae, divide=mode == "divide_affine")
                result["modified_norm_modules"] = count
            times = []
            with torch.no_grad():
                for _ in range(5):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    encoded = vae.batch_encode(pixels, device="cuda")
                    torch.cuda.synchronize()
                    times.append(time.perf_counter()-start)
            if mode == "baseline":
                reference = encoded.clone()
            delta = (reference.float() - encoded.float())
            row = dict(batch=batch, exact_equal=torch.equal(reference, encoded),
                       changed_elements=int((reference != encoded).sum()),
                       total_elements=encoded.numel(), max_abs=float(delta.abs().max()),
                       relative_l2=float(delta.norm()/reference.float().norm()),
                       median_seconds=statistics.median(times[2:]), seconds=times)
            result["modes"].setdefault(mode, []).append(row)
            print(mode, json.dumps(row), flush=True)
            output.write_text(json.dumps(result, indent=2))
        del pixels, samples, reference, encoded
    result["accepted_modes"] = [mode for mode in ("affine", "divide_affine")
                               if all(row["exact_equal"] for row in result["modes"][mode])]
    assert all(row["exact_equal"] for row in result["modes"]["baseline_recheck"])
    result["status"] = "passed" if result["accepted_modes"] else "rejected"
    output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
