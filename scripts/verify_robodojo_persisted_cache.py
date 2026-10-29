"""Check persisted cache against native VAE; then validate published reader."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from cache_robodojo_vae import atomic_json, load_state, make_dataset
from openwam.dataloader.robodojo_latent_cache import SHAPE, sha256_file, state_digest
from openwam.dataloader.robodojo_prepared import PreparedRoboDojoDataset
from openwam.model.video_backbone.wan.encode import preprocess_video_batch
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38

root = Path("/mnt/vepfs01/output/tiecheng.guo/RoboDojo_openwam_vae_cache_v1")
build = json.loads((root / "build.json").read_text())
config = yaml.safe_load(Path("configs/dataloader/robodojo_prepared.yaml").read_text())
ds = make_dataset(config)
ds._open()
torch.set_num_threads(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.set_per_process_memory_fraction(16 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
state = load_state(build["checkpoint"])
assert state_digest(state) == build["vae_weights_sha256"]
vae = WanVideoVAE38()
vae.load_state_dict(state, strict=True, assign=True)
del state
vae.cuda().bfloat16().eval().requires_grad_(False)
indices = [0, len(ds) - 1]
for length in (2, 4, 5, 9, 17, 32, 33):
    found = np.flatnonzero(ds._windows[:, 2] == length)
    if len(found):
        indices.append(int(found[len(found) // 2]))
indices += np.random.default_rng(20261021).integers(0, len(ds), 32 - len(indices)).tolist()
with torch.inference_mode():
    for start in range(0, len(indices), 16):
        chosen = indices[start:start + 16]
        pixels = preprocess_video_batch([ds[i]["video"] for i in chosen],
                                        dtype=torch.bfloat16, device="cuda")
        native = vae.batch_encode(pixels, device="cuda").cpu()
        for row, index in enumerate(chosen):
            shard_start = index // build["chunk_size"] * build["chunk_size"]
            receipt = json.loads((root / f"{shard_start:08d}.json").read_text())
            data = np.memmap(root / receipt["path"], mode="r", dtype=np.uint16,
                             shape=(receipt["end"] - shard_start, *SHAPE))
            persisted = torch.from_numpy(np.array(data[index - shard_start], copy=True)).view(torch.bfloat16)
            assert torch.equal(native[row], persisted), f"Native/cache mismatch at {index}"
            assert torch.isfinite(persisted).all()
            del data
        print("NATIVE_PERSISTED_EQUAL", chosen, flush=True)
        del native, pixels
del vae
torch.cuda.empty_cache()
deadline = time.monotonic() + 1800
while not (root / "validation.json").exists():
    assert time.monotonic() < deadline, "Finalizer did not publish within 30 minutes"
    time.sleep(5)
validation = json.loads((root / "validation.json").read_text())
assert validation["status"] == "passed" and validation["all_shards_sha256_checked"]
cached = PreparedRoboDojoDataset(dict(config, color_jitter={"enabled": False}, latent_cache_dir=str(root)))
for index in indices:
    online, sample = ds[index], cached[index]
    for key in ("action", "proprio", "action_mask", "video_mask", "proprio_mask"):
        assert torch.equal(online[key], sample[key]), (index, key)
    assert online["prompt"] == sample["prompt"] and sample["video"] == []
boundary_samples = 0
for shard in cached._latent_cache.manifest["shards"]:
    for index in (shard["start"], shard["end"] - 1):
        assert torch.isfinite(cached._latent_cache[index]).all(), index
        boundary_samples += 1
report = dict(status="passed", unix_time=time.time(), windows=len(ds),
              shards=validation["shards"], all_shards_sha256_checked=True,
              manifest_sha256=sha256_file(root / "manifest.json"),
              native_persisted_bitwise_equal=True, native_checked_indices=indices,
              actions_masks_prompts_equal=True, finite_boundary_samples=boundary_samples)
atomic_json(root / "post_completion_validation.json", report)
print(json.dumps(report), flush=True)
