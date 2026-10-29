# OpenWAM VAV

Private source snapshot of the OpenWAM VAV mutual-attention experiment.

## Source

The root `openwam/` tree is the isolated source used by `libero-mutual-robot-rgb-32gpu-fix-20261026`, based on OpenWAM upstream commit `1e103e1`. It includes VAV mutual routing, robot RGB conditioning, latent cache support, and training runtime changes. See [the upstream README](README.upstream.md) for installation and upstream documentation. The original Apache-2.0 license is retained.

- `openwam/model/architectures/dual_system/vav.py`: VAV architecture.
- `openwam/model/architectures/dual_system/vav_driver.py`: routed mutual attention.
- `openwam/model/architectures/dual_system/vav_attention.py`: attention implementation.
- `experiments/libero_mutual_robot_rgb_32gpu/`: original four-node/eight-GPU-per-node training launcher, configuration and validation helpers.
- `evaluation/libero_plus/`: archived Plus supervisor/model server and evaluation protocol.

## Runtime notes

This is a source archive, not a self-contained runtime image. Training/evaluation configuration preserves original server paths and requires the original assets, cached data, environments and benchmark dependencies. Adjust paths before using it on another machine. No training is started by this upload.

For the archived training package, `source` points back to this repository root. The launcher requires four nodes and eight 80-GB-class GPUs per node, with reachable `MASTER_ADDR` and unique node ranks. It starts a fresh run; review its checks before executing.

Weights, datasets, logs, run state, virtual environments, credentials and temporary files are not included. `UPLOAD_MANIFEST.json` records file hashes for this source snapshot.
