"""Validate immutable per-window LIBERO latents consumed in larger DiT batches.

The encoder/cache builder and original validation source remain byte-identical.
Only the consumer grouping changes; no VAE encoding is performed during SFT.
"""
import json
import os
from pathlib import Path

from omegaconf import OmegaConf, open_dict
from .libero_robot_rgb_pair import PairedRobotRGBCachedLiberoDataset as CachedLiberoDataset
from .robodojo_latent_cache import sha256_file


def validate_vav_cache_consumer(cfg):
    os.environ.pop('OPENWAM_VALIDATED_LIBERO_LATENT_CACHE', None)
    root = OmegaConf.select(cfg, 'dataloader.libero_latent_cache_dir')
    if not root or cfg.training.mixed_precision != 'bf16' or cfg.dataloader.type != 'libero_cached':
        raise ValueError('VAV requires the pinned native BF16 per-window LIBERO cache')
    ds = CachedLiberoDataset.from_config(cfg.dataloader, split='train')
    manifest = ds._latent_cache.manifest
    if manifest.get('execution_mode') != 'native_training_vae':
        raise ValueError('VAV requires the native training VAE cache')
    batch = int(cfg.training.batch_size)
    if batch < 1:
        raise ValueError('Invalid VAV consumer batch')
    digest = sha256_file(Path(root)/'manifest.json')
    receipt_path = Path(__file__).resolve().parents[2]/'rebatched_cache_validation.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'passed' or receipt.get('manifest_sha256') != digest:
        raise ValueError('VAV regrouping requires a matching cache equivalence receipt')
    if batch not in receipt.get('validated_consumer_batch_sizes', []):
        raise ValueError('VAV consumer batch grouping has not been validated')
    os.environ['OPENWAM_VALIDATED_LIBERO_LATENT_CACHE'] = str(Path(root).resolve())
    with open_dict(cfg.project):
        cfg.project.vae_cache_directory = str(root)
        cfg.project.vae_cache_manifest_sha256 = digest
        cfg.project.vae_cache_weights_sha256 = manifest['vae_weights_sha256']
        cfg.project.vae_cache_encoding_batch_size = int(manifest['batch_size'])
        cfg.project.vae_cache_consumer_batch_size = batch
        cfg.project.vae_cache_consumer_validation = str(receipt_path)
        robot = ds._robot_dataset._latent_cache
        paired_path=Path(__import__('os').environ.get('LIBERO_PAIRED_CACHE_PROOF', str(Path(__file__).resolve().parents[3]/'paired_cache_grouping.json')))
        paired=json.loads(paired_path.read_text())
        if (paired.get('status')!='passed' or not paired.get('both_streams_bitwise_equal')
                or paired.get('scene_manifest_sha256')!=digest
                or paired.get('robot_manifest_sha256')!=sha256_file(robot.root/'manifest.json')
                or batch not in paired.get('validated_consumer_batch_sizes',[])):
            raise ValueError('Both RGB streams require matching cached regrouping proof')
        cfg.project.robot_rgb_cache_directory = str(robot.root)
        cfg.project.robot_rgb_cache_manifest_sha256 = sha256_file(robot.root/'manifest.json')
        cfg.project.robot_rgb_cache_vae_weights_sha256 = robot.manifest['vae_weights_sha256']
