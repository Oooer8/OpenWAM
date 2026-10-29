"""CPU integrity, boundary mask, no-decode, and model-input parity checks."""
import json
from pathlib import Path
from types import SimpleNamespace
import pickle

import numpy as np
import pytest
import torch

from test_robodojo_prepared import pair
from openwam.dataloader.robodojo_prepared import PreparedRoboDojoDataset
from openwam.dataloader.robodojo_latent_cache import LatentCache, SHAPE, sha256_file
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.video_backbone.wan_backbone import WanBase
from openwam.model.video_backbone.wan.variants.ti2v import TI2VVariant


def create_cache(tmp_path, cfg, length):
    root = tmp_path / 'latent'
    root.mkdir()
    values = torch.arange(length, dtype=torch.float32).bfloat16()[:, None, None, None, None].expand(length, *SHAPE).contiguous()
    shards = []
    for start, end in ((0, 7), (7, length)):
        path = root / f'{start}.bf16'
        values[start:end].view(torch.uint16).numpy().tofile(path)
        shards.append(dict(start=start, end=end, path=path.name, size=path.stat().st_size, sha256=sha256_file(path)))
    manifest = dict(status='ready', format=1, dtype='bfloat16_uint16_bits', shape=list(SHAPE), windows=length,
                    augmentation='none', encoding='independent_9_frame_windows',
                    prepared_manifest_sha256=sha256_file(Path(cfg['prepared_dir'])/'manifest.json'), shards=shards)
    publish(root, manifest)
    return root, values, manifest


def publish(root, manifest):
    (root/'manifest.json').write_text(json.dumps(manifest))
    (root/'validation.json').write_text(json.dumps(dict(status='passed', manifest_sha256=sha256_file(root/'manifest.json'))))


def test_all_windows_actions_masks_no_jpeg_decode(pair, tmp_path, monkeypatch):
    _, _, cfg = pair
    cfg = dict(cfg, color_jitter={'enabled': False})
    online = PreparedRoboDojoDataset(cfg)
    root, values, _ = create_cache(tmp_path, cfg, len(online))
    cached = PreparedRoboDojoDataset(dict(cfg, latent_cache_dir=str(root)))
    reference = [online[i] for i in range(len(online))]
    def no_decode(*args, **kwargs):
        raise AssertionError('Cached training must never decode images')
    monkeypatch.setattr('cv2.imdecode', no_decode)
    for i, a in enumerate(reference):
        b = cached[i]
        for key in ('action', 'proprio', 'action_mask', 'video_mask', 'proprio_mask'):
            assert torch.equal(a[key], b[key]), (i, key)
        assert a['prompt'] == b['prompt'] and b['video'] == []
        assert torch.equal(values[i], b['video_latents'])
    assert torch.equal(pickle.loads(pickle.dumps(cached))[-1]['video_latents'], values[-1])
    with pytest.raises(ValueError, match='color_jitter'):
        PreparedRoboDojoDataset(dict(cfg, latent_cache_dir=str(root), color_jitter={'enabled': True}))


def test_reject_partial_and_wrong_source(pair, tmp_path):
    _, ds, cfg = pair
    root, _, manifest = create_cache(tmp_path, cfg, len(ds))
    for key, bad in (('status', 'building'), ('shape', [1]), ('augmentation', 'jitter'),
                     ('prepared_manifest_sha256', 'bad'), ('windows', len(ds)-1)):
        publish(root, dict(manifest, **{key: bad}))
        with pytest.raises(ValueError):
            LatentCache(root, ds.root, len(ds))
    publish(root, manifest)
    manifest['shards'][1]['start'] += 1
    publish(root, manifest)
    with pytest.raises(ValueError, match='Gap/overlap'):
        LatentCache(root, ds.root, len(ds))
    manifest['shards'][1]['start'] -= 1
    publish(root, manifest)
    with (root/manifest['shards'][1]['path']).open('r+b') as f:
        f.truncate(2)
    with pytest.raises(ValueError, match='Truncated'):
        LatentCache(root, ds.root, len(ds))


def test_cached_native_conditioning_and_model_masks(pair, tmp_path, monkeypatch):
    _, ds, cfg = pair
    root, values, _ = create_cache(tmp_path, cfg, len(ds))
    cfg = dict(cfg, color_jitter={'enabled': False})
    online = PreparedRoboDojoDataset(cfg)
    cached = PreparedRoboDojoDataset(dict(cfg, latent_cache_dir=str(root)))
    bb = SimpleNamespace(_variant=TI2VVariant(), video_encoder=None, _robodojo_vae_cache_validated=True,
        dtype=torch.bfloat16, device='cpu', _tokenizer=None, text_encoder=None,
        needs_first_frame_skip=True, temporal_compression=4)
    context = torch.zeros((3, 512, 4), dtype=torch.bfloat16)
    monkeypatch.setattr('openwam.model.video_backbone.wan.encode.encode_text', lambda *a, **kw: (context, torch.tensor([9,9,9])))
    bb.preprocess_cached_robodojo = lambda x, text: WanBase.preprocess_cached_robodojo(bb, x, text)
    samples = [0, 30, len(ds)-1]
    expected = bb.preprocess_cached_robodojo(values[samples], ['a']*3)
    assert torch.equal(expected['first_frame_latents'], values[samples, :, :1])
    assert expected['fuse_vae_embedding_in_latents'] is True
    arch = SimpleNamespace(dtype=torch.bfloat16, device='cpu', video_backbone=bb,
        _use_gradient_checkpointing=True, _use_gradient_checkpointing_offload=False,
        _max_timestep_boundary=1., _min_timestep_boundary=0., preprocess=lambda **kw: expected)
    a = BaseWAMArchitecture.prepare_inputs(arch, [online[i] for i in samples])
    b = BaseWAMArchitecture.prepare_inputs(arch, [cached[i] for i in samples])
    assert a.keys() == b.keys()
    for key in a:
        if torch.is_tensor(a[key]):
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key
    with pytest.raises(ValueError, match='Mixed'):
        BaseWAMArchitecture.prepare_inputs(arch, [online[0], cached[0]])
