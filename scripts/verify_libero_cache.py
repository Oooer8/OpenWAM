"""Strict persisted-cache and canonical-training-input verification."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.libero_latent_cache import CachedLiberoDataset, state_digest
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38
from openwam.model.video_backbone.wan.encode import preprocess_video
from openwam.model.video_backbone.wan.variants.ti2v import TI2VVariant
from openwam.model.video_backbone.wan_backbone import WanBase
from openwam.train.utils.seeding import seed_everything
from scripts.cache_robodojo_vae import atomic_json, load_state
from scripts.cache_libero_vae import validation_indices


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--config',default='configs/libero_cache_data.yaml')
    p.add_argument('--wait-ready',action='store_true')
    args=p.parse_args()
    if args.wait_ready:
        while not (args.root/'manifest.json').is_file():
            status=json.loads((args.root/'run_status.json').read_text())
            if status['status']=='failed':
                raise RuntimeError('Cache generation failed; see worker logs')
            time.sleep(20)
    torch.set_num_threads(2)
    seed_everything(42,rank=0)
    config=yaml.safe_load(Path(args.config).read_text())
    ds=LiberoDataset.from_config(config,split='train')
    cached=CachedLiberoDataset.from_config(dict(config,libero_latent_cache_dir=str(args.root)),split='train')
    indices=validation_indices(ds)
    for shard in cached._latent_cache.manifest['shards']:
        indices += [shard['start'],shard['end']-1]
    indices=sorted(set(indices))
    for index in indices:
        before=ds._wrist_fail_count
        online=ds._getitem_impl(index)
        stored=cached[index]
        assert ds._wrist_fail_count==before
        assert stored['video']==[] and stored['first_frame_image'] is None
        for key in ('action','action_mask','proprio','proprio_mask','video_mask'):
            assert torch.equal(online[key],stored[key]),(index,key)
        assert online['prompt']==stored['prompt']

    state=load_state(cached._latent_cache.manifest['checkpoint'])
    assert state_digest(state)==cached._latent_cache.manifest['vae_weights_sha256']
    vae=WanVideoVAE38()
    vae.load_state_dict(state,strict=True,assign=True)
    del state
    vae.cuda().bfloat16().eval().requires_grad_(False)
    shards=cached._latent_cache.manifest['shards']
    selected=[shards[0],shards[len(shards)//2],shards[-1]]
    comparisons=[]
    with torch.inference_mode():
        for shard in selected:
            bs=cached._latent_cache.manifest['batch_size']
            start=shard['start']
            batch=list(range(start,min(start+bs,shard['end'])))
            frames=[ds._getitem_impl(i)['video'] for i in batch]
            frames += [frames[-1]]*(bs-len(frames))
            pixels=torch.cat([preprocess_video(f,dtype=torch.bfloat16,device='cuda') for f in frames])
            live=vae.batch_encode(pixels,device='cuda')[:len(batch)].cpu()
            persisted=torch.stack([cached[i]['video_latents'] for i in batch])
            assert torch.equal(live,persisted),f'Persisted latent differs from online encoding: {start}'
            comparisons.append(dict(start=start,windows=len(batch),bitwise_equal=True))

        # Verify the exact existing cached-TI2V path without loading the DiT or
        # text model: inject immutable text tensors and compare conditioning.
        from unittest.mock import patch
        import openwam.model.video_backbone.wan_backbone as bb_module
        bb=SimpleNamespace(_variant=TI2VVariant(),video_encoder=None,
            _robodojo_vae_cache_validated=True,device='cpu',dtype=torch.bfloat16,
            _tokenizer=None,text_encoder=None)
        latent=torch.stack([cached[0]['video_latents']])
        context=torch.zeros(1,512,4096,dtype=torch.bfloat16)
        seq_lens=torch.tensor([12])
        with patch.object(bb_module.wan_encode,'encode_text',return_value=(context,seq_lens)):
            prepared=WanBase.preprocess_cached_robodojo(bb,latent,['example'])
        assert torch.equal(prepared['input_latents'],latent)
        assert torch.equal(prepared['first_frame_latents'],latent[:,:,:1])
        assert prepared['fuse_vae_embedding_in_latents'] and prepared['num_frames']==9

    # Fail closed on accidental augmentation or a changed temporal sampling.
    rejected=[]
    for changes in ({'color_jitter':{'enabled':True}},{'video_stride':2}):
        try:
            CachedLiberoDataset.from_config(dict(config,libero_latent_cache_dir=str(args.root),**changes))
        except ValueError:
            rejected.append(changes)
        else:
            raise AssertionError('Unsafe cache config was accepted')
    report=dict(status='passed',canonical_sample_checks=len(indices),
        persisted_online_comparisons=comparisons,action_proprio_masks_prompt_exact=True,
        clean_first_frame_conditioning_exact=True,rejected_configurations=rejected,
        training_started=False)
    atomic_json(args.root/'integration_validation.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
