"""CPU-only exact regrouping checks using real cached latents and T5 vectors."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from openwam.dataloader.libero_latent_cache import CachedLiberoDataset
from openwam.dataloader.robodojo_latent_cache import sha256_file
from openwam.model.video_backbone.wan.text_cache import FrozenTextCache
from openwam.model.video_backbone.wan.variants.ti2v import TI2VVariant
from openwam.model.video_backbone.wan_backbone import WanBase
import openwam.model.video_backbone.wan_backbone as backbone_module


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root/'mutual_baseline_config.yaml')
    ds = CachedLiberoDataset.from_config(cfg.dataloader,split='train')
    indices = np.linspace(0,len(ds)-1,32,dtype=np.int64).tolist()
    samples = [ds[i] for i in indices]
    latent = torch.stack([s['video_latents'] for s in samples])
    prompts = [s['prompt'] for s in samples]
    cache = FrozenTextCache('/mnt/vepfs01/output/tiecheng.guo/artifacts/waw/libero_4suite_text_cache_v1')
    encoder = torch.nn.Linear(1,1).eval().requires_grad_(False)
    backbone = NS(_variant=TI2VVariant(),video_encoder=None,_robodojo_vae_cache_validated=True,
                  device='cpu',dtype=torch.bfloat16,_tokenizer=None,text_encoder=encoder)
    def text_encode(prompts, **kwargs):
        return cache.encode(prompts,text_encoder=encoder,device='cpu',compact=False)
    with patch.object(backbone_module.wan_encode,'encode_text',side_effect=text_encode):
        whole = WanBase.preprocess_cached_robodojo(backbone,latent,prompts)
        for batch in (4,8,16,32):
            parts = [WanBase.preprocess_cached_robodojo(backbone,latent[i:i+batch],prompts[i:i+batch])
                     for i in range(0,32,batch)]
            for key in ('input_latents','first_frame_latents','context','seq_lens'):
                assert torch.equal(whole[key],torch.cat([p[key] for p in parts])), (batch,key)
    manifest = ds._latent_cache.manifest
    source = Path(__file__).resolve().parents[1]
    for name,digest in manifest['encoding_source_sha256'].items():
        assert sha256_file(source/name)==digest,name
    receipt=dict(status='passed',manifest_sha256=sha256_file(ds._latent_cache.root/'manifest.json'),
        validated_consumer_batch_sizes=[4,8,16,32],sample_indices=indices,
        cached_latents_clean_prefix_t5_context_lengths_bitwise_equal=True,
        cache_encoding_batch_size=manifest['batch_size'],encoding_source_unchanged=True,
        changed_encoder_execution=False,gpu_used=False)
    (root/'rebatched_cache_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
