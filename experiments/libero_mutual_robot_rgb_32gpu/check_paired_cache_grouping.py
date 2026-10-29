"""CPU regrouping proof for both immutable per-window RGB latent streams."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))


def main():
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from openwam.dataloader.libero_robot_rgb_pair import PairedRobotRGBCachedLiberoDataset
    from openwam.model.video_backbone.wan.text_cache import FrozenTextCache
    from openwam.model.video_backbone.wan.variants.ti2v import TI2VVariant
    from openwam.model.video_backbone.wan_backbone import WanBase
    import openwam.model.video_backbone.wan_backbone as module
    from runtime_common import sha,write
    torch.set_num_threads(2)
    cfg=OmegaConf.load(ROOT/'config.yaml')
    ds=PairedRobotRGBCachedLiberoDataset.from_config(cfg.dataloader,split='train')
    indices=np.linspace(0,len(ds)-1,32,dtype=np.int64).tolist()
    samples=[ds[i] for i in indices]
    prompts=[s['prompt'] for s in samples]
    cache=FrozenTextCache(str(cfg.project.text_cache_directory))
    encoder=torch.nn.Linear(1,1).eval().requires_grad_(False)
    backbone=NS(_variant=TI2VVariant(),video_encoder=None,_robodojo_vae_cache_validated=True,
        device='cpu',dtype=torch.bfloat16,_tokenizer=None,text_encoder=encoder)
    def text_encode(prompts,**kw):
        return cache.encode(prompts,text_encoder=encoder,device='cpu',compact=False)
    with patch.object(module.wan_encode,'encode_text',side_effect=text_encode):
        for key in ['video_latents','robot_video_latents']:
            latent=torch.stack([s[key] for s in samples])
            whole=WanBase.preprocess_cached_robodojo(backbone,latent,prompts)
            for size in [4,8,16,32]:
                parts=[WanBase.preprocess_cached_robodojo(backbone,latent[i:i+size],prompts[i:i+size]) for i in range(0,32,size)]
                for name in ['input_latents','first_frame_latents','context','seq_lens']:
                    assert torch.equal(whole[name],torch.cat([p[name] for p in parts])),(key,size,name)
    write(ROOT/'paired_cache_grouping.json',dict(status='passed',indices=indices,
        scene_manifest_sha256=sha(ds._latent_cache.root/'manifest.json'),
        robot_manifest_sha256=sha(ds._robot_dataset._latent_cache.root/'manifest.json'),
        text_index_sha256=sha(Path(cfg.project.text_cache_directory)/'index.json'),
        validated_consumer_batch_sizes=[4,8,16,32],both_streams_bitwise_equal=True,
        changed_encoder_execution=False,gpu_used=False))


if __name__=='__main__':main()
