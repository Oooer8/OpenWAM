"""CPU decode audit: same native window/action/prompt; different RGB signal."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))


def main():
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from openwam.dataloader.libero import LiberoDataset
    torch.set_num_threads(2)
    cfg=OmegaConf.load(ROOT/'config.yaml')
    a=LiberoDataset.from_config(cfg.dataloader,split='train')
    b=LiberoDataset.from_config(OmegaConf.load(ROOT/'robot_cache_dataset.yaml'),split='train')
    assert len(a)==len(b)==273465
    assert a._camera_layout==['observation.images.image','observation.images.wrist_image',None]
    assert b._camera_layout==['observation.images.robot_rgb','observation.images.wrist_robot_rgb',None]
    for name in ['_cum_n_starts','_ep_valid_start','_ep_valid_end','_video_sample_indices']:
        assert np.array_equal(getattr(a,name),getattr(b,name)),name
    indices={0,len(a)//2,len(a)-1}
    conversion=json.loads((a._dataset_dir/'meta/conversion.json').read_text())
    for item in conversion.get('sources',[]):
        for i in [item['frame_offset']-1,item['frame_offset']]:
            if 0<=i<len(a):indices.add(i)
    rows=[]
    for index in sorted(indices):
        before=(a._wrist_fail_count,b._wrist_fail_count)
        scene,robot=a._getitem_impl(index),b._getitem_impl(index)
        assert before==(a._wrist_fail_count,b._wrist_fail_count),'Camera fallback prohibited'
        assert scene['prompt']==robot['prompt']
        for key in ['action','action_mask','proprio','proprio_mask','video_mask']:
            assert torch.equal(scene[key],robot[key]),(index,key)
        assert len(scene['video'])==len(robot['video'])==9
        x=np.stack([np.asarray(v) for v in scene['video']])
        y=np.stack([np.asarray(v) for v in robot['video']])
        assert x.shape==y.shape==(9,384,320,3) and not np.array_equal(x,y)
        rows.append(dict(index=index,scene_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            robot_rgb_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
            same_actions_masks_proprio_prompt=True))
    report=dict(status='passed',windows=len(a),robot_rgb_is_simulator_rendered=True,
        v1_cameras=a._camera_layout,v2_cameras=b._camera_layout,checks=rows,gpu_used=False)
    (ROOT/'raw_rgb_pair_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
