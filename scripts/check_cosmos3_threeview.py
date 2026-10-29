"""Verify source identities, synchronization, slot pixels and unchanged actions."""
import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from openwam.dataloader.registry import build_dataset
from openwam.dataloader.cosmos3_droid import HEAD, HEAD2, WRIST

with hydra.initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
    cfg = hydra.compose(config_name='train_cosmos3_droid_fast', overrides=[
        'dataloader=cosmos3_droid_threeview', 'dataloader.window_stride=3'])
assert cfg.training.resume_ckpt_path is None
assert 'Foundation-Model' in cfg.training.finetune_ckpt_path
random.seed(42)
train = build_dataset(cfg.dataloader, split='train')
val = build_dataset(cfg.dataloader, split='val')
assert len(train) == 4274500
assert train._head_camera_choices is None
assert train._camera_layout == val._camera_layout == [WRIST, HEAD, HEAD2]
assert (train._head_camera, train._left_wrist_camera, train._right_wrist_camera) == (WRIST, HEAD, HEAD2)
assert not set(train._eps_df.episode_index) & set(val._eps_df.episode_index)

# Independent colors make camera swaps, missing/duplicated slots obvious.
real_decode = train._decode_one_camera
calls = []
colors = {WRIST: (220, 30, 20), HEAD: (10, 210, 30), HEAD2: (20, 30, 230)}
def synthetic(camera, row, ep_local, offset, real_local_indices, *, kind):
    calls.append((camera, ep_local, offset, tuple(real_local_indices), kind))
    return [Image.new('RGB', (640, 360), colors[camera]) for _ in real_local_indices]
train._decode_one_camera = synthetic
clip = train._decode_window_video(None, 7, 12, np.arange(0,33,4), 0)
a = np.asarray(clip[0])
assert len(clip) == 9 and clip[0].size == (320,384)
assert (a[:256] == colors[WRIST]).all()
assert (a[256:,:160] == colors[HEAD]).all()
assert (a[256:,160:] == colors[HEAD2]).all()
assert [c[0] for c in calls] == [WRIST,HEAD,HEAD2]
assert len({c[1:4] for c in calls}) == 1
train._decode_one_camera = real_decode

old_cfg = OmegaConf.load('configs/dataloader/cosmos3_droid.yaml')
old_cfg.window_stride = 3
old = build_dataset(old_cfg, split='train')
panels, results = [], []
for split, ds in [('train',train),('val',val)]:
    indices = [0,len(ds)-1] + random.sample(range(len(ds)),6)
    for index in indices:
        sample = ds[index]
        assert len(sample['video']) == 9
        assert all(im.size == (320,384) for im in sample['video'])
        assert sample['action'].shape == (32,80)
        assert sample['proprio'].shape == (1,80)
        assert sample['action_mask'][:,:10].all() and not sample['action_mask'][:,10:].any()
        assert sample['video_mask'].all()
        assert torch.isfinite(sample['action']).all() and torch.isfinite(sample['proprio']).all()
        im = np.asarray(sample['video'][0])
        results.append(dict(split=split,index=index,slot_std=[float(im[:256].std()),
            float(im[256:,:160].std()),float(im[256:,160:].std())]))
        if split == 'train' and len(panels)<8:
            previous = old[index]
            for k in ['action','proprio','action_mask','proprio_mask','video_mask']:
                assert torch.equal(sample[k],previous[k]), k
            panel = Image.new('RGB',(320,414),'white')
            panel.paste(sample['video'][0],(0,30))
            ImageDraw.Draw(panel).text((4,5),f'window {index} | wrist / ext1 / ext2',fill='black')
            panels.append(panel)
        print(json.dumps(results[-1]),flush=True)
out = Path('work/setup_logs')
sheet = Image.new('RGB',(1280,828),'white')
for i,p in enumerate(panels): sheet.paste(p,((i%4)*320,(i//4)*414))
sheet.save(out/'threeview_samples.png')
summary=dict(status='passed',train_windows=len(train),val_windows=len(val),
    camera_layout=train._camera_layout,canvas_width_height=[320,384],
    training_view_sampling=False,source_black_streams='preserved, not substituted',
    slot_and_sync_test='passed',action_equivalence_test='passed',samples=results)
(out/'threeview_preflight.json').write_text(json.dumps(summary,indent=2))
print('THREEVIEW_PREFLIGHT_PASSED',flush=True)
