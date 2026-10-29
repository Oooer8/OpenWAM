#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Fresh foundation warm-start at peak LR 5e-5. Do not resume superseded runs.
/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin/python - <<'CHECK'
import json
import os
from pathlib import Path
p = json.loads(Path('work/setup_logs/threeview_preflight.json').read_text())
assert p['status'] == 'passed' and p['train_windows'] == 4274500
assert p['camera_layout'] == ['observation.image.wrist_image_left', 'observation.image.exterior_image_1_left', 'observation.image.exterior_image_2_left']
assert p['canvas_width_height'] == [320, 384]
if os.environ.get('OPENWAM_RESUME_RUN'):
    run = Path(os.environ['OPENWAM_RESUME_RUN']).resolve()
    active = json.loads(Path('work/setup_logs/active_formal_training.json').read_text())
    assert str(run) == active['run_directory']
    assert run.parent.name == 'cosmos3_droid_alpha_stride3_threeview_lr5e5'
    assert list(run.glob('accel_state_step_*/trainer_state.json'))
    assert os.environ.get('OPENWAM_FINETUNE_RUN') == 'null'

CHECK
exec bash scripts/train_cosmos3_droid_fast.sh \
    dataloader=cosmos3_droid_threeview dataloader.window_stride=3 \
    training.num_epochs=2 training.max_steps=null \
    training.batch_size=16 training.gradient_accumulation_steps=1 \
    training.save_steps=33395 training.keep_last_k_ckpts=2 \
    training.save_full_states_for_resume=true training.resume_ckpt_path="${OPENWAM_RESUME_RUN:-null}" \
    training.learning_rate=5e-5 training.action_lr=null training.video_lr=null \
    training.finetune_ckpt_path="${OPENWAM_FINETUNE_RUN:-/mnt/vepfs01/output/tiecheng.guo/openwam/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}" \
    training.output_path=/mnt/vepfs01/output/tiecheng.guo/openwam/outputs/cosmos3_droid_alpha_stride3_threeview_lr5e5
