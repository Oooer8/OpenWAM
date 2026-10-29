#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir work/cosmos3_pipeline.lock
trap 'rmdir work/cosmos3_pipeline.lock' EXIT
date -u '+WAITING_FOR_VERIFIED_FOUNDATION %FT%TZ'
while [[ ! -f work/setup_logs/foundation.ready ]]; do
    sleep 15
done
date -u '+SMOKE_START %FT%TZ'
# Twelve full optimizer updates, same memory/batch settings as the full run.
# Profiling run writes metrics and config, but no large model checkpoint.
bash scripts/train_cosmos3_droid.sh \
    training.num_epochs=null training.max_steps=12 training.save_steps=0 \
    training.save_full_states_for_resume=false training.lr_scheduler=null \
    training.output_path=/mnt/vepfs01/output/tiecheng.guo/openwam/outputs/cosmos3_droid_smoke \
    >work/setup_logs/cosmos3_smoke.log 2>&1
/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin/python - <<'PY'
import json, math, pathlib
root = pathlib.Path('outputs/cosmos3_droid_smoke')
paths = sorted(root.glob('*/metrics.jsonl'), key=lambda p: p.stat().st_mtime)
if not paths:
    raise RuntimeError('Smoke produced no local metrics')
rows = [json.loads(line) for line in paths[-1].read_text().splitlines()]
if len(rows) != 12 or rows[-1]['optimizer_step'] != 12:
    raise RuntimeError('Smoke did not finish all optimizer updates')
for row in rows:
    if not all(math.isfinite(float(row[k])) for k in ('loss_total','loss_video','loss_action')):
        raise RuntimeError('Non-finite smoke loss')
result = {'status': 'passed', 'metrics': str(paths[-1]), 'steps': len(rows),
          'optimizer_updates': rows[-1]['optimizer_step'], 'last_loss': rows[-1]['loss_total'],
          'median_micro_step_seconds': sorted(1/r['micro_steps_per_sec'] for r in rows[4:])[4]}
pathlib.Path('work/setup_logs/cosmos3_smoke_passed.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
PY
date -u '+FORMAL_TRAINING_START %FT%TZ'
bash scripts/train_cosmos3_droid.sh >work/setup_logs/cosmos3_train.log 2>&1
date -u '+FORMAL_TRAINING_COMPLETE %FT%TZ'
touch work/setup_logs/cosmos3_training.complete
