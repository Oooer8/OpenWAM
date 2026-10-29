"""Summarize 8x16 candidates only after all real optimizer updates finish."""
import argparse
import json
import math
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
results = []
for run in sorted(a.root.glob('*/run.json')):
    receipt = json.loads(run.read_text())
    if receipt['status'] != 'completed':
        continue
    assert receipt['gpus'] == 8 and receipt['effective_global_batch'] == 128
    ranks = []
    for rank in range(8):
        rows = [json.loads(x) for x in (run.parent/f'phases_rank{rank}.jsonl').read_text().splitlines()]
        assert len(rows) == 24
        assert [r['actual_parameter_updates'] for r in rows] == list(range(1, 25))
        assert all(r.get('vae_encoding_seconds', 0) == 0 for r in rows)
        ranks.append(rows)
    timings = [max(rank[i]['trainer_micro_step_seconds'] for rank in ranks) for i in range(4,20)]
    assert all(math.isfinite(t) and t > 0 for t in timings)
    result = dict(case=receipt['case'], batch_size_per_gpu=16, gpus=8, accumulation=1,
                  effective_global_batch=128, seconds_per_update=statistics.mean(timings),
                  median_seconds_per_update=statistics.median(timings),
                  samples_per_second=128/statistics.mean(timings),
                  peak_allocated_gib=max(r['peak_allocated_gib'] for rows in ranks for r in rows),
                  peak_reserved_gib=max(r['peak_reserved_gib'] for rows in ranks for r in rows),
                  vae_online_calls=0, checkpoint_pattern=receipt['checkpoint_pattern'],
                  block_checkpointing=receipt['block_checkpointing'])
    results.append(result)
summary = dict(status='measured' if results else 'no_completed_candidates', candidates=results,
               fastest=min(results,key=lambda r:r['seconds_per_update']) if results else None)
(a.root/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
