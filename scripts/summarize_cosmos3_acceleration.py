"""Summarize measured short runs without treating them as completed training."""
import json
import statistics
from pathlib import Path
import yaml
import math
import pyarrow.parquet as pq

root = Path('.')
names = {'all_layers_checkpoint':'cosmos3_droid_b16_benchmark',
         'alternate_layers_checkpoint':'cosmos3_droid_b16_half_gc_benchmark',
         'text_cache':'cosmos3_droid_b16_cache_benchmark',
         'text_cache_and_compute':'cosmos3_droid_b16_compute_benchmark',
         'fused_norm_gate_and_uint8':'cosmos3_droid_b16_fused_benchmark',
         'selective_matmul_checkpoint':'cosmos3_droid_b16_selective_benchmark',
         'alternate_ffn_checkpoint':'cosmos3_droid_b16_ffn_checkpoint_benchmark',
         'alternate_attention_checkpoint':'cosmos3_droid_b16_attention_checkpoint_benchmark',
         'no_checkpoint':'cosmos3_droid_b16_no_checkpoint_benchmark',
         'no_checkpoint_batch8_accumulate2':'cosmos3_droid_b8_ga2_no_checkpoint_benchmark'}
runs = {}
first = {}
for label,name in names.items():
    candidates = list((root/'outputs'/name).glob('*/metrics.jsonl'))
    if not candidates:
        continue
    path = max(candidates,key=lambda p:p.stat().st_mtime)
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    cfg = yaml.safe_load((path.parent/'config.yaml').read_text())
    if not rows or len(rows) < int(cfg['training'].get('max_steps') or 0):
        continue  # Partial/failed runs never qualify for selection.
    execution = cfg.get('project',{}).get('runtime_optimizations',{})
    accum = int(cfg['training']['gradient_accumulation_steps'])
    normal_limit = execution.get('memory_stress_after_step') or len(rows)
    normal = [r for r in rows if r['micro_step'] <= normal_limit]
    grouped = [normal[i:i+accum] for i in range(0,len(normal)-accum+1,accum)]
    tail = grouped[-min(12,len(grouped)-2):]
    seconds = [sum(1/r['micro_steps_per_sec'] for r in group) for group in tail]
    assert all(math.isfinite(r[k]) for r in rows for k in ('loss_total','loss_video','loss_action','grad_norm'))
    first[label] = rows[0]
    runs[label] = dict(metrics=str(path.resolve()),steps=len(rows),steady_steps=[group[-1]['micro_step'] for group in tail],
                       batch_per_gpu=int(cfg['training']['batch_size']),gradient_accumulation_steps=accum,
                       seconds_per_optimizer_update=statistics.median(seconds),
                       seconds_per_step_median=statistics.median(seconds),
                       seconds_per_step_min=min(seconds),seconds_per_step_max=max(seconds),
                       peak_allocated_gib=max(r['gpu_peak_allocated_gib'] for r in rows),
                       peak_reserved_gib=max(r['gpu_peak_reserved_gib'] for r in rows),
                       final_loss=rows[-1]['loss_total'],final_gradient_norm=rows[-1]['grad_norm'],
                       normal_peak_allocated_gib=max(r['gpu_peak_allocated_gib'] for r in normal),
                       normal_optimizer_updates=normal[-1]['optimizer_step'],
                       explicit_gradient_checkpointing=cfg['training']['use_gradient_checkpointing'],
                       mixed_attention_checkpointing=cfg['model']['architecture'].get('mot_checkpoint_mixed_attn',True),
                       memory_stress_micro_steps=len(rows)-len(normal),
                       execution=execution)
cache = json.loads((root/'work/cosmos3_text_cache/manifest.json').read_text())
validation = json.loads((root/'work/cosmos3_text_cache/validation.json').read_text())
compact = json.loads((root/'work/setup_logs/compact_text_context_validation.json').read_text())
ffn = json.loads((root/'work/setup_logs/training_optimization_numerics.json').read_text())
result = dict(server='192.168.52.168:2222',gpus='8 x NVIDIA A800 80GB',batch_per_gpu=16,global_batch=128,
              foundation_revision=cache['foundation_revision'],text_cache_prompts=cache['num_prompts'],
              text_cache_bytes=cache['total_bytes'],text_cache_directory=str((root/'work/cosmos3_text_cache').resolve()),
              text_cache_validation=validation,compiled_ffn_validation=ffn,compact_context_validation=compact,
              benchmarks=runs,formal_training_started=False,window_stride=1,epochs=2,
              timing_limitation='Short-run medians; estimates exclude checkpoint writes, validation, stalls, and startup.')
if 'text_cache_and_compute' in runs:
    selected = min(runs, key=lambda name:runs[name]['seconds_per_step_median'])
    result['selected_variant']=selected
    result['batch_per_gpu']=runs[selected]['batch_per_gpu']
    result['gradient_accumulation_steps']=runs[selected]['gradient_accumulation_steps']
    best=runs[selected]['seconds_per_step_median']
    episodes = pq.read_table('work/cosmos3_droid_prepared/episodes.parquet', columns=['length','training_split']).to_pylist()
    counts = [(s,2*math.ceil(sum((r['length']-33)//s+1 for r in episodes if r['training_split']=='train')/128)) for s in (1,2,3,4)]
    result['time_estimates']={str(stride):dict(window_start_stride=stride,optimizer_updates=updates,
         pure_training_hours=best*updates/3600,pure_training_days=best*updates/86400)
         for stride,updates in counts}
    result['throughput_speedup_vs_all_layers_checkpoint']=runs['all_layers_checkpoint']['seconds_per_step_median']/best
    result['throughput_speedup_vs_alternate_layers_checkpoint']=runs['alternate_layers_checkpoint']['seconds_per_step_median']/best
    a,b=first['text_cache'],first[selected]
    result['first_step_comparison']={k:dict(cache_only=a[k],optimized=b[k],absolute_difference=abs(a[k]-b[k]))
                                     for k in ('loss_total','loss_video','loss_action','grad_norm')}
phase = root/'work/phase_profile_optimized_b16/summary.json'
if phase.exists():
    result['second_round_phase_profile'] = json.loads(phase.read_text())
for key,filename in [('pixel_transport_check','pixel_transport_check.json'),('norm_gate_compile_check','norm_gate_compile_check.json'),('vae_autotune_check','vae_autotune_check.json'),
                     ('selective_checkpoint_check','selective_checkpoint_check.json'),('vae_memory_format_check','vae_memory_format_check.json'),('vae_norm_compile_check','vae_norm_compile_check.json'),('cache_lookup_batch_check','cache_lookup_batch_check.json')]:
    path=root/'work/setup_logs'/filename
    if path.exists(): result[key]=json.loads(path.read_text())
result['parallelism'] = {'type':'data_parallel', 'zero_stage':2, 'replicated':'model parameters',
                         'partitioned':'trainable gradients and optimizer states', 'tensor_or_pipeline_parallel':False}
result['first_step_by_variant'] = {label:{k:row[k] for k in ('loss_total','loss_video','loss_action','grad_norm')} for label,row in first.items()}
memory_path = root/'work/setup_logs/no_checkpoint_physical_memory.json'
if 'no_checkpoint' in runs and memory_path.exists():
    memory = json.loads(memory_path.read_text())
    run = runs['no_checkpoint']
    execution = run['execution']
    peaks = memory['peaks']['batch16']
    assert len(peaks) == 8 and memory['samples']['batch16'] > 0
    assert run['steps'] == 48 and run['memory_stress_micro_steps'] == 12
    assert not run['explicit_gradient_checkpointing'] and not run['mixed_attention_checkpointing']
    assert execution['cached_text_encoder_on_cpu'] and execution['zero_overlap_comm']
    assert execution['zero_reduce_bucket_size'] == 50_000_000
    assert all(card['peak_used_gib'] < card['total_reported_gib'] for card in peaks.values())
    run_cfg = yaml.safe_load((Path(run['metrics']).parent/'config.yaml').read_text())
    checked = dict(status='passed', foundation_revision=cache['foundation_revision'],
                   text_cache_index_sha256=run_cfg['project']['text_cache_index_sha256'],
                   benchmark=run, physical_memory=memory,
                   maximum_tested_text_context=128, maximum_dataset_valid_text_tokens=114,
                   minimum_sampled_physical_headroom_gib=min(card['total_reported_gib']-card['peak_used_gib'] for card in peaks.values()),
                   limitations='48-step finite-loss/gradient and memory check; not a convergence test. Physical memory sampled once per second.')
    (root/'work/setup_logs/no_checkpoint_validation.json').write_text(json.dumps(checked,indent=2))
    result['no_checkpoint_validation'] = checked
    result['batch8_accumulate2'] = {'status':'cancelled_before_training', 'reason':'User requested no batch-8 test if batch 16 fits.', 'measured_micro_steps':0}
result['parallelism']['frozen_text_encoder_location'] = 'CPU on each rank; no forward during cached training'
output=root/'work/setup_logs/training_acceleration_results.json'
output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k in ['benchmarks','time_estimates','first_step_comparison']},indent=2))
