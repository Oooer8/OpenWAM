#!/usr/bin/env python3
"""Copy-verified NVIDIA LIBERO -> canonical OpenWAM EEF10, entirely on CPU.

Raw copies and output are explicit external paths. Videos are hard-linked only
from the user's independent raw copy, never from the shared frontier source.
All rewritten metadata/parquet files use atomic replacement (no link writes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90')
FOUR = SUITES[:4]
STATE_NAMES = ['eef_x', 'eef_y', 'eef_z', 'rot6d_col0_x', 'rot6d_col0_y',
               'rot6d_col0_z', 'rot6d_col1_x', 'rot6d_col1_y', 'rot6d_col1_z', 'gripper_open_scale']
ACTION_NAMES = ['eef_native_delta_x', 'eef_native_delta_y', 'eef_native_delta_z',
                'eef_native_delta_rot6d_col0_x', 'eef_native_delta_rot6d_col0_y',
                'eef_native_delta_rot6d_col0_z', 'eef_native_delta_rot6d_col1_x',
                'eef_native_delta_rot6d_col1_y', 'eef_native_delta_rot6d_col1_z', 'gripper_open_command']


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def write_table(path, table, *, keep_metadata=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    pq.write_table(table if keep_metadata else table.replace_schema_metadata(None), temp, compression='zstd')
    os.replace(temp, path)


def link_copy(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return str(dst)


def feature_stats(a):
    return {'min': a.min(0).tolist(), 'max': a.max(0).tolist(),
            'mean': a.mean(0, dtype=np.float64).tolist(),
            'std': a.std(0, dtype=np.float64).tolist(), 'count': [len(a)]}


def convert_vectors(action, state):
    if action.ndim != 2 or action.shape[1] != 7 or state.shape != (len(action), 8):
        raise ValueError(f'Unexpected source shapes {action.shape}, {state.shape}')
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError('Non-finite source action/state')
    if action[:, 6].min() < -1e-6 or action[:, 6].max() > 1 + 1e-6:
        raise ValueError('This converter is specific to NVIDIA [0,1] open-scale gripper')
    ar = Rotation.from_rotvec(action[:, 3:6].astype(np.float64)).as_matrix()
    sr = Rotation.from_rotvec(state[:, 3:6].astype(np.float64)).as_matrix()
    a6 = np.concatenate([ar[:, :, 0], ar[:, :, 1]], -1).astype(np.float32)
    s6 = np.concatenate([sr[:, :, 0], sr[:, :, 1]], -1).astype(np.float32)
    aa = np.concatenate([action[:, :3], a6, 2 * action[:, 6:7] - 1], -1).astype(np.float32)
    ss = np.concatenate([state[:, :3], s6,
                         np.clip(2 * (state[:, 6:7] - state[:, 7:8]) / 0.08 - 1, -1, 1)], -1).astype(np.float32)
    # An independent matrix reconstruction checks every row, not just samples.
    restored = np.stack([aa[:, 3:6], aa[:, 6:9], np.cross(aa[:, 3:6], aa[:, 6:9])], -1)
    rv = Rotation.from_matrix(restored.astype(np.float64)).as_rotvec()
    err = float(np.max(np.abs(rv - action[:, 3:6])))
    if err > 2e-6:
        raise ValueError(f'Action rotation round-trip failed: {err}')
    sm = np.stack([ss[:, 3:6], ss[:, 6:9], np.cross(ss[:, 3:6], ss[:, 6:9])], -1)
    serr = float(np.max(np.abs(sm - sr)))
    if serr > 2e-6:
        raise ValueError(f'State rotation reconstruction failed: {serr}')
    assert np.array_equal(aa[:, :3], action[:, :3])
    return aa, ss, err, serr


def replace_col(tab, name, values):
    arr = pa.array(values, type=pa.list_(pa.float32(), 10))
    return tab.set_column(tab.schema.get_field_index(name), name, arr)


def source_inventory(shared, raw, output):
    rows = []
    for suite in SUITES:
        for p in sorted((raw / suite).rglob('*')):
            if not p.is_file():
                continue
            relative = p.relative_to(raw)
            original = shared / relative
            a = sha(original)
            b = sha(p)
            if a != b or original.stat().st_size != p.stat().st_size:
                raise ValueError(f'Copy verification mismatch: {relative}')
            if (original.stat().st_dev, original.stat().st_ino) == (p.stat().st_dev, p.stat().st_ino):
                raise ValueError(f'Raw copy is hard-linked to shared source: {relative}')
            rows.append({'path': str(relative), 'bytes': p.stat().st_size, 'sha256': b})
        log('Verified independent raw copy:', suite)
    write_json(output / 'raw_copy_manifest.json', {'source': str(shared), 'raw_copy': str(raw), 'files': rows})
    return rows


def convert_suite(raw, dst, name, stats_builder):
    src = raw / name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=link_copy)
    info = json.loads((src / 'meta/info.json').read_text())
    episode_stats = {}
    action_all, state_all = [], []
    max_aerr = max_serr = 0.0
    for p in sorted((src / 'data').rglob('*.parquet')):
        tab = pq.read_table(p)
        a = np.stack(tab['action'].to_pylist()).astype(np.float32)
        s = np.stack(tab['observation.state'].to_pylist()).astype(np.float32)
        aa, ss, aerr, serr = convert_vectors(a, s)
        max_aerr, max_serr = max(max_aerr, aerr), max(max_serr, serr)
        ids = np.asarray(tab['episode_index'])
        frames = np.asarray(tab['frame_index'])
        tasks = np.asarray(tab['task_index'])
        for eid in np.unique(ids):
            mask = ids == eid
            if int(eid) in episode_stats:
                raise ValueError(f'Episode spans data shards; conversion needs explicit support: {eid}')
            fi = frames[mask]
            if not np.array_equal(fi, np.arange(len(fi))):
                raise ValueError(f'Non-contiguous source frame indices: {name}/{eid}')
            tids = np.unique(tasks[mask])
            if len(tids) != 1:
                raise ValueError(f'Multiple task indices in episode {name}/{eid}')
            episode_stats[int(eid)] = {'length': int(mask.sum()), 'task_index': int(tids[0]),
                                      'action': feature_stats(aa[mask]), 'observation.state': feature_stats(ss[mask])}
        action_all.append(aa)
        state_all.append(ss)
        tab = replace_col(replace_col(tab, 'action', aa.tolist()), 'observation.state', ss.tolist())
        write_table(dst / p.relative_to(src), tab)
    task_frame = pd.read_parquet(src / 'meta/tasks.parquet')
    task_text = {int(row.task_index): str(text) for text, row in task_frame.iterrows()}
    ep_count = 0
    for p in sorted((src / 'meta/episodes').rglob('*.parquet')):
        df = pd.read_parquet(p)
        for idx, row in df.iterrows():
            st = episode_stats[int(row.episode_index)]
            if st['length'] != int(row.length):
                raise ValueError(f'Episode length mismatch: {name}/{row.episode_index}')
            if task_text[st['task_index']] not in list(row.tasks):
                raise ValueError(f'Episode prompt mismatch: {name}/{row.episode_index}')
        for field in ('action', 'observation.state'):
            drop = [c for c in df.columns if c.startswith(f'stats/{field}/')]
            df = df.drop(columns=drop)
            for stat in ('min', 'max', 'mean', 'std', 'count'):
                df[f'stats/{field}/{stat}'] = [episode_stats[int(e)][field][stat] for e in df.episode_index]
        ep_count += len(df)
        write_table(dst / p.relative_to(src), pa.Table.from_pandas(df, preserve_index=False))
    aa, ss = np.concatenate(action_all), np.concatenate(state_all)
    if len(aa) != info['total_frames'] or ep_count != info['total_episodes']:
        raise ValueError(f'Source totals disagree: {name}')
    for key, names in (('action', ACTION_NAMES), ('observation.state', STATE_NAMES)):
        info['features'][key].update(shape=[10], names=names, dtype='float32')
    write_json(dst / 'meta/info.json', info)
    original_stats = json.loads((src / 'meta/stats.json').read_text())
    original_stats.update({'action': feature_stats(aa), 'observation.state': feature_stats(ss)})
    write_json(dst / 'meta/stats.json', original_stats)
    conversion = {'output_representation': 'native_delta_eef10', 'source': str(src),
                  'source_dataset_url': 'https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3',
                  'source_info_sha256': sha(src / 'meta/info.json'),
                  'source_action_gripper': '0=closed,1=open', 'action_gripper_conversion': '2*g-1',
                  'action_xyz_conversion': 'unchanged native normalized delta; no OSC scaling or pose composition',
                  'rotation_conversion': 'axis-angle to first two rotation-matrix columns; column-major rot6d',
                  'state_gripper_conversion': 'clip(2*(qpos0-qpos1)/0.08-1,-1,1)',
                  'fps': info['fps'], 'fps_policy': 'preserve source timestamps and videos; no temporal resampling',
                  'episodes': ep_count, 'frames': len(aa), 'action_roundtrip_max_abs_error': max_aerr,
                  'state_matrix_max_abs_error': max_serr,
                  'gripper_convention': 'minus1_closed_plus1_open'}
    write_json(dst / 'meta/conversion.json', conversion)
    stats_builder(dst)
    log('Converted:', name, ep_count, 'episodes', len(aa), 'frames')
    return conversion


def merge_four(output, stats_builder):
    dst = output / 'libero_4suite'
    dst.mkdir(parents=True, exist_ok=True)
    tasks, episode_offset, frame_offset, chunk_offset = {}, 0, 0, 0
    sources = []
    for name in FOUR:
        src = output / 'suites' / name
        info = json.loads((src / 'meta/info.json').read_text())
        tf = pd.read_parquet(src / 'meta/tasks.parquet')
        task_map = {}
        for text, row in tf.iterrows():
            if str(text) not in tasks:
                tasks[str(text)] = len(tasks)
            task_map[int(row.task_index)] = tasks[str(text)]
        used_chunks = [int(re.search(r'chunk-(\d+)', str(p.relative_to(src))).group(1))
                       for p in src.rglob('*.parquet') if 'chunk-' in str(p.relative_to(src))]
        def new_rel(path):
            rel = str(path.relative_to(src))
            return re.sub(r'chunk-(\d+)', lambda m: f'chunk-{int(m[1]) + chunk_offset:03d}', rel)
        for p in sorted((src / 'data').rglob('*.parquet')):
            df = pd.read_parquet(p)
            df['episode_index'] += episode_offset
            df['index'] += frame_offset
            df['task_index'] = df['task_index'].map(task_map).astype('int64')
            write_table(dst / new_rel(p), pa.Table.from_pandas(df, preserve_index=False))
        for p in sorted((src / 'meta/episodes').rglob('*.parquet')):
            df = pd.read_parquet(p)
            df['episode_index'] += episode_offset
            for c in ('dataset_from_index', 'dataset_to_index'):
                df[c] += frame_offset
            for c in df.columns:
                if c.endswith('/chunk_index'):
                    df[c] += chunk_offset
            write_table(dst / new_rel(p), pa.Table.from_pandas(df, preserve_index=False))
        for p in (src / 'videos').rglob('*.mp4'):
            link_copy(p, dst / new_rel(p))
        sources.append({'suite': name, 'episode_offset': episode_offset, 'frame_offset': frame_offset,
                        'chunk_offset': chunk_offset, 'task_map': task_map})
        episode_offset += info['total_episodes']
        frame_offset += info['total_frames']
        chunk_offset += max(used_chunks) + 1
    info = json.loads((output / 'suites' / FOUR[0] / 'meta/info.json').read_text())
    info.update(total_episodes=episode_offset, total_frames=frame_offset, total_tasks=len(tasks),
                total_chunks=chunk_offset, total_videos=2 * episode_offset, splits={'train': f'0:{episode_offset}'})
    info['data_files'] = [str(p.relative_to(dst)) for p in sorted((dst / 'data').rglob('*.parquet'))]
    write_json(dst / 'meta/info.json', info)
    tdf = pd.DataFrame({'task_index': list(tasks.values())}, index=list(tasks.keys()))
    p = dst / 'meta/tasks.parquet'
    # The official reader resolves prompts from the pandas text index. Its
    # schema metadata must survive serialization; data-table HF metadata is
    # intentionally discarded elsewhere because its vector sizes are stale.
    write_table(p, pa.Table.from_pandas(tdf, preserve_index=True), keep_metadata=True)
    conversion = {'output_representation': 'native_delta_eef10', 'gripper_convention': 'minus1_closed_plus1_open',
                  'sources': sources, 'episodes': episode_offset, 'frames': frame_offset, 'tasks': len(tasks),
                  'fps': info['fps'], 'fps_policy': 'preserved source 20 FPS; differs from official 10 FPS release',
                  'excluded_from_four_suite_training': ['libero_90'], 'videos': 'hardlinks to independent copied source'}
    write_json(dst / 'meta/conversion.json', conversion)
    statpath = stats_builder(dst)
    blocks = np.load(statpath, allow_pickle=True).item()
    write_json(dst / 'meta/stats.json', {'action': {k: blocks['eef'][k] for k in ('min','max','mean','std','count')},
                                        'observation.state': {k: blocks['eef_state'][k] for k in ('min','max','mean','std','count')}})
    log('Merged four-suite bucket:', episode_offset, 'episodes', frame_offset, 'frames', len(tasks), 'tasks')
    return conversion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-source', type=Path, required=True)
    parser.add_argument('--raw-copy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    for p in (args.shared_source, args.raw_copy, args.output):
        if not p.is_absolute():
            raise ValueError('All dataset paths must be absolute')
    if args.shared_source.resolve() == args.raw_copy.resolve() or args.output.resolve() in (args.shared_source.resolve(), args.raw_copy.resolve()):
        raise ValueError('Source, raw copy and derived output must be distinct')
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / '.openwam_libero_conversion'
    if not marker.exists() and list(args.output.iterdir()):
        raise ValueError('Refusing to reuse unrelated nonempty output')
    marker.touch()
    sys.path.insert(0, str(args.repo))
    from openwam.dataloader.utils.stats_computation.libero_stats_computation import build_and_save_libero_stats
    inventory = source_inventory(args.shared_source, args.raw_copy, args.output)
    report = {'source': str(args.shared_source), 'raw_copy': str(args.raw_copy), 'output': str(args.output),
              'converter_sha256': sha(__file__), 'raw_verified_files': len(inventory),
              'raw_verified_bytes': sum(x['bytes'] for x in inventory), 'suites': {}}
    for name in SUITES:
        report['suites'][name] = convert_suite(args.raw_copy, args.output / 'suites' / name, name, build_and_save_libero_stats)
    report['training_bucket'] = merge_four(args.output, build_and_save_libero_stats)
    write_json(args.output / 'conversion_report.json', report)
    log('Conversion complete:', args.output)


if __name__ == '__main__':
    main()
