#!/usr/bin/env python3
"""Validate full metadata, video coverage and actual OpenWAM reader outputs."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo))
    from openwam.dataloader.libero import LiberoDataset
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    roots = sorted((args.output / 'suites').iterdir()) + [args.output / 'libero_4suite']
    report = {'buckets': [], 'training_started': False, 'validation_scope':
              'All table rows and episode metadata; video header frame coverage for every referenced video; '
              'strict direct OpenWAM reader decode for first/middle/last windows of each bucket, '
              'plus every suite boundary in merged bucket. No GPU/model execution.'}
    previews = []
    for root in roots:
        info = json.loads((root / 'meta/info.json').read_text())
        eps = pd.concat([pd.read_parquet(x) for x in sorted((root / 'meta/episodes').rglob('*.parquet'))], ignore_index=True)
        tasks = pd.read_parquet(root / 'meta/tasks.parquet')
        task_text = {int(r.task_index): str(text) for text, r in tasks.iterrows()}
        assert len(eps) == info['total_episodes']
        assert eps.episode_index.is_unique
        videos, tables, sample_count = {}, {}, 0
        for path in sorted((root / 'data').rglob('*.parquet')):
            table = pq.read_table(path)
            frame = table.to_pandas()
            for col in ('action', 'observation.state'):
                array = np.stack(frame[col].values)
                assert array.shape == (len(frame), 10) and np.isfinite(array).all()
                c0, c1 = array[:, 3:6], array[:, 6:9]
                np.testing.assert_allclose(np.linalg.norm(c0, axis=1), 1, atol=2e-6)
                np.testing.assert_allclose(np.linalg.norm(c1, axis=1), 1, atol=2e-6)
                assert np.max(np.abs(np.sum(c0 * c1, axis=1))) < 2e-6
                assert array[:, 9].min() >= -1 and array[:, 9].max() <= 1
            sample_count += len(frame)
            tables[str(path)] = frame
        assert sample_count == info['total_frames']
        all_indices = np.concatenate([t['index'].to_numpy() for t in tables.values()])
        assert np.array_equal(np.sort(all_indices), np.arange(sample_count))
        for _, row in eps.iterrows():
            data_path = root / info['data_path'].format(chunk_index=int(row['data/chunk_index']), file_index=int(row['data/file_index']))
            table = tables[str(data_path)]
            rows = table[table.episode_index == row.episode_index]
            assert len(rows) == int(row.length)
            assert np.array_equal(rows.frame_index.to_numpy(), np.arange(int(row.length)))
            assert np.isfinite(rows.timestamp.to_numpy()).all()
            assert np.all(np.diff(rows.timestamp.to_numpy()) > 0)
            tids = rows.task_index.unique()
            assert len(tids) == 1 and task_text[int(tids[0])] in list(row.tasks)
            for camera in ('observation.images.image', 'observation.images.wrist_image'):
                key = f'videos/{camera}'
                path = root / info['video_path'].format(video_key=camera, chunk_index=int(row[f'{key}/chunk_index']), file_index=int(row[f'{key}/file_index']))
                assert path.is_file()
                first = int(round(float(row[f'{key}/from_timestamp']) * info['fps']))
                last = first + int(row.length)
                value = videos.setdefault(str(path), {'required_frames': 0, 'episodes': []})
                value['required_frames'] = max(value['required_frames'], last)
                value['episodes'].append([first, last])
        video_headers = []
        for path, required in videos.items():
            with av.open(path) as container:
                stream = container.streams.video[0]
                nframes = int(stream.frames)
                fps = float(stream.average_rate)
                if nframes <= 0:
                    nframes = sum(1 for packet in container.demux(stream) if packet.size > 0)
                assert nframes >= required['required_frames'], (path, nframes, required)
                assert abs(fps - info['fps']) < 0.01, (path, fps, info['fps'])
                ranges = sorted(required['episodes'])
                assert ranges[0][0] == 0
                assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])), (path, ranges[:3])
                video_headers.append({'file': str(Path(path).relative_to(root)), 'frames': nframes, 'fps': fps})
        reader = LiberoDataset.from_config({'type': 'libero', 'dataset_dir': str(root), 'action_mode': 'eef',
                    'unify_action': True, 'unify_action_map': ['0-9'], 'num_frames': 33, 'video_stride': 4,
                    'window_stride': 1, 'height': 384, 'width': 320, 'multiview': True, 'split': 'train',
                    'camera_layout': ['observation.images.image', 'observation.images.wrist_image', None],
                    'normalize_mode': 'min-max', 'color_jitter': {'enabled': False}})
        indices = {0, len(reader) // 2, len(reader) - 1}
        if root.name == 'libero_4suite':
            conv = json.loads((root / 'meta/conversion.json').read_text())
            for src in conv['sources']:
                indices.add(src['frame_offset'])
                if src['frame_offset']:
                    indices.add(src['frame_offset'] - 1)
        checked = []
        for idx in sorted(indices):
            # Bypass retry/skip and tolerate no missing wrist views during validation.
            failures_before = reader._wrist_fail_count
            sample = reader._getitem_impl(idx)
            assert reader._wrist_fail_count == failures_before
            assert sample['action'].shape == (32, 80)
            assert sample['proprio'].shape == (1, 80)
            assert sample['action_mask'].shape == (32, 80)
            assert not sample['action_mask'][:, 10:].any()
            assert not sample['proprio_mask'][:, 10:].any()
            assert len(sample['video']) == 9
            assert all(img.size == (320, 384) for img in sample['video'])
            assert np.isfinite(sample['action'].numpy()).all()
            canvas = np.asarray(sample['video'][0])
            assert canvas[256:, :160].max() > 0
            assert canvas[256:, 160:].max() == 0
            assert sample['prompt'].strip()
            checked.append({'window': idx, 'prompt': sample['prompt'], 'valid_video_frames': int(sample['video_mask'].sum()),
                            'valid_action_steps': int(sample['action_mask'][:, 0].sum())})
            if idx == 0:
                previews.append((root.name, sample['video'][0].copy()))
        # Check official deployment conversion, including gripper sign, on 128 actions.
        frame = next(iter(tables.values()))
        for value in frame.action.iloc[np.linspace(0, len(frame) - 1, min(128, len(frame)), dtype=int)]:
            out = native_eef10_to_libero7d(np.asarray(value))
            np.testing.assert_allclose(out[:3], np.asarray(value)[:3], atol=1e-6)
            np.testing.assert_allclose(out[6], -float(value[9]), atol=1e-6)
        report['buckets'].append({'name': root.name, 'episodes': len(eps), 'frames': sample_count,
                'reader_windows': len(reader), 'tasks': len(tasks), 'fps': info['fps'],
                'video_files': video_headers, 'strict_reader_samples': checked})
        print(time.strftime('%H:%M:%S'), 'Validated', root.name, sample_count, 'rows,', len(eps), 'episodes,', len(videos), 'video headers,', len(checked), 'reader samples', flush=True)
        del tables, reader
    sheet = Image.new('RGB', (3 * 320, 2 * 410), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, (name, image) in enumerate(previews):
        x, y = (i % 3) * 320, (i // 3) * 410
        draw.text((x + 5, y + 4), name, fill=(20, 20, 20))
        sheet.paste(image, (x, y + 24))
    sheet.save(args.output / 'reader_preview.jpg', quality=90)
    report['status'] = 'PASS'
    target = args.output / 'validation_report.json'
    temp = target.with_suffix('.json.writing')
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, target)


if __name__ == '__main__':
    main()
