"""Resume-safe eight-GPU LIBERO VAE cache. No training is started."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.libero_latent_cache import dataset_binding, validate_binding
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.robodojo_latent_cache import SHAPE, sha256_file, state_digest
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38
from openwam.model.video_backbone.wan.encode import preprocess_video, encode_video
from openwam.train.utils.seeding import seed_everything
from scripts.cache_robodojo_vae import atomic_json, load_state, claim_shard, validate_shard

SOURCES = ('openwam/model/video_backbone/wan/models/vae.py',
    'openwam/model/video_backbone/wan/preprocess.py', 'openwam/model/video_backbone/wan/encode.py',
    'openwam/train/utils/seeding.py', 'openwam/dataloader/libero.py',
    'openwam/dataloader/bases/lerobot_v3_reader.py', 'openwam/dataloader/utils/video_io.py',
    'openwam/dataloader/transforms/multiview.py', 'openwam/dataloader/libero_latent_cache.py',
    'scripts/cache_libero_vae.py', 'scripts/cache_robodojo_vae.py')


class WindowVideos:
    """Decode each episode once, assemble once, encode every window separately."""
    def __init__(self, ds):
        self.ds = ds
        self.ep = None
        self.canvases = None

    def __getitem__(self, index):
        ds = self.ds
        ep = int(np.searchsorted(ds._cum_n_starts, index, side='right') - 1)
        offset = int(ds._ep_valid_start[ep]) + (index - int(ds._cum_n_starts[ep])) * ds._window_stride
        end = int(ds._ep_valid_end[ep])
        if ep != self.ep:
            row = ds._eps_df.iloc[ep]
            cameras = [ds._head_camera, ds._left_wrist_camera, ds._right_wrist_camera]
            before = ds._wrist_fail_count
            decoded = {}
            for slot, camera in enumerate(cameras):
                if camera:
                    frames = ds._decode_one_camera(camera, row, ep, 0, np.arange(end), kind='head' if slot == 0 else 'wrist')
                    if len(frames) != end:
                        raise RuntimeError(f'Incomplete episode decode {ep}, {camera}: {len(frames)} != {end}')
                    decoded[camera] = frames
            assert ds._wrist_fail_count == before, 'Wrist fallback prohibited during cache generation'
            self.canvases = [assemble_multiview_layout({c: v[i] for c, v in decoded.items()},
                ds._camera_layout, out_h=ds._height, out_w=ds._width) for i in range(end)]
            self.ep = ep
        actual = min(ds._num_frames, end-offset)
        frames = [self.canvases[offset+int(i)] for i in ds._video_sample_indices if i < actual]
        frames += [frames[-1]] * (ds._num_video_frames-len(frames))
        return frames


def validation_indices(ds):
    indices = [0, len(ds)-1]
    for remaining in (1, 2, 4, 5, 9, 17, 32, 33):
        ep = len(ds._eps_df)//2
        start = int(ds._cum_n_starts[ep])
        end = int(ds._cum_n_starts[ep+1])
        if end-start >= remaining:
            indices.append(end-remaining)
    # Include every suite boundary in the merged bucket.
    conversion = json.loads((ds._dataset_dir/'meta/conversion.json').read_text())
    for source in conversion.get('sources', []):
        for index in (source['frame_offset']-1, source['frame_offset']):
            if 0 <= index < len(ds):
                indices.append(index)
    return sorted(set(indices))


def finalize(root, ds, build, chunks):
    with open(root/'finalize.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        digest = sha256_file(root/'build.json')
        shards = [validate_shard(root, a, b, digest) for a,b in chunks]
        checks = sorted(root.glob('worker_*_validation.json'))
        assert len(checks) == build['world_size'], 'Not every GPU passed validation'
        for check in checks:
            r = json.loads(check.read_text())
            assert r['status'] == 'passed' and r['build_sha256'] == digest
        # Check actual immutable dataset content again before publication.
        assert dataset_binding(ds) == build['dataset_binding'], 'Source data changed during cache generation'
        atomic_json(root/'manifest.json', dict(build, status='ready', shards=shards))
        atomic_json(root/'validation.json', dict(status='passed', windows=len(ds), shards=len(shards),
            all_shards_sha256_checked=True, all_dataset_files_sha256_checked=True,
            manifest_sha256=sha256_file(root/'manifest.json'), checks=[p.name for p in checks]))
    print('CACHE_READY', len(ds), flush=True)


def worker(args, root, ds, build, chunks):
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    seed_everything(42, rank=args.worker)
    state = load_state(args.checkpoint)
    assert state_digest(state) == build['vae_weights_sha256']
    vae = WanVideoVAE38()
    vae.load_state_dict(state, strict=True, assign=True)
    del state
    vae.cuda().bfloat16().eval().requires_grad_(False)
    videos = WindowVideos(ds)
    indices = validation_indices(ds)
    checked_elements = 0
    with torch.inference_mode():
        # Use the training implementation unchanged: per-image preprocessing,
        # torch.cat over the configured training microbatch, native encode_video.
        # No fused normalization, CUDA Graph, packed-pixel path or larger batch.
        for start in range(0, len(indices), build['batch_size']):
            selected = indices[start:start+build['batch_size']]
            selected += [selected[-1]]*(build['batch_size']-len(selected))
            canonical_frames, cached_decode_frames = [], []
            for index in selected:
                before = ds._wrist_fail_count
                canonical = ds._getitem_impl(index)['video']
                fast = videos[index]
                assert ds._wrist_fail_count == before
                assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a,b in zip(canonical,fast))
                canonical_frames.append(canonical)
                cached_decode_frames.append(fast)
            original_pixels = torch.cat([preprocess_video(f,dtype=torch.bfloat16,device='cuda') for f in canonical_frames])
            cache_pixels = torch.cat([preprocess_video(f,dtype=torch.bfloat16,device='cuda') for f in cached_decode_frames])
            assert torch.equal(original_pixels,cache_pixels)
            online = encode_video(original_pixels,vae=vae)
            offline = encode_video(cache_pixels,vae=vae)
            assert tuple(online.shape)==(build['batch_size'],*SHAPE)
            assert torch.equal(online,offline), 'Native training/cache VAE mismatch'
            assert torch.isfinite(offline).all()
            bits = offline.cpu().contiguous().view(torch.uint16).numpy()
            assert torch.equal(online.cpu(),torch.from_numpy(bits.copy()).view(torch.bfloat16))
            checked_elements += online.numel()
        del canonical_frames,cached_decode_frames,original_pixels,cache_pixels,online,offline,bits
    atomic_json(root/f'worker_{args.worker}_validation.json',dict(status='passed',
        build_sha256=sha256_file(root/'build.json'),indices=indices,
        canonical_reader_pixels_bitwise_equal=True,original_preprocess_bitwise_equal=True,
        native_training_latents_bitwise_equal=True,bf16_roundtrip_bitwise_equal=True,
        training_batch_size=build['batch_size'],cache_batch_size=build['batch_size'],
        fused_normalization=False,cuda_graph=False,execution_mode='native_training_vae',
        gpu=torch.cuda.get_device_name(),elements=checked_elements))
    print(json.dumps(dict(status='validated',worker=args.worker,execution_mode='native_training_vae',
        batch_size=build['batch_size'],bitwise_equal=True)),flush=True)

    def read_batch(indices):
        frames = [videos[i] for i in indices]
        frames += [frames[-1]]*(build['batch_size']-len(frames))
        return frames

    digest = sha256_file(root/'build.json')
    count = 0
    with ThreadPoolExecutor(max_workers=1) as pool, torch.inference_mode():
        for cid, (start,end) in enumerate(chunks):
            if cid % build['world_size'] != args.worker:
                continue
            with claim_shard(root, start, end, digest) as claimed:
                if not claimed:
                    continue
                began = time.perf_counter()
                dest = root/f'{start:08d}.bf16'
                partial = dest.with_suffix('.bf16.partial')
                out = np.memmap(partial, mode='w+', dtype=np.uint16, shape=(end-start,*SHAPE))
                batches = [list(range(i,min(i+build['batch_size'],end))) for i in range(start,end,build['batch_size'])]
                future = pool.submit(read_batch, batches[0])
                for b, batch_indices in enumerate(batches):
                    frames = future.result()
                    if b+1 < len(batches):
                        future = pool.submit(read_batch,batches[b+1])
                    pixels = torch.cat([preprocess_video(f,dtype=torch.bfloat16,device='cuda') for f in frames])
                    encoded = encode_video(pixels,vae=vae)
                    assert torch.isfinite(encoded).all(), batch_indices[0]
                    bits = encoded[:len(batch_indices)].cpu().contiguous().view(torch.uint16).numpy()
                    out[batch_indices[0]-start:batch_indices[-1]-start+1] = bits
                    del frames, pixels, encoded, bits
                    if b % 32 == 0 or b == len(batches)-1:
                        n = batch_indices[-1]-start+1
                        print(json.dumps(dict(status='encoding', worker=args.worker, chunk_start=start,
                            windows_in_chunk=n, windows_per_second=n/(time.perf_counter()-began),
                            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)),flush=True)
                out.flush()
                del out
                with open(partial,'rb') as f:
                    os.fsync(f.fileno())
                os.replace(partial,dest)
                receipt = dict(start=start,end=end,path=dest.name,size=dest.stat().st_size,
                    sha256=sha256_file(dest),build_sha256=digest,worker=args.worker,
                    seconds=time.perf_counter()-began,producer_sha256=sha256_file(__file__))
                atomic_json(root/f'{start:08d}.json',receipt)
                print(json.dumps(receipt),flush=True)
                count += 1
                if args.limit_chunks and count >= args.limit_chunks:
                    break


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--init',action='store_true')
    p.add_argument('--finalize',action='store_true')
    p.add_argument('--worker',type=int)
    p.add_argument('--world-size',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--chunk-size',type=int,default=1024)
    p.add_argument('--max-gpu-memory-gib',type=float,default=24)
    p.add_argument('--limit-chunks',type=int,default=0)
    p.add_argument('--training-config',default='/mnt/vepfs01/output/tiecheng.guo/artifacts/waw/libero_cache_integration_receipts/resolved_config.yaml')
    args=p.parse_args()
    torch.set_num_threads(2)
    train_config=yaml.safe_load(Path(args.training_config).read_text())
    assert args.batch_size == int(train_config['training']['batch_size']) == 4
    assert train_config['training']['mixed_precision']=='bf16'
    assert not any(os.environ.get(k) for k in ('OPENWAM_EXACT_VAE_NORM','OPENWAM_VAE_CHANNELS_LAST','OPENWAM_BATCH_VIDEO_PREPROCESS','OPENWAM_COMPILE_VAE_NORM'))
    root=Path(args.root)
    root.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load(Path(args.config).read_text())
    ds=LiberoDataset.from_config(config,split='train')
    assert ds._color_jitter is None and ds._head_camera_choices is None
    assert (ds._num_frames,ds._video_stride,ds._window_stride,ds._height,ds._width,ds._multiview)==(33,4,1,384,320,True)
    if args.init:
        state=load_state(args.checkpoint)
        build=dict(format='libero_window_v1',dtype='bfloat16_uint16_bits',shape=list(SHAPE),
            windows=len(ds),augmentation='none',encoding='independent_9_frame_windows',
            batch_size=args.batch_size,chunk_size=args.chunk_size,world_size=args.world_size,
            dataset_binding=dataset_binding(ds),vae_weights_sha256=state_digest(state),
            checkpoint=str(Path(args.checkpoint).resolve()),torch_version=str(torch.__version__),
            pixel_preprocess='original_per_image_preprocess_video',encoder_norm='native_unmodified',
            execution_mode='native_training_vae',fused_normalization=False,cuda_graph=False,
            training_config_sha256=sha256_file(args.training_config),
            encoding_source_sha256={name:sha256_file(name) for name in SOURCES})
        if (root/'build.json').exists():
            assert json.loads((root/'build.json').read_text())==build,'Existing cache provenance differs'
        else:
            atomic_json(root/'build.json',build)
        print(json.dumps(dict(status='initialized',windows=len(ds),bytes=len(ds)*int(np.prod(SHAPE))*2)),flush=True)
        return
    build=json.loads((root/'build.json').read_text())
    validate_binding(ds,build['dataset_binding'])
    assert all(sha256_file(name)==digest for name,digest in build['encoding_source_sha256'].items())
    chunks=[(s,min(s+build['chunk_size'],len(ds))) for s in range(0,len(ds),build['chunk_size'])]
    if args.finalize:
        finalize(root,ds,build,chunks)
    else:
        assert args.worker is not None and 0<=args.worker<build['world_size']
        with open(root/f'worker_{args.worker}.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            worker(args,root,ds,build,chunks)


if __name__=='__main__':
    main()
