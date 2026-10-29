"""Resume-safe offline BF16 VAE cache. One worker per explicitly selected GPU.

--init creates immutable provenance; --worker writes assigned independent shards;
--finalize validates every shard, then atomically publishes the reader manifest.
No training or automatic process termination is performed by this script.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import fcntl

import numpy as np
import torch
import yaml
from safetensors import safe_open

from openwam.dataloader.robodojo_prepared import PreparedRoboDojoDataset
from openwam.dataloader.robodojo_latent_cache import SHAPE, sha256_file, state_digest
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38
from openwam.model.video_backbone.wan.encode import preprocess_video_batch
from openwam.train.utils.robodojo_exact_optimizations import install_exact_vae_norms

DEFAULT_CHECKPOINT = 'assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors'


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def load_state(checkpoint):
    prefix = 'video_backbone.vae.'
    with safe_open(checkpoint, framework='pt', device='cpu') as f:
        return {key[len(prefix):]: f.get_tensor(key) for key in f.keys() if key.startswith(prefix)}


def make_dataset(config):
    config = dict(config)
    config['color_jitter'] = {'enabled': False}
    config.pop('latent_cache_dir', None)
    config['decode_cache_size'] = 512
    return PreparedRoboDojoDataset(config, split='train')


def validate_shard(root, start, end, build_hash, *, verify_hash=True):
    receipt = json.loads((root / f'{start:08d}.json').read_text())
    path = root / receipt['path']
    assert receipt['start'] == start and receipt['end'] == end
    assert receipt['build_sha256'] == build_hash
    assert receipt['size'] == path.stat().st_size == (end-start) * int(np.prod(SHAPE)) * 2
    if verify_hash:
        assert receipt['sha256'] == sha256_file(path)
    return receipt


def pack_video_batch(frames):
    """Move CPU image packing into the existing prefetch thread."""
    shape = np.asarray(frames[0][0]).shape
    array = np.empty((len(frames), len(frames[0]), *shape), dtype=np.uint8)
    for i, clip in enumerate(frames):
        for j, image in enumerate(clip):
            array[i, j] = np.asarray(image)
    return torch.from_numpy(array).pin_memory()


def pixels_from_packed(packed):
    # Same operation sequence and BF16 roundings as preprocess_video_batch.
    pixels = packed.to(device='cuda', non_blocking=True).to(dtype=torch.bfloat16)
    pixels = pixels * (2 / 255) - 1
    return pixels.permute(0, 4, 1, 2, 3).contiguous()


def capture_exact_encoder(vae, pixels, reference):
    """Replay unchanged kernels; avoid any CPU→GPU constant copy in capture."""
    vae.scale = [value.to(device=pixels.device, dtype=pixels.dtype) for value in vae.scale]
    static_input = pixels.clone()
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            temporary = vae.batch_encode(static_input, device=static_input.device)
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()
    assert torch.equal(reference, temporary.cpu()), 'Pre-staging VAE scale changed values'
    del temporary
    torch.cuda.empty_cache()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = vae.batch_encode(static_input, device=static_input.device)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(reference, static_output.cpu()), 'Graph differs from eager VAE'

    def encode(value):
        static_input.copy_(value)
        graph.replay()
        return static_output
    return encode


@contextmanager
def claim_shard(root, start, end, build_hash):
    """All dynamic writers hold a per-shard advisory lock until atomic publish."""
    with open(root / f'{start:08d}.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            if (root / f'{start:08d}.json').exists():
                # Each published receipt was SHA-verified at write time; the
                # finalizer hashes every shard again. Do not reread 256GB per worker.
                validate_shard(root, start, end, build_hash, verify_hash=False)
                yield False
            else:
                yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', default='configs/dataloader/robodojo_prepared.yaml')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--max-gpu-memory-gib', type=float, default=16.)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--schedule', choices=('strided','dynamic'), default='strided')
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--finalize-on-complete', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--limit-chunks', type=int, default=0, help='Pilot only; incomplete caches cannot finalize')
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    ds = make_dataset(config)
    torch.set_num_threads(2)
    if args.init:
        state = load_state(args.checkpoint)
        build = dict(format=1, dtype='bfloat16_uint16_bits', shape=list(SHAPE), windows=len(ds),
                     augmentation='none', encoding='independent_9_frame_windows', batch_size=16,
                     chunk_size=args.chunk_size, prepared_root=str(ds.root.resolve()),
                     prepared_manifest_sha256=sha256_file(ds.root / 'manifest.json'),
                     vae_weights_sha256=state_digest(state), checkpoint=str(Path(args.checkpoint).resolve()),
                     torch_version=str(torch.__version__), pixel_preprocess='uint8_to_bfloat16_native',
                     encoder_norm='divide_affine_exact',
                     encoding_source_sha256={name: sha256_file(name) for name in (
                         'openwam/model/video_backbone/wan/models/vae.py',
                         'openwam/model/video_backbone/wan/preprocess.py',
                         'openwam/model/video_backbone/wan/encode.py',
                         'openwam/train/utils/robodojo_exact_optimizations.py')})
        existing = root / 'build.json'
        if existing.exists():
            assert json.loads(existing.read_text()) == build, 'Existing cache provenance differs'
        else:
            atomic_json(existing, build)
        print(json.dumps(build), flush=True)
        return
    build = json.loads((root / 'build.json').read_text())
    build_hash = sha256_file(root / 'build.json')
    assert build['prepared_manifest_sha256'] == sha256_file(ds.root / 'manifest.json')
    assert all(sha256_file(name) == digest for name, digest in build['encoding_source_sha256'].items())
    assert build['windows'] == len(ds) and build['shape'] == list(SHAPE)
    chunks = [(start, min(start+build['chunk_size'], len(ds)))
              for start in range(0, len(ds), build['chunk_size'])]
    if args.finalize:
        finalize_cache(root, ds, build, build_hash, chunks)
        return
    assert args.worker is not None and 0 <= args.worker < args.world_size
    with open(root / f'worker_{args.worker}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker(args, root, ds, build, build_hash, chunks)
    torch.cuda.empty_cache()
    if args.finalize_on_complete and all((root / f'{start:08d}.json').exists() for start, _ in chunks):
        try:
            finalize_cache(root, ds, build, build_hash, chunks)
        except BlockingIOError:
            print('Another cache worker is finalizing the completed cache', flush=True)


def finalize_cache(root, ds, build, build_hash, chunks):
    # Exclusive publication: no partial manifest is ever accepted by training.
    with open(root / 'finalize.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shards = [validate_shard(root, start, end, build_hash) for start, end in chunks]
        checks = list(root.glob('worker_*_validation.json'))
        assert checks, 'Missing online/cache numerical verification'
        for path in checks:
            report = json.loads(path.read_text())
            assert report['status'] == 'passed' and report['build_sha256'] == build_hash
        manifest = dict(build, status='ready', shards=shards)
        atomic_json(root / 'manifest.json', manifest)
        atomic_json(root / 'validation.json', dict(status='passed', windows=len(ds),
            shards=len(shards), all_shards_sha256_checked=True,
            manifest_sha256=sha256_file(root / 'manifest.json'), checks=[p.name for p in checks]))
    print('CACHE_READY', len(ds), flush=True)

def worker(args, root, ds, build, build_hash, chunks):
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(args.max_gpu_memory_gib * 2**30 / total, 1.0))
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    state = load_state(args.checkpoint)
    assert state_digest(state) == build['vae_weights_sha256']
    vae = WanVideoVAE38()
    vae.load_state_dict(state, strict=True, assign=True)
    del state
    vae.cuda().bfloat16().eval().requires_grad_(False)
    ds._open()
    # Include short tail windows (2, 4, 5, 9, 17, 32 frames) and corpus boundaries.
    indices = [0, len(ds)-1]
    for length in (2, 4, 5, 9, 17, 32, 33):
        found = np.flatnonzero(ds._windows[:, 2] == length)
        if len(found):
            indices.append(int(found[len(found)//2]))
    indices += np.random.default_rng(42).integers(0, len(ds), 16-len(indices)).tolist()
    validation_frames = [ds[i]['video'] for i in indices]
    pixels = preprocess_video_batch(validation_frames, dtype=torch.bfloat16, device='cuda')
    packed_reference = pixels_from_packed(pack_video_batch(validation_frames))
    assert torch.equal(pixels, packed_reference), 'Pixel prefetch changed BF16 input'
    del packed_reference, validation_frames
    with torch.inference_mode():
        native = vae.batch_encode(pixels, device='cuda')
        install_exact_vae_norms(vae, divide=True)
        fused = vae.batch_encode(pixels, device='cuda')
        assert tuple(native.shape) == (16, *SHAPE)
        assert torch.equal(native, fused), 'Encoder optimization is not bitwise equal on this GPU'
        assert torch.isfinite(fused).all()
        bits = fused.cpu().view(torch.uint16).numpy()
        roundtrip = torch.from_numpy(bits.copy()).view(torch.bfloat16).cuda()
        assert torch.equal(native, roundtrip)
    checked_elements = native.numel()
    reference_cpu = fused.cpu()
    del native, fused, bits, roundtrip
    encode = lambda value: vae.batch_encode(value, device=value.device)
    if args.cuda_graph:
        with torch.inference_mode():
            encode = capture_exact_encoder(vae, pixels, reference_cpu)
    atomic_json(root / f'worker_{args.worker}_validation.json', dict(status='passed',
        build_sha256=build_hash, indices=indices, native_fused_bitwise_equal=True,
        bf16_roundtrip_bitwise_equal=True, pixel_prefetch_bitwise_equal=True, cuda_graph_bitwise_equal=bool(args.cuda_graph),
        execution_mode='cuda_graph' if args.cuda_graph else 'eager',
        gpu=torch.cuda.get_device_name(), elements=checked_elements))
    del pixels, reference_cpu
    def read_batch(indices):
        frames = [ds[i]['video'] for i in indices]
        frames += [frames[-1]] * (16-len(frames))
        return pack_video_batch(frames)
    count = 0
    producer_digest = sha256_file(Path(__file__))
    # Graph activations are allocated once and retained; replay needs no new
    # activation workspace. Keep 2GiB free for pixel transfer and small checks.
    required_free = (2 if args.cuda_graph else 12) * 2**30
    with ThreadPoolExecutor(max_workers=1) as pool, torch.inference_mode():
        for cid, (start, end) in enumerate(chunks):
            if args.schedule == 'strided' and cid % args.world_size != args.worker:
                continue
            with claim_shard(root, start, end, build_hash) as claimed:
                if not claimed:
                    continue
                began = time.perf_counter()
                partial = root / f'{start:08d}.bf16.partial'
                dest = root / f'{start:08d}.bf16'
                out = np.memmap(partial, mode='w+', dtype=np.uint16, shape=(end-start, *SHAPE))
                batches = [list(range(i, min(i+16, end))) for i in range(start, end, 16)]
                future = pool.submit(read_batch, batches[0])
                for b, indices in enumerate(batches):
                    free, _ = torch.cuda.mem_get_info()
                    while free < required_free:
                        torch.cuda.empty_cache()
                        free, _ = torch.cuda.mem_get_info()
                        if free >= required_free:
                            break
                        print(json.dumps(dict(status='waiting_for_memory', free_gib=free/2**30)), flush=True)
                        time.sleep(30)
                        free, _ = torch.cuda.mem_get_info()
                    frames = future.result()
                    if b+1 < len(batches):
                        future = pool.submit(read_batch, batches[b+1])
                    pixels = pixels_from_packed(frames)
                    encoded = encode(pixels)
                    assert torch.isfinite(encoded).all(), f'Nonfinite VAE output at {indices[0]}'
                    bits = encoded[:len(indices)].cpu().contiguous().view(torch.uint16).numpy()
                    out[indices[0]-start:indices[-1]-start+1] = bits
                    if b in (0, len(batches)-1):
                        assert np.array_equal(out[indices[0]-start:indices[-1]-start+1], bits)
                    del pixels, encoded, bits, frames
                    if (b+1) % 16 == 0:
                        elapsed = time.perf_counter()-began
                        print(json.dumps(dict(status='encoding', execution_mode='cuda_graph' if args.cuda_graph else 'eager',
                            schedule=args.schedule, chunk_start=start, windows_in_chunk=indices[-1]-start+1,
                            windows_per_second=(indices[-1]-start+1)/elapsed,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                            current_reserved_gib=torch.cuda.memory_reserved()/2**30)), flush=True)
                out.flush()
                del out
                with open(partial, 'rb') as f:
                    os.fsync(f.fileno())
                os.replace(partial, dest)
                receipt = dict(start=start, end=end, path=dest.name, size=dest.stat().st_size,
                               sha256=sha256_file(dest), build_sha256=build_hash,
                               seconds=time.perf_counter()-began, worker=args.worker,
                               schedule=args.schedule, execution_mode='cuda_graph' if args.cuda_graph else 'eager',
                               producer_sha256=producer_digest)
                atomic_json(root / f'{start:08d}.json', receipt)
                print(json.dumps(receipt), flush=True)
                count += 1
                if args.limit_chunks and count >= args.limit_chunks:
                    return



if __name__ == '__main__':
    main()
