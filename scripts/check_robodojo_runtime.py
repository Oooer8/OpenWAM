"""Check the real container, imports and cached dataset without starting training."""
import argparse
import datetime
import faulthandler
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# Importing Inductor can otherwise create32 compiler workers before any model
# is loaded. This probe needs no compiler pool and never trains a model.
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
PROGRESS_PATH = None
PHASE_TIMINGS = {}


def progress(stage, status, **extra):
    value = dict(stage=stage, status=status, pid=os.getpid(), hostname=socket.gethostname(),
                 updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 phase_seconds=PHASE_TIMINGS, **extra)
    if PROGRESS_PATH is not None:
        temporary = PROGRESS_PATH.with_name(PROGRESS_PATH.name + f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(PROGRESS_PATH)


def measured(stage, function, *args, **kwargs):
    print('PREFLIGHT BEGIN: ' + stage, flush=True)
    progress(stage, 'running')
    started = time.monotonic()
    try:
        value = function(*args, **kwargs)
    except BaseException as error:
        PHASE_TIMINGS[stage] = round(time.monotonic()-started, 3)
        progress(stage, 'failed', error_type=type(error).__name__)
        raise
    PHASE_TIMINGS[stage] = round(time.monotonic()-started, 3)
    progress(stage, 'passed')
    print(f'PREFLIGHT END: {stage} ({PHASE_TIMINGS[stage]:.3f}s)', flush=True)
    return value


def gpu_probe(torch, gpus):
    assert torch.cuda.device_count() == gpus, 'Visible GPU count differs from allocation'
    assert torch.cuda.is_available(), 'CUDA driver initialization failed'
    # A tiny allocation/synchronize isolates driver/runtime failures before DeepSpeed.
    value = torch.empty(1, device='cuda:0')
    torch.cuda.synchronize(0)
    del value
    return True


def memory_capacity():
    host = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemTotal:'))) * 1024
    limits = []
    for name in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        p = Path(name)
        if p.is_file():
            raw = p.read_text().strip()
            if raw.isdigit() and int(raw) < 1 << 60:
                limits.append(int(raw))
    return min([host] + limits), min(limits) if limits else None


def check(folder, gpus, cpu_only):
    torch = measured('import torch', importlib.import_module, 'torch')
    if not cpu_only:
        measured('CUDA inventory/context', gpu_probe, torch, gpus)
    measured('import hydra', importlib.import_module, 'hydra')
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from monitor_robodojo_training import weights_valid, state_valid

    folder = folder.resolve(strict=True)
    warnings = []
    versions = {}
    for module, package in [('deepspeed', 'deepspeed'), ('accelerate', 'accelerate'),
                            ('transformers', 'transformers'), ('diffusers', 'diffusers'),
                            ('swanlab', 'swanlab'), ('cv2', 'opencv-python'),
                            ('h5py', 'h5py'), ('safetensors', 'safetensors')]:
        measured('import ' + module, importlib.import_module, module)
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'importable'
    measured('import trainer', importlib.import_module, 'openwam.train.openwam_trainer')
    from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: F401
    from openwam.dataloader.registry import build_dataset
    with initialize_config_dir(config_dir=str(REPO/'configs'), version_base=None):
        config = compose(config_name='train_robodojo_cached_swanlab')
    print('PREFLIGHT: validating cache and reading a real sample', flush=True)
    dataset = measured('build cached dataset', build_dataset, config.dataloader, split='train')
    sample = measured('read cached sample', dataset.__getitem__, 0)
    assert tuple(sample['video_latents'].shape) == (48, 3, 24, 20)
    assert tuple(sample['action'].shape) == (32, 80)
    assert torch.isfinite(sample['video_latents']).all() and torch.isfinite(sample['action']).all()
    text = Path(config.dataloader.prepared_dir)/'text_cache'
    assert json.loads((text/'validation.json').read_text())['status'] == 'passed'
    assert (text/'index.json').is_file()
    saved = OmegaConf.load(folder/'config.yaml')
    spec = saved.model.video_backbone.tokenizer
    tokenizer_path = folder/spec.subdir
    from transformers import AutoTokenizer
    print('PREFLIGHT: loading local tokenizer', flush=True)
    tokenizer = measured('load local tokenizer', AutoTokenizer.from_pretrained, str(tokenizer_path), local_files_only=True)
    assert tokenizer(sample['prompt'])['input_ids']
    steps = [int(x.name.split('_')[-1]) for x in folder.glob('accel_state_step_*')
             if x.name.split('_')[-1].isdigit() and (x/'trainer_state.json').is_file()]
    assert steps, 'No completed resume state'
    step = max(steps)
    print(f'PREFLIGHT: checkpoint structure at step{step} and shared write/fsync', flush=True)
    assert measured('validate weights header', weights_valid, folder/f'checkpoint_step_{step}.safetensors')
    assert measured('validate full state structure', state_valid, folder/f'accel_state_step_{step}', step)
    with tempfile.NamedTemporaryFile(prefix='.startup_write_probe_', dir=folder) as probe:
        probe.write(b'OpenWAM startup write/fsync probe\n')
        probe.flush()
        os.fsync(probe.fileno())
    free = shutil.disk_usage(folder).free
    assert free >= 250_000_000_000, 'Less than250GB checkpoint disk space'
    capacity, limit = memory_capacity()
    shm = shutil.disk_usage('/dev/shm').total
    if limit is None:
        warnings.append('Container memory limit is not exposed: set platform RAM explicitly, recommended512GiB.')
    if cpu_only:
        warnings.append('CPU-only check: GPU kernels/NCCL and target GPU-job RAM/SHM limits are NOT verified.')
    else:
        assert capacity >= 384 * (1 << 30), 'Container RAM too small; request512GiB (guard minimum384GiB)'
        assert shm >= 8 * (1 << 30), 'Set platform /dev/shm to at least8GiB'
    devices = []
    if not cpu_only:
        assert torch.cuda.device_count() == gpus, 'Visible GPU count differs from allocation'
        for index in range(gpus):
            prop = torch.cuda.get_device_properties(index)
            available, total = torch.cuda.mem_get_info(index)
            assert total >= 78 * (1 << 30), f'GPU{index} must be an80GB-class device'
            assert available >= 76 * (1 << 30), f'GPU{index} is busy or lacks memory'
            assert prop.major >= 8, f'GPU{index} lacks the expected BF16 capability'
            devices.append(dict(index=index, name=prop.name, total_bytes=total, free_bytes=available))
    key = os.environ.get('SWANLAB_API_KEY', '').strip()
    if not key:
        path = Path(os.environ.get('OPENWAM_SWANLAB_KEY_FILE',
                         '/mnt/vepfs01/output/tiecheng.guo/.secrets/swanlab_api_key'))
        assert path.read_text().strip(), 'Empty SwanLab key file'
    return dict(status='passed', cpu_only=cpu_only, checkpoint_step=step,
                checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                python=sys.version, python_executable=sys.executable, base_prefix=sys.base_prefix,
                libc=platform.libc_ver(), torch=torch.__version__, torch_cuda=torch.version.cuda,
                versions=versions, dataset_windows=len(dataset), sample_latents_shape=list(sample['video_latents'].shape),
                phase_seconds=PHASE_TIMINGS, hostname=socket.gethostname(),
                cache_paths={name: os.environ.get(name) for name in ('TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH')},
                effective_memory_capacity_bytes=capacity, container_memory_limit_bytes=limit,
                shm_capacity_bytes=shm, disk_free_bytes=free, devices=devices, warnings=warnings,
                scope='Imports, CPU sample/cache/tokenizer, checkpoint structure, write/fsync; no full optimizer load or NCCL test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--report-path', type=Path, help='Separate diagnostic receipt; does not mark the platform job passed')
    args = parser.parse_args()
    PROGRESS_PATH = (args.report_path.with_suffix('.progress.json') if args.report_path else
                     args.checkpoint_dir.parent.parent / ('runtime_progress_node_' + os.environ.get('NODE_RANK', '0') + '.json'))
    # Active before torch/DeepSpeed imports; a timeout now leaves Python stacks.
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    if hasattr(signal, 'SIGTERM'):
        faulthandler.register(signal.SIGTERM, all_threads=True, chain=True)
    progress('bootstrap', 'running', python=sys.version, python_executable=sys.executable,
             libc=platform.libc_ver())
    if not args.cpu_only:
        def driver_snapshot():
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.free',
                                     '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=20)
            assert result.returncode == 0, 'nvidia-smi failed in this container'
            print('PREFLIGHT NVIDIA:\n' + result.stdout, flush=True)
        measured('NVIDIA driver inventory', driver_snapshot)
    report = check(args.checkpoint_dir, args.gpus, args.cpu_only)
    progress('all checks', 'passed')
    print(json.dumps(report, indent=2), flush=True)
    path = args.report_path or args.checkpoint_dir.parent.parent / ('runtime_preflight_' + ('cpu' if args.cpu_only else 'node_'+os.environ.get('NODE_RANK', '0')) + '.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(path)
