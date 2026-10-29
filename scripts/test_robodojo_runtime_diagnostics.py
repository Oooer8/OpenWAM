"""Check phase evidence and worker-local caches without importing GPU packages."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile

scripts = Path(__file__).parent
spec = importlib.util.spec_from_file_location('runtime_probe_under_test', scripts/'check_robodojo_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    module.PROGRESS_PATH = root/'progress.json'
    with contextlib.redirect_stdout(io.StringIO()):
        assert module.measured('success', lambda: 7) == 7
        good = json.loads(module.PROGRESS_PATH.read_text())
        assert good['status'] == 'passed' and good['stage'] == 'success'
        def fail():
            raise RuntimeError('test-secret-must-not-enter-receipt')
        try:
            module.measured('failure', fail)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Import failure was swallowed')
    bad = json.loads(module.PROGRESS_PATH.read_text())
    assert bad['status'] == 'failed' and bad['error_type'] == 'RuntimeError'
    assert 'test-secret' not in module.PROGRESS_PATH.read_text()
    old = root/'old_cache'
    old.mkdir()
    marker = old/'keep'
    marker.write_text('existing cache must remain untouched')
    command = '''set -euo pipefail
source "$1"
[[ $DS_ACCELERATOR == cuda && $TORCHINDUCTOR_COMPILE_THREADS == 1 && $MAX_JOBS == 1 ]]
[[ $TRITON_CACHE_DIR == "$OPENWAM_LOCAL_CACHE_ROOT/triton" ]]
[[ $TORCH_EXTENSIONS_DIR == "$OPENWAM_LOCAL_CACHE_ROOT/extensions" ]]
[[ $TORCHINDUCTOR_CACHE_DIR == "$OPENWAM_LOCAL_CACHE_ROOT/inductor" ]]
[[ $CUDA_CACHE_PATH == "$OPENWAM_LOCAL_CACHE_ROOT/cuda" ]]
[[ $HOME == "$EXPECTED_HOME" ]]
'''
    result = subprocess.run(['bash', '-c', command, 'cache-test', str((scripts/'robodojo_runtime_env.sh').resolve())],
        env=dict(os.environ, OPENWAM_LOCAL_CACHE_ROOT=str(root/'worker_local'),
                 TRITON_CACHE_DIR=str(old), EXPECTED_HOME=os.environ['HOME']),
        text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == 'existing cache must remain untouched'
print('PASS: timed success/failure evidence, no exception secret in receipt, local cache routing, HOME/old caches preserved')
