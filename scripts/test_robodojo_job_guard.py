"""Real shell tests of bounded failure retention and intentional termination."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

guard = Path(__file__).parent/'robodojo_job_guard.sh'
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    binary = root/'bin'
    binary.mkdir()
    (binary/'python').symlink_to(sys.executable)
    env = dict(os.environ, PATH=str(binary)+':'+os.environ['PATH'],
               OPENWAM_JOB_STATUS_DIR=str(root), OPENWAM_FAILURE_HOLD_SECONDS='1',
               DRY_RUN='0', NODE_RANK='0')
    script = 'set -euo pipefail\nsource "$1"\nexport OPENWAM_JOB_STAGE=unit_test\nexit 7\n'
    start = time.monotonic()
    result = subprocess.run(['bash', '-c', script, 'guard-test', str(guard)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 7 and time.monotonic()-start >= 1
    status = root/'volcano_job_status_node_0.json'
    value = json.loads(status.read_text())
    assert value['status'] == 'failed_hold_expired' and value['exit_code'] == 7
    assert 'training is NOT running' in result.stderr
    status.unlink()
    script = ('set -euo pipefail\nsource "$1"\n'
              'bash -c "exit 23" &\nOPENWAM_TRAIN_PID=$!\nopenwam_wait_child\n')
    result = subprocess.run(['bash', '-c', script, 'guard-test', str(guard)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 23
    assert json.loads(status.read_text())['exit_code'] == 23
    status.unlink()
    script = 'set -euo pipefail\nsource "$1"\nsleep 60 &\nOPENWAM_TRAIN_PID=$!\necho "$OPENWAM_TRAIN_PID" > "$OPENWAM_JOB_STATUS_DIR/child"\nwait "$OPENWAM_TRAIN_PID"\n'
    child = subprocess.Popen(['bash', '-c', script, 'guard-test', str(guard)], env=dict(env, OPENWAM_FAILURE_HOLD_SECONDS='30'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        if (root/'child').exists(): break
        time.sleep(.02)
    assert (root/'child').exists()
    child.send_signal(signal.SIGTERM)
    child.communicate(timeout=5)
    assert child.returncode == 143 and not status.exists(), 'Intentional stop must not hold resources'
    retry_entry = root/'retry_entry.sh'
    retry_entry.write_text('set -euo pipefail\nsource "'+str(guard.resolve())+'"\n'
        'if [[ ! -f "$OPENWAM_JOB_STATUS_DIR/repaired" ]]; then exit 9; fi\n'
        'echo "$$" > "$OPENWAM_JOB_STATUS_DIR/recovered_pid"\n')
    child = subprocess.Popen(['bash', str(retry_entry)], env=dict(env, OPENWAM_FAILURE_HOLD_SECONDS='2', OPENWAM_JOB_ENTRY=str(retry_entry)), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(200):
        if status.exists() and json.loads(status.read_text())['status'] == 'failed_holding': break
        time.sleep(.02)
    assert status.exists()
    (root/'repaired').touch()
    (root/'volcano_retry_node_0').write_text(str(child.pid)+'\n')
    child.communicate(timeout=8)
    assert child.returncode == 0 and int((root/'recovered_pid').read_text()) == child.pid
    assert json.loads(status.read_text())['status'] == 'completed'
print('PASS: bounded failure hold and exit code; operator TERM; explicit repaired retry keeps the same wrapper/allocation')
