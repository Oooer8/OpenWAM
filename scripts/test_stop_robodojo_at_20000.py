"""Exercise the stop watcher without sending any operating-system signals."""
import json
from pathlib import Path
import signal
import tempfile
from unittest.mock import patch

import stop_robodojo_at_20000 as watcher

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    folder = root/'training'
    state = folder/'accel_state_step_20000'
    state.mkdir(parents=True)
    (state/'trainer_state.json').write_text('{"global_step":20000,"opt_step":20000}')
    (folder/'metrics.jsonl').write_text('{"micro_step":20000}\n')
    signals = []
    def send(group, sig):
        assert group == watcher.PID
        signals.append(sig)
    def identity(pid):
        if signal.SIGTERM in signals:
            return None
        return (watcher.START_TICKS, 'T', b'train_robodojo_cached_swanlab '+str(folder).encode())
    original_iterdir = Path.iterdir
    def iterdir(path):
        return iter([Path('/proc')/str(watcher.PID)]) if path == Path('/proc') else original_iterdir(path)
    with patch.multiple(watcher, RUN=root, FOLDER=folder, RECEIPT=root/'planned_stop_20000.json'), \
         patch.object(watcher, 'identity', identity), patch.object(watcher.os, 'killpg', send), \
         patch.object(watcher.os, 'getpgid', lambda pid: watcher.PID), \
         patch.object(watcher, 'weights_valid', lambda p: True), \
         patch.object(watcher, 'state_valid', lambda p,s: s==20000), patch.object(Path,'iterdir',iterdir):
        watcher.main()
        assert signals == [signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT]
        assert json.loads(watcher.RECEIPT.read_text())['status']=='stopped'
        with patch.object(watcher, 'identity', lambda pid: ('reused', 'R', b'other task')):
            try:
                watcher.verify_identity()
            except AssertionError:
                pass
            else:
                raise AssertionError('Reused PID was not rejected')
        watcher.RECEIPT.unlink()
        signals.clear()
        with patch.object(watcher, 'weights_valid', lambda p: False):
            try:
                watcher.main()
            except AssertionError:
                pass
            else:
                raise AssertionError('Invalid checkpoint was not rejected')
        assert signals == [signal.SIGSTOP, signal.SIGCONT], 'Validation failure must resume without TERM'
        watcher.RECEIPT.unlink()
        (state/'trainer_state.json').unlink()
        signals.clear()
        with patch.object(watcher, 'original_alive', lambda: False):
            watcher.main()
        assert not signals, 'Incomplete checkpoint must never trigger signals'
        assert json.loads(watcher.RECEIPT.read_text())['status'] == 'failed'
print('PASS: completed checkpoint, exact process group, reused PID rejection, invalid/incomplete save guards')
