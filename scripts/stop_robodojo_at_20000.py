"""Finite watcher: stop only the authorized developer-machine job after save20k."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from monitor_robodojo_training import RUN, FOLDER, weights_valid, state_valid

PID = 2928795
START_TICKS = '1120864002'
TARGET = 20000
RECEIPT = RUN / 'planned_stop_20000.json'


def identity(pid):
    root = Path(f'/proc/{pid}')
    try:
        text = (root / 'stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return fields[19], fields[0], (root / 'cmdline').read_bytes()
    except FileNotFoundError:
        return None


def original_alive():
    current = identity(PID)
    return current is not None and current[0] == START_TICKS and current[1] != 'Z'


def verify_identity():
    current = identity(PID)
    assert current and current[0] == START_TICKS
    assert b'train_robodojo_cached_swanlab' in current[2] and str(FOLDER).encode() in current[2]
    assert os.getpgid(PID) == PID


def record(status, **extra):
    value = dict(status=status, pid=PID, start_ticks=START_TICKS, target_step=TARGET,
                 run_directory=str(FOLDER), updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), **extra)
    temporary = RECEIPT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(RECEIPT)
    print(json.dumps(value), flush=True)


def main():
    with (RUN / 'planned_stop_20000.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if RECEIPT.exists() and json.loads(RECEIPT.read_text())['status'] == 'stopped':
            return
        verify_identity()
        record('armed')
        state = FOLDER / 'accel_state_step_20000'
        marker = state / 'trainer_state.json'
        weight = FOLDER / 'checkpoint_step_20000.safetensors'
        while not marker.is_file():
            if not original_alive():
                record('failed', reason='original training exited before checkpoint20k completion')
                return
            time.sleep(0.5)
        verify_identity()
        # Stop the exact original group as soon as the all-rank save completion
        # marker appears. Validation never races continued training or writes.
        os.killpg(PID, signal.SIGSTOP)
        try:
            assert weights_valid(weight) and state_valid(state, TARGET)
            members = {}
            for entry in Path('/proc').iterdir():
                if entry.name.isdigit():
                    try:
                        if os.getpgid(int(entry.name)) == PID:
                            item = identity(int(entry.name))
                            if item and item[1] != 'Z':
                                members[entry.name] = item[0]
                    except ProcessLookupError:
                        pass
            assert str(PID) in members
            latest = json.loads((FOLDER / 'metrics.jsonl').read_text().splitlines()[-1])
            record('stopping', checkpoint_validated=True, members=members, last_observed_step=latest['micro_step'])
            os.killpg(PID, signal.SIGTERM)
        except BaseException:
            os.killpg(PID, signal.SIGCONT)
            raise
        os.killpg(PID, signal.SIGCONT)
        for _ in range(120):
            remaining = {pid: tick for pid, tick in members.items()
                         if (item := identity(int(pid))) and item[0] == tick and item[1] != 'Z'}
            if not remaining:
                record('stopped', checkpoint_validated=True, members=members,
                       last_observed_step=latest['micro_step'], resume_step=TARGET)
                return
            time.sleep(0.5)
        record('needs_attention', reason='SIGTERM issued; some original group members remain', remaining=remaining)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        record('failed', reason=f'{type(error).__name__}: {error}')
        raise
