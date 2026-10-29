"""Exercise resume selection and stop guards without GPUs or model payloads."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

source = (Path(__file__).parent/'launch_robodojo_volcano_resume.sh').read_text()
preflight = source.split("python - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    folder = root/'training'/'run'
    folder.mkdir(parents=True)
    for step in (22000, 24000):  # The keep3 policy may already have removed20k.
        state = folder/f'accel_state_step_{step}'
        state.mkdir()
        (state/'trainer_state.json').write_text(json.dumps(dict(global_step=step, opt_step=step)))
        (folder/f'checkpoint_step_{step}.safetensors').touch()
    (folder/'accel_state_step_26000').mkdir()  # In-progress save must not be selected.
    stop = root/'planned_stop_20000.json'
    stop.write_text('{"status":"stopped"}')
    seen = []
    fake_monitor = SimpleNamespace(weights_valid=lambda p: p.is_file(),
                                   state_valid=lambda p, step: seen.append(step) is None)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 8))
    env = dict(RESUME_CKPT_DIR=str(folder), NPROC_PER_NODE='8', NODE_RANK='1',
               SWANLAB_API_KEY='unit-test-only', OPENWAM_SWANLAB_KEY_FILE=str(root/'absent'))
    with patch.dict(os.environ, env), patch.dict(sys.modules, {'torch': fake_torch, 'monitor_robodojo_training': fake_monitor}):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(preflight, '<resume-preflight>', 'exec'), {})
        assert seen == [24000]
        stop.write_text('{"status":"armed"}')
        try:
            exec(compile(preflight, '<resume-preflight>', 'exec'), {})
        except AssertionError as error:
            assert 'not confirmed stopped' in str(error)
        else:
            raise AssertionError('A still-active original job was accepted')
        stop.write_text('{"status":"stopped"}')
        (folder/'checkpoint_step_24000.safetensors').unlink()
        try:
            exec(compile(preflight, '<resume-preflight>', 'exec'), {})
        except AssertionError as error:
            assert 'not complete/readable' in str(error)
        else:
            raise AssertionError('Invalid latest completion silently fell back to older state')
print('PASS: latest complete state after20k retention, incomplete-save skip, original-stop guard, corrupt-latest rejection')
