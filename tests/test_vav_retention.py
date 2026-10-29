import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1]/'openwam/train/utils/checkpointing.py'
spec = importlib.util.spec_from_file_location('vav_checkpointing_test',path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pair(root, step, complete=True):
    (root/f'checkpoint_step_{step}.safetensors').write_bytes(b'weights')
    state=root/f'accel_state_step_{step}'
    state.mkdir()
    if complete:(state/'trainer_state.json').write_text('{}')


def test_new_incomplete_state_never_prunes_recoverable_pairs(tmp_path):
    for step in (1,2,3):pair(tmp_path,step)
    pair(tmp_path,4,complete=False)
    with pytest.raises(RuntimeError,match='refusing retention'):
        module.manage_checkpoints(str(tmp_path),3)
    assert len(list(tmp_path.glob('checkpoint_step_*')))==4
    assert (tmp_path/'accel_state_step_1/trainer_state.json').is_file()
    (tmp_path/'accel_state_step_4/trainer_state.json').write_text('{}')
    module.manage_checkpoints(str(tmp_path),3)
    assert sorted(p.name for p in tmp_path.glob('checkpoint_step_*'))==[
        f'checkpoint_step_{i}.safetensors' for i in (2,3,4)]
    assert sorted(p.name for p in tmp_path.glob('accel_state_step_*'))==[
        f'accel_state_step_{i}' for i in (2,3,4)]
