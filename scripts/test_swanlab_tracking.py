"""Verify logging lifecycle and credential separation without cloud writes."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

path = Path(__file__).resolve().parents[1] / 'openwam/train/utils/swanlab_tracking.py'
spec = importlib.util.spec_from_file_location('swanlab_tracking_under_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    key = root / 'key'
    key.write_text('unit-test-only-credential')
    run = SimpleNamespace(log=MagicMock(), id='test-run', url='https://example.invalid/test-run')
    sdk = SimpleNamespace(login=MagicMock(), init=MagicMock(return_value=run),
                          finish=MagicMock(), __version__='test')
    env = dict(SWANLAB_API_KEY='', OPENWAM_SWANLAB_RESUME_ID='', OPENWAM_SWANLAB_KEY_FILE=str(key), OPENWAM_SWANLAB_RUN_NAME='test',
               OPENWAM_SWANLAB_LOGDIR=str(root/'logs'), OPENWAM_SWANLAB_PROJECT='test-project')
    with patch.dict(os.environ, env), patch.dict(sys.modules, {'swanlab': sdk}):
        tracker = module.init_swanlab(OmegaConf.create({'training': {'batch_size': 16}}))
        tracker.log({'train/loss': 0.5}, step=3)
        tracker.finish()
    sdk.login.assert_called_once_with(api_key='unit-test-only-credential', save=False)
    assert sdk.init.call_args.kwargs['mode'] == 'online'
    assert sdk.init.call_args.kwargs['public'] is False
    run.log.assert_called_once_with({'train/loss': 0.5}, step=3)
    sdk.finish.assert_called_once()
    assert 'unit-test-only-credential' not in json.dumps(sdk.init.call_args.kwargs)
    receipt = (root/'logs/openwam_run.json').read_text()
    assert 'unit-test-only-credential' not in receipt
    assert json.loads(receipt)['id'] == 'test-run'
    sdk.login.reset_mock()
    env.update(SWANLAB_API_KEY='unit-test-env-credential', OPENWAM_SWANLAB_KEY_FILE=str(root/'missing'),
               OPENWAM_SWANLAB_RESUME_ID='existing-test-run')
    with patch.dict(os.environ, env), patch.dict(sys.modules, {'swanlab': sdk}):
        module.init_swanlab(OmegaConf.create({'training': {'batch_size': 16}}))
    sdk.login.assert_called_once_with(api_key='unit-test-env-credential', save=False)
    assert sdk.init.call_args.kwargs['id'] == 'existing-test-run'
    assert sdk.init.call_args.kwargs['resume'] == 'must'
    assert 'unit-test-env-credential' not in json.dumps(sdk.init.call_args.kwargs)
    assert 'unit-test-env-credential' not in (root/'logs/openwam_run.json').read_text()
print('PASS: SwanLab lifecycle, private project, file/env credential separation and same-run resume')
