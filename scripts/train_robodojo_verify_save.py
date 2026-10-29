"""Train with the real 60k-step schedule, save at step20, then exit for a resume test."""
import functools
import json
from pathlib import Path

from openwam.train import openwam_trainer as module
from openwam.train.utils.checkpointing import save_weights, save_full_state

original_log = module.OpenWAMTrainer.log_step


@functools.wraps(original_log)
def log_and_verify(self, **kwargs):
    original_log(self, **kwargs)
    if kwargs['global_step'] != 20:
        return
    assert kwargs['opt_step'] == 20 and self.cfg.training.gradient_accumulation_steps == 1
    save_weights(self.accelerator, self.architecture, kwargs['output_path'], 20, final=False)
    save_full_state(self.accelerator, kwargs['output_path'], 20, 20, kwargs['epoch'])
    self.accelerator.wait_for_everyone()
    if self.accelerator.is_main_process:
        Path(kwargs['output_path'], 'early_save_passed.json').write_text(json.dumps(
            dict(status='saved',global_step=20,opt_step=20,world_size=self.accelerator.num_processes),indent=2)+'\n')
    if kwargs['wandb_run'] is not None:
        kwargs['wandb_run'].finish()
    self.accelerator.wait_for_everyone()
    print('EARLY_CHECKPOINT_SAVED', self.accelerator.process_index, flush=True)
    # Do not call finish_training: it removes the optimizer states needed for this test.
    raise SystemExit(0)


module.OpenWAMTrainer.log_step = log_and_verify
import train
train.main()
