"""Rank-zero SwanLab tracking through the trainer's existing log/finish interface."""
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


class SwanLabTracker:
    def __init__(self, run):
        self.run = run

    def log(self, metrics, step):
        self.run.log(metrics, step=step)

    def finish(self):
        import swanlab
        swanlab.finish()


def init_swanlab(cfg):
    import swanlab
    key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if not key:
        key = Path(os.environ["OPENWAM_SWANLAB_KEY_FILE"]).read_text().strip()
    swanlab.login(api_key=key, save=False)
    name = os.environ["OPENWAM_SWANLAB_RUN_NAME"]
    logdir = Path(os.environ["OPENWAM_SWANLAB_LOGDIR"])
    logdir.mkdir(parents=True, exist_ok=True)
    resume = {}
    if os.environ.get("OPENWAM_SWANLAB_RESUME_ID"):
        resume = dict(id=os.environ["OPENWAM_SWANLAB_RESUME_ID"], resume="must")
    run = swanlab.init(
        project=os.environ.get("OPENWAM_SWANLAB_PROJECT", "openwam-robodojo"),
        experiment_name=name,
        description="RoboDojo OpenWAM fine-tuning: 8x16, global batch128, frozen VAE cache, no augmentation",
        config=OmegaConf.to_container(cfg, resolve=True),
        logdir=str(logdir), mode="online", public=False, **resume,
    )
    # Credentials stay outside config, source, and the experiment receipt.
    receipt = dict(experiment_name=name, project=os.environ.get("OPENWAM_SWANLAB_PROJECT", "openwam-robodojo"),
                   swanlab_version=swanlab.__version__, mode="online")
    for field in ("id", "url"):
        value = getattr(run, field, None)
        if isinstance(value, str):
            receipt[field] = value
    path = logdir / "openwam_run.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("SWANLAB_INITIALIZED " + json.dumps(receipt), flush=True)
    return SwanLabTracker(run)
