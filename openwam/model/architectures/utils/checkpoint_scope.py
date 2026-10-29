"""Restrict checkpoint recomputation to the FFN in selected joint layers."""
from contextlib import contextmanager
from contextvars import ContextVar


ffn_checkpoint_active = ContextVar('openwam_ffn_checkpoint_active', default=False)


@contextmanager
def checkpoint_ffn_scope():
    token = ffn_checkpoint_active.set(True)
    try:
        yield
    finally:
        ffn_checkpoint_active.reset(token)
