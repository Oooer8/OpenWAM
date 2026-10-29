"""Validation-only gradient reads at the ZeRO2 backward/step boundary."""
import types
import torch
from parameter_sampling import sample_parameter

CAPTURE_METHOD = 'deepspeed_safe_get_full_grad_after_backward_before_step'


def collect_gradient_samples(chosen):
    # All ranks must call in the same parameter order: the supported ZeRO2 API
    # assembles reduced gradient partitions collectively. Release each full
    # tensor before reading the next, retaining only 512 CPU samples per weight.
    from deepspeed.utils import safe_get_full_grad
    samples = {}
    for name, parameter in chosen:
        gradient = safe_get_full_grad(parameter)
        if gradient is None or tuple(gradient.shape) != tuple(parameter.shape):
            raise AssertionError(f'Missing or incorrectly shaped actual gradient: {name}')
        if not torch.isfinite(gradient).all().item():
            raise AssertionError(f'Nonfinite actual gradient: {name}')
        samples[name] = sample_parameter(gradient)
        del gradient
    return samples


def install_gradient_capture(engine, chosen, callback):
    if engine.zero_optimization_stage() != 2:
        raise AssertionError('This validation reader requires actual ZeRO stage 2')
    original_backward = engine.backward

    def backward(self, *args, **kwargs):
        result = original_backward(*args, **kwargs)
        # Accelerate calls engine.step() immediately after this returns. At this
        # exact point reduced gradients still exist and have not been clipped,
        # updated or cleared. Reading them never replaces the backward operation.
        callback(collect_gradient_samples(chosen))
        return result

    engine.backward = types.MethodType(backward, engine)
