"""Lightweight multi-layer integration tests without constructing foundation weights."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1] / 'openwam/model/architectures/dual_system'
PACKAGE = '_vav_driver_test'
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + '.vav_driver', ROOT / 'vav_driver.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
VAVDriver = module.VAVDriver


class Expert(nn.Module):
    num_layers, num_heads, head_dim = 3, 2, 2
    def __init__(self, action=False):
        super().__init__()
        self.action = action
        self.qkv = nn.ModuleList([nn.Linear(4, 12, bias=False) for _ in range(3)])
        self.out = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)])
    def pre_attn_at_layer(self, layer, state):
        x = state.payload.x_action if self.action else state.hidden_states
        return (*self.qkv[layer](x).chunk(3, dim=-1), x)
    def post_attn_at_layer(self, layer, state, out, residual):
        x = residual + self.out[layer](out)
        if self.action:
            state.payload.x_action = x
        else:
            state.hidden_states = x
        return state


def states():
    def video():
        return NS(hidden_states=torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True),
                  prefix_kv_len=0, grid_height=1, grid_width=2, grid_frames=3)
    return video(), NS(payload=NS(x_action=torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True))), video()


def run(experts, inputs, checkpoint=False):
    # State copies preserve leaf input tensors for comparing all input gradients.
    v1, a, v2 = copy.copy(inputs[0]), copy.copy(inputs[1]), copy.copy(inputs[2])
    a.payload = copy.copy(inputs[1].payload)
    output = VAVDriver(*experts).run(v1, a, v2, use_gradient_checkpointing=checkpoint)
    return (output[0].hidden_states, output[1].payload.x_action, output[2].hidden_states)


def test_checkpoint_outputs_parameter_gradients_and_outer_states():
    torch.manual_seed(12)
    experts = [Expert().double(), Expert(True).double(), Expert().double()]
    other = copy.deepcopy(experts)
    inputs, other_inputs = states(), None
    other_inputs = copy.deepcopy(inputs)
    plain, checked = run(experts, inputs), run(other, other_inputs, True)
    for x, y in zip(plain, checked):
        torch.testing.assert_close(x, y)
    sum(x.square().sum() for x in plain).backward()
    sum(x.square().sum() for x in checked).backward()
    for ex, ref in zip(experts, other):
        for p, q in zip(ex.parameters(), ref.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
    for x, y in zip(checked, run(other, other_inputs, True)):
        torch.testing.assert_close(x, y)  # backward did not clobber final state
    for x, y in zip((inputs[0].hidden_states, inputs[1].payload.x_action, inputs[2].hidden_states),
                    (other_inputs[0].hidden_states, other_inputs[1].payload.x_action, other_inputs[2].hidden_states)):
        torch.testing.assert_close(x.grad, y.grad)


def test_multiple_layers_v2_does_not_change_v1_or_action():
    torch.manual_seed(18)
    experts = [Expert().double(), Expert(True).double(), Expert().double()]
    inputs = states()
    original = run(experts, inputs)
    grad_v1, grad_action = torch.autograd.grad(original[2].square().sum(),
        (inputs[0].hidden_states, inputs[1].payload.x_action))
    assert grad_v1.abs().sum() > 0  # indirect V1 -> A -> V2 across layers
    assert grad_action.abs().sum() > 0

    with torch.no_grad():
        for p in experts[2].parameters():
            p.add_(torch.randn_like(p) * 10)
    changed = run(experts, inputs)
    torch.testing.assert_close(original[0], changed[0], rtol=0, atol=0)
    torch.testing.assert_close(original[1], changed[1], rtol=0, atol=0)
    assert not torch.allclose(original[2], changed[2])
