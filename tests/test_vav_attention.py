"""CPU mathematical checks for mutual V1/A and one-way A to V2."""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


path = Path(__file__).resolve().parents[1] / 'openwam/model/architectures/dual_system/vav_attention.py'
spec = importlib.util.spec_from_file_location('vav_attention_under_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def dense_reference(groups, heads, prefix):
    sizes = [g[0].shape[1] for g in groups]
    edges = [0, sizes[0], sizes[0] + sizes[1], sum(sizes)]
    mask = torch.zeros(sum(sizes), sum(sizes), dtype=torch.bool, device=groups[0][0].device)
    for target, sources in enumerate(((0, 1), (0, 1), (1, 2))):
        for source in sources:
            mask[edges[target]:edges[target + 1], edges[source]:edges[source + 1]] = True
    for video in (0, 2):
        start = edges[video]
        mask[start:start + prefix] = False
        mask[start:start + prefix, start:start + prefix] = True
    b, _, width = groups[0][0].shape
    q, k, v = [torch.cat([g[i] for g in groups], dim=1) for i in range(3)]
    q, k, v = [x.reshape(b, -1, heads, width // heads).transpose(1, 2) for x in (q, k, v)]
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    return result.transpose(1, 2).reshape(b, -1, width).split(sizes, dim=1)


@pytest.mark.parametrize('prefix', [1, 2, 5])
def test_routed_matches_dense_outputs_and_gradients(prefix):
    gen = torch.Generator().manual_seed(197)
    groups = [tuple(torch.randn(2, n, 12, generator=gen, dtype=torch.float64,
                               requires_grad=True) for _ in range(3)) for n in (5, 3, 5)]
    ref_groups = [tuple(x.detach().clone().requires_grad_() for x in g) for g in groups]
    actual = module.routed_vav_attention(*groups, num_heads=3, first_frame_tokens=prefix)
    expected = dense_reference(ref_groups, 3, prefix)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-10)
    sum(x.square().sum() for x in actual).backward()
    sum(x.square().sum() for x in expected).backward()
    for a, b in zip(groups, ref_groups):
        for x, y in zip(a, b):
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-9, atol=1e-9)


def test_forbidden_paths_and_clean_prefix_are_invariant():
    gen = torch.Generator().manual_seed(211)
    groups = [tuple(torch.randn(1, n, 8, generator=gen, dtype=torch.float64)
                    for _ in range(3)) for n in (5, 3, 5)]
    def run(gs):
        return module.routed_vav_attention(*gs, num_heads=2, first_frame_tokens=2)
    original = run(groups)
    changed_v2 = [groups[0], groups[1], tuple(x + 7 for x in groups[2])]
    changed_action = [groups[0], tuple(x + 5 for x in groups[1]), groups[2]]
    changed_v1 = [tuple(x + 3 for x in groups[0]), groups[1], groups[2]]
    for i in (0, 1):
        torch.testing.assert_close(run(changed_v2)[i], original[i], rtol=0, atol=0)
    torch.testing.assert_close(run(changed_action)[0][:, :2], original[0][:, :2], rtol=0, atol=0)
    assert not torch.allclose(run(changed_action)[0][:, 2:], original[0][:, 2:])
    assert not torch.allclose(run(changed_v1)[1], original[1])
    assert not torch.allclose(run(changed_action)[2][:, 2:], original[2][:, 2:])
    torch.testing.assert_close(run(changed_v1)[2], original[2], rtol=0, atol=0)
    torch.testing.assert_close(run(changed_action)[2][:, :2], original[2][:, :2], rtol=0, atol=0)
    future_changed = [tuple(x.clone() for x in g) for g in groups]
    for video in (0, 2):
        for tensor in future_changed[video]:
            tensor[:, 2:] += 11
    for video in (0, 2):
        torch.testing.assert_close(run(future_changed)[video][:, :2], original[video][:, :2], rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_pruned_attention_preserves_routed_calls_and_gradients(dtype):
    torch.manual_seed(42)
    groups=[tuple(torch.randn(2,n,48,dtype=dtype,requires_grad=True) for _ in range(3)) for n in (60,16,60)]
    independent=[tuple(x.detach().clone().requires_grad_() for x in g) for g in groups[:2]]
    full=module.routed_vav_attention(*groups,num_heads=2,first_frame_tokens=20)
    pruned=module.routed_mutual_attention(*independent,num_heads=2,first_frame_tokens=20)
    for a,b in zip(full[:2],pruned):torch.testing.assert_close(a,b,rtol=0,atol=0)
    sum(x.float().square().mean() for x in full[:2]).backward()
    sum(x.float().square().mean() for x in pruned).backward()
    for left,right in zip(groups[:2],independent):
        for a,b in zip(left,right):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
