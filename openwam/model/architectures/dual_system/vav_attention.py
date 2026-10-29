"""Routed V1/A/V2 attention for Wan training, without a dense 3-stream mask.

All Q/K/V tensors must be computed from the previous layer before calling.
Inputs use (batch, sequence, heads * head_dim), matching Wan/ActionDiT.
This module owns no parameters and never mutates its inputs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


ATTENTION_SOURCES = ((0, 1), (0, 1), (1, 2))


def _attend(q, k, v, num_heads):
    batch, _, width = q.shape
    def split(x):
        return x.reshape(batch, x.shape[1], num_heads, width // num_heads).transpose(1, 2)
    out = F.scaled_dot_product_attention(split(q), split(k), split(v), dropout_p=0.0)
    return out.transpose(1, 2).reshape(batch, q.shape[1], width)


def routed_mutual_attention(qkv_video, qkv_action, *, num_heads, first_frame_tokens):
    """Pruned V1/A path with the SAME SDPA call shapes as VAV training.

    A dense boolean-mask SDPA has the same mathematical visibility, but can
    select a different CUDA kernel. BF16 differences then accumulate through
    the full DiT. Physical V2 removal must preserve the routed kernel layout.
    """
    q, k, v = qkv_video
    qa, ka, va = qkv_action
    prefix = first_frame_tokens
    if not 1 <= prefix <= q.shape[1]:
        raise ValueError('Invalid clean video prefix')
    out = _attend(q[:, :prefix], k[:, :prefix], v[:, :prefix], num_heads)
    keys, values = torch.cat((k, ka), dim=1), torch.cat((v, va), dim=1)
    if prefix < q.shape[1]:
        out = torch.cat((out, _attend(q[:, prefix:], keys, values, num_heads)), dim=1)
    return out, _attend(qa, keys, values, num_heads)


def routed_vav_attention(
    qkv_v1: tuple[Tensor, Tensor, Tensor],
    qkv_action: tuple[Tensor, Tensor, Tensor],
    qkv_v2: tuple[Tensor, Tensor, Tensor],
    *,
    num_heads: int,
    first_frame_tokens: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """V1 reads V1+A; A reads V1+A; V2 reads A+V2.

    Each clean video prefix reads only its own clean prefix. Future video
    queries read their full allowed streams. No within-layer expert updates
    occur here. Cached extra prefix KV is deliberately unsupported: this
    helper targets the pinned Wan LIBERO training configuration.
    """
    tensors = (*qkv_v1, *qkv_action, *qkv_v2)
    if len(tensors) != 9 or any(x.ndim != 3 for x in tensors):
        raise ValueError('Expected three (Q,K,V) tuples of rank-three tensors')
    batch, _, width = tensors[0].shape
    if num_heads < 1 or width % num_heads:
        raise ValueError('Attention width must be divisible by num_heads')
    if any(x.shape[0] != batch or x.shape[2] != width for x in tensors):
        raise ValueError('All experts must share batch and attention width')
    if any(x.device != tensors[0].device or x.dtype != tensors[0].dtype for x in tensors):
        raise ValueError('All experts must share device and dtype')
    for group in (qkv_v1, qkv_action, qkv_v2):
        if any(x.shape != group[0].shape for x in group):
            raise ValueError('Extra prefix KV and differing Q/K/V lengths are unsupported')
        if group[0].shape[1] < 1:
            raise ValueError('Each expert needs at least one token')
    if qkv_v1[0].shape[1] != qkv_v2[0].shape[1]:
        raise ValueError('V1 and V2 must have identical sequence lengths')
    if not 1 <= first_frame_tokens <= qkv_v1[0].shape[1]:
        raise ValueError('first_frame_tokens must identify a nonempty video prefix')

    def attend(q, k, v):
        return _attend(q, k, v, num_heads)

    q1, k1, v1 = qkv_v1
    qa, ka, va = qkv_action
    q2, k2, v2 = qkv_v2
    prefix = first_frame_tokens
    out1 = attend(q1[:, :prefix], k1[:, :prefix], v1[:, :prefix])
    out2 = attend(q2[:, :prefix], k2[:, :prefix], v2[:, :prefix])
    if prefix < q1.shape[1]:
        out1 = torch.cat((out1, attend(q1[:, prefix:], torch.cat((k1, ka), dim=1),
                                     torch.cat((v1, va), dim=1))), dim=1)
        out2 = torch.cat((out2, attend(q2[:, prefix:], torch.cat((ka, k2), dim=1),
                                     torch.cat((va, v2), dim=1))), dim=1)
    outa = attend(qa, torch.cat((k1, ka), dim=1), torch.cat((v1, va), dim=1))
    return out1, outa, out2
