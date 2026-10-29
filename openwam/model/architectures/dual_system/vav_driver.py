"""Synchronous V1/A mutual with A -> V2 layers, owning no parameters."""
from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
from torch.utils.checkpoint import checkpoint

from .vav_attention import routed_vav_attention, routed_mutual_attention
from .mot_driver import DualSystemMoTDriver


class RoutedMutualMoTDriver(DualSystemMoTDriver):
    """VAV deployment subgraph; no V2 module, tensors or parameters required."""
    def __init__(self, video, action, **kwargs):
        super().__init__(video, action, **kwargs)
        if self.attention_mask_mode != 'mutual' or video.video_attention_mask_mode != 'first_frame_causal':
            raise ValueError('Routed VAV deployment requires mutual/first_frame_causal')

    def _step_impl(self, layer_id, vstate, astate, attn_mask=None, *, suppress_inner_attn_ckpt=False):
        if vstate.prefix_kv_len:
            raise ValueError('Routed VAV deployment does not support extra prefix KV')
        pv = self.vb.pre_attn_at_layer(layer_id, vstate)
        pa = self.ab.pre_attn_at_layer(layer_id, astate)
        ov, oa = routed_mutual_attention(pv[:3], pa[:3], num_heads=self.num_heads,
            first_frame_tokens=self._video_tokens_per_frame(vstate))
        vstate = self.vb.post_attn_at_layer(layer_id, vstate, ov.contiguous(), pv[3])
        astate = self.ab.post_attn_at_layer(layer_id, astate, oa.contiguous(), pa[3])
        return vstate, astate


class VAVDriver:
    def __init__(self, video, action, video2):
        self.video, self.action, self.video2 = video, action, video2
        for attr in ('num_layers', 'num_heads', 'head_dim'):
            if len({getattr(x, attr) for x in (video, action, video2)}) != 1:
                raise ValueError(f'VAV experts must have equal {attr}')
        self.num_layers, self.num_heads = video.num_layers, video.num_heads

    def _step(self, layer, v1, a, v2, prefix):
        # All nine projections precede every expert update. In particular V2
        # reads A from layer l, never the already updated A from layer l+1.
        p1 = self.video.pre_attn_at_layer(layer, v1)
        pa = self.action.pre_attn_at_layer(layer, a)
        p2 = self.video2.pre_attn_at_layer(layer, v2)
        outputs = routed_vav_attention(p1[:3], pa[:3], p2[:3],
                                       num_heads=self.num_heads, first_frame_tokens=prefix)
        v1 = self.video.post_attn_at_layer(layer, v1, outputs[0].contiguous(), p1[3])
        a = self.action.post_attn_at_layer(layer, a, outputs[1].contiguous(), pa[3])
        v2 = self.video2.post_attn_at_layer(layer, v2, outputs[2].contiguous(), p2[3])
        return v1, a, v2

    def _checkpointed_step(self, layer, v1, a, v2, prefix, offload):
        # Checkpoint recomputation must not mutate the outer state containers.
        def run(x1, xa, x2):
            s1, sa, s2 = copy.copy(v1), copy.copy(a), copy.copy(v2)
            sa.payload = copy.copy(a.payload)
            s1.hidden_states, sa.payload.x_action, s2.hidden_states = x1, xa, x2
            s1, sa, s2 = self._step(layer, s1, sa, s2, prefix)
            return s1.hidden_states, sa.payload.x_action, s2.hidden_states
        context = torch.autograd.graph.save_on_cpu() if offload else nullcontext()
        with context:
            x1, xa, x2 = checkpoint(run, v1.hidden_states, a.payload.x_action,
                                    v2.hidden_states, use_reentrant=False)
        v1.hidden_states, a.payload.x_action, v2.hidden_states = x1, xa, x2
        return v1, a, v2

    def run(self, v1, a, v2, *, use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False):
        if v1.prefix_kv_len or v2.prefix_kv_len:
            raise ValueError('VAV currently supports native Wan without extra prefix KV')
        prefix = v1.grid_height * v1.grid_width
        if (v1.grid_frames, v1.grid_height, v1.grid_width) != (v2.grid_frames, v2.grid_height, v2.grid_width):
            raise ValueError('VAV video grids differ')
        if use_gradient_checkpointing_offload and not use_gradient_checkpointing:
            raise ValueError('Activation offload requires gradient checkpointing')
        for layer in range(self.num_layers):
            if use_gradient_checkpointing:
                v1, a, v2 = self._checkpointed_step(layer, v1, a, v2, prefix,
                                                    use_gradient_checkpointing_offload)
            else:
                v1, a, v2 = self._step(layer, v1, a, v2, prefix)
        return v1, a, v2
