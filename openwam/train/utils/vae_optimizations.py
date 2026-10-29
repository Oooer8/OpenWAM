"""Optional frozen native-VAE inference optimizations, preserving state keys."""
import types

import torch
import torch.nn.functional as F


def _vae_norm(x, weight):
    return F.normalize(x, dim=1) * (x.shape[1] ** .5) * weight


def compile_vae_encoder_norms(vae):
    from openwam.model.video_backbone.wan.models.vae import RMS_norm
    compiled = torch.compile(_vae_norm, fullgraph=True, dynamic=True)

    def forward(module, x):
        return compiled(x, module.gamma)

    count = 0
    for module in vae.model.encoder.modules():
        if isinstance(module, RMS_norm):
            if not module.channel_first or isinstance(module.bias, torch.Tensor) or module.bias != 0:
                raise ValueError('VAE fusion requires channel-first, bias-free RMS normalization')
            module.forward = types.MethodType(forward, module)
            count += 1
    return count


def use_vae_channels_last(vae):
    torch.nn.utils.memory_format.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
