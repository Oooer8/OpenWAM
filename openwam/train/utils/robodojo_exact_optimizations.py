"""Opt-in frozen-VAE fusion retaining BF16 rounding at each eager boundary."""
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rounded_rms_affine(X, D, G, Y, N, C: tl.constexpr, S: tl.constexpr,
                        SCALE: tl.constexpr, DIVIDE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    channel = (i // S) % C
    x = tl.load(X + i, valid, 0).to(tl.float32)
    if DIVIDE:
        denominator_index = (i // (C * S)) * S + i % S
        denominator = tl.load(D + denominator_index, valid, 1).to(tl.float32)
        x = tl.div_rn(x, denominator).to(tl.bfloat16).to(tl.float32)
    x = (x * SCALE).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(G + channel, valid, 0).to(tl.float32)
    x = (x * weight).to(tl.bfloat16).to(tl.float32)
    x = (x + 0.0).to(tl.bfloat16)
    tl.store(Y + i, x, valid)


def exact_vae_norm(x, gamma, scale, *, divide=True):
    if (x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous()
            or not gamma.is_contiguous() or gamma.dtype != x.dtype):
        return F.normalize(x, dim=1) * scale * gamma + 0.0
    if torch.is_grad_enabled() and (x.requires_grad or gamma.requires_grad):
        raise RuntimeError("Exact fused VAE norm is only for frozen no-grad encoding")
    if divide:
        value = x
        denominator = x.norm(2, dim=1, keepdim=True).clamp_min(1e-12)
    else:
        value = F.normalize(x, dim=1)
        denominator = gamma  # unused in this compiled specialization
    output = torch.empty_like(x)
    channels = x.shape[1]
    spatial = x.numel() // (x.shape[0] * channels)
    _rounded_rms_affine[(triton.cdiv(x.numel(), 1024),)](
        value, denominator, gamma, output, x.numel(), channels, spatial,
        float(scale), divide, 1024, num_warps=4, enable_fp_fusion=False,
    )
    return output


def install_exact_vae_norms(vae, *, divide=True):
    from openwam.model.video_backbone.wan.models.vae import RMS_norm
    if vae.training or any(p.requires_grad for p in vae.parameters()):
        raise ValueError("Only a frozen eval-mode VAE can use this optimization")

    def forward(module, x):
        return exact_vae_norm(x, module.gamma, module.scale, divide=divide)

    count = 0
    for module in vae.model.encoder.modules():
        if isinstance(module, RMS_norm):
            if not module.channel_first or isinstance(module.bias, torch.Tensor) or module.bias != 0:
                raise ValueError("Expected channel-first, bias-free native VAE RMS normalization")
            module.forward = types.MethodType(forward, module)
            count += 1
    return count
