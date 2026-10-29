"""Wan VAE / text-encoder IO with native-VAE ↔ external-encoder routing.

Free functions — ``vae`` / ``encoder`` / ``tokenizer`` / ``text_encoder`` passed
explicitly so the backbone class holds no IO logic. ``encoder is not None``
selects the external-encoder path; otherwise the native Wan VAE is used.
"""

from __future__ import annotations

from typing import Tuple
import os
import numpy as np

import torch
from torch import Tensor

from openwam.model.video_backbone.wan.preprocess import preprocess_video as _preprocess_video_native
from openwam.model.video_backbone.wan.preprocess import vae_output_to_video


def encode_text(prompts: list, *, tokenizer, text_encoder, device) -> Tuple[Tensor, Tensor]:
    from openwam.model.video_backbone.wan.text_cache import configured_text_cache
    cache = configured_text_cache()
    if cache is not None:
        return cache.encode(prompts, text_encoder=text_encoder, device=device,
                            compact=os.environ.get('OPENWAM_COMPACT_TEXT_CONTEXT') == '1')
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        context[i, v:] = 0
    return context, seq_lens


def compact_masked_text_context(context: Tensor, seq_lens: Tensor) -> Tensor:
    """Drop only universally masked suffix tokens AFTER the frozen T5 encoder.

    Text/proprio cross-attention has no positional encoding. Proprio is appended
    after this operation and retains its explicit True mask. T5 itself always
    receives the original 512 tokens, preserving its full encoding numerics.
    """
    longest = int(seq_lens.max())
    width = min(context.shape[1], max(32, ((longest + 31) // 32) * 32))
    return context[:, :width].contiguous()


def encode_text_for_inference(
    prompt, *, vace_cache, prompt_embed_cache, tokenizer, text_encoder, device
) -> Tuple[Tensor, Tensor]:
    """Deploy ``(context, seq_lens)``, reusing a prompt-keyed cached embed."""
    if vace_cache and vace_cache.get("populated") and vace_cache.get("prompt_key") == prompt:
        return vace_cache["context"], vace_cache["seq_lens"]
    if prompt_embed_cache is not None and prompt in prompt_embed_cache:
        return prompt_embed_cache[prompt]
    context, seq_lens = encode_text([prompt], tokenizer=tokenizer, text_encoder=text_encoder, device=device)
    if prompt_embed_cache is not None:
        prompt_embed_cache[prompt] = (context, seq_lens)
    return context, seq_lens


def preprocess_video(frames, *, encoder=None, dtype, device) -> Tensor:
    if encoder is not None:
        return encoder.preprocess_video(frames)
    return _preprocess_video_native(frames, dtype=dtype, device=device)


def preprocess_video_batch(frames, *, encoder=None, dtype, device) -> Tensor:
    if encoder is not None:
        return torch.cat([preprocess_video(x, encoder=encoder, dtype=dtype, device=device) for x in frames], dim=0)
    # RGB uint8 values 0..255 are represented exactly in BF16. Transfer the
    # compact uint8 batch, then cast on GPU before the unchanged BF16 arithmetic.
    # Building/transferring a float32 pixel batch wastes four times the bandwidth.
    arrays = [[np.asarray(image) for image in clip] for clip in frames]
    shape = arrays[0][0].shape
    if any(a.dtype != np.uint8 or a.shape != shape for clip in arrays for a in clip):
        return torch.cat([preprocess_video(x, dtype=dtype, device=device) for x in frames], dim=0)
    array = np.empty((len(frames), len(frames[0]), *shape), dtype=np.uint8)
    for i, clip in enumerate(arrays):
        for j, pixels in enumerate(clip):
            array[i, j] = pixels
    pixels = torch.from_numpy(array).to(device=device).to(dtype=dtype)
    pixels = pixels * (2 / 255) - 1
    return pixels.permute(0, 4, 1, 2, 3).contiguous()


def encode_video(video_tensor: Tensor, *, vae, encoder=None) -> Tensor:
    if encoder is not None:
        return encoder.batch_encode(video_tensor)
    if os.environ.get('OPENWAM_VAE_CHANNELS_LAST') == '1':
        video_tensor = video_tensor.contiguous(memory_format=torch.channels_last_3d)
    return vae.batch_encode(video_tensor, device=video_tensor.device)


def encode_video_for_vace(
    pixels: Tensor,
    *,
    vae,
    encoder=None,
    device,
    tiled: bool,
    tile_size: tuple,
    tile_stride: tuple,
) -> Tensor:
    """Tiled-aware VAE encode for the VACE pixel→latent helper, returning
    ``(B, z_dim, T_lat, H_lat, W_lat)``. ``tiled=False`` (training) batches in
    one call; ``tiled=True`` (deploy) loops per-sample with bounded peak memory,
    mirroring the vendored unit so large-frame deploy does not OOM.
    """
    if not tiled:
        return encode_video(pixels, vae=vae, encoder=encoder).to(dtype=pixels.dtype, device=pixels.device)
    if encoder is not None:
        # No generic tiled-encode contract; fall back to batch_encode.
        # Unreachable today (VACE + external_encoder is fail-fast).
        return encoder.batch_encode(pixels).to(dtype=pixels.dtype, device=pixels.device)
    # Native Wan VAE: per-sample tiled encode (deploy is B=1).
    outs = []
    for i in range(pixels.shape[0]):
        lat = vae.encode(
            [pixels[i]],
            device=device,
            tiled=True,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        outs.append(lat)
    return torch.cat(outs, dim=0).to(dtype=pixels.dtype, device=pixels.device)


def decode_latents(latents: Tensor, *, vae, encoder=None, device, tiled: bool = True) -> Tensor:
    if encoder is not None:
        return encoder.decode(latents.to(device), tiled=tiled)
    return vae.decode(latents.to(device), device=device, tiled=tiled)


def latents_to_frames(video_tensor: Tensor, *, encoder=None) -> list:
    if encoder is not None:
        return encoder.to_frames(video_tensor)
    return vae_output_to_video(video_tensor)
