"""Opt-in, measured training accelerations for the Cosmos3-DROID run."""
import json
import logging
import os
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from openwam.model.video_backbone.wan.text_cache import configured_text_cache

logger = logging.getLogger(__name__)


def validate_text_cache_config(cfg):
    from openwam.dataloader.libero_latent_cache import validate_training_config
    if OmegaConf.select(cfg, 'model.architecture.variant') == 'vav':
        from openwam.dataloader.vav_cache_consumer import validate_vav_cache_consumer
        validate_vav_cache_consumer(cfg)
    else:
        validate_training_config(cfg)
    # Save the effective execution settings alongside the normal training config.
    with open_dict(cfg.project):
        cfg.project.runtime_optimizations = {
            'checkpoint_every_n_layer': int(os.environ.get('OPENWAM_CHECKPOINT_EVERY_N_LAYER', '1')),
            'checkpoint_pattern': os.environ.get('OPENWAM_CHECKPOINT_PATTERN', ''),
            'exact_vae_norm': os.environ.get('OPENWAM_EXACT_VAE_NORM', 'none'),
            'selective_checkpoint': os.environ.get('OPENWAM_SELECTIVE_CHECKPOINT', 'none'),
            'checkpoint_region': os.environ.get('OPENWAM_CHECKPOINT_REGION', 'block'),
            'cached_text_encoder_on_cpu': os.environ.get('OPENWAM_CACHED_TEXT_ENCODER_CPU') == '1',
            'vae_channels_last': os.environ.get('OPENWAM_VAE_CHANNELS_LAST') == '1',
            'compiled_vae_norm': os.environ.get('OPENWAM_COMPILE_VAE_NORM') == '1',
            'zero_overlap_comm': os.environ.get('OPENWAM_ZERO_OVERLAP') == '1',
            'zero_reduce_bucket_size': 50_000_000 if os.environ.get('OPENWAM_ZERO_OVERLAP') == '1' else 500_000_000,
            'memory_stress_after_step': int(os.environ['OPENWAM_MEMORY_STRESS_AFTER_STEP']) if os.environ.get('OPENWAM_MEMORY_STRESS_AFTER_STEP') else None,
            'batch_video_preprocess': os.environ.get('OPENWAM_BATCH_VIDEO_PREPROCESS') == '1',
            'pixel_transport': 'uint8_to_bfloat16_on_device' if os.environ.get('OPENWAM_BATCH_VIDEO_PREPROCESS') == '1' else 'original_per_image',
            'compiled_train_ffn': os.environ.get('OPENWAM_COMPILE_TRAIN_FFN') == '1',
            'compact_masked_text_context': os.environ.get('OPENWAM_COMPACT_TEXT_CONTEXT') == '1',
            'compiled_norm_and_gate': os.environ.get('OPENWAM_COMPILE_NORM_GATE') == '1',
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'torch_version': str(torch.__version__),
        }
    latent_root = OmegaConf.select(cfg, 'dataloader.latent_cache_dir')
    if latent_root:
        if cfg.training.mixed_precision != 'bf16':
            raise ValueError('RoboDojo latent cache is validated for BF16 training only')
        from openwam.dataloader.robodojo_latent_cache import LatentCache
        from openwam.dataloader.transforms.video import color_jitter_enabled
        if color_jitter_enabled(cfg.dataloader.get('color_jitter')):
            raise ValueError('VAE cache requires disabled augmentation')
        prepared_root = Path(cfg.dataloader.prepared_dir)
        count = json.loads((prepared_root / 'manifest.json').read_text())['windows']
        latent_cache = LatentCache(latent_root, prepared_root, count)
        os.environ['OPENWAM_VALIDATED_LATENT_CACHE'] = str(Path(latent_root).resolve())
        with open_dict(cfg.project):
            cfg.project.vae_cache_directory = str(latent_root)
            cfg.project.vae_cache_weights_sha256 = latent_cache.manifest['vae_weights_sha256']
    else:
        os.environ.pop('OPENWAM_VALIDATED_LATENT_CACHE', None)
    cache = configured_text_cache()
    if cache is None:
        if os.environ.get('OPENWAM_CACHED_TEXT_ENCODER_CPU') == '1':
            raise ValueError('CPU text-encoder retention requires a validated complete text cache')
        return
    # The manifest batch size belongs to the offline T5 encoding pass. Cached
    # per-prompt vectors are immutable and lookup supports any training batch.
    result = json.loads((cache.root / 'validation.json').read_text())
    digest = cache.manifest['index_sha256']
    if result.get('status') != 'passed' or result.get('index_sha256') != digest:
        raise ValueError('Text cache has not passed numerical/integrity validation')
    resume = cfg.training.get('resume_ckpt_path')
    if resume:
        source = Path(resume)
        config = next((p / 'config.yaml' for p in (source, source.parent) if (p/'config.yaml').is_file()), None)
        if config is None or OmegaConf.select(OmegaConf.load(config), 'project.text_cache_index_sha256') != digest:
            raise ValueError('Resume source is not bound to this text cache')
        cache.validate_checkpoint(cache.manifest['checkpoint'])
    else:
        cache.validate_checkpoint(cfg.training.finetune_ckpt_path)
    with open_dict(cfg.project):
        cfg.project.text_cache_index_sha256 = digest
        cfg.project.text_cache_directory = str(cache.root)
    logger.info('Validated frozen-text cache: %d prompts', cache.manifest['num_prompts'])


def _functional_ffn(x, first_weight, first_bias, last_weight, last_bias):
    return F.linear(F.gelu(F.linear(x, first_weight, first_bias), approximate='tanh'),
                    last_weight, last_bias)


def compile_training_ffns(backbones):
    # One shared tensor-only graph per shape. Compiling a bound module method
    # generates identity guards for every layer and exhausts Dynamo's graph cache.
    compiled = torch.compile(_functional_ffn, backend='inductor', fullgraph=True,
                             dynamic=False, mode='default')

    def forward(module, x):
        from openwam.model.architectures.utils.checkpoint_scope import ffn_checkpoint_active
        args = (x, module[0].weight, module[0].bias, module[2].weight, module[2].bias)
        if ffn_checkpoint_active.get() and module.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(compiled, *args, use_reentrant=False)
        return compiled(*args)

    count = 0
    for backbone in backbones:
        for block in backbone.blocks:
            ffn = block.ffn
            if not (len(ffn) == 3 and isinstance(ffn[0], torch.nn.Linear)
                    and isinstance(ffn[1], torch.nn.GELU) and ffn[1].approximate == 'tanh'
                    and isinstance(ffn[2], torch.nn.Linear)):
                raise ValueError('Compiled FFN requires the exact Linear/GELU(tanh)/Linear layout')
            if any(layer._forward_hooks or layer._forward_pre_hooks for layer in (ffn[0], ffn[1], ffn[2])):
                raise ValueError('Functional FFN cannot bypass submodule forward hooks')
            # Only change the callable: modules, parameters, and checkpoint keys
            # remain unchanged. Parameter autograd hooks still run normally.
            ffn.forward = types.MethodType(forward, ffn)
            count += 1
    return count


def apply_training_optimizations(architecture):
    from openwam.dataloader.libero_latent_cache import apply_training_cache
    apply_training_cache(architecture)
    latent_root = os.environ.get('OPENWAM_VALIDATED_LATENT_CACHE')
    if latent_root:
        from openwam.dataloader.robodojo_latent_cache import state_digest, sha256_file
        manifest = json.loads((Path(latent_root) / 'manifest.json').read_text())
        for name, digest in manifest['encoding_source_sha256'].items():
            if sha256_file(Path(__file__).resolve().parents[3] / name) != digest:
                raise ValueError(f'VAE cache encoder source differs: {name}')
        vae = architecture.video_backbone.vae
        if vae.training or any(p.requires_grad for p in vae.parameters()):
            raise ValueError('Cached VAE must be frozen and in eval mode')
        if state_digest(vae.state_dict()) != manifest['vae_weights_sha256']:
            raise ValueError('Loaded VAE weights differ from offline cache')
        vae.to(device='cpu')
        architecture.video_backbone._robodojo_vae_cache_validated = True
        torch.cuda.empty_cache()
        logger.info('Validated cached VAE weights; retained unused VAE on CPU')
    exact_vae = os.environ.get('OPENWAM_EXACT_VAE_NORM', 'none')
    if exact_vae not in ('none', 'affine', 'divide_affine'):
        raise ValueError('Unknown OPENWAM_EXACT_VAE_NORM mode')
    if exact_vae != 'none':
        if (os.environ.get('OPENWAM_VAE_CHANNELS_LAST') == '1'
                or os.environ.get('OPENWAM_COMPILE_VAE_NORM') == '1'):
            raise ValueError('Exact VAE norms require the validated native layout and no compiled norms')
        from openwam.train.utils.robodojo_exact_optimizations import install_exact_vae_norms
        count = install_exact_vae_norms(architecture.video_backbone.vae, divide=exact_vae == 'divide_affine')
        logger.info('Enabled frozen-VAE fusion preserving eager BF16 rounding: %s, %d modules', exact_vae, count)
    if os.environ.get('OPENWAM_CACHED_TEXT_ENCODER_CPU') == '1':
        if configured_text_cache() is None:
            raise ValueError('CPU text encoder requires the validated frozen-text cache')
        encoder = architecture.video_backbone.text_encoder
        if encoder.training or any(p.requires_grad for p in encoder.parameters()):
            raise ValueError('Only unused frozen eval-mode text encoders may be retained on CPU')
        size = sum(p.numel() * p.element_size() for p in encoder.parameters())
        encoder.to(device='cpu')
        torch.cuda.empty_cache()
        logger.info('Kept cached, unused frozen text encoder on CPU: %.3f GiB freed per GPU', size / 2**30)
    if os.environ.get('OPENWAM_COMPILE_TRAIN_FFN') == '1':
        backbones = [architecture.video_backbone.dit, architecture.action_backbone]
        if getattr(architecture, 'video_backbone_v2', None) is not None:
            backbones.append(architecture.video_backbone_v2.dit)
        count = compile_training_ffns(backbones)
        logger.info('Enabled shared compiled FFN forward/backward for %d blocks', count)
    if os.environ.get('OPENWAM_COMPILE_NORM_GATE') == '1':
        count = compile_norm_and_gates(architecture)
        logger.info('Enabled compiled normalization/gates: %s', count)
    if os.environ.get('OPENWAM_VAE_CHANNELS_LAST') == '1':
        from openwam.train.utils.vae_optimizations import use_vae_channels_last
        use_vae_channels_last(architecture.video_backbone.vae)
    if os.environ.get('OPENWAM_COMPILE_VAE_NORM') == '1':
        from openwam.train.utils.vae_optimizations import compile_vae_encoder_norms
        count = compile_vae_encoder_norms(architecture.video_backbone.vae)
        logger.info('Enabled compiled frozen-VAE normalization: %d modules', count)



def _functional_rms(x, weight, eps):
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return normed.to(x.dtype) * weight


def _functional_gate(x, gate, residual):
    return x + gate * residual


def compile_norm_and_gates(architecture):
    from openwam.model.video_backbone.wan.models.dit import RMSNorm as VideoRMS, GateModule
    from openwam.model.action_backbone.components import RMSNorm as ActionRMS
    # Text context length varies by batch; share dynamic-length tensor graphs.
    rms = torch.compile(_functional_rms, fullgraph=True, dynamic=True)
    gate = torch.compile(_functional_gate, fullgraph=True, dynamic=True)

    def norm_forward(module, x):
        return rms(x, module.weight, module.eps)

    def gate_forward(module, x, scale, residual):
        return gate(x, scale, residual)

    counts = {'rms_norm': 0, 'gate': 0}
    for module in architecture.modules():
        if isinstance(module, (VideoRMS, ActionRMS)):
            module.forward = types.MethodType(norm_forward, module)
            counts['rms_norm'] += 1
        elif isinstance(module, GateModule):
            module.forward = types.MethodType(gate_forward, module)
            counts['gate'] += 1
    for block in architecture.action_backbone.blocks:
        if callable(getattr(block, 'gate', None)) and not isinstance(block.gate, torch.nn.Module):
            block.gate = types.MethodType(gate_forward, block)
            counts['gate'] += 1
    return counts
