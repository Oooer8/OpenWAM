"""LIBERO mutual VAV: scene RGB in V1; paired robot-only RGB in V2."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from openwam.model.architectures.registry import register_architecture
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from .joint_self_attn import DualSystemSelfAttnArchitecture
from .vav_driver import VAVDriver, RoutedMutualMoTDriver


@register_architecture('dual_system_vav', framework='dual_system', variant='vav',
                       note='V1/A mutual; V2 reads A+V2; V1/A never read V2. Independent video experts.')
class DualSystemVAVArchitecture(DualSystemSelfAttnArchitecture):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.video_backbone_v2 = None
        self._vav_driver = None
        if self.video_backbone is None:
            return
        if not isinstance(self.video_backbone, Wan22Ti2v):
            raise ValueError('VAV is validated only for the pinned native Wan LIBERO backbone')
        if (self.video_backbone.video_encoder is not None or self.video_backbone.vace is not None
                or self.video_backbone.motion_controller is not None):
            raise ValueError('VAV does not implement VACE/motion-controller duplication')
        # Clone only the DiT. VAE, T5, tokenizer and caches belong to V1 and are
        # reused for preprocessing; no duplicate frozen modules enter state_dict.
        self.video_backbone_v2 = Wan22Ti2v(
            SimpleNamespace(dit=copy.deepcopy(self.video_backbone.dit)),
            shift_video=self.video_backbone.shift_video, text_dim=self.video_backbone.text_dim)
        self._vav_driver = VAVDriver(self.video_backbone, self.action_backbone, self.video_backbone_v2)

    def build_mot_driver(self):
        # Deployment discards V2. The remaining two experts must preserve the
        # exact directed V1/A subgraph, independent of caller's legacy mask.
        self._mot_driver_kwargs['attention_mask_mode'] = 'mutual'
        self._mot_driver_kwargs['video_attention_mask_mode'] = 'first_frame_causal'
        self._mot_driver = RoutedMutualMoTDriver(self.video_backbone, self.action_backbone,
            **self._mot_driver_kwargs)
        return self._mot_driver

    def set_dtype_device(self, dtype, device):
        super().set_dtype_device(dtype, device)
        if self.video_backbone_v2 is not None:
            self.video_backbone_v2.set_dtype_device(dtype, device)

    def load_checkpoint(self, path, strict=True):
        from safetensors.torch import load_file
        state = load_file(path)
        v2_prefix = 'video_backbone_v2.dit.'
        if not any(k.startswith('video_backbone_v2.') for k in state):
            # Foundation warm start: both experts start from the SAME official
            # loaded video weights, but have independent tensor storage.
            source = 'video_backbone.dit.'
            state.update({v2_prefix + k[len(source):]: v.clone()
                          for k, v in list(state.items()) if k.startswith(source)})
        assign = any(p.device.type == 'meta' for p in self.parameters())
        missing, unexpected = self.load_state_dict(state, strict=False, assign=assign)
        if strict and (missing or unexpected):
            raise RuntimeError(f'Strict VAV load failed: missing={missing}, unexpected={unexpected}')

    def forward(self, noisy_actions, action_timestep, *, proprio=None,
                use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
                return_auxiliary_video=False, **pipeline_inputs):
        if not return_auxiliary_video:
            pipeline_inputs.pop("robot_pipeline_inputs", None)
            return super().forward(noisy_actions, action_timestep, proprio=proprio,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload, **pipeline_inputs)
        if noisy_actions is None:
            raise ValueError('Three-expert VAV training requires action inputs')
        if pipeline_inputs.get('first_frame_latents') is None:
            raise ValueError('VAV LIBERO training requires the clean first-frame condition')
        pipeline_inputs = dict(pipeline_inputs)
        robot_inputs = pipeline_inputs.pop('robot_pipeline_inputs', None)
        if robot_inputs is None:
            raise ValueError('V2 robot RGB inputs are required for three-expert training')
        inputs = self._append_proprio_context_token(pipeline_inputs, proprio)
        context, context_mask = inputs.get('context'), inputs.get('context_mask')
        if context is not None and context_mask is None and inputs.get('seq_lens') is not None:
            positions = torch.arange(context.shape[1], device=context.device)
            context_mask = positions.unsqueeze(0) < inputs['seq_lens'].to(context.device).unsqueeze(1)
        inputs.setdefault('force_per_token_t_mod', True)
        inputs.setdefault('zero_clean_prefix_t_mod', True)
        options = dict(use_gradient_checkpointing=use_gradient_checkpointing,
                       use_gradient_checkpointing_offload=use_gradient_checkpointing_offload)
        v1 = self.video_backbone.prepare(**inputs, **options)
        v2_inputs = dict(inputs)
        v2_inputs.update(robot_inputs)
        v2 = self.video_backbone_v2.prepare(**v2_inputs, **options)
        a = self.action_backbone.prepare_state(noisy_actions, action_timestep,
                                               context=context, context_mask=context_mask, **options)
        v1, a, v2 = self._vav_driver.run(v1, a, v2, **options)
        return (self.video_backbone.finalize(v1), self.video_backbone_v2.finalize(v2)), self.action_backbone.extract_prediction(a)

    def prepare_inputs(self, batch):
        samples = [batch] if isinstance(batch, dict) else batch
        if not samples or any('robot_video_latents' not in sample for sample in samples):
            raise ValueError('Robot-RGB VAV requires paired robot_video_latents; scene-latent fallback is forbidden')
        result = super().prepare_inputs(batch)
        robot = torch.stack([sample['robot_video_latents'] for sample in samples]).to(
            dtype=self.dtype, device=self.device)
        if robot.shape != result['input_latents'].shape:
            raise ValueError('Scene/robot RGB latent geometry differs')
        if not torch.isfinite(robot).all():
            raise ValueError('Nonfinite robot RGB latents')
        result['robot_input_latents'] = robot
        result['robot_first_frame_latents'] = robot[:, :, 0:1].clone()
        return result

    def compute_loss(
        self,
        *,
        actions: Optional[torch.Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        **inputs,
    ) -> dict:
        """Compute joint video-action flow matching loss.

        This is the single entry point for training loss computation.
        Handles timestep sampling, noise injection, forward pass, and
        loss calculation internally.

        Callers should produce ``inputs`` via ``self.prepare_inputs(batch)``
        (preferred) or assemble it manually with the same keys: the output of
        ``self.preprocess()`` plus any of ``actions / proprio /
        action_is_pad / video_is_pad / use_gradient_checkpointing[_offload] /
        max_timestep_boundary / min_timestep_boundary``.

        Args:
            actions: (B, T_action, action_dim) ground truth actions. May also
                be passed via ``inputs["actions"]``.
            lambda_video: Weight for video loss term.
            lambda_action: Weight for action loss term.
            **inputs: Preprocessed video/text tensors plus forward-time flags.

        Returns:
            dict with keys: loss, loss_video, loss_action.
        """
        if lambda_action <= 0:
            raise ValueError("VAV training requires positive action loss")
        vb = self.video_backbone
        action_scheduler = self.action_backbone.scheduler
        _dtype = self.dtype
        _device = self.device

        if actions is None:
            actions = inputs.pop("actions", None)
        else:
            inputs.pop("actions", None)

        max_tb = int(inputs.pop("max_timestep_boundary", 1) * len(vb.scheduler.timesteps))
        min_tb = int(inputs.pop("min_timestep_boundary", 0) * len(vb.scheduler.timesteps))
        B = inputs["input_latents"].shape[0]

        robot_clean = inputs.pop('robot_input_latents', None)
        robot_first = inputs.pop('robot_first_frame_latents', None)
        if robot_clean is None or robot_first is None:
            raise ValueError('Missing paired robot RGB latents or robot clean prefix')
        if robot_clean.shape != inputs['input_latents'].shape:
            raise ValueError('Robot RGB latent shape differs from scene RGB')

        # --- Sample video timesteps ---
        video_timestep_ids = torch.randint(min_tb, max_tb, (B,))

        video_timesteps = vb.scheduler.timesteps[video_timestep_ids].to(dtype=_dtype, device=_device)
        video_sigmas = vb.scheduler.sigmas[video_timestep_ids].to(dtype=_dtype, device=_device)

        # --- Add video noise (flow-matching: linear interp + velocity target) ---
        video_noise = torch.randn_like(inputs["input_latents"])
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        inputs["latents"] = (1 - sigma_bc) * inputs["input_latents"] + sigma_bc * video_noise
        video_target = video_noise - inputs["input_latents"]
        # Share timestep and sampled epsilon, but each video has its own clean
        # signal and velocity target. This adds no new RNG draws.
        robot_noisy = (1 - sigma_bc) * robot_clean + sigma_bc * video_noise
        robot_target = video_noise - robot_clean
        robot_noisy[:, :, 0:1] = robot_first

        if inputs.get("first_frame_latents") is not None:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

        # --- Prepare action noise ---
        noisy_actions, action_target, action_timesteps, action_timestep_ids, action_sigmas, a_sigma_bc = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if lambda_action > 0 and actions is not None:
            action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))

            action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(dtype=_dtype, device=_device)
            action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(dtype=_dtype, device=_device)

            actions = actions.to(dtype=_dtype, device=_device)
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)

            action_noise = torch.randn_like(actions)
            if action_sigmas.dim() == 1:
                a_sigma_bc = action_sigmas.view(B, 1, 1)
            else:
                a_sigma_bc = action_sigmas.unsqueeze(-1)
            noisy_actions = action_scheduler.add_noise(actions, action_noise, a_sigma_bc)
            action_target = action_scheduler.training_target(actions, action_noise)

        # --- Joint forward pass ---
        forward_inputs = dict(inputs)
        forward_inputs['robot_pipeline_inputs'] = dict(input_latents=robot_clean,
            latents=robot_noisy, first_frame_latents=robot_first)
        proprio = forward_inputs.pop("proprio", None)
        proprio_mask = forward_inputs.pop("proprio_mask", None)
        use_grad_ckpt = forward_inputs.pop("use_gradient_checkpointing", False)
        use_grad_ckpt_offload = forward_inputs.pop("use_gradient_checkpointing_offload", False)
        # Padding masks are kept in `inputs` for loss-side masking but dropped
        # from `forward_inputs` so they don't leak into vb.prepare(). Per FastWAM
        # MoT design, attention itself does not consume sample-level padding.
        forward_inputs.pop("action_is_pad", None)
        forward_inputs.pop("video_is_pad", None)

        # Route per-sample proprio_mask through pipeline_inputs to
        # ``_append_proprio_context_token``. Internal-only key; pop'd there.
        if proprio_mask is not None:
            forward_inputs["_proprio_sample_mask"] = proprio_mask

        # Use ``self(...)`` (not ``self.forward(...)``) so ``nn.Module.__call__``
        # is invoked and any architecture-level forward-pre-hooks fire.
        video_noise_pred, action_noise_pred = self(
            noisy_actions if lambda_action > 0 else None,
            action_timesteps if lambda_action > 0 else None,
            proprio=proprio,
            use_gradient_checkpointing=use_grad_ckpt,
            use_gradient_checkpointing_offload=use_grad_ckpt_offload,
            **forward_inputs,
            return_auxiliary_video=True,
            timestep=video_timesteps,
        )

        # --- Video loss ---
        pred_v1, pred_v2 = video_noise_pred
        loss_v1 = self._compute_video_loss(pred_v1, video_target, video_timestep_ids, inputs, _device)
        robot_loss_inputs = dict(inputs, input_latents=robot_clean, first_frame_latents=robot_first)
        loss_v2 = self._compute_video_loss(pred_v2, robot_target, video_timestep_ids, robot_loss_inputs, _device)
        loss_video = loss_v1 + loss_v2

        if lambda_action == 0 or action_noise_pred is None:
            return {
                "loss": lambda_video * loss_video,
                "loss_video": lambda_video * loss_video.detach(),
                "loss_action": torch.tensor(0.0, device=loss_video.device),
            }

        # --- Action loss ---
        loss_action = self._compute_action_loss(
            action_noise_pred,
            action_target,
            action_timestep_ids,
            action_scheduler,
            inputs,
            _device,
        )

        if lambda_video == 0:
            loss = lambda_action * loss_action
        else:
            loss = lambda_video * loss_video + lambda_action * loss_action

        result = {
            "loss": loss,
            "loss_video_v1": lambda_video * loss_v1.detach(),
            "loss_video_v2": lambda_video * loss_v2.detach(),
            "loss_video": lambda_video * loss_video.detach(),
            "loss_action": lambda_action * loss_action.detach(),
        }

        return result

