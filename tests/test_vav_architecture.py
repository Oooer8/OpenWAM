"""Real Wan/ActionDiT integration at small dimensions, on CPU."""
from types import SimpleNamespace as NS
import copy

import pytest
import torch
from safetensors.torch import save_file

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.vav import DualSystemVAVArchitecture
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler


@pytest.fixture
def architecture(monkeypatch):
    torch.set_num_threads(2)
    def build(self, cfg):
        dit = WanModel(dim=48, in_dim=4, ffn_dim=96, out_dim=4, text_dim=16,
            freq_dim=16, eps=1e-6, patch_size=(1,2,2), num_heads=2, num_layers=2,
            has_image_input=False, seperated_timestep=True, require_vae_embedding=False,
            require_clip_embedding=False, fuse_vae_embedding_in_latents=True)
        self.video_backbone = Wan22Ti2v(NS(dit=dit, scheduler=FlowMatchScheduler()))
    monkeypatch.setattr(BaseWAMArchitecture, '_init_video_backbone', build)
    model = DualSystemVAVArchitecture(dict(dim=48, ffn_dim=96, action_dim=4,
        bridge_interval=1, text_dim=16, use_proprioception=True, state_dim=4))
    model.set_dtype_device(torch.float32, torch.device('cpu'))
    model.init_training_schedulers()
    return model


def inputs():
    latents = torch.randn(2,4,3,4,4)
    return dict(latents=latents, input_latents=latents.clone(), first_frame_latents=latents[:,:,:1].clone(),
                context=torch.randn(2,5,16), seq_lens=torch.tensor([5,3]), timestep=torch.tensor([250.,750.]),
                proprio=torch.randn(2,4))


def test_three_expert_forward_backward_and_deployment(architecture):
    model = architecture
    batch = inputs()
    actions, t = torch.randn(2,4,4), torch.tensor([100.,600.])
    (v1, v2), a = model(actions, t, return_auxiliary_video=True, **batch)
    deployed_v1, deployed_a = model(actions, t, **batch)
    torch.testing.assert_close(v1, deployed_v1, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(a, deployed_a, rtol=2e-5, atol=2e-5)
    (v1.square().mean()+v2.square().mean()+a.square().mean()).backward()
    for module in (model.video_backbone.dit, model.action_backbone, model.video_backbone_v2.dit):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    def forbidden(*args, **kwargs):
        raise AssertionError('Deployment executed V2')
    model.video_backbone_v2.prepare = forbidden
    model(actions, t, **batch)
    # Deployment must remain identical after physical removal, with no V2 driver.
    model.video_backbone_v2 = None
    model._vav_driver = None
    pruned_v1, pruned_a = model(actions, t, **batch)
    torch.testing.assert_close(v1, pruned_v1, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(a, pruned_a, rtol=2e-5, atol=2e-5)


def test_foundation_clone_independent_and_full_weight_restore(architecture, tmp_path):
    model = architecture
    foundation = {k:v.contiguous() for k,v in model.state_dict().items() if not k.startswith('video_backbone_v2.')}
    source = tmp_path/'foundation.safetensors'
    save_file(foundation, source)
    model.load_checkpoint(source)
    for (name,p), (name2,q) in zip(model.video_backbone.dit.named_parameters(), model.video_backbone_v2.dit.named_parameters()):
        assert name==name2 and p.data_ptr()!=q.data_ptr()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert not any(k.startswith('video_backbone_v2.vae') or k.startswith('video_backbone_v2.text_encoder') for k in model.state_dict())
    with torch.no_grad():
        next(model.video_backbone_v2.parameters()).add_(1)
    expected = copy.deepcopy(model.state_dict())
    saved = tmp_path/'vav.safetensors'
    model.save_checkpoint(saved)
    with torch.no_grad():
        for p in model.parameters():p.zero_()
    model.load_checkpoint(saved)
    for key,value in expected.items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)


def test_three_loss_terms_and_checkpointed_backward(architecture):
    model = architecture
    batch = inputs()
    batch.pop('latents'); batch.pop('timestep')
    result = model.compute_loss(actions=torch.randn(2,4,4), use_gradient_checkpointing=True, **batch)
    torch.testing.assert_close(result['loss_video'],result['loss_video_v1']+result['loss_video_v2'])
    torch.testing.assert_close(result['loss'].detach(),result['loss_video']+result['loss_action'])
    result['loss'].backward()
    assert all(torch.isfinite(v) for v in result.values())
    assert next(model.video_backbone_v2.parameters()).grad is not None
