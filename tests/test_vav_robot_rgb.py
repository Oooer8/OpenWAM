"""Real tiny Wan/ActionDiT tests for separate V1/V2 RGB signals."""
import copy
import pytest
import torch

from tests.test_vav_architecture import architecture, inputs


def paired_forward_inputs():
    batch = inputs()
    robot = torch.randn_like(batch['input_latents'])
    batch['robot_pipeline_inputs'] = dict(input_latents=robot,
        latents=robot.clone(), first_frame_latents=robot[:,:,:1].clone())
    return batch


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_robot_signal_changes_only_v2_and_pruning_is_exact(architecture, dtype):
    model=architecture
    model.set_dtype_device(dtype, torch.device('cpu'))
    batch=paired_forward_inputs()
    def cast(value):
        if isinstance(value, dict):return {k:cast(v) for k,v in value.items()}
        return value.to(dtype) if torch.is_tensor(value) and value.is_floating_point() else value
    batch=cast(batch)
    actions,t=torch.randn(2,4,4),torch.tensor([100.,600.])
    actions,t=actions.to(dtype),t.to(dtype)
    (v1,v2),a=model(actions,t,return_auxiliary_video=True,**batch)
    changed=copy.deepcopy(batch)
    changed['robot_pipeline_inputs']['latents'].add_(7)
    changed['robot_pipeline_inputs']['first_frame_latents'].add_(7)
    (v1b,v2b),ab=model(actions,t,return_auxiliary_video=True,**changed)
    torch.testing.assert_close(v1,v1b,rtol=0,atol=0)
    torch.testing.assert_close(a,ab,rtol=0,atol=0)
    assert not torch.equal(v2,v2b)
    (v1.square().mean()+v2.square().mean()+a.square().mean()).backward()
    for module in (model.video_backbone.dit,model.action_backbone,model.video_backbone_v2.dit):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    model.video_backbone_v2=None;model._vav_driver=None
    # Deployment uses scene RGB and action only; no robot RGB required.
    batch.pop('robot_pipeline_inputs')
    with torch.no_grad():vp,ap=model(actions,t,**batch)
    torch.testing.assert_close(v1,vp,rtol=0,atol=0)
    torch.testing.assert_close(a,ap,rtol=0,atol=0)


def test_same_epsilon_but_separate_targets_and_clean_prefix(architecture,monkeypatch):
    model=architecture;batch=inputs()
    batch.pop('latents');batch.pop('timestep')
    scene=batch['input_latents'];robot=torch.randn_like(scene)
    batch.update(robot_input_latents=robot,robot_first_frame_latents=robot[:,:,:1].clone())
    captures=[];forwards=[]
    def video_loss(pred,target,ids,side,device):
        captures.append((target.clone(),side['first_frame_latents'].clone()))
        return (pred-target).square().mean()
    original=model.forward
    def checked(*args,**kw):
        forwards.append({k:v.clone() for k,v in kw['robot_pipeline_inputs'].items()})
        return original(*args,**kw)
    monkeypatch.setattr(model,'forward',checked)
    monkeypatch.setattr(model,'_compute_video_loss',video_loss)
    result=model.compute_loss(actions=torch.randn(2,4,4),use_gradient_checkpointing=True,**batch)
    torch.testing.assert_close(captures[0][0]-captures[1][0],robot-scene,rtol=2e-6,atol=2e-6)
    torch.testing.assert_close(captures[0][1],scene[:,:,:1],rtol=0,atol=0)
    torch.testing.assert_close(captures[1][1],robot[:,:,:1],rtol=0,atol=0)
    torch.testing.assert_close(forwards[0]['latents'][:,:,:1],robot[:,:,:1],rtol=0,atol=0)
    torch.testing.assert_close(result['loss'].detach(),result['loss_video_v1']+result['loss_video_v2']+result['loss_action'])
    result['loss'].backward()
    assert next(model.video_backbone_v2.parameters()).grad is not None


def test_no_scene_fallback(architecture):
    batch=inputs();batch.pop('latents');batch.pop('timestep')
    with pytest.raises(ValueError,match='Missing paired robot RGB'):
        architecture.compute_loss(actions=torch.randn(2,4,4),**batch)
    batch=inputs()
    with pytest.raises(ValueError,match='V2 robot RGB inputs are required'):
        architecture(torch.randn(2,4,4),torch.tensor([100.,600.]),return_auxiliary_video=True,**batch)
    with pytest.raises(ValueError,match='scene-latent fallback is forbidden'):
        architecture.prepare_inputs([dict(video_latents=torch.randn(48,3,24,20))])
