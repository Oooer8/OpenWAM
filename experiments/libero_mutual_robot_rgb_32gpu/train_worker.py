"""Fixed global256 recipe: 32 ranks x local8; bounded gates never seed training."""
import json
import math
import os
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'source/scripts')]


class GateComplete(Exception):pass


def main():
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    import train
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from runtime_common import write
    phase=os.environ['LIBERO_PHASE']
    assert phase in ['gate_save','gate_restore','formal']
    cfg=OmegaConf.load(os.environ['LIBERO_CONFIG'])
    rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==32
    gate=phase!='formal';offset=2 if phase=='gate_restore' else 0
    state=dict(captures=0,rows=[],batch_sizes=[],pruning=None)
    if phase=='gate_save':os.environ['OPENWAM_VALIDATE_CHECKPOINT_STEP']='2'
    if phase=='gate_restore':
        os.environ['OPENWAM_VERIFY_CHECKPOINT_RESTORE']='1'
        from openwam.dataloader.vav_aligned_batch_sampler import AlignedVAVBatchSampler
        iterator=AlignedVAVBatchSampler.__iter__
        def bounded(self):
            batches=list(iterator(self))
            assert len(batches)==1069*32
            yield from batches[:3*32]
            yield from batches[-32:]
        AlignedVAVBatchSampler.__iter__=bounded
    original_prepare=OpenWAMTrainer.prepare_accelerate
    def prepare(self,optimizer,loader,scheduler):
        result=original_prepare(self,optimizer,loader,scheduler)
        assert self.accelerator.num_processes==32 and len(result[1])==1069
        runtime={k:getattr(self.architecture,k)() for k in
            ['train_batch_size','train_micro_batch_size_per_gpu','gradient_accumulation_steps']}
        assert runtime==dict(train_batch_size=256,train_micro_batch_size_per_gpu=8,gradient_accumulation_steps=1),runtime
        if gate:
            from gradient_capture import install_gradient_capture
            arch=self.accelerator.unwrap_model(self.architecture)
            chosen=[(n,p) for n,p in arch.named_parameters() if p.requires_grad and
                (n.endswith('ffn.0.weight') or n.endswith('ffn.2.weight'))]
            assert len(chosen)>=180
            def captured(samples):
                assert len(samples)==len(chosen) and all(torch.isfinite(x).all() for x in samples.values())
                state['captures']+=1
            install_gradient_capture(self.architecture,chosen,captured)
            if phase=='gate_save':
                original=arch.forward
                def forward(this,*args,**kw):
                    output=original(*args,**kw)
                    if state['pruning'] is None and kw.get('return_auxiliary_video'):
                        v2,driver=this.video_backbone_v2,this._vav_driver
                        try:
                            this.video_backbone_v2=None;this._vav_driver=None
                            options=dict(kw,return_auxiliary_video=False,use_gradient_checkpointing=False)
                            options.pop('robot_pipeline_inputs',None)
                            with torch.no_grad():pruned=original(*args,**options)
                            pairs=[(output[0][0],pruned[0]),(output[1],pruned[1])]
                            comparison=dict(physical_v2_removed=True,rtol=.02,atol=.02,
                                attention_layout='same_routed_sdpa_shapes_as_training',
                                max_absolute_error=[float((a.detach()-b).abs().max()) for a,b in pairs])
                            write(Path(os.environ['LIBERO_GATE_REPORT_DIR'])/f'pruning_rank{rank}.json',comparison)
                            for a,b in pairs:torch.testing.assert_close(a.detach(),b,rtol=.02,atol=.02)
                            state['pruning']=comparison
                        finally:this.video_backbone_v2=v2;this._vav_driver=driver
                    return output
                arch.forward=types.MethodType(forward,arch)
        return result
    OpenWAMTrainer.prepare_accelerate=prepare
    if gate:
        original_compute=OpenWAMTrainer.compute_loss
        def compute(self,batch):
            state['batch_sizes'].append(len(batch))
            return original_compute(self,batch)
        OpenWAMTrainer.compute_loss=compute
        original_log=OpenWAMTrainer.log_step
        def log(self,**kw):
            step=kw['global_step']
            assert step==kw['opt_step']==state['captures']+offset
            assert all(math.isfinite(float(v)) for v in kw['metrics'].values())
            assert math.isfinite(float(kw['lr'])) and kw['metrics']['grad_norm']>0
            expected=2 if phase=='gate_restore' and step==4 else 8
            assert state['batch_sizes'][-1]==expected
            state['rows'].append(dict(step=step,lr=kw['lr'],metrics=kw['metrics'],local_batch=expected,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            original_log(self,**kw)
            if phase=='gate_restore' and step==4:raise GateComplete()
        OpenWAMTrainer.log_step=log
    try:
        try:train._train(cfg)
        except GateComplete:assert phase=='gate_restore'
        if gate:
            assert state['captures']==2
            if phase=='gate_save':assert state['pruning'] is not None
            write(Path(os.environ['LIBERO_GATE_REPORT_DIR'])/f'rank{rank}.json',
                dict(status='passed',phase=phase,world_size=32,prepared_steps_per_epoch=1069,
                    total_updates=10690,finite_loss_gradients=True,**state))
    finally:
        if dist.is_available() and dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
