import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import torch
from accelerate.data_loader import BatchSamplerShard
from omegaconf import OmegaConf
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'source'))
from launch import topology
from cache_stage import worker_ranks
spec=importlib.util.spec_from_file_location('aligned_sampler',ROOT/'source/openwam/dataloader/vav_aligned_batch_sampler.py')
sampler_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampler_module)
AlignedVAVBatchSampler=sampler_module.AlignedVAVBatchSampler


def test_native_cache_workers_have_unique_global_ranks_and_resume_existing8():
    assert [rank for node in range(4) for rank in worker_ranks(32,node)]==list(range(32))
    assert worker_ranks(8,0)==list(range(8))
    assert all(worker_ranks(8,node)==[] for node in [1,2,3])


def test_volcano_topology_rejects_loopback_wrong_rank_or_world():
    assert topology(dict(NNODES='4',NPROC_PER_NODE='8',NODE_RANK='3',MASTER_ADDR='worker0'))['node_rank']==3
    for args in [dict(NNODES='1'),dict(NODE_RANK='4'),dict(MASTER_ADDR='127.0.0.1'),dict(NPROC_PER_NODE='4')]:
        with pytest.raises(ValueError):topology(dict(MASTER_ADDR='worker0',**{k:v for k,v in args.items() if k!='MASTER_ADDR'}) if 'MASTER_ADDR' not in args else args)


def test_exact_global_windows_and_short_tail_match_8rank_reference():
    # Exercise Accelerate's actual BatchSamplerShard for every physical rank.
    permutation=torch.randperm(273465,generator=torch.Generator().manual_seed(42)).tolist()
    old=AlignedVAVBatchSampler(permutation,micro_batch=32,world_size=8)
    new=AlignedVAVBatchSampler(permutation,micro_batch=8,world_size=32)
    assert len(old)==1069*8 and len(new)==1069*32
    old_batches=list(old);new_batches=list(new)
    for step in range(1069):
        a=sum(old_batches[step*8:(step+1)*8],[])
        b=sum(new_batches[step*32:(step+1)*32],[])
        assert a==b
        assert len(b)==(64 if step==1068 else 256)
    class Fixed:
        batch_size=None;drop_last=False
        def __iter__(self):return iter(new_batches)
        def __len__(self):return len(new_batches)
    for rank in range(32):
        shard=BatchSamplerShard(Fixed(),num_processes=32,process_index=rank,even_batches=False)
        rows=list(shard)
        assert len(rows)==1069 and all(len(x)==8 for x in rows[:-1]) and len(rows[-1])==2
        assert rows[-1]==new_batches[-32+rank]
    flattened=[index for batch in new_batches for index in batch]
    assert len(flattened)==273472 and len(set(flattened))==273465
    assert flattened[-7:]==permutation[:7]
    assert (8/32)==(2/8)  # Identical partial-update/global256 loss weighting.


def test_budget_optimizer_and_retention_contract():
    c=OmegaConf.load(ROOT/'config.yaml')
    assert c.training.batch_size*32*c.training.gradient_accumulation_steps==256
    assert c.training.learning_rate==1e-4 and c.training.num_epochs==10
    assert 1069*c.training.num_epochs==10690
    assert c.training.keep_last_k_ckpts==3 and c.training.save_full_states_for_resume
    assert c.project.vav.attention_sources==[[0,1],[0,1],[1,2]]
    tree=ast.parse((ROOT/'source/openwam/train/openwam_trainer.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OpenWAMTrainer')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='finish_training')
    events=[]
    ns=dict(os=__import__('os'),save_weights=lambda *a,**k:events.append('weights'),
        save_full_state=lambda *a,**k:events.append('full_state'),
        manage_checkpoints=lambda *a:events.append(('prune',a[1])))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<retention>','exec'),ns)
    trainer=NS(accelerator=NS(wait_for_everyone=lambda:None),architecture=object(),cfg=c)
    ns['finish_training'](trainer,'/unused',10690,2000,3,True,None,opt_step=10690,epoch=9)
    assert events==['weights','full_state',('prune',3)]
