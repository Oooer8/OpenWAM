"""Real multi-process flock test: dynamic workers never publish duplicate shards."""
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
from cache_robodojo_vae import claim_shard, atomic_json, validate_shard
from openwam.dataloader.robodojo_latent_cache import SHAPE, sha256_file


def writer(root, worker, gate, result):
    gate.wait()
    claimed=[]
    for start in range(12):
        with claim_shard(root,start,start+1,'test') as acquired:
            if not acquired: continue
            time.sleep(.02)
            path=root/f'{start:08d}.bf16'
            np.full(SHAPE,worker,np.uint16).tofile(path)
            atomic_json(root/f'{start:08d}.json',dict(start=start,end=start+1,path=path.name,
                size=path.stat().st_size,sha256=sha256_file(path),build_sha256='test',worker=worker))
            claimed.append(start)
    result.put(claimed)


def test_dynamic_claims_are_disjoint_complete_and_resumable(tmp_path):
    ctx=mp.get_context('fork');gate=ctx.Event();result=ctx.Queue()
    processes=[ctx.Process(target=writer,args=(tmp_path,i,gate,result)) for i in range(4)]
    for p in processes:p.start()
    gate.set()
    claims=[result.get(timeout=20) for _ in processes]
    for p in processes:p.join(20);assert p.exitcode==0
    flattened=[i for row in claims for i in row]
    assert sorted(flattened)==list(range(12))
    assert sum(bool(row) for row in claims)>1
    for start in range(12):
        validate_shard(tmp_path,start,start+1,'test')
        with claim_shard(tmp_path,start,start+1,'test') as acquired:
            assert not acquired
    # A crashed worker's unpublished partial file is safely reclaimable.
    (tmp_path/'00000012.bf16.partial').write_bytes(b'partial')
    with claim_shard(tmp_path,12,13,'test') as acquired:
        assert acquired
        with claim_shard(tmp_path,12,13,'test') as duplicate:
            assert not duplicate
    with claim_shard(tmp_path,12,13,'test') as acquired:
        assert acquired


def test_finalizer_rejects_incomplete_or_corrupt_then_publishes(tmp_path):
    import pytest
    from cache_robodojo_vae import finalize_cache
    class Dataset:
        def __len__(self):return 2
    ds=Dataset();build={'windows':2};chunks=[(0,1),(1,2)]
    atomic_json(tmp_path/'worker_0_validation.json',dict(status='passed',build_sha256='test'))
    def make_shard(start):
        path=tmp_path/f'{start:08d}.bf16'
        np.zeros(SHAPE,np.uint16).tofile(path)
        atomic_json(tmp_path/f'{start:08d}.json',dict(start=start,end=start+1,path=path.name,
            size=path.stat().st_size,sha256=sha256_file(path),build_sha256='test',worker=0))
        return path
    make_shard(0)
    with pytest.raises(FileNotFoundError):finalize_cache(tmp_path,ds,build,'test',chunks)
    assert not (tmp_path/'manifest.json').exists()
    path=make_shard(1)
    with path.open('r+b') as f:f.write(b'xx')
    with pytest.raises(AssertionError):finalize_cache(tmp_path,ds,build,'test',chunks)
    assert not (tmp_path/'manifest.json').exists()
    make_shard(1)
    finalize_cache(tmp_path,ds,build,'test',chunks)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    receipt=json.loads((tmp_path/'validation.json').read_text())
    assert manifest['status']=='ready' and len(manifest['shards'])==2
    assert receipt['status']=='passed' and receipt['manifest_sha256']==sha256_file(tmp_path/'manifest.json')
