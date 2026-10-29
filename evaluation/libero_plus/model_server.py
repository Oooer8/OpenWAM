"""One model per GPU, with independent policy/caches/RNG per WebSocket session."""
import asyncio, copy, json, logging, os, time
from pathlib import Path

E = Path('/mnt/vepfs01/output/tiecheng.guo/artifacts/waw/openwam_libero_mutual_robot_rgb_32gpu_20261025/runs/libero-mutual-robot-rgb-32gpu-fix-20261026/libero_plus_evaluation')

def session_server(base):
    from openwam.deploy.server import PolicyServer
    from benchmarks.utils.rng_domain import GlobalRngDomain
    engine = copy.copy(base.engine)
    engine._vace_cache = {}
    # Prompt embeddings are immutable, keyed by the complete prompt, and may
    # be shared; action buffers and mutable temporal caches may not be shared.
    assert engine._dit_cache is None
    server = PolicyServer(engine, base.cfg, readiness_token=base.readiness_token)
    server._init_policy()
    assert not server._policy._async
    return server, GlobalRngDomain.from_seed(0)

def multisession_run(base, host='127.0.0.1', port=19520):
    import websockets
    from openwam.deploy.server import MAX_MESSAGE_BYTES, ObsValidationError
    base._init_policy()
    stats = {'pid': os.getpid(), 'port': port, 'max_connections': 20,
             'active_connections': 0, 'peak_connections': 0, 'requests': 0,
             'shared_model': True, 'session_isolation': 'policy/action_buffer/vace_cache/Python_NumPy_Torch_CPU_CUDA_RNG',
             'gpu_inference': 'serialized atomic requests; environments run concurrently'}
    stat_path = E/f'server_{port}_status.json'
    stat_path.parent.mkdir(parents=True, exist_ok=True)
    def save():
        stats['checked_unix'] = time.time()
        tmp=stat_path.with_suffix('.tmp');tmp.write_text(json.dumps(stats,indent=2));tmp.replace(stat_path)
    save()
    async def handler(ws):
        if stats['active_connections'] >= 20:
            await ws.close(code=1013, reason='twenty environment sessions already active');return
        session, domain = session_server(base)
        stats['active_connections'] += 1
        stats['peak_connections'] = max(stats['active_connections'], stats['peak_connections']);save()
        try:
            async for raw in ws:
                try:
                    data=json.loads(raw);kind=data.get('type','obs')
                    # There is no await inside this domain. CUDA operations,
                    # mutable architecture schedulers, RNG and caches cannot
                    # interleave between sessions.
                    with domain.activate():
                        if kind == 'reset':
                            session.reset();session.engine._vace_cache.clear();result={'type':'reset_ack'}
                        elif kind == 'obs':
                            result=dict(session.predict(data),type='action')
                            stats['requests']+=1;save()
                        elif kind == 'ping':
                            result={'type':'pong',**session._pong_payload()}
                        else: raise ValueError('unknown_message_type')
                    await ws.send(json.dumps(result))
                except Exception as exc:
                    logging.exception('session request failed')
                    await ws.send(json.dumps({'type':'error','code':'internal_error','message':str(exc)}))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            session.shutdown();stats['active_connections']-=1;save()
    async def serve():
        async with websockets.serve(handler,host,port,max_size=MAX_MESSAGE_BYTES,ping_interval=None,max_queue=16):
            logging.info('Twenty-session shared-model server ready on %s:%d',host,port)
            await asyncio.Future()
    asyncio.run(serve())

if __name__ == '__main__':
    # Plus includes unseen perturbed prompts; always use the live frozen T5 encoder.
    os.environ.pop('OPENWAM_TEXT_CACHE_DIR', None)
    os.environ.pop('OPENWAM_COMPACT_TEXT_CONTEXT', None)
    assert json.loads((E/'environment_validation.json').read_text())['status']=='passed'
    from openwam.deploy import server
    server.PolicyServer.run = multisession_run
    server.main()
