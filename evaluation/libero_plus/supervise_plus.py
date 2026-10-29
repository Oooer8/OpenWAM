"""Prepare and validate Plus on CPU; automatically run only after standard LIBERO finishes."""
from common import *
import fcntl,subprocess,concurrent.futures,traceback,hashlib,signal,socket,queue,collections,statistics
lock=(D/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'status':'waiting_environment','supervisor_pid':os.getpid(),'started_unix':time.time(),'checkpoint':str(CKPT),'checkpoint_sha256':CHECKPOINT_SHA256,'total_rollouts':10030,'completed_rollouts':0,'successes':0,'model_replicas':16,'environment_workers':320,'environments_per_model':20,'prerequisite':str(R/'completed.json')}
children=[];servers=[]
def save(**kw):
 with mutex:state.update(kw,checked_unix=time.time());atomic(D/'status.json',state)
def run(cmd,env,logfile,timeout=None):
 logfile.parent.mkdir(parents=True,exist_ok=True)
 with logfile.open('a',buffering=1) as log:
  c=subprocess.Popen(cmd,cwd=D,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  with mutex:children.append(c)
  try:rc=c.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(c.pid,signal.SIGTERM)
   try:rc=c.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);rc=c.wait()
  atomic(logfile.with_suffix('.exit.json'),{'pid':c.pid,'returncode':rc,'finished_unix':time.time(),'command':cmd})
 assert rc==0,f'{logfile}: OS returncode {rc}'
def prepare_validation():
 while True:
  e=read(D/'environment_status.json') if (D/'environment_status.json').exists() else {}
  save(status='waiting_environment',preparation=e)
  if e.get('status')=='failed':raise RuntimeError('environment preparation failed: '+e.get('error',''))
  if e.get('status') in ['installed_pending_validation','ready'] and (D/'native_receipt.json').exists():break
  time.sleep(10)
 if not (D/'environment_validation.json').exists():
  save(status='validating_environment',validation_stage='inventory')
  run([LP,str(D/'validate_environment.py')],clientenv('inventory'),D/'logs/inventory.log',1800)
  cases=read(D/'environment_case_plan.json');seen=set();reports=[]
  for case in cases:
   key=case['suite'],case['task_id']
   if key in seen:continue
   seen.add(key);save(validation_stage='real_cpu_environment',validation_case={'suite':key[0],'task_id':key[1]},validated_cases=len(reports))
   run([LP,str(D/'validate_environment.py'),'case',key[0],str(key[1])],clientenv('env_'+key[0]+'_'+str(key[1])),D/f'logs/environment_{key[0]}_{key[1]}.log',1800)
   reports.append(read(D/f'environment_cases/{key[0]}_{key[1]}.json'))
  # One cold-process replay checks the official global seed path and real observations.
  case=cases[0];p=D/f'environment_cases/{case["suite"]}_{case["task_id"]}.json';first=read(p)
  run([LP,str(D/'validate_environment.py'),'case',case['suite'],str(case['task_id'])],clientenv('env_repeat'),D/'logs/environment_repeat.log',1800)
  assert first['hashes']==read(p)['hashes'],'fresh-process replay differs'
  h=hashlib.sha256()
  with CKPT.open('rb') as f:
   for buf in iter(lambda:f.read(64*1024*1024),b''):h.update(buf)
  assert h.hexdigest()==CHECKPOINT_SHA256
  atomic(D/'environment_validation.json',{'status':'passed','real_rendered_cases':reports,'fresh_process_repeat_exact':True,'checkpoint':str(CKPT),'checkpoint_sha256':h.hexdigest(),'inventory':str(D/'task_inventory.json'),'finished_unix':time.time()})
 assert read(D/'environment_validation.json')['status']=='passed'
 e=read(D/'environment_status.json');e.update(status='ready',stage='validated_waiting_standard_libero',checked_unix=time.time());atomic(D/'environment_status.json',e)
def audit_standard(root):
 summary=read(root/'summary.json');assert summary['status']=='completed' and summary['rollouts']==2000
 unique={};files=[]
 for filename in summary['results']:
  if isinstance(filename,dict):filename=filename['file']
  path=Path(filename);r=read(path)
  for t in r['trials']:
   key=r['suite'],r['task_id'],t['trial'];assert key not in unique,key
   assert t['model_eval_seed']==42+100000*SUITES.index(key[0])+1000*key[1]+key[2]
   assert t['policy_steps']>0 and t['inference_latencies_ms'];unique[key]=t
  files.append({'file':filename,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rollouts':len(r['trials'])})
 expected={(s,t,i) for s in SUITES for t in range(10) for i in range(50)}
 assert set(unique)==expected and read(root/'exit_receipt.json')['returncode']==0
 atomic(D/'prerequisite_libero_audit.json',{'status':'passed','unique_rollouts':2000,'successes':sum(x['success'] for x in unique.values()),'task_counts':{s:{str(t):50 for t in range(10)} for s in SUITES},'results':files,'exit_receipt':read(root/'exit_receipt.json'),'checked_unix':time.time()})
def await_standard():
 while True:
  std=read(R/'evaluation/status.json');launch=read(R/'evaluation/launch_receipt.json');root=Path(launch['script']).parent
  ex=read(root/'exit_receipt.json') if (root/'exit_receipt.json').exists() else None
  save(status='waiting_standard_libero',standard_status=std.get('status'),standard_completed_rollouts=std.get('completed_rollouts'),standard_checked_unix=std.get('checked_unix'))
  if std['status']=='completed' and ex and ex['returncode']==0 and (root/'summary.json').exists():
   pids=[std['supervisor_pid'],*std.get('server_pids',{}).values(),*(int(p) for p in ex.get('child_exit_codes',{}))]
   if not any(running(p) for p in pids):break
  # A standard evaluation failure is never a trigger to stop it or bypass its gate.
  time.sleep(15)
 audit_standard(root)
 while True:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  free={int(x.split(',')[0]):float(x.split(',')[1]) for x in rows.strip().splitlines()}
  save(status='waiting_gpu_capacity',free_gpu_memory_mib=free)
  if all(free.get(g,0)>=56000 for g in range(8)):break
  time.sleep(15)
def start_server(gpu,replica):
 port=19920+2*gpu+replica
 with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0,f'occupied port {port}'
 cmd=[P,str(D/'model_server.py'),'--ckpt-dir',str(CKPT.parent),'--ckpt-name',CKPT.name,'--device','cuda:0','--host','127.0.0.1','--port',str(port),'--denoise-steps','10','--denoise-mode','sync','--inference-mode','sync','--inference-horizon','10','--compile-enabled','false','optimization.dit_cache.enabled=false']
 log=(D/f'logs/servers/gpu{gpu}_replica{replica}.log');log.parent.mkdir(parents=True,exist_ok=True);handle=log.open('a',buffering=1)
 c=subprocess.Popen(cmd,cwd=S,env=dict(baseenv,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
 children.append(c);servers.append((gpu,replica,port,c,handle));save(server_pids={f'{g}:{r}':c.pid for g,r,_,c,_ in servers})
def wait_ready():
 start=time.time()
 while True:
  assert all(c.poll() is None for _,_,_,c,_ in servers),'Plus server exited; inspect server logs'
  count=0
  for _,_,port,_,_ in servers:
   with socket.socket() as sock:sock.settimeout(1);count+=sock.connect_ex(('127.0.0.1',port))==0
  save(ready_servers=count)
  if count==len(servers):return
  assert time.time()-start<1800,'model initialization exceeded 30 minutes; inspect logs'
  time.sleep(5)
def valid(path,job):
 if not path.exists():return None
 r=read(path);assert r['suite']==job['suite'] and r['task_id']==job['task_id'] and r['task']==job['task'] and r['instruction']==job['instruction']
 assert r['seed']==10000 and r['rng_mode']=='official_global' and r['trial_start']==0 and r['trial_stop']==1
 assert r['policy_config_sha256']==hashlib.sha256((D/'policy_config.yml').read_bytes()).hexdigest()
 assert r['settle_steps']==30 and r['max_steps']==(700 if job['suite']=='libero_10' else 600)
 assert len(r['trials'])==1;t=r['trials'][0]
 assert t['trial']==0 and t['model_eval_seed']==10000 and 0<t['policy_steps']<=r['max_steps']
 assert t['inference_latencies_ms'] and len(t['inference_latencies_ms'])==t['policy_steps'] and len(t['inference_round_trip_latencies_ms'])==t['policy_steps']
 assert all(0<=x<1e9 for x in t['inference_latencies_ms']+t['inference_round_trip_latencies_ms'])
 return r
def run_job(gpu,replica,job,smoke=False):
 suite=job['suite'];idx=job['task_id'];name=f'{suite}_task{idx:05d}';phase='smoke' if smoke else 'full'
 for attempt in range(1,4):
  dest=D/phase/suite/f'task_{idx:05d}'/f'attempt_{attempt:02d}';log=D/f'logs/{phase}/{name}_attempt{attempt:02d}.log';ep=log.with_suffix('.exit.json')
  try:
   if ep.exists() and read(ep)['returncode']==0:
    result=valid(dest/'results.json',job)
    if result:return result,str(dest/'results.json')
   if ep.exists():continue
   if str(dest) in interrupted_attempts:continue
   cmd=[LP,str(S/'benchmarks/libero-plus/single_eval.py'),'--config',str(D/'policy_config.yml'),'--suite',suite,'--task-id',str(idx),'--host','127.0.0.1','--port',str(19920+2*gpu+replica),'--trial-start','0','--num-trials','1','--result-dir',str(dest),'--seed','10000']
   run(cmd,clientenv(name+f'_{phase}_{attempt}'),log,7200)
   result=valid(dest/'results.json',job);assert result;return result,str(dest/'results.json')
  except Exception as exc:
   atomic(dest/'infrastructure_error.json',{'error':str(exc),'traceback':traceback.format_exc(),'checked_unix':time.time()})
   if attempt==3:raise
 raise RuntimeError(f'{name}: all three attempts failed; inspect preserved logs')
def aggregate(results,jobs):
 expected={(x['suite'],x['task_id']) for x in jobs};unique={};inventory=[];groups={k:collections.defaultdict(lambda:[0,0]) for k in ['suite','category','difficulty_level']};service=[];roundtrip=[];wall=[]
 lookup={(j['suite'],j['task_id']):j for j in jobs}
 for r,path in results:
  job=lookup[(r['suite'],r['task_id'])];valid(Path(path),job);key=(r['suite'],r['task_id']);assert key not in unique;unique[key]=r
  success=int(r['trials'][0]['success'])
  for dim in groups:groups[dim][job[dim]][0]+=success;groups[dim][job[dim]][1]+=1
  t=r['trials'][0];service+=t['inference_latencies_ms'];roundtrip+=t['inference_round_trip_latencies_ms'];wall.append(t['rollout_wall_seconds'])
  inventory.append({'file':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'suite':key[0],'task_id':key[1],'success':bool(success)})
 assert set(unique)==expected and len(unique)==10030
 def stats(v):
  s=sorted(v);return {'count':len(v),'mean':statistics.mean(v),'p50':statistics.median(v),'p95':s[min(len(s)-1,int(len(s)*.95))]}
 groups={k:{str(name):{'successes':v[0],'rollouts':v[1],'success_rate':v[0]/v[1]} for name,v in g.items()} for k,g in groups.items()}
 successes=sum(r['successes'] for r in unique.values());cat=list(groups['category'].values())
 atomic(D/'summary.json',{'status':'completed','rollouts':10030,'successes':successes,'success_rate':successes/10030,'macro_category_success_rate':statistics.mean(v['success_rate'] for v in cat),'breakdowns':groups,'inference_service_latency_ms':stats(service),'inference_roundtrip_latency_ms':stats(roundtrip),'rollout_wall_seconds':stats(wall),'results':inventory,'protocol':read(D/'protocol.json'),'full_started_unix':state['full_started_unix'],'finished_unix':time.time()})
try:
 assert read(R/'completed.json')['actual_node_exit_codes']==[0,0,0,0]
 assert read(D/'source_sha256.json')=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in read(D/'source_sha256.json')}
 assert read(D/'environment_validation.json')['status']=='passed'
 interrupted_attempts=set()
 jobs=read(D/'full_manifest.json')['jobs'];assert len(jobs)==10030
 q=queue.Queue();results=[]
 for job in jobs:q.put(job)
 state['active_environments_per_model']=[0]*16
 def free_memory():
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  return {int(x.split(',')[0]):float(x.split(',')[1]) for x in rows.strip().splitlines()}
 def footprint():
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
  mem={int(x.split(',')[0]):float(x.split(',')[1]) for x in rows.strip().splitlines() if x.split(',')[0].strip().isdigit() and x.split(',')[1].strip().isdigit()}
  return max([mem[c.pid] for _,_,_,c,_ in servers if c.pid in mem],default=22000)
 def worker(g,replica,slot):
  while True:
   try:job=q.get_nowait()
   except queue.Empty:return
   with mutex:state['active_environments_per_model'][2*g+replica]+=1;save()
   try:
    r,path=run_job(g,replica,job)
    with mutex:results.append((r,path));state['completed_rollouts']+=1;state['successes']+=r['successes'];save()
   finally:
    with mutex:state['active_environments_per_model'][2*g+replica]-=1;save()
    q.task_done()
 state.update(model_replicas=0,environment_workers=0)
 save(status='waiting_shared_gpu_capacity',shared_with_existing_tasks=True,minimum_free_memory_mib=26000)
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=320);fs=[]
 try:
  while state['completed_rollouts']<len(jobs):
   for f in fs:
    if f.done():f.result()
   assert all(c.poll() is None for _,_,_,c,_ in servers),'model service exited'
   used={(g,r) for g,r,_,_,_ in servers};free=free_memory();reserve=max(26000,footprint()+4096)
   save(shared_gpu_free_memory_mib=free,additional_replica_required_mib=reserve)
   # Every card receives its first replica before any card receives a second.
   first=[(g,0) for g in range(8) if (g,0) not in used]
   candidates=first if first else [(g,1) for g in range(8) if (g,1) not in used]
   candidate=next(((g,r) for g,r in candidates if free.get(g,0)>=reserve),None)
   if candidate and not q.empty():
    g,replica=candidate;save(status='loading_model',loading_gpu=g,loading_replica=replica)
    start_server(g,replica);wait_ready()
    if not fs:
     save(status='new_checkpoint_policy_smoke')
     result,path=run_job(g,replica,jobs[0],smoke=True)
     atomic(D/'policy_smoke.json',{'status':'passed','result':path,'success':result['successes']==1,'finite_actions':True,'excluded_from_formal_10030':True,'checkpoint':str(CKPT),'finished_unix':time.time()})
     state['full_started_unix']=time.time()
    state.update(model_replicas=len(servers),environment_workers=20*len(servers))
    fs += [pool.submit(worker,g,replica,n) for n in range(20)]
    save(status='full_running')
   else:time.sleep(15)
  for f in fs:f.result()
 finally:pool.shutdown(wait=True)
 save(status='auditing_results');aggregate(results,jobs);save(status='completed',returncode=0,finished_unix=time.time())
except BaseException as exc:save(status='failed',returncode=1,error=str(exc),traceback=traceback.format_exc(),finished_unix=time.time());raise
finally:
 for c in children:
  if c.poll() is None:
   try:os.killpg(c.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 for c in children:
  if c.poll() is None:
   try:c.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
 atomic(D/'exit_receipt.json',{'pid':os.getpid(),'returncode':state.get('returncode',1),'finished_unix':time.time(),'child_exit_codes':{str(c.pid):c.poll() for c in children}})
