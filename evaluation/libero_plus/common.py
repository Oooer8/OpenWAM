from pathlib import Path
import os,json,time,threading
R=Path('/mnt/vepfs01/output/tiecheng.guo/artifacts/waw/openwam_libero_mutual_robot_rgb_32gpu_20261025/runs/libero-mutual-robot-rgb-32gpu-fix-20261026')
D=R/'libero_plus_evaluation';S=D/'source';B=R/'evaluation/parallel320';A=Path('/mnt/vepfs01/output/tiecheng.guo/artifacts/waw/openwam_libero_vav_20261023/evaluation/parallel80')
P='/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin/python';LP='/mnt/vepfs01/output/tiecheng.guo/artifacts/waw/openwam_libero_vav_20261023/libero_plus_evaluation/client_env/bin/python'
CKPT=Path('/mnt/vepfs01/output/tiecheng.guo/artifacts/waw/openwam_libero_mutual_robot_rgb_32gpu_20261025/runs/libero-mutual-robot-rgb-32gpu-fix-20261026/artifacts/2026-10-26_02-40-31/checkpoint_step_10690.safetensors')
CHECKPOINT_SHA256=None
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
mutex=threading.RLock()
def atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with mutex:
  tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())
def running(pid):
 try:
  p=Path('/proc')/str(pid);st=p.joinpath('stat').read_text().split(') ',1)[1].split()
  if st[0]=='Z':return False
  return p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return False
baseenv=dict(os.environ,PYTHONPATH=str(S)+os.pathsep+str(S/'benchmarks/libero-plus'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NO_PROXY='127.0.0.1,localhost')
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:baseenv.pop(k,None)
def clientenv(name):
 native=D/'native/root'
 return dict(baseenv,MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',CUDA_VISIBLE_DEVICES='',LIBERO_PLUS_PATH=str(D/'LIBERO-plus'),LIBERO_PLUS_CONFIG_ROOT=str(D/'libero_config'/name),LIBERO_CONFIG_PATH=str(D/'libero_config'/name),LP_NUM_THREADS='1',GALLIUM_DRIVER='llvmpipe',MAGICK_HOME=str(native/'usr'),MAGICK_CONFIGURE_PATH=str(native/'etc/ImageMagick-6')+os.pathsep+str(native/'usr/share/ImageMagick-6'),MAGICK_CODER_MODULE_PATH=str(native/'usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders'),LD_LIBRARY_PATH=str(native/'usr/lib/x86_64-linux-gnu')+os.pathsep+str(A/'render_dependencies/root/usr/lib/x86_64-linux-gnu')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
