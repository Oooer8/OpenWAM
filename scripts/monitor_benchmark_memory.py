"""Collect physical memory peaks for the two explicitly named benchmark jobs."""
import json
import subprocess
import sys
import time
from pathlib import Path


jobs = dict(zip(('batch16', 'batch8_accumulate2'), map(int, sys.argv[1:])))
peaks = {name:{} for name in jobs}
samples = {name:0 for name in jobs}


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


for _ in range(1200):
    current = next((name for name,pid in jobs.items() if alive(pid)), None)
    if current is None:
        break
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total',
                                      '--format=csv,noheader,nounits'],text=True)
    for line in output.splitlines():
        index,used,total = map(float,line.split(','))
        key = str(int(index))
        peaks[current][key] = {'peak_used_gib':max(peaks[current].get(key,{}).get('peak_used_gib',0),used/1024),
                               'total_reported_gib':total/1024}
    samples[current] += 1
    Path('work/setup_logs/no_checkpoint_physical_memory.json').write_text(json.dumps({'peaks':peaks,'samples':samples},indent=2))
    time.sleep(1)
