from concurrent.futures import ThreadPoolExecutor
import json, os, subprocess
from pathlib import Path
base=Path(__file__).resolve().parent
spec=json.loads((base/'launch.json').read_text())
env=os.environ | spec['environment']
def run(kind):
    for entry in spec['commands']:
        if entry['kind'] != kind:
            continue
        name=entry['task']+'-'+kind
        with (base/(name+'.log')).open('x') as log:
            result=subprocess.run(entry['argv'],env=env,stdout=log,stderr=subprocess.STDOUT)
        status=dict(entry,returncode=result.returncode)
        if kind=='segmentation':
            metrics=base/name/'metrics.json'
            status['metrics']=json.loads(metrics.read_text()) if metrics.exists() else None
            status['accepted']=result.returncode==0 and status['metrics'] is not None and status['metrics']['complete'] and status['metrics']['failure']==0
        else:
            status['accepted']=result.returncode==0
        (base/(name+'-status.json')).write_text(json.dumps(status,indent=2)+'\n')
        print(name, 'returncode', result.returncode, 'accepted',status['accepted'],flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run,['numerics','segmentation']))
