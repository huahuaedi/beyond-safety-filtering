import json,sys,tarfile
from pathlib import Path
pkg=Path(sys.argv[1]);run=Path(json.loads((pkg/'config.json').read_text())['run'])
def clean(i):i.uid=i.gid=0;i.uname=i.gname='';i.pax_headers={};return i
files=[(p,'logs/'+p.name) for pat in ('*.log','*.err','jobs.txt','config.json') for p in pkg.glob(pat)]
if run.exists():
 files += [(p,p.name) for p in run.glob('*.json')]
 files += [(p,str(p.relative_to(run))) for p in (run/'metrics').rglob('*.csv')]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',format=tarfile.PAX_FORMAT) as tf:
 for p,name in files:tf.add(str(p),arcname=name,recursive=False,filter=clean)
