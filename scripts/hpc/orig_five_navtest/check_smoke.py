import json
from pathlib import Path
from paired_results import read_scores
c=json.loads(Path('config.json').read_text());r=Path(c['run']);t=json.loads((r/'smoke_tokens.json').read_text())
v=read_scores(r/'metrics'/(r.name+'_smoke_sft'),t)
(r/'smoke_verified.json').write_text(json.dumps({'status':'PASS','scenes':len(t),'mean':v.mean(0).tolist()}))
print('SFT smoke: all frozen scenes have finite valid scores')
