"""Validate full scene coverage, select on validation only, report final holdout."""
import argparse
import csv
import json
from pathlib import Path

METRICS=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort')


def read_metrics(root, name, tokens):
    files=sorted((root/'metrics'/name).rglob('*.csv'))
    if len(files)!=1: raise ValueError(f'{name}: expected one score CSV, got {files}')
    with files[0].open() as f: rows=[r for r in csv.DictReader(f) if r['token']!='average']
    if len(rows)!=len(tokens) or {r['token'] for r in rows}!=set(tokens): raise ValueError(f'{name}: incomplete or duplicate evaluation')
    if any(r['valid'].lower() not in ('true','1') for r in rows): raise ValueError(f'{name}: scoring failures')
    import math
    metrics={k:sum(float(r[k]) for r in rows)/len(rows) for k in METRICS}
    if not all(math.isfinite(v) for v in metrics.values()): raise ValueError('nonfinite metrics')
    return metrics


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--stage',choices=['select','final'],required=True); p.add_argument('--baseline',required=True); a=p.parse_args()
    run=a.run; prefix=run.name
    if a.stage=='select':
        tokens=json.loads((run/'val_tokens.json').read_text())
        base=read_metrics(run,prefix+'_val_baseline',tokens)
        rows={'baseline':base}; selected='baseline'; checkpoint=a.baseline; best=base['score']
        for step in (50,100,150,200):
            row=read_metrics(run,f'{prefix}_val_{step}',tokens); rows[str(step)]=row
            safe=all(row[k]>=base[k]-1e-8 for k in METRICS[1:4])
            if safe and row['score']>best:
                selected=str(step); best=row['score']; checkpoint=str(run/'checkpoints'/(f'step_{step:06d}'+('_final' if step==200 else '')))
        result=dict(selected=selected,checkpoint=checkpoint,validation=rows,selection_rule='best PDMS with validation mean NC/DAC/TTC >= baseline; not a safety guarantee')
        (run/'selected.json').write_text(json.dumps(result,indent=2))
    else:
        tokens=json.loads((run/'test_tokens.json').read_text())
        base=read_metrics(run,prefix+'_test_baseline',tokens); trained=read_metrics(run,prefix+'_test_selected',tokens)
        result=dict(selection=json.loads((run/'selected.json').read_text()),final_eval_baseline=base,final_eval_selected=trained,delta={k:trained[k]-base[k] for k in METRICS})
        (run/'final_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
