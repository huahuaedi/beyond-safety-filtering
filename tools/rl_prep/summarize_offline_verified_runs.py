"""Pair validation-only pilot results; require identical scenes and schedules."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    report={'split':'checkpoint_selection_val','pairs':[],
        'caveat':'Two-seed pilot on validation only; not a final-test performance claim.'}
    for seed in (20,21):
        a=args.root/f'pilot_v1_unconditional_seed{seed}'
        b=args.root/f'pilot_v1_conditioned_seed{seed}'
        if json.loads((a/'schedule.json').read_text())!=json.loads((b/'schedule.json').read_text()):
            raise ValueError('ablation arms saw different candidate schedules')
        sa=json.loads((a/'summary.json').read_text());sb=json.loads((b/'summary.json').read_text())
        if sa['updates']!=sb['updates'] or sa['examples']!=sb['examples']:
            raise ValueError('ablation budgets differ')
        ra={r['token']:r for r in json.loads((a/'validation/scores.json').read_text())['rows']}
        rb={r['token']:r for r in json.loads((b/'validation/scores.json').read_text())['rows']}
        if set(ra)!=set(rb): raise ValueError('validation scene sets differ')
        keys=sorted(ra);delta=np.array([rb[k]['score']-ra[k]['score'] for k in keys])
        metrics={k:sb['validation'][k]-sa['validation'][k] for k in sb['validation']}
        report['pairs'].append({'seed':seed,'scenes':len(keys),'unconditional':sa['validation'],
            'conditioned':sb['validation'],'conditioned_minus_unconditional':metrics,
            'improve_rate':float(np.mean(delta>1e-8)),'regression_rate':float(np.mean(delta < -1e-8)),
            'worst_10pct_delta':float(np.sort(delta)[:max(1,int(np.ceil(.1*len(delta))))].mean())})
    report['mean_paired_score_delta']=float(np.mean([x['conditioned_minus_unconditional']['score'] for x in report['pairs']]))
    baseline=args.root/'baseline_v1_baseline_seed20'/'summary.json'
    if baseline.is_file(): report['same_protocol_pretrained_baseline']=json.loads(baseline.read_text())['validation']
    path=args.root/'pilot_comparison.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
