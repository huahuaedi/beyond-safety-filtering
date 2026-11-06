"""Report matched-budget replay comparison without attributing it to WM selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from summarize_posttrain import METRICS, read_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline-run',type=Path,required=True);a=p.parse_args()
    tokens=sorted(json.loads((a.baseline_run/'test_tokens.json').read_text()))
    result={}; arrays={}; budgets={}
    for mode in ('matched_reference','rescored_rl'):
        values=[]; result[mode]=[]; budgets[mode]=[]
        for seed in range(3):
            arm=f'{mode}_seed{seed}'; name=a.run.name+'_'+arm
            metrics=read_metrics(a.run,name,tokens)
            path=next((a.run/'metrics'/name).rglob('*.csv'))
            with path.open() as f: rows={r['token']:r for r in csv.DictReader(f) if r['token']!='average'}
            values.append(np.asarray([[float(rows[t][k]) for k in METRICS] for t in tokens]))
            updates=[json.loads(s) for s in (a.run/arm/'replay_updates.jsonl').read_text().splitlines()]
            if len(updates)!=100: raise ValueError(f'{arm}: incomplete updates')
            budgets[mode].append({k:updates[-1][k] for k in ('score_calls','scored_trajectories')})
            result[mode].append(metrics)
        arrays[mode]=np.stack(values)
    if budgets['matched_reference']!=budgets['rescored_rl']:raise ValueError(f'scoring budget mismatch: {budgets}')
    delta=(arrays['rescored_rl']-arrays['matched_reference']).mean(axis=0)
    rng=np.random.default_rng(23); indices=rng.integers(0,len(tokens),(10000,len(tokens)))
    ci=np.quantile(delta[indices].mean(axis=1),[.025,.975],axis=0)
    baseline=read_metrics(a.baseline_run,a.baseline_run.name+'_test_baseline',tokens)
    report=dict(metrics=list(METRICS),baseline=baseline,per_seed=result,scoring_budgets=budgets,rl_minus_reference_mean=delta.mean(axis=0).tolist(),paired_scene_bootstrap_95_ci=ci.tolist(),interpretation='Fixed-manifest mechanism comparison; not WM selector benefit. Intervals condition on fitted models; scene correlation/model fitting uncertainty not included.')
    (a.run/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__=='__main__':main()
