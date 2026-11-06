"""Pair-file target clearance and empirical target/output distance; no safety certificate.
Predictions: <prediction-dir>/<token>.npz, key absolute, shape (8,3),
matching candidate ego coordinates. Norm: mean planar waypoint distance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def trajectory(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (8, 3) or not np.isfinite(x).all():
        raise ValueError(f'Expected finite ego-frame (8,3) trajectory, got {x.shape}')
    return x


def distance(a, b):
    return float(np.linalg.norm(trajectory(a)[:, :2]-trajectory(b)[:, :2], axis=1).mean())


def summary(x):
    x = np.asarray(x, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), p10=float(np.quantile(x,.1)),
                median=float(np.median(x)), p90=float(np.quantile(x,.9)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-root',type=Path,required=True)
    p.add_argument('--geom-root',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--prediction-dir',type=Path)
    p.add_argument('--cohort',required=True,choices=['train','heldout'])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    pairs=[json.loads(t) for t in a.pairs.read_text().splitlines() if t.strip()]
    if not pairs or len({r['token'] for r in pairs})!=len(pairs):
        raise ValueError('Empty or duplicate-token pair file')
    rows=[]
    for r in pairs:
        token=r['token']; split=r.get('split','train'); w=int(r['winner'])
        if Path(token).name!=token or Path(split).name!=split: raise ValueError('Invalid token/split')
        with np.load(a.candidate_root/split/f'{token}.npz',allow_pickle=False) as z:
            target=trajectory(z['absolute_trajectories'][w])
            ref=trajectory(z['reference_probe_absolute_trajectories'].mean(axis=0))
        with np.load(a.geom_root/split/f'{token}.npz',allow_pickle=False) as z:
            names=[str(t) for t in z['feature_names']]
            # geometric_features.py stores minimum clearance divided by ten.
            margin=float(z['geom_features'][w,names.index('min_clearance')])*10.
        if not np.isfinite(margin): raise ValueError(f'Nonfinite margin {token}')
        row=dict(token=token,winner=w,target_clearance_m=margin,target_reference_distance_m=distance(target,ref))
        if a.prediction_dir:
            with np.load(a.prediction_dir/f'{token}.npz',allow_pickle=False) as z:
                row['output_target_distance_m']=distance(z['absolute'],target)
        rows.append(row)
    report=dict(cohort=a.cohort,pairs_sha256=hashlib.sha256(a.pairs.read_bytes()).hexdigest(),
                target_clearance_m=summary([r['target_clearance_m'] for r in rows]),
                target_reference_distance_m=summary([r['target_reference_distance_m'] for r in rows]),rows=rows,
                interpretation='Finite-cohort empirical pairing; not a flow-loss error, population bound, or certified simulator safety result. Compare rules only on identical token sets and latent tags.')
    if a.prediction_dir:
        e=np.array([r['output_target_distance_m'] for r in rows])
        report['epsilon_w_m']=float(np.sqrt(np.mean(e**2)))
        report['output_target_distance_m']=summary(e)
    else: report['epsilon_w_m']=None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__': main()
