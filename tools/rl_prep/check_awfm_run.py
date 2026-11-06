"""Fail closed on incomplete smoke output, including preservation probe checks."""
import argparse
import json
import math
from pathlib import Path


def check(root, preserve=None, expected_scenes=None, expected_updates=None):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    run = json.loads((root/'run.json').read_text())
    scores = json.loads((root/'validation/scores.json').read_text())
    plan = json.loads((root/'awfm_plan.json').read_text())
    if summary.get('complete') is not True or summary.get('updates', 0) < 1:
        raise ValueError('training/summary incomplete')
    if expected_updates is not None and summary['updates'] != expected_updates:
        raise ValueError('unexpected smoke update count')
    if not (root/'action_policy.pt').is_file():
        raise ValueError('checkpoint absent')
    rows = scores.get('rows', [])
    tokens = [r['token'] for r in rows]
    if not rows or len(tokens) != len(set(tokens)) or set(tokens) != set(run['eval_tokens']):
        raise ValueError('incomplete/duplicate evaluation scenes')
    if expected_scenes is not None and len(rows) != expected_scenes:
        raise ValueError('unexpected smoke scene count')
    for row in rows:
        for key in ('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress'):
            if not math.isfinite(row[key]): raise ValueError('nonfinite evaluation')
    if not plan or not any(not p.get('skipped', False) for p in plan.values()):
        raise ValueError('no usable candidate targets')
    actual = run.get('preserve', 'none')
    if preserve is not None and actual != preserve:
        raise ValueError('wrong preservation arm')
    if actual != 'none':
        p = summary.get('preservation', {})
        if p.get('kind') != actual or p.get('cached',0) < 1 or p.get('terms',0) < 1:
            raise ValueError('preservation path did not run')
        if p.get('initial_identity_checked') is not True:
            raise ValueError('teacher identity check absent')
        if not math.isfinite(p['first_drift']) or not 0 <= p['first_drift'] <= 1e-7:
            raise ValueError('invalid initial preservation drift')
        for key in ('sum_drift','max_drift'):
            if not math.isfinite(p[key]) or p[key] < 0: raise ValueError('invalid drift')
    return {'status':'PASS', 'preserve':actual, 'scenes':len(rows), 'updates':summary['updates']}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--preserve', choices=['none','global','targeted'])
    ap.add_argument('--expected-scenes', type=int)
    ap.add_argument('--expected-updates', type=int)
    args = ap.parse_args()
    print(json.dumps(check(args.root,args.preserve,args.expected_scenes,args.expected_updates)))
