"""Fixed-schedule, equal-scoring-budget replay mechanism experiment."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def step_policy(mode, kind):
    if mode not in ('matched_reference','rescored_rl','rescored_rl_reference') or kind not in ('main','risk'):
        raise ValueError((mode,kind))
    return dict(train_on_group=(kind=='main' or mode in ('rescored_rl','rescored_rl_reference')),
                rehearse=(kind=='risk' and mode in ('matched_reference','rescored_rl_reference')))


def make_schedule(train_tokens, risk_tokens, *, seed, main_steps=50):
    if not train_tokens or not risk_tokens or main_steps<1: raise ValueError('empty schedule inputs')
    if not set(risk_tokens)<=set(train_tokens): raise ValueError('risk scenes must be training scenes')
    regular=random.Random(seed); replay=random.Random(seed+100003)
    return [item for _ in range(main_steps) for item in
            [dict(kind='main',token=regular.choice(sorted(train_tokens))),
             dict(kind='risk',token=replay.choice(sorted(risk_tokens)))]]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline-run',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    train=set(json.loads((a.baseline_run/'train_tokens.json').read_text()))
    excluded=set(json.loads((a.baseline_run/'val_tokens.json').read_text()))|set(json.loads((a.baseline_run/'test_tokens.json').read_text()))
    if train&excluded: raise ValueError('training/evaluation overlap')
    original=json.loads(a.manifest.read_text())
    entries=[e for e in original['entries'] if e['scene_token'] in train and e['verified_unsafe_indices'] and e['risk_priority']>0]
    entries=sorted(entries,key=lambda e:(-e['risk_priority'],e['scene_token']))[:32]
    if not entries: raise ValueError('no verified training risk scenes')
    tokens=[e['scene_token'] for e in entries]
    if len(tokens)!=len(set(tokens)):raise ValueError('duplicate risk scene')
    payload={**original,'entries':entries,'development_scenes':len(entries),'verified_risk_scenes':len(entries)}
    payload['verified_unsafe_candidates']=sum(len(e['verified_unsafe_indices']) for e in entries)
    payload['verification_calls']=len(entries)*payload['verification_budget_per_scene']
    (a.output/'fixed_manifest.json').write_text(json.dumps(payload,indent=2))
    for seed in range(3):
        schedule=make_schedule(train,tokens,seed=seed)
        (a.output/f'schedule_seed{seed}.json').write_text(json.dumps(schedule,indent=2))
    protocol=dict(comparison=['matched_reference','rescored_rl'],seeds=[0,1,2],main_groups=50,risk_groups=50,candidates_per_group=16,risk_buffer_size=len(tokens),online_insert=False,manifest_source=str(a.manifest),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),risk_tokens=tokens,scoring='Both arms resample and score complete groups plus probes. Reference arm discards risk-group RL signal and performs reference-path rehearsal. RL arm uses the existing guarded PG/KL/rollback update.',scope='Mechanism experiment with a fixed manifest, NOT evidence that WM beats random/action selection. NAVSIM scoring budgets matched; wall time and backward compute may differ.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))


if __name__=='__main__':main()
