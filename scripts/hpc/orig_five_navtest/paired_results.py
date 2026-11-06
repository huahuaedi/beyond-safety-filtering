"""Validate matched execution; descriptive and paired log-bootstrap reports."""
import csv,json,math,sys
from pathlib import Path
import numpy as np
ARMS=('baseline','pareto')
METRICS=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort')

def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]

def check(run,smoke=False):
    budget={};activation={}
    for seed in ([0] if smoke else range(3)):
        schedule=json.loads((run/f'schedule_seed{seed}.json').read_text())[:2 if smoke else 100];paired=[]
        for mode in ARMS:
            arm=('smoke_' if smoke else '')+f'{mode}_seed{seed}';d=run/arm
            updates=rows(d/'replay_updates.jsonl');audits=rows(d/'optimization_audit.jsonl')
            if len(updates)!=len(schedule) or len(audits)!=len(schedule):raise ValueError('Incomplete '+arm)
            replay=0
            for i,(item,u,a) in enumerate(zip(schedule,updates,audits)):
                replay+=int(item['kind']=='risk')
                if any(u.get(k)!=v for k,v in {'step':i,'scene':item['token'],'kind':item['kind'],'train_on_group':True,'rehearsal':item['kind']=='risk','mode':'rescored_rl_reference','policy_optimizer_steps':i+1,'rehearsal_optimizer_steps':replay}.items()):raise ValueError('Incorrect update '+arm)
                if a['step']!=i or a['scene']!=item['token'] or (a['positive_guard'] or a['pareto_coef']!=(0.25 if mode=='pareto' else 0.0)) or a['kl_mode']!='legacy':raise ValueError('Incorrect mode audit '+arm)
                if not all(math.isfinite(float(u[k])) for k in ('pg_loss','replay_loss','grad_norm','replay_grad_norm')) or not math.isfinite(a['observed_kl']) or not math.isfinite(a['pareto_loss']):raise ValueError('Nonfinite update')
                if mode=='baseline' and a['pareto_active_candidates']!=0:raise ValueError('Baseline changed')
            if not (d/f'step_{len(schedule):06d}_final/config.json').is_file():raise ValueError('No final checkpoint '+arm)
            budget[arm]={k:updates[-1][k] for k in ('score_calls','scored_trajectories','policy_optimizer_steps','rehearsal_optimizer_steps')};paired.append(budget[arm])
            activation[arm]={'preference_pairs':sum(a['pareto_pairs'] for a in audits),'active_changed':sum(a['pareto_active_candidates'] for a in audits),'groups_with_pairs':sum(a['pareto_pairs']>0 for a in audits),'max_gpu_allocated_bytes':max(a['peak_gpu_allocated_bytes'] for a in audits)}
        if paired[0]!=paired[1]:raise ValueError('Unequal oracle/update budgets '+str(paired))
    result={'budget':budget,'activation':activation}
    (run/('smoke_verified.json' if smoke else 'training_verified.json')).write_text(json.dumps(result,indent=2))
    if not smoke and sum(activation[f'pareto_seed{s}']['active_changed'] for s in range(3))==0:
        (run/'no_activation.json').write_text(json.dumps({'status':'NO_ACTIVE_CHANGES','meaning':'Rule never affected a nonzero policy-gradient weight; no efficacy claim. Evaluation skipped.'},indent=2))
    print(json.dumps(result,indent=2))

def read_scores(path,tokens):
    files=list(path.rglob('*.csv'))
    if len(files)!=1:raise ValueError('Expected one score CSV '+str(path))
    with files[0].open() as f:rs=[r for r in csv.DictReader(f) if r['token']!='average']
    if len(rs)!=len(tokens) or {r['token'] for r in rs}!=set(tokens):raise ValueError('Missing/duplicate score rows')
    if any(r['valid'].lower() not in ('1','true') for r in rs):raise ValueError('Invalid scored trajectory')
    lookup={r['token']:r for r in rs};v=np.array([[float(lookup[t][k]) for k in METRICS] for t in tokens])
    if not np.isfinite(v).all():raise ValueError('Nonfinite scores')
    return v

def compare(after,before,tokens,mapping,draws=5000):
    # Inputs seeds x scenes x metrics; cluster by log after seed averaging.
    delta=after-before;scene=delta.mean(0);logs=sorted({mapping[t] for t in tokens})
    if len(logs)<2:raise ValueError('Too few logs for CI')
    sums=np.stack([scene[[mapping[t]==log for t in tokens]].sum(0) for log in logs]);counts=np.array([sum(mapping[t]==log for t in tokens) for log in logs])
    rng=np.random.default_rng(20260930);boot=[]
    for start in range(0,draws,250):
        ix=rng.integers(0,len(logs),size=(min(250,draws-start),len(logs)))
        boot.append(sums[ix].sum(1)/counts[ix].sum(1)[:,None])
    ci=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
    transitions=[]
    for s in range(after.shape[0]):
        item={}
        for j,name in [(1,'NC'),(2,'DAC'),(3,'TTC_strict')]:
            a=after[s,:,j]>=1-1e-8;b=before[s,:,j]>=1-1e-8
            item[name]={'new_failures':int((b&~a).sum()),'recovered':int((~b&a).sum())}
        a=after[s,:,3]>=.5;b=before[s,:,3]>=.5
        item['TTC_training_gate']={'new_failures':int((b&~a).sum()),'recovered':int((~b&a).sum())};transitions.append(item)
    return {'mean_delta':scene.mean(0).tolist(),'per_seed_delta':delta.mean(1).tolist(),'log_bootstrap_95ci':ci.tolist(),'safety_transitions':transitions,'ci_scope':'Scene-weighted whole-log bootstrap, averaging 3 paired training seeds; not full training uncertainty.'}

def report(run):
    if (run/'no_activation.json').exists():
        result=json.loads((run/'no_activation.json').read_text());(run/'report.json').write_text(json.dumps(result,indent=2));(run/'report.md').write_text('规则未影响有效策略梯度，本轮没有验证其效果；评估已跳过。\n');return
    protocol=json.loads((run/'protocol.json').read_text());tokens=sorted(json.loads((run/'test_tokens.json').read_text()));mapping=json.loads((run/'token_log_map.json').read_text())
    v={mode:np.stack([read_scores(run/'metrics'/f'{run.name}_{mode}_seed{s}',tokens) for s in range(3)]) for mode in ARMS}
    sft=read_scores(run/'metrics'/f'{run.name}_sft',tokens);v['sft']=np.repeat(sft[None],3,axis=0)
    result={'protocol':protocol,'metrics':METRICS,'scenes':len(tokens),'training':json.loads((run/'training_verified.json').read_text()),'mean':{k:a.mean((0,1)).tolist() for k,a in v.items()},'comparisons':{a+'_minus_'+b:compare(v[a],v[b],tokens,mapping) for a,b in [('pareto','baseline'),('baseline','sft'),('pareto','sft')]}}
    (run/'report.json').write_text(json.dumps(result,indent=2))
    lines=['# RL Pareto auxiliary policy-gradient development ablation','','全部数值乘100；固定最终checkpoint，报告全部三个种子。','','| Arm | PDMS | NC | DAC | TTC | Progress | Comfort |','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('sft','baseline','pareto'):lines.append('| '+name+' | '+' | '.join(f'{x*100:.3f}' for x in result['mean'][name])+' |')
    lines+=['','主比较是 pareto−baseline；同时报告对SFT的变化。逐种子新增失败和log级区间见report.json。','开发集已被查看过，不作全新泛化结论。训练log与评估隔离，但SFT训练来源未核实。','若进度下降或安全无改善，应如实判为无收益；不按测试分数选择checkpoint。']
    (run/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':
    c=json.loads(Path(__file__).with_name('config.json').read_text());run=Path(c['run']);task=sys.argv[1]
    if task in ('check_smoke','check_train'):check(run,task=='check_smoke')
    elif task=='report':report(run)
    else:raise ValueError(task)
