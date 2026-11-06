"""Offline waypoint-policy extraction from NAVSIM-verified candidate data.

Both ablation arms use identical candidates, flow noise/times, unconditional
loss and expert anchoring. Only the quality indicator differs. Frozen VLM;
no learned world-model reward or online GRPO trajectory likelihood is used.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time
import numpy as np
import torch
from torch import nn

from offline_verified_data import (read_scene, verified_labels, topk_safe_labels, normalize_action,
    build_schedule, stable_seed, POSITIVE, UNCONDITIONAL)
# Reuse the exact main-branch Action Expert forward and environment setup.
import rl_stage1_prototype as core
from train_native_control_action_expert import sample_flow_time
from pair_preference import load_pairs, build_pair_schedule, run_pair_training
import budgeted_awfm as awfm
from awfm_preservation import preservation_loss, teacher_mode_velocity

MODULES = ('action_expert','tau_emb','action_projector','state_projector',
           'action_decoder','advantage_condition_embedding')


def action_state(policy):
    return {name:{k:v.detach().cpu().clone() for k,v in getattr(policy,name).state_dict().items()}
            for name in MODULES if hasattr(policy,name)}


def load_action_state(policy, state):
    for name in MODULES:
        if name not in state: raise ValueError(f'missing checkpoint module {name}')
        getattr(policy,name).load_state_dict(state[name], strict=True)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_model_dataset(base):
    tokenizer=core.Emu3Tokenizer.from_pretrained(str(core.VLM_MODEL_PATH),model_max_length=1400,padding_side='right',use_fast=False)
    config=core.Emu3Pi0Config.from_pretrained(str(base/'config.json'))
    config.advantage_conditioning=True;config.advantage_condition_count=3
    policy,info=core.Emu3Pi0.from_pretrained(str(base),config=config,
        pretrain_vlm_path=str(core.VLM_MODEL_PATH),attn_implementation='sdpa',
        torch_dtype=core.DTYPE,trust_remote_code=True,output_loading_info=True)
    unexpected=info.get('unexpected_keys',[])
    missing=[k for k in info.get('missing_keys',[]) if 'advantage_condition_embedding' not in k]
    if missing or unexpected or info.get('mismatched_keys'):
        raise ValueError(f'unexpected base loading: missing={missing} unexpected={unexpected}')
    policy=policy.to(core.DEVICE)
    # HF's missing-key initializer may run after __init__; explicitly preserve
    # the original policy for both arms before the first gradient update.
    nn.init.zeros_(policy.advantage_condition_embedding.weight)
    trainable=core.freeze_vlm(policy);policy.vlm.eval()
    heads=core.build_policy_heads(policy)
    data_args=core.DataArguments(data_path=str(core.SCENE_PKL),actions=True,driving=True,
        use_previous_actions=True,actions_format='fast',action_tokenizer_path=os.environ['VLA_ACTION_TOKENIZER'],
        frames=1,action_frames=8,action_dim=3,cur_frame_idx=3,pre_action_frames=3,video_format=None,use_flip=False)
    dataset=core.Emu3DrivingVAVADataset(data_args,tokenizer=tokenizer)
    token_index={s['token']:i for i,s in enumerate(dataset.data)}
    norm=json.loads(core.NORM_STATS_PATH.read_text())['norm_stats']['libero']
    low=np.asarray(norm['q01'],np.float32);high=np.asarray(norm['q99'],np.float32)
    return policy,heads,trainable,dataset,token_index,low,high


def inputs(sample):
    return (sample['input_ids'].unsqueeze(0).to(core.DEVICE),
        sample['attention_mask'].unsqueeze(0).to(core.DEVICE),
        sample['pre_action'].unsqueeze(0).to(core.DEVICE,dtype=core.DTYPE),
        sample['cmd'].unsqueeze(0).to(core.DEVICE,dtype=core.DTYPE))


def flow_velocity(policy,heads,sample,target,indicator,draw_seed):
    ids,mask,pre,cmd=inputs(sample)
    action=torch.as_tensor(target,device=core.DEVICE,dtype=core.DTYPE).reshape(1,8,3)
    # Independent per-example RNG: adding conditional passes never changes
    # later candidates, flow noise or timesteps in the paired ablation.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if action.is_cuda else []):
        torch.manual_seed(draw_seed);torch.cuda.manual_seed_all(draw_seed)
        noise=torch.randn_like(action)
        tau=sample_flow_time(policy.rf,1,torch.device(core.DEVICE)).to(core.DTYPE)
        z=policy.rf.add_noise(action,noise,tau)
        vlm_h=policy.vlm.model.embed_tokens(ids)
        state=torch.cat([pre.reshape(1,-1),cmd],dim=1)
        indicator=torch.tensor([indicator],device=core.DEVICE,dtype=torch.long)
        state_embedding=core.conditioned_state_embedding(heads,state,indicator)
        positions=torch.arange(ids.shape[1],device=core.DEVICE).unsqueeze(0)
        velocity=core.denoise_step_velocity(heads,vlm_h,state_embedding,z,float(tau.item()),ids,mask,positions,ids.shape[1],1)
        return velocity,(noise-action)


def flow_loss(policy,heads,sample,target,indicator,draw_seed):
    velocity,goal=flow_velocity(policy,heads,sample,target,indicator,draw_seed)
    loss=nn.functional.mse_loss(velocity.float(),goal.float())
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite flow loss')
    return loss


def evaluate(policy,heads,dataset,index,tokens,low,high,out,mode,split_name='checkpoint_selection_val',latent_tag='offline_eval_v1'):
    """Fresh deterministic proposals, real NAVSIM scores, explicit errors.

    Evaluation is on the checkpoint-selection validation subset only. No
    projection in this first ablation so any effect is policy learning.
    """
    policy.eval()
    rows=[]
    for i,token in enumerate(tokens):
        ids,mask,pre,cmd=inputs(dataset[index[token]])
        indicator=POSITIVE if mode=='conditioned' else UNCONDITIONAL
        with torch.no_grad():
            generator=torch.Generator(device=core.DEVICE).manual_seed(stable_seed(token,latent_tag))
            initial=torch.randn((1,8,3),generator=generator,device=core.DEVICE,dtype=core.DTYPE)
            action,_,_=core.rollout_and_score(policy,heads,ids,mask,pre,cmd,8,3,10,0.,
                initial_z=initial,advantage_indicator=torch.tensor([indicator],device=core.DEVICE))
        relative=.5*(action[0].float().cpu().numpy()+1)*(high-low)+low
        absolute=core.relative_to_absolute_se2(relative)
        with tempfile.TemporaryDirectory(prefix='offline_score_') as tmp:
            np.save(Path(tmp)/'traj.npy',absolute[None])
            cmdline=[os.environ['NAVSIM_ENV_PYTHON'],str(core.NAVSIM_WORKER_SCRIPT),
                '--metric-cache-path',os.environ['NAVSIM_METRIC_CACHE_PATH'],
                '--scene-token',token,'--trajectories-npy',str(Path(tmp)/'traj.npy'),
                '--output-json',str(Path(tmp)/'scores.json')]
            result=subprocess.run(cmdline,capture_output=True,text=True,timeout=180)
            if result.returncode: raise RuntimeError(f'NAVSIM {token}: {result.stderr[-1800:]}')
            payload=json.loads((Path(tmp)/'scores.json').read_text())
            if 'error' in payload or any('error' in x for x in payload.get('results',[])):
                raise RuntimeError(f'NAVSIM scoring error for {token}: {payload}')
            if len(payload.get('results',[]))!=1: raise ValueError('unexpected scorer result count')
            row={'token':token,**payload['results'][0]}
        rows.append(row)
        np.savez_compressed(out/f'{token}.npz',relative=relative,absolute=absolute)
        if (i+1)%10==0 or i==0: print(f'validation {i+1}/{len(tokens)} score={row["score"]:.6f}',flush=True)
    metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='token'}
    (out/'scores.json').write_text(json.dumps({'split':split_name,'mode':mode,
        'num_scenes':len(rows),'metrics':metrics,'rows':rows},indent=2)+'\n')
    print('VALIDATION '+json.dumps(metrics),flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['unconditional','conditioned','filtered','pair_pref','awfm','baseline'],required=True)
    p.add_argument('--label-mode',choices=['margin','topk_safe'],default='margin')
    p.add_argument('--topk',type=int,default=2)
    p.add_argument('--pairs',type=Path,default=None,help='pairs.jsonl from build_margin_preferences.py (pair_pref mode)')
    p.add_argument('--pref-beta',type=float,default=200.0,help='preference temperature on the mean flow-matching loss scale')
    p.add_argument('--winner-weight',type=float,default=1.0,help='flow-matching weight on the winner (anchor)')
    p.add_argument('--eps-unsafe',type=float,default=0.3,help='awfm: KL(tilt||empirical uniform candidates) budget in nats for scenes with a failing candidate')
    p.add_argument('--eps-safe',type=float,default=0.0,help='awfm: budget for scenes whose candidates all pass the gates')
    p.add_argument('--utility-weight',type=float,default=0.0,help='awfm: weight of the centred-progress term among gate passers (0 = gate only)')
    p.add_argument('--preserve',choices=['none','global','targeted'],default='none',help='awfm: keep the frozen SFT velocity field on noised candidates (targeted = gate passers only, global = all candidates)')
    p.add_argument('--preserve-weight',type=float,default=20.0,help='awfm: fixed soft-regularisation weight on re-noised candidate velocities; not a hard budget')
    p.add_argument('--positive-fraction',type=float,default=None,help='share of POSITIVE candidates per scene; None keeps the original random sampling')
    p.add_argument('--eval-set',choices=['manifest_val','heldout'],default='manifest_val',help='heldout = log-disjoint tokens from a schema-2 manifest')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path,default=core.EMU_HUB)
    p.add_argument('--seed',type=int,default=20)
    p.add_argument('--max-scenes',type=int,default=64,help='0 uses all audited train scenes')
    p.add_argument('--candidates-per-scene',type=int,default=8)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--max-updates',type=int,default=128)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-6)
    p.add_argument('--unconditional-weight',type=float,default=.1)
    p.add_argument('--expert-weight',type=float,default=.25)
    p.add_argument('--eval-scenes',type=int,default=150,help='cap on evaluated scenes')
    args=p.parse_args()
    if args.lr<=0 or args.unconditional_weight<0 or args.expert_weight<0:
        raise ValueError('invalid optimizer/loss weights')
    if args.mode=='pair_pref' and (args.pairs is None or args.pref_beta<=0 or args.winner_weight<0):
        raise ValueError('pair_pref needs --pairs, a positive --pref-beta and a nonnegative --winner-weight')
    if args.mode=='awfm' and (args.eps_unsafe<0 or args.eps_safe<0 or args.utility_weight<0):
        raise ValueError('awfm budgets and utility weight must be nonnegative')
    if args.preserve!='none' and (args.mode!='awfm' or args.preserve_weight<=0):
        raise ValueError('--preserve needs --mode awfm and a positive --preserve-weight')
    if args.topk<1 or (args.positive_fraction is not None and not 0<=args.positive_fraction<=1):
        raise ValueError('invalid topk or positive fraction')
    if args.max_scenes<0 or args.candidates_per_scene<1 or args.accumulation<1 or args.epochs<1 or args.max_updates<0 or args.eval_scenes<1:
        raise ValueError('invalid budget')
    if not torch.cuda.is_available(): raise RuntimeError('submit to a GPU allocation')
    if args.output.exists(): raise FileExistsError(f'refusing to overwrite {args.output}')
    args.output.mkdir(parents=True)
    manifest=json.loads(args.manifest.read_text());records=manifest['train']
    if not records: raise ValueError('empty audited data')
    # Fixed data subset for all seeds/arms, stratified by positive availability.
    rng_data=np.random.default_rng(0)
    pos=[r for r in records if r['positive']];other=[r for r in records if not r['positive']]
    rng_data.shuffle(pos);rng_data.shuffle(other)
    if args.max_scenes:
        n=min(args.max_scenes,len(records));n_pos=min(len(pos),n//2)
        selected=pos[:n_pos]+other[:n-n_pos]
        selected+=(pos[n_pos:]+other[n-n_pos:])[:n-len(selected)]
        records=sorted(selected,key=lambda r:r['token'])
    else: records=sorted(records,key=lambda r:r['token'])
    source=Path(manifest['source_root'])
    split_of={r['token']:r.get('split','train') for r in records}
    @lru_cache(maxsize=4)
    def get_record(token,sha):
        path=source/split_of[token]/f'{token}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha: raise ValueError(f'candidate data changed after audit: {token}')
        action,scores,reference=read_scene(path)
        if args.label_mode=='topk_safe':
            labels=topk_safe_labels(scores,reference,k=args.topk,margin=manifest['margin'])['labels']
        else:
            labels=verified_labels(scores,reference,manifest['margin'])['labels']
        return action,labels
    @lru_cache(maxsize=4)
    def get_scores(token,sha):
        get_record(token,sha)  # hash check
        return read_scene(source/split_of[token]/f'{token}.npz')[1]
    seed_all(args.seed)
    policy,heads,trainable,dataset,index,low,high=build_model_dataset(args.base)
    for r in records:
        if r['token'] not in index: raise ValueError(f'missing scene {r["token"]}')
    if args.eval_set=='heldout':
        if manifest.get('schema_version')!=2 or 'heldout_eval_tokens' not in manifest:
            raise ValueError('--eval-set heldout needs a schema-2 log-disjoint manifest')
        eval_tokens=manifest['heldout_eval_tokens'][:args.eval_scenes]
        eval_name='heldout_log_disjoint'
    else:
        eval_tokens=manifest['validation_tokens'][:args.eval_scenes]
        eval_name='checkpoint_selection_val'
    missing_eval=[t for t in eval_tokens if t not in index]
    if missing_eval: raise ValueError(f'{len(missing_eval)} evaluation scenes missing from inputs, e.g. {missing_eval[0]}')
    if set(eval_tokens)&{r['token'] for r in records}: raise ValueError('training/validation overlap')
    run={**vars(args),'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'github_base':'18190e36a7c83141f04af36196ace0b9e04766e1',
        'source_sha256':{str(path.relative_to(core.REPO_ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(),Path(core.__file__).resolve(),
                         Path(__file__).resolve().with_name('offline_verified_data.py'),
                         Path(__file__).resolve().with_name('budgeted_awfm.py'),
                         Path(__file__).resolve().with_name('awfm_preservation.py'),
                         core.REPO_ROOT/'reference/Emu3/emu3/mllm/modeling_emu3.py')},
        'selected_train_tokens':[r['token'] for r in records], 'eval_tokens':eval_tokens,
        'selection':f'fixed update budget; evaluation set={eval_name}; no checkpoint selection',
        'normalization':{'low':low.tolist(),'high':high.tolist()},'loss':('weighted candidate FM + expert FM + optional fixed-weight re-noised velocity preservation' if args.mode=='awfm' else 'conditional FM + unconditional FM + expert FM')}
    run={k:str(v) if isinstance(v,Path) else v for k,v in run.items()}
    (args.output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    print('RUN '+json.dumps({k:v for k,v in run.items() if not k.endswith('tokens')}),flush=True)
    updates=0;examples=0;positives=0;losses=[]
    preservation_stats={"kind":args.preserve,"cached":0,"terms":0,"first_drift":None,"sum_drift":0.,"max_drift":0.,"initial_identity_checked":False}
    if args.mode!='baseline':
        optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=1e-4)
        if args.mode=='pair_pref':
            sha={r['token']:r['sha256'] for r in records}
            pairs=load_pairs(args.pairs,set(sha))
            schedule=build_pair_schedule(pairs,args.epochs,args.seed)
            (args.output/'schedule.json').write_text(json.dumps([(q['token'],q['winner'],q['loser'],e) for q,e in schedule])+'\n')
            policy.train();policy.vlm.eval()
            def pair_targets(q):
                actions,_=get_record(q['token'],sha[q['token']])
                return normalize_action(actions[q['winner']],low,high),normalize_action(actions[q['loser']],low,high)
            def expert_loss(sample,update_index):
                expert=sample['action'].detach().cpu().numpy()
                if expert.shape!=(8,3): raise ValueError(f'expert shape {expert.shape}')
                return flow_loss(policy,heads,sample,expert,UNCONDITIONAL,stable_seed(args.seed,update_index,'expert'))
            started=time.time()
            def pair_log(n,loss,grad,seen,info):
                if n==1 or n%10==0:
                    print(f'update={n} loss={loss:.6f} pref={info["pref"]:.4f} delta={info["delta"]:.6f} grad={grad:.6f} pairs={seen} elapsed={time.time()-started:.1f} gpu_gb={torch.cuda.max_memory_allocated()/1e9:.2f}',flush=True)
            updates,examples,losses,_=run_pair_training(schedule=schedule,accumulation=args.accumulation,max_updates=args.max_updates,
                beta=args.pref_beta,winner_weight=args.winner_weight,expert_weight=args.expert_weight,
                get_pair_targets=pair_targets,get_sample=lambda t:dataset[index[t]],
                loss_fn=lambda sample,target,seed:flow_loss(policy,heads,sample,target,UNCONDITIONAL,seed),
                seed_fn=lambda epoch,token:stable_seed(args.seed,epoch,token,'pair'),expert_loss_fn=expert_loss,
                optimizer=optimizer,trainable=trainable,clip_fn=lambda ps:nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True),log=pair_log)
        else:
            if args.mode=='awfm':
                plan={}
                def tilt_fn(r):
                    sc=get_scores(r['token'],r['sha256'])
                    p0=float(awfm.gate_pass(sc).mean())
                    if p0==0.: plan[r['token']]={'p':0.,'skipped':True};return None
                    eps=awfm.budget_for_scene(p0,args.eps_unsafe,args.eps_safe)
                    w,tau,kl=awfm.tilt_weights(awfm.scene_reward(sc,utility_weight=args.utility_weight),eps)
                    plan[r['token']]={'p':p0,'q':float((w*awfm.gate_pass(sc)).sum()),'kl':kl,'eps':eps,'skipped':False}
                    return w
                schedule=awfm.build_tilted_schedule(records,tilt_fn,candidates_per_scene=args.candidates_per_scene,
                    epochs=args.epochs,seed=args.seed)
                ref_velocity={}
                if args.preserve!='none':
                    kind='passers' if args.preserve=='targeted' else 'all'
                    limit=len(schedule) if not args.max_updates else min(len(schedule),args.max_updates*args.accumulation)
                    pres_pick={}
                    policy.train();policy.vlm.eval()
                    with torch.no_grad():
                        for k in range(limit):
                            r,c,epoch=schedule[k]
                            j=awfm.preserve_candidate(get_scores(r['token'],r['sha256']),kind,stable_seed(args.seed,epoch,r['token'],c,k,'presj'))
                            if j is None: continue
                            sample=dataset[index[r['token']]]
                            target=normalize_action(get_record(r['token'],r['sha256'])[0][j],low,high)
                            v,_=teacher_mode_velocity(policy,flow_velocity,policy,heads,sample,target,UNCONDITIONAL,stable_seed(args.seed,epoch,r['token'],j,k,'pres'))
                            if not torch.isfinite(v).all(): raise FloatingPointError("nonfinite teacher cache")
                            ref_velocity[k]=(j,v.detach().float().cpu())
                    preservation_stats['cached']=len(ref_velocity)
                    if not ref_velocity: raise RuntimeError('preservation enabled but no probes cached')
                    print(f'PRESERVE kind={kind} cached={len(ref_velocity)}/{limit}',flush=True)
                (args.output/'awfm_plan.json').write_text(json.dumps(plan,indent=1)+'\n')
                used=[v for v in plan.values() if not v['skipped']]
                print('AWFM_PLAN '+json.dumps({'scenes':len(plan),'skipped_no_passer':len(plan)-len(used),
                    'target_pool_pass_gain':float(np.mean([v['q']-v['p'] for v in used])) if used else 0.}),flush=True)
            else:
                schedule=build_schedule(records,lambda r:get_record(r['token'],r['sha256'])[1],
                    mode=args.mode,candidates_per_scene=args.candidates_per_scene,epochs=args.epochs,
                    seed=args.seed,positive_fraction=args.positive_fraction)
            if not schedule: raise ValueError('empty training schedule')
            (args.output/'schedule.json').write_text(json.dumps([(r['token'],c,e) for r,c,e in schedule])+'\n')
            policy.train();policy.vlm.eval()
            optimizer.zero_grad(set_to_none=True)
            started=time.time()
            for start in range(0,len(schedule),args.accumulation):
                if args.max_updates and updates>=args.max_updates: break
                group=schedule[start:start+args.accumulation]
                group_loss=0.
                preserve_terms=[]
                for gi,(r,c,epoch) in enumerate(group):
                    actions,labels=get_record(r['token'],r['sha256'])
                    target=normalize_action(actions[c],low,high)
                    sample=dataset[index[r['token']]]
                    draw_seed=stable_seed(args.seed,epoch,r['token'],c)
                    label=int(labels[c]);indicator=label if args.mode=='conditioned' else UNCONDITIONAL
                    if args.mode=='awfm': label=int(awfm.gate_pass(get_scores(r['token'],r['sha256']))[c])*POSITIVE
                    if args.mode=='filtered' and label!=POSITIVE: raise RuntimeError('filtered schedule contained a non-positive candidate')
                    loss=flow_loss(policy,heads,sample,target,indicator,draw_seed)
                    group_loss+=float(loss.detach())/len(group)
                    (loss/len(group)).backward()
                    if args.unconditional_weight and args.mode!='awfm':
                        uncond=flow_loss(policy,heads,sample,target,UNCONDITIONAL,draw_seed)
                        (args.unconditional_weight*uncond/len(group)).backward()
                    examples+=1;positives+=int(label==POSITIVE)
                    k=start+gi
                    if args.mode=='awfm' and k in ref_velocity:
                        j,ref=ref_velocity[k]
                        v,_=teacher_mode_velocity(policy,flow_velocity,policy,heads,sample,normalize_action(actions[j],low,high),UNCONDITIONAL,stable_seed(args.seed,epoch,r['token'],j,k,'pres'))
                        drift=preservation_loss(v,ref)
                        value=float(drift.detach())
                        if preservation_stats['first_drift'] is None:
                            preservation_stats['first_drift']=value
                            if value>1e-7: raise RuntimeError(f'teacher/student identity failed before update: {value}')
                            preservation_stats['initial_identity_checked']=True
                        preservation_stats['terms']+=1
                        preservation_stats['sum_drift']+=value
                        preservation_stats['max_drift']=max(preservation_stats['max_drift'],value)
                        (args.preserve_weight*drift/len(group)).backward()
                        preserve_terms.append(float(drift.detach()))
                # Same expert anchor and cost for both arms. The dataset returns
                # normalized ground truth action; it is never relabeled positive.
                if args.expert_weight:
                    expert=sample['action'].detach().cpu().numpy()
                    if expert.shape!=(8,3): raise ValueError(f'expert shape {expert.shape}')
                    anchor=flow_loss(policy,heads,sample,expert,UNCONDITIONAL,stable_seed(args.seed,updates,'expert'))
                    (args.expert_weight*anchor).backward()
                grad=nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                if float(grad)==0: raise RuntimeError('zero policy gradient')
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;losses.append(group_loss)
                if updates==1 or updates%10==0:
                    print(f'update={updates} drift={np.mean(preserve_terms) if preserve_terms else 0.:.3e} fm={group_loss:.6f} grad={float(grad):.6f} examples={examples} positives={positives} elapsed={time.time()-started:.1f} gpu_gb={torch.cuda.max_memory_allocated()/1e9:.2f}',flush=True)
        if not updates: raise RuntimeError('no training updates')
        state=action_state(policy)
        checkpoint=args.output/'action_policy.pt'
        torch.save({'action_state_dict':state,'metadata':run},checkpoint)
        # Verify serialized weights and strict module loading before scoring.
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for name in state:
            for k,v in state[name].items():
                if not torch.equal(v,loaded['action_state_dict'][name][k]): raise RuntimeError('checkpoint roundtrip mismatch')
        load_action_state(policy,loaded['action_state_dict'])
        del state,loaded,optimizer
        torch.cuda.empty_cache()
    scores_dir=args.output/'validation';scores_dir.mkdir()
    metrics=evaluate(policy,heads,dataset,index,eval_tokens,low,high,scores_dir,args.mode,eval_name)
    (args.output/'summary.json').write_text(json.dumps({'updates':updates,'examples':examples,
        'positive_examples':positives,'train_fm_mean':float(np.mean(losses)) if losses else None,
        'validation':metrics,'eval_set':eval_name,'complete':True,'preservation':preservation_stats},indent=2)+'\n')
    print(f'COMPLETE {args.output}',flush=True)

if __name__=='__main__': main()
