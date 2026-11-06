"""Frozen-model evaluation on full/test excluding every pilot scene token.

No test-label model selection. Report mean performance of 15 individual models
(3 seeds x 5 training folds), not an ensemble chosen after test inspection.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    import train_world_action_risk_ranker as r
    class LinearLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net=torch.nn.Sequential(torch.nn.LayerNorm(4096,elementwise_affine=False),torch.nn.Linear(4096,4))
        def forward(self,action,layer1,layer2): return self.net(layer1)
    old={p.stem for p in (a.source/'world_action_v1').glob('*/*.npz')}
    files=[p for p in sorted((a.source/'world_action_full_v1/test').glob('*.npz')) if p.stem not in old]
    if not files: raise ValueError('No new test scenes')
    if len({p.stem for p in files})!=len(files): raise ValueError('Duplicate scene tokens')
    r.LATENT_ROOT=a.source/'world_action_latents_full_v1'
    models=[]
    for cond,mode in [('action','action'),('linear_layer1','layer1')]:
        for seed in range(3):
            for fold in range(5):
                root=a.checkpoints/cond/'n_128'/mode/f'seed_{seed}'/f'fold_{fold}'
                metrics=json.loads((root/'metrics.json').read_text())
                seen=set(metrics['training_scenes'])|set(metrics['selection_scenes'])|set(metrics['outer_scenes'])
                if seen & {p.stem for p in files}: raise ValueError('Model/test overlap')
                model=(r.RiskRanker('action') if cond=='action' else LinearLayer()).to(r.DEVICE)
                model.load_state_dict(torch.load(root/'ranker.pt',map_location=r.DEVICE,weights_only=False)['model'])
                model.eval(); models.append((cond,seed,fold,model))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'protocol.json').write_text(json.dumps(dict(scenes=[p.stem for p in files], excluded_pilot_scenes=sorted(old),checkpoint_root=str(a.checkpoints), budget=8, models='all 3 seeds x 5 folds, fixed n128 action and linear_layer1; no tuning', caveat='new to this diagnostic cycle by scene token; log independence and use in older research are not established'),indent=2))
    targets=[]; predictions=[[] for _ in models]
    for start in range(0,len(files),8):
        batch=r.load_paths(files[start:start+8]); targets.append(batch[3])
        for i,(_,_,_,model) in enumerate(models): predictions[i].append(r.predict(model,batch[:4]))
        print(f'evaluated {min(start+8,len(files))}/{len(files)} scenes',flush=True)
    labels=np.concatenate(targets); counts={}; output={}
    for cond in ('action','linear_layer1'):
        found=[]
        for i,(name,seed,fold,_) in enumerate(models):
            if name!=cond: continue
            probs=np.concatenate(predictions[i])
            heads=[]
            for h in range(4):
                top=np.argsort(-probs[:,:,h],axis=1,kind='stable')[:,:8]
                heads.append(np.take_along_axis(labels[:,:,h],top,axis=1).sum(axis=1))
            found.append(np.stack(heads,axis=1))
            np.savez_compressed(a.output/f'{cond}_seed{seed}_fold{fold}.npz',scene_names=np.asarray([p.stem for p in files]),probabilities=probs,labels=labels)
        counts[cond]=np.stack(found)
        output[cond]={'selected_by_model_and_head':counts[cond].sum(axis=1).tolist(),'mean_selected_by_head':counts[cond].mean(axis=0).sum(axis=0).tolist()}
    random=labels.sum(axis=1)*8/64
    counts['random_expectation']=random[None,:,:]
    output['random_expectation']={'mean_selected_by_head':random.sum(axis=0).tolist()}
    rng=np.random.default_rng(23); indices=rng.integers(0,len(files),(5000,len(files)))
    comparisons={}
    for base in ('action','random_expectation'):
        delta=counts['linear_layer1'].mean(axis=0)-counts[base].mean(axis=0)
        comparisons[base]={'difference_by_head':delta.sum(axis=0).tolist(),'paired_scene_bootstrap_95_ci':np.quantile(delta[indices].sum(axis=1),[.025,.975],axis=0).tolist()}
    report=dict(scenes=len(files),total_candidates=int(labels.shape[0]*labels.shape[1]),verification_budget_per_model=len(files)*8,label_order=list(r.LABEL_NAMES),total_positives=labels.sum(axis=(0,1)).tolist(),results=output,linear_minus=comparisons,caveat='Exploratory fixed-model comparison; scene-bootstrap intervals do not include model-fitting or driving-log uncertainty. Each head selects its own top 8; not a routed selector.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
