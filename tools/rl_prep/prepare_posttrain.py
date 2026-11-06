"""Prepare immutable scene splits and inference inputs for policy post-training."""
import argparse
import json
import pickle
from pathlib import Path
import yaml


def main():
    p=argparse.ArgumentParser(); p.add_argument('--assets',type=Path,required=True); p.add_argument('--run',type=Path,required=True); a=p.parse_args()
    anchor=a.assets/'data/navsim/processed_data/anchor'
    train=set(json.loads((anchor/'trainval_rl_scene_tokens_cached.json').read_text()))
    val=set(json.loads((anchor/'checkpoint_selection_val_tokens_cached.json').read_text()))
    test=set(json.loads((anchor/'trainval_rl_eval_tokens_cached.json').read_text()))
    if val & test: raise ValueError('validation/test overlap')
    train-=val|test
    if not train or not val or not test: raise ValueError('empty split')
    with (a.assets/'pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl').open('rb') as f: scenes=pickle.load(f)
    tokens=[s['token'] for s in scenes]
    if len(tokens)!=len(set(tokens)): raise ValueError('duplicate metadata token')
    known=set(tokens)
    for name, group in [('train',train),('val',val),('test',test)]:
        if group-known: raise ValueError(f'{name}: {len(group-known)} scenes lack metadata')
        (a.run/f'{name}_tokens.json').write_text(json.dumps(sorted(group)))
        if name!='train':
            subset=[s for s in scenes if s['token'] in group]
            with (a.run/f'{name}.pkl').open('wb') as f: pickle.dump(subset,f)
            (a.run/f'{name}.yaml').write_text(yaml.safe_dump({'tokens':[s['token'] for s in subset]}))
    protocol=dict(train_scenes=len(train),validation_scenes=len(val),final_eval_scenes=len(test),training_steps=200,group_size=16,seed=0,learning_rate=1e-5,reward='real NAVSIM sub-scores',advantage='reference_anchored hard_sign_bounded_v1',kl='normalized mean-log-ratio surrogate; not exact policy KL',trainable='Action Expert and action-side modules; VLM frozen',wm_manifest=False,checkpoints=[50,100,150,200],selection='highest validation PDMS among checkpoints not below baseline mean NC/DAC/TTC; baseline fallback',note='final evaluation is disjoint from this run training, but this dataset has prior research use')
    (a.run/'protocol.json').write_text(json.dumps(protocol,indent=2)); print(json.dumps(protocol,indent=2))


if __name__=='__main__': main()
