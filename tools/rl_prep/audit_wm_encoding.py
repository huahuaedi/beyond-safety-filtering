"""Offline action clipping/token alias audit, without loading the large WM."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    cfg=json.loads((a.repo/'configs/normalizer_navsim_trainval/norm_stats.json').read_text())['norm_stats']['libero']
    lo,hi=np.array(cfg['q01'],dtype=np.float32),np.array(cfg['q99'],dtype=np.float32)
    if (hi<=lo).any(): raise ValueError('invalid normalization bounds')
    from transformers import AutoProcessor
    import torch
    tokenizer=AutoProcessor.from_pretrained(str(a.repo/'pretrained_models/fast'), trust_remote_code=True, local_files_only=True)
    meta_path=a.repo/'pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl'
    with meta_path.open('rb') as f: metadata=pickle.load(f)
    by_token={str(s['token']):s for s in metadata}
    rows=[]
    for split in ('train','val'):
        for path in sorted((a.repo/'navsim_data/exp/world_action_v1'/split).glob('*.npz')):
            with np.load(path,allow_pickle=False) as d:
                raw=d['relative_trajectories'].astype(np.float32)
                unsafe=~d['reference_feasible'].astype(bool)
                nc=d['score__no_at_fault_collisions']<1-1e-8
            norm=2*(raw-lo)/(hi-lo)-1
            clipped=np.clip(norm,-1,1)
            changed=(np.abs(norm)>1).any(axis=(1,2))
            restored=(clipped+1)*.5*(hi-lo)+lo
            groups={}
            for i,action in enumerate(clipped):
                tokens=tokenizer(torch.tensor(action,dtype=torch.float32))[0]
                key=tuple(int(v) for v in tokens)
                groups.setdefault(key,[]).append(i)
            alias=[g for g in groups.values() if len(g)>1]
            conflict=[g for g in alias if len(set(unsafe[g].tolist()))>1]
            with np.load(a.repo/'navsim_data/exp/world_action_latents_v1'/split/path.name,allow_pickle=False) as latent_file:
                latent=latent_file['eoa_latents'].astype(np.float32)
                if not np.isfinite(latent).all(): raise ValueError(f'nonfinite latent: {path}')
                variation=np.mean((latent-latent.mean(axis=0,keepdims=True))**2,axis=(0,2))
                energy=np.mean(latent**2,axis=(0,2))
                ratio=(variation/np.maximum(energy,1e-12)).tolist()
            scene=by_token.get(path.stem,{})
            log=next((str(scene[k]) for k in ('log_name','log_id','log_token') if k in scene),None)
            rows.append(dict(within_scene_latent_variance_fraction_by_layer=ratio, scene=path.stem, split=split, log=log, metadata_keys=sorted(scene.keys()), candidates=len(raw), clipped_candidates=int(changed.sum()), clipped_elements=int((np.abs(norm)>1).sum()), elements=norm.size, unsafe=int(unsafe.sum()), collision=int(nc.sum()), clipped_unsafe=int((changed&unsafe).sum()), unclipped_unsafe=int((~changed&unsafe).sum()), max_abs_change_per_coordinate=np.abs(restored-raw).max(axis=(0,1)).tolist(), unique_token_sequences=len(groups), aliased_candidates=sum(map(len,alias)), conflicting_label_candidates=sum(map(len,conflict))))
    known=[r for r in rows if r['log'] is not None]
    overlap=sorted({r['log'] for r in known if r['split']=='train'} & {r['log'] for r in known if r['split']=='val'})
    summary={k:sum(r[k] for r in rows) for k in ('candidates','clipped_candidates','clipped_elements','elements','unsafe','collision','clipped_unsafe','unclipped_unsafe','aliased_candidates','conflicting_label_candidates')}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(summary=summary, scenes=rows, known_log_scenes=len(known), train_val_log_overlap=overlap, log_isolation_status='unknown if log metadata absent; CV is scene-isolated only'),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
