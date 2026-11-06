"""CPU-only audit and immutable training manifest; no model loading."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from offline_verified_data import read_scene, verified_labels, split_records, POSITIVE


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scene-pkl', type=Path, required=True)
    p.add_argument('--eval-tokens', type=Path, required=True)
    p.add_argument('--val-tokens', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--margin', type=float, default=0.01)
    args=p.parse_args()
    forbidden=set(json.loads(args.eval_tokens.read_text()))
    val_tokens=set(json.loads(args.val_tokens.read_text()))
    if forbidden & val_tokens: raise ValueError('validation/final-evaluation overlap')
    records,excluded=split_records(args.root, forbidden | val_tokens)
    with args.scene_pkl.open('rb') as f: scenes=pickle.load(f)
    input_tokens={s['token'] for s in scenes}
    if len(input_tokens)!=len(scenes): raise ValueError('duplicate input scene tokens')
    if not val_tokens <= input_tokens: raise ValueError('validation token missing from inputs')
    report={'schema_version':1,'margin':args.margin,'source_root':str(args.root.resolve()),
            'excluded_training_tokens':excluded,'splits':{},'train':[],
            'validation_tokens':sorted(val_tokens),'final_eval_count':len(forbidden)}
    for split, paths in records.items():
        stats={'scenes':len(paths),'candidates':0,'positive':0,'safe':0,'scenes_with_positive':0}
        for i,path in enumerate(paths):
            if path.stem not in input_tokens: raise ValueError(f'missing input: {path.stem}')
            action,scores,ref=read_scene(path)
            m=verified_labels(scores,ref,args.margin)
            n=int((m['labels']==POSITIVE).sum())
            stats['candidates']+=len(action);stats['positive']+=n;stats['safe']+=int(m['safe'].sum());stats['scenes_with_positive']+=int(n>0)
            if split=='train':
                report['train'].append({'token':path.stem,'candidates':len(action),'positive':n,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            if (i+1)%1000==0: print(split,i+1,stats,flush=True)
        report['splits'][split]=stats
    if report['splits']['train']['positive']==0: raise ValueError('no positive training candidates')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='train'},indent=2),flush=True)

if __name__=='__main__': main()
