"""Summarize paired exploratory diagnostics; intervals condition on fitted models."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_world_action_mining import load_oof


def main():
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args()
    conditions=['action','layer1','fusion','linear_layer1','shuffled_fusion']
    rng=np.random.default_rng(23); output={}; ref_names=None
    for size in (32,64,128):
        counts={}; stats={}
        for cond in conditions:
            mode='fusion' if cond=='shuffled_fusion' else 'layer1' if cond=='linear_layer1' else cond
            by_seed=[]
            for seed in (0,1,2):
                names,probs,labels=load_oof(a.root/cond/f'n_{size}',mode,5,seed=seed)
                order=np.argsort(names); names=np.asarray(names)[order].tolist(); probs=probs[order]; labels=labels[order]
                if ref_names is None: ref_names=names; ref_labels=labels
                if names!=ref_names or not np.array_equal(labels,ref_labels): raise ValueError('unaligned comparisons')
                head=[]
                for h in range(4):
                    chosen=np.argsort(-probs[:,:,h],axis=1,kind='stable')[:,:8]
                    head.append(np.take_along_axis(labels[:,:,h],chosen,axis=1).sum(axis=1))
                by_seed.append(np.stack(head,axis=1))
            counts[cond]=np.stack(by_seed)
            stats[cond]={'selected_positives_by_seed_and_head':counts[cond].sum(axis=1).tolist(),'total_positives_by_head':labels.sum(axis=(0,1)).tolist()}
        indices=rng.integers(0,len(ref_names),(10000,len(ref_names)))
        for cond in conditions:
            delta=(counts[cond]-counts['action']).mean(axis=0)
            boot=delta[indices].sum(axis=1)
            stats[cond]['mean_selected_delta_vs_action']=delta.sum(axis=0).tolist()
            stats[cond]['paired_scene_bootstrap_95_ci']=np.quantile(boot,[.025,.975],axis=0).tolist()
        output[str(size)]=stats
    result={'label_order':['unsafe','ttc_failure','dac_failure','nc_failure'],'interpretation':'Exploratory reused scenes; intervals condition on fitted models and average 3 seeds, not independent-log uncertainty. Each risk head selects its own top 8; this is not routed manifest performance.','results':output}
    (a.root/'diagnostic_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
