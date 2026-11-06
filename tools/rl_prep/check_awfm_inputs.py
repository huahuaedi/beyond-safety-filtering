"""Read-only preflight before spending GPU time; verifies the reused SFT baseline."""
import argparse
import hashlib
import json
from pathlib import Path
from awfm_statistics import log_labels


def check_inputs(root, cache, n=600):
    root = Path(root)
    manifest_path = root/'manifest_v2.json'
    manifest = json.loads(manifest_path.read_text())
    tokens = manifest['heldout_eval_tokens'][:n]
    if manifest.get('schema_version') != 2 or not tokens or len(set(tokens)) != len(tokens):
        raise ValueError('invalid log-disjoint evaluation manifest')
    train = {r['token'] for r in manifest['train']}
    if train & set(tokens): raise ValueError('train/evaluation token overlap')
    all_tokens=sorted(train | set(tokens))
    all_logs=log_labels(all_tokens, cache_dir=cache)
    mapping=dict(zip(all_tokens,all_logs))
    logs=[mapping[t] for t in tokens]
    if {mapping[t] for t in train} & set(logs):
        raise ValueError('train/evaluation driving-log overlap')
    baselines = sorted(root.glob('scaled_v1_baseline_seed*'))
    if not baselines: raise ValueError('missing existing scaled_v1_baseline_seed*; do not train before fixing baseline')
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    base_paths = set()
    for b in baselines:
        summary = json.loads((b/'summary.json').read_text())
        run = json.loads((b/'run.json').read_text())
        scores = json.loads((b/'validation/scores.json').read_text())
        observed = [r['token'] for r in scores['rows']]
        if summary.get('complete') is not True or run.get('mode') != 'baseline':
            raise ValueError(f'incomplete/non-SFT baseline: {b}')
        if set(observed) != set(tokens) or len(observed) != len(tokens):
            raise ValueError(f'baseline scenes differ from requested {len(tokens)} scenes: {b}')
        if run.get('manifest_sha256') != digest or set(run['eval_tokens']) != set(tokens):
            raise ValueError(f'baseline manifest/protocol mismatch: {b}')
        base_paths.add(run['base'])
    if len(base_paths) != 1 or not Path(next(iter(base_paths))).is_dir():
        raise ValueError('missing/inconsistent baseline model path')
    return {'status':'PASS','scenes':len(tokens),'logs':len(set(logs)),
            'baseline_runs':[str(b) for b in baselines], 'base':next(iter(base_paths))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--metric-cache-dir', type=Path, required=True)
    ap.add_argument('--eval-scenes', type=int, default=600)
    ap.add_argument('--base-only', action='store_true')
    a=ap.parse_args()
    result=check_inputs(a.root,a.metric_cache_dir,a.eval_scenes)
    print(result['base'] if a.base_only else json.dumps(result))
