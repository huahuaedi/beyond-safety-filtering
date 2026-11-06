"""Read-only reuse of the inspected 1000-scene cohort; never select new tokens.

Differences from the IL package: the evaluated checkpoints are the original direct-RL runs
(no training_verified.json exists for them), and the training/evaluation log-isolation check
uses the full RL training pool (every cached trainval scene), because that is what those
runs sampled from.
"""
import hashlib, json, os, pickle, shutil, sys
from pathlib import Path
import yaml
from log_identity import NumpyCoreCompatUnpickler
import arms as arm_list


def validate_cohort(tokens, mapping, records, yaml_tokens):
    if len(tokens) != 1000 or len(set(tokens)) != 1000:
        raise ValueError('Expected 1000 unique frozen tokens')
    if set(tokens) - set(mapping) or len({mapping[t] for t in tokens}) != 136:
        raise ValueError('Frozen log mapping differs')
    if [r['token'] for r in records] != tokens or yaml_tokens != tokens:
        raise ValueError('Metadata/YAML order mismatch')


def log_overlap(mapping, tokens, pool_logs):
    cohort_logs = {mapping[t].split('log_name:')[-1] for t in tokens}
    return sorted(cohort_logs & set(pool_logs))


def main():
    p = Path(__file__).resolve().parent
    c = json.loads((p / 'config.json').read_text())
    run, cohort, assets, repo = Path(c['run']), Path(c['cohort_run']), Path(c['assets']), Path(c['repo'])
    if run.exists():
        raise ValueError('Refuse overwrite')
    for name, digest in json.loads((p / 'frozen_cohort_hashes.json').read_text()).items():
        if hashlib.sha256((cohort / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen cohort hash differs: ' + name)
    tokens = json.loads((cohort / 'test_tokens.json').read_text())
    mapping = json.loads((cohort / 'token_log_map.json').read_text())
    with (cohort / 'test.pkl').open('rb') as f:
        records = NumpyCoreCompatUnpickler(f).load()
    validate_cohort(tokens, mapping, records, yaml.safe_load((cohort / 'test.yaml').read_text())['tokens'])

    model_root = assets / 'pretrained_models/DriveVLA-W0'
    checkpoints = {}
    for name, rel, evalseed in arm_list.arms():
        ck = model_root / rel
        if not (ck / 'config.json').is_file() or not list(ck.glob('*.safetensors')):
            raise FileNotFoundError(ck)
        index = ck / 'model.safetensors.index.json'
        if index.exists():
            for shard in set(json.loads(index.read_text())['weight_map'].values()):
                if not (ck / shard).is_file():
                    raise FileNotFoundError(ck / shard)
        checkpoints[name] = {'path': str(ck), 'eval_seed': evalseed,
                             'config_sha256': hashlib.sha256((ck / 'config.json').read_bytes()).hexdigest(),
                             'shards': {f.name: f.stat().st_size for f in ck.glob('*.safetensors')}}

    pool_logs = os.listdir(assets / 'navsim_data/exp/metric_cache_trainval_rl')
    overlap = log_overlap(mapping, tokens, pool_logs)
    if overlap:
        raise ValueError(f'Training pool / evaluation log overlap: {len(overlap)} logs, e.g. {overlap[0]}')

    sys.path.insert(0, str(repo / 'tools/rl_prep'))
    from navsim_score_worker import MetricCacheLoader
    if not set(tokens) <= set(MetricCacheLoader(cohort / 'cache').metric_cache_paths):
        raise ValueError('Frozen cache incomplete')
    if not (cohort / 'navsim_data/navsim_logs/test').is_dir():
        raise FileNotFoundError('Frozen raw-log root')
    for r in records:
        for field in ('image', 'pre_1s_image'):
            if not Path(r[field][3]).is_file():
                raise FileNotFoundError(r[field][3])

    run.mkdir()
    (run / 'metrics').mkdir()
    for name in ['test.pkl', 'test.yaml', 'test_tokens.json', 'token_log_map.json']:
        shutil.copy2(cohort / name, run / name)
    (run / 'cache').symlink_to(cohort / 'cache', target_is_directory=True)
    (run / 'navsim_data').symlink_to(cohort / 'navsim_data', target_is_directory=True)
    smoke = tokens[:8]
    with (run / 'smoke.pkl').open('wb') as f:
        pickle.dump(records[:8], f, protocol=4)
    (run / 'smoke_tokens.json').write_text(json.dumps(smoke))
    (run / 'smoke.yaml').write_text(yaml.safe_dump({'tokens': smoke}))
    protocol = {
        **c,
        'scope': 'Frozen evaluation of the five originally selected direct-RL checkpoints on a previously inspected navtest subset; '
                 'not fresh confirmation. SFT training provenance unverified. No training, no reselection.',
        'scenes': 1000, 'logs': 136, 'training_updates': 0, 'eval_seed': 0,
        'primary_arms': arm_list.selected_names(),
        'descriptive_arms': [a[0] for a in arm_list.arms() if a[0].startswith('desc_') or a[0].startswith('sft_evalseed')],
        'cohort_sha256': hashlib.sha256((run / 'test_tokens.json').read_bytes()).hexdigest(),
        'training_pool_log_overlap': 0,
        'decision_rule': ('Primary quantity: five-seed average PDMS difference vs SFT with a whole-log paired bootstrap 95% interval. '
                          'Lower bound > 0: original recipe also helps on this cohort. Upper bound < 0: its 286-scene gain does not extend. '
                          'Otherwise inconclusive. Other steps and SFT re-evaluations are descriptive only.'),
        'checkpoints': checkpoints,
    }
    (run / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (run / 'ready.json').write_text(json.dumps({'status': 'READY', 'scenes': 1000, 'logs': 136}))
    print(json.dumps(protocol, indent=2))


if __name__ == '__main__':
    main()
