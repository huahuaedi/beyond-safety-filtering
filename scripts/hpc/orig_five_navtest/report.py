"""Paired report for the original direct-RL checkpoints on the frozen navtest cohort."""
import json
from pathlib import Path
import numpy as np
from paired_results import METRICS, read_scores, compare
import arms as A

PDMS, NC, DAC, TTC, PROG, COMF = range(6)
assert METRICS[PDMS] == 'score' and METRICS[PROG] == 'ego_progress'


def verdict(lower, upper):
    if lower > 0:
        return 'EFFECTIVE_ON_NAVTEST'
    if upper < 0:
        return 'GAIN_DOES_NOT_EXTEND'
    return 'INCONCLUSIVE'


def try_load(metrics, run_name, arm, tokens):
    """Descriptive arms may be missing; primary arms must exist, so their errors propagate."""
    if arm.startswith('desc_') or arm.startswith('sft_evalseed'):
        try:
            return read_scores(metrics / f'{run_name}_{arm}', tokens)
        except Exception:
            return None
    return read_scores(metrics / f'{run_name}_{arm}', tokens)


def build_report(run):
    run = Path(run)
    tokens = sorted(json.loads((run / 'test_tokens.json').read_text()))
    mapping = json.loads((run / 'token_log_map.json').read_text())
    metrics = run / 'metrics'
    sft = read_scores(metrics / f'{run.name}_sft', tokens)
    sel_names = A.selected_names()
    sel = np.stack([read_scores(metrics / f'{run.name}_{n}', tokens) for n in sel_names])
    base = np.repeat(sft[None], len(sel_names), axis=0)
    main = compare(sel, base, tokens, mapping)
    lo, hi = main['log_bootstrap_95ci'][0][PDMS], main['log_bootstrap_95ci'][1][PDMS]
    result = {
        'protocol': json.loads((run / 'protocol.json').read_text()),
        'metrics': list(METRICS), 'scenes': len(tokens),
        'mean': {'sft': sft.mean(0).tolist(), 'selected_five_avg': sel.mean((0, 1)).tolist()},
        'selected_minus_sft': main,
        'pdms_delta_x100': main['mean_delta'][PDMS] * 100,
        'pdms_log_ci95_x100': [lo * 100, hi * 100],
        'verdict': verdict(lo, hi),
        'per_seed_pdms_delta_x100': [float((sel[i, :, PDMS] - sft[:, PDMS]).mean() * 100) for i in range(len(sel_names))],
    }
    noise = {}
    for name in ('sft_evalseed1', 'sft_evalseed2'):
        other = try_load(metrics, run.name, name, tokens)
        if other is not None:
            noise[name] = {'mean_delta_x100': [float(x * 100) for x in (other.mean(0) - sft.mean(0))],
                           'scenes_with_different_pdms': int((np.abs(other[:, PDMS] - sft[:, PDMS]) > 1e-9).sum())}
    result['sft_reevaluation_noise'] = noise
    steps = {}
    for step in A.STEPS:
        arrays = []
        for seed in A.SELECTED:
            arr = try_load(metrics, run.name, A.step_arm(seed, step), tokens)
            if arr is None:
                arrays = None
                break
            arrays.append(arr)
        if arrays is not None:
            c = compare(np.stack(arrays), np.repeat(sft[None], len(arrays), axis=0), tokens, mapping)
            steps[str(step)] = {'pdms_delta_x100': c['mean_delta'][PDMS] * 100,
                                'ttc_delta_x100': c['mean_delta'][TTC] * 100,
                                'progress_delta_x100': c['mean_delta'][PROG] * 100,
                                'pdms_log_ci95_x100': [c['log_bootstrap_95ci'][0][PDMS] * 100, c['log_bootstrap_95ci'][1][PDMS] * 100]}
    result['descriptive_by_step'] = steps
    result['ci_scope'] = ('Scene-weighted whole-log bootstrap after averaging the five paired seeds; not full training uncertainty. '
                          'The cohort has been inspected before. Other steps are descriptive only.')
    (run / 'report.json').write_text(json.dumps(result, indent=2))
    lines = ['# Original direct-RL checkpoints on the frozen navtest cohort', '',
             '数值乘100；检查点为原版按验证集选定的五个，不重训、不重选。', '',
             '| Arm | PDMS | NC | DAC | TTC | Progress | Comfort |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, key in (('sft', 'sft'), ('five selected (avg)', 'selected_five_avg')):
        lines.append('| ' + name + ' | ' + ' | '.join(f'{x * 100:.3f}' for x in result['mean'][key]) + ' |')
    lines += ['', f"PDMS 差 {result['pdms_delta_x100']:+.3f}，log 级 95% 区间 [{lo * 100:+.3f}, {hi * 100:+.3f}]，判定：**{result['verdict']}**。",
              '逐种子 PDMS 差：' + ', '.join(f'{x:+.3f}' for x in result['per_seed_pdms_delta_x100']),
              'Progress/TTC/NC/DAC 的差和新增失败见 report.json。该子集已被查看过，不是盲测；其他步数只作描述。']
    (run / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return result


if __name__ == '__main__':
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    build_report(cfg['run'])
