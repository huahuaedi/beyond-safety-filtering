import csv, json, tempfile, unittest
from pathlib import Path
import numpy as np
import arms as A
import report
from paired_results import METRICS
from prepare import log_overlap


def write_scores(path, tokens, offset=0.0, ttc_fail=()):
    path.mkdir(parents=True)
    with open(path / 'x.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['', 'token', 'valid'] + list(METRICS))
        for i, t in enumerate(tokens):
            ttc = 0.0 if i in ttc_fail else 1.0
            ep = min(1.0, 0.7 + offset)
            w.writerow([i, t, 'True', (5 * ep + 5 * ttc + 2) / 12, 1.0, 1.0, ttc, ep, 1.0])
        w.writerow(['', 'average', '', 0, 0, 0, 0, 0, 0])


class PackageTest(unittest.TestCase):
    def test_arm_table(self):
        a = A.arms()
        self.assertEqual(len(a), 23)
        self.assertEqual(a[0][0], 'sft')
        sel = A.selected_names()
        self.assertEqual(sel, ['sel_seed20_step000025', 'sel_seed21_step000025', 'sel_seed22_step000050',
                               'sel_seed23_step000010', 'sel_seed24_step000040'])
        self.assertEqual([x[0] for x in a[1:6]], sel)                  # primary arms come first in the array
        self.assertEqual({x[2] for x in a if x[0].startswith('sft_evalseed')}, {1, 2})
        self.assertEqual(sum(x[0].startswith('desc_') for x in a), 15)
        self.assertTrue(all(x[2] == 0 for x in a if x[0].startswith(('sel_', 'desc_'))))

    def test_every_saved_step_is_covered_exactly_once(self):
        got = {(s, st) for s in A.SELECTED for st in A.STEPS}
        used = set()
        for name, rel, _ in A.arms():
            if name.startswith(('sel_', 'desc_')):
                seed = int(name.split('seed')[1].split('_')[0]); step = int(name.split('step')[1])
                self.assertEqual(rel, f'direct_rl_seed{seed}/step_{step:06d}')
                self.assertNotIn((seed, step), used); used.add((seed, step))
        self.assertEqual(used, got)

    def test_verdicts(self):
        self.assertEqual(report.verdict(0.01, 1.0), 'EFFECTIVE_ON_NAVTEST')
        self.assertEqual(report.verdict(-1.0, -0.01), 'GAIN_DOES_NOT_EXTEND')
        self.assertEqual(report.verdict(-0.5, 0.5), 'INCONCLUSIVE')

    def test_log_overlap(self):
        m = {'a': 'log_name:L1', 'b': 'log_name:L2'}
        self.assertEqual(log_overlap(m, ['a', 'b'], ['L3', 'L4']), [])
        self.assertEqual(log_overlap(m, ['a', 'b'], ['L2', 'L4']), ['L2'])

    def test_report_on_synthetic_run(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / 'orig_five_navtest_x'; (run / 'metrics').mkdir(parents=True)
            tokens = [f't{i}' for i in range(60)]
            (run / 'test_tokens.json').write_text(json.dumps(tokens))
            (run / 'token_log_map.json').write_text(json.dumps({x: f'log_name:L{i % 12}' for i, x in enumerate(tokens)}))
            (run / 'protocol.json').write_text('{}')
            m = run / 'metrics'
            write_scores(m / f'{run.name}_sft', tokens)
            for n in A.selected_names():
                write_scores(m / f'{run.name}_{n}', tokens, offset=0.05, ttc_fail=(1,))
            write_scores(m / f'{run.name}_sft_evalseed1', tokens)
            write_scores(m / f'{run.name}_sft_evalseed2', tokens)
            for seed in A.SELECTED:                                  # only some descriptive steps exist
                for st in (10, 25):
                    if A.SELECTED[seed] != st:
                        write_scores(m / f'{run.name}_{A.step_arm(seed, st)}', tokens, offset=0.02)
            r = report.build_report(run)
            self.assertTrue((run / 'report.json').is_file() and (run / 'report.md').is_file())
        self.assertEqual(r['scenes'], 60)
        self.assertGreater(r['mean']['selected_five_avg'][4], r['mean']['sft'][4])   # progress up
        self.assertLess(r['selected_minus_sft']['mean_delta'][3], 0)                 # TTC down
        self.assertEqual(len(r['per_seed_pdms_delta_x100']), 5)
        self.assertEqual(r['sft_reevaluation_noise']['sft_evalseed1']['scenes_with_different_pdms'], 0)
        self.assertNotIn('40', r['descriptive_by_step'])                              # incomplete steps skipped


if __name__ == '__main__':
    unittest.main()
