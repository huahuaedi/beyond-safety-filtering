import os
from pathlib import Path
import subprocess
import sys
import tempfile
import json
import unittest
import importlib.util
import numpy as np

repo = Path(__file__).resolve().parents[2]
@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for training smoke test")
class RankerPipelineTest(unittest.TestCase):
    def test_nested_oof_to_verified_manifests(self):
        with tempfile.TemporaryDirectory(prefix='drivevla-smoke-') as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(7)
            for i in range(12):
                split = 'train' if i < 8 else 'val'
                data = root/'data'/split
                latent = root/'latents'/split
                data.mkdir(parents=True, exist_ok=True)
                latent.mkdir(parents=True, exist_ok=True)
                failures = rng.random((64, 3)) < .3
                violation = failures.sum(axis=1)
                np.savez(data/f'scene_{i}.npz',
                         relative_trajectories=rng.normal(size=(64, 8, 3)).astype('float32'),
                         reference_feasible=violation == 0, reference_violation=violation,
                         score__time_to_collision_within_bound=1-failures[:, 0].astype(int),
                         score__drivable_area_compliance=1-failures[:, 1].astype(int),
                         score__no_at_fault_collisions=1-failures[:, 2].astype(int))
                np.savez(latent/f'scene_{i}.npz', eoa_latents=rng.normal(size=(64,4,4096)).astype('float32'))
            env = dict(os.environ, WM_ACTION_DATA_ROOT=str(root/'data'), WM_LATENT_ROOT=str(root/'latents'),
                       WM_RISK_RANKER_OUTPUT=str(root/'oof'), WM_RISK_CV_FOLDS='2', WM_RISK_EPOCHS='2',
                       WM_RISK_SEED='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            for mode in ('action', 'layer1', 'fusion'):
                for fold in range(2):
                    run = subprocess.run([sys.executable, 'tools/rl_prep/train_world_action_risk_ranker.py'],
                         env=dict(env, WM_RISK_FEATURE_MODE=mode, WM_RISK_CV_FOLD=str(fold)), capture_output=True, text=True, cwd=repo)
                    if run.returncode:
                        raise RuntimeError(run.stdout+run.stderr)
                    print(f'trained {mode} fold {fold}', flush=True)
            for script, args in [
                ('evaluate_world_action_mining.py', [str(root/'oof'), '--modes', 'action', 'layer1', 'fusion', '--folds', '2', '--seed', '3']),
                ('build_verified_risk_manifests.py', [str(root/'oof'), str(root/'data'), str(root/'manifests'), '--folds', '2', '--ranker-seed', '3'])]:
                run = subprocess.run([sys.executable, f'tools/rl_prep/{script}', *args], capture_output=True, text=True, env=env, cwd=repo)
                if run.returncode:
                    raise RuntimeError(run.stdout+run.stderr)
                print(f'{script}: passed', flush=True)
            run = subprocess.run([sys.executable, 'tools/rl_prep/run_wm_diagnostics.py',
                '--fold', '0', '--seed', '0', '--output', str(root/'diagnostics'), '--sizes', '2', '4'],
                env=env, cwd=repo, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            artifacts = list((root/'diagnostics').glob('*/n_*/*/seed_0/fold_0/metrics.json'))
            self.assertEqual(len(artifacts), 10)
            rows = [json.loads(p.read_text()) for p in artifacts]
            self.assertTrue(all(r['selection_scenes'] == rows[0]['selection_scenes'] for r in rows))
            self.assertTrue(all(r['outer_scenes'] == rows[0]['outer_scenes'] for r in rows))
            small = next(r for r in rows if len(r['training_scenes']) == 2)
            large = next(r for r in rows if len(r['training_scenes']) == 4)
            self.assertTrue(set(small['training_scenes']) < set(large['training_scenes']))
            files = list((root/'manifests').glob('*.json'))
            assert len(files) == 3
            for path in files:
                manifest=json.loads(path.read_text())
                assert manifest['development_scenes'] == 12
                assert manifest['verification_calls'] == 96
                assert all(len(e['selected_indices']) == 8 for e in manifest['entries'])
            print('All three manifests: 12 scenes, exactly 8 verifications per scene; nonzero ranker seed works.', flush=True)


if __name__ == "__main__":
    unittest.main()
