import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from awfm_statistics import bootstrap, log_labels
from check_awfm_run import check

class ValidationTest(unittest.TestCase):
    def fixture(self, root):
        (root/'validation').mkdir()
        (root/'action_policy.pt').write_bytes(b'test placeholder')
        (root/'run.json').write_text(json.dumps({'preserve':'targeted','eval_tokens':['a','b']}))
        summary={'complete':True,'updates':2,'preservation':{'kind':'targeted','cached':2,'terms':2,
          'first_drift':0.,'sum_drift':.1,'max_drift':.1,'initial_identity_checked':True}}
        (root/'summary.json').write_text(json.dumps(summary))
        (root/'awfm_plan.json').write_text(json.dumps({'a':{'skipped':False}}))
        rows=[{'token':t,**{k:1. for k in ('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress')}} for t in ('a','b')]
        (root/'validation/scores.json').write_text(json.dumps({'rows':rows}))
        return summary

    def test_directory_alone_is_not_completion(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'validation').mkdir()
            with self.assertRaises(FileNotFoundError): check(root)

    def test_complete_and_missing_preservation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);s=self.fixture(root)
            self.assertEqual(check(root,'targeted',2,2)['status'],'PASS')
            s['preservation']['terms']=0
            (root/'summary.json').write_text(json.dumps(s))
            with self.assertRaises(ValueError): check(root,'targeted',2,2)

    def test_incomplete_summary_and_wrong_counts(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);s=self.fixture(root)
            with self.assertRaises(ValueError): check(root,'targeted',3,2)
            s['complete']=False
            (root/'summary.json').write_text(json.dumps(s))
            with self.assertRaises(ValueError): check(root)

    def test_log_cluster_bootstrap_and_mapping(self):
        # Correlated frames: resample logs, not individual frames.
        d=np.array([1.]*50+[-1.]*50)
        labels=['a']*50+['b']*50
        ci=bootstrap(d,np.random.default_rng(0),labels)
        self.assertEqual(ci,[-1.,1.])
        scene=bootstrap(d,np.random.default_rng(0))
        self.assertLess(scene[1]-scene[0],ci[1]-ci[0])
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'map.json';p.write_text(json.dumps({'x':'a','y':'b'}))
            self.assertEqual(log_labels(['x','y'],p),['a','b'])
            with self.assertRaises(ValueError): log_labels(['x','z'],p)


class InputPreflightTest(unittest.TestCase):
    def test_baseline_and_cache_preflight(self):
        import hashlib
        from check_awfm_inputs import check_inputs
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cache=root/'cache';base=root/'model';base.mkdir()
            manifest={'schema_version':2,'heldout_eval_tokens':['a','b'],'train':[{'token':'c'}]}
            mp=root/'manifest_v2.json';mp.write_text(json.dumps(manifest))
            for token,log in [('a','log_a'),('b','log_b'),('c','log_c')]:
                d=cache/log/'type'/token;d.mkdir(parents=True);(d/'metric_cache.pkl').touch()
            b=root/'scaled_v1_baseline_seed20';(b/'validation').mkdir(parents=True)
            (b/'summary.json').write_text(json.dumps({'complete':True}))
            (b/'run.json').write_text(json.dumps({'mode':'baseline','eval_tokens':['a','b'],
              'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'base':str(base)}))
            (b/'validation/scores.json').write_text(json.dumps({'rows':[{'token':'a'},{'token':'b'}]}))
            self.assertEqual(check_inputs(root,cache)['status'],'PASS')
            mp.write_text(json.dumps({**manifest,'train':[{'token':'a'}]}))
            with self.assertRaises(ValueError): check_inputs(root,cache)

if __name__=='__main__': unittest.main()
