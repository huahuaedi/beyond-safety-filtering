import tempfile
import unittest
from pathlib import Path
import numpy as np
from offline_verified_data import (verified_labels, normalize_action, absolute_from_relative, read_scene, split_records, POSITIVE, NEGATIVE, SAFETY)


class VerifiedDataTests(unittest.TestCase):
    def scores(self, progress):
        return {**{k:np.ones(len(progress)) for k in SAFETY},'ego_progress':np.array(progress)}

    def test_no_forced_positive_when_all_worse_than_reference(self):
        m=verified_labels(self.scores([.1,.2,.3,.4]),{'ego_progress':[1.]})
        self.assertEqual(m['labels'].tolist(),[NEGATIVE]*4)

    def test_each_safety_failure_blocks_high_progress(self):
        for key in SAFETY:
            s=self.scores([1.,.9]);s[key][0]=.5
            m=verified_labels(s,{'ego_progress':[.5]})
            self.assertEqual(m['labels'].tolist(),[NEGATIVE,POSITIVE])

    def test_same_score_and_margin_are_not_improvement(self):
        m=verified_labels(self.scores([.5,.505,.6]),{'ego_progress':[.5]},margin=.01)
        self.assertEqual(m['labels'].tolist(),[NEGATIVE,NEGATIVE,POSITIVE])

    def test_bad_labels_fail_closed(self):
        for value in [np.nan,np.inf,-.1,1.1]:
            s=self.scores([.9]);s['ego_progress'][0]=value
            with self.assertRaises(ValueError): verified_labels(s,{'ego_progress':[.5]})

    def test_normalization_roundtrip_without_clipping(self):
        a=np.arange(24,dtype=np.float32).reshape(8,3)
        low=np.array([0,-.2,-.1]);high=np.array([6,.2,.1])
        z=normalize_action(a,low,high)
        self.assertGreater(z.max(),1.)
        np.testing.assert_allclose(.5*(z+1)*(high-low)+low,a,atol=2e-6)

    def test_scored_target_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'train'/'token.npz';p.parent.mkdir()
            a=np.zeros((2,8,3),np.float32);a[:,:,0]=1
            absolute=absolute_from_relative(a);absolute[0,0,0]+=1
            np.savez(p,scene_token='token',split='train',relative_trajectories=a,absolute_trajectories=absolute)
            with self.assertRaisesRegex(ValueError,'scored trajectory'): read_scene(p)

    def test_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for split in ['train','val']:
                p=Path(d)/split;p.mkdir();(p/'same.npz').touch()
            with self.assertRaisesRegex(ValueError,'overlapping'): split_records(d)

if __name__=='__main__': unittest.main()
