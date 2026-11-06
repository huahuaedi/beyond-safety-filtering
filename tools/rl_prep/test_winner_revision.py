import unittest
import numpy as np
from build_margin_preferences import scene_pair
from analyze_winner_transfer import distance
class Tests(unittest.TestCase):
    def test_random_only_eligible_and_shared_loser(self):
        scores={k:np.array([1.,1.,0.]) for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')}
        scores.update(ego_progress=np.array([.2,.8,.9]),comfort=np.ones(3))
        args=(np.zeros((3,8,3)),scores,{}, {'min_clearance':np.array([2.,1.,0.])},np.zeros((8,3)),np.zeros(3),np.ones(3),'simulator')
        seen=set()
        for seed in range(40):
            pair=scene_pair(*args,winner_rule='random',random_seed=seed)
            self.assertEqual(pair,scene_pair(*args,winner_rule='random',random_seed=seed))
            self.assertIn(pair[0],[0,1]);self.assertEqual(pair[1],2);seen.add(pair[0])
        self.assertEqual(seen,{0,1})
    def test_distance_units(self):
        a=np.zeros((8,3));b=a.copy();b[:,:2]=[3,4]
        self.assertEqual(distance(a,b),5.)
        with self.assertRaises(ValueError):distance(a,np.zeros((7,3)))
if __name__=='__main__':unittest.main()
