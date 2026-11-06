import unittest
from risk_replay_experiment import make_schedule,step_policy


class RiskReplayTest(unittest.TestCase):
    def test_modes_share_main_update_but_distinct_risk_update(self):
        for mode in ('matched_reference','rescored_rl'):
            self.assertEqual(step_policy(mode,'main'),dict(train_on_group=True,rehearse=False))
        self.assertEqual(step_policy('matched_reference','risk'),dict(train_on_group=False,rehearse=True))
        self.assertEqual(step_policy('rescored_rl','risk'),dict(train_on_group=True,rehearse=False))

    def test_fixed_schedule_is_order_invariant_and_equal_budget(self):
        a=make_schedule(['a','b','c'],['b','c'],seed=1)
        self.assertEqual(a,make_schedule(['c','b','a'],['c','b'],seed=1))
        self.assertEqual(len(a),100)
        self.assertEqual(sum(x['kind']=='risk' for x in a),50)
        self.assertTrue(all(x['token'] in ('b','c') for x in a if x['kind']=='risk'))
        self.assertNotEqual(a,make_schedule(['a','b','c'],['b','c'],seed=2))

    def test_reject_nontraining_risk_tokens(self):
        with self.assertRaises(ValueError):make_schedule(['a'],['test'],seed=0)


if __name__=='__main__':unittest.main()
