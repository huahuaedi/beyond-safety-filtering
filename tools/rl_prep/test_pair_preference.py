import json, tempfile, unittest
from pathlib import Path
import torch
import pair_preference as P


class PairPreferenceTest(unittest.TestCase):
    def test_zero_delta_gives_log2_and_gradient_signs(self):
        lw = torch.tensor(0.05, requires_grad=True); ll = torch.tensor(0.05, requires_grad=True)
        pref, delta = P.preference_loss(lw, ll, torch.tensor(0.05), torch.tensor(0.05), beta=100.0)
        self.assertAlmostEqual(float(pref), 0.6931, places=3)           # -log sigmoid(0)
        pref.backward()
        self.assertGreater(float(lw.grad), 0.0)                          # lowering the winner loss lowers the objective
        self.assertLess(float(ll.grad), 0.0)                             # raising the loser loss lowers the objective

    def test_objective_decreases_when_winner_improves_relative_to_loser(self):
        ref_w, ref_l = torch.tensor(0.06), torch.tensor(0.06)
        better, _ = P.pair_objective(torch.tensor(0.05), torch.tensor(0.07), ref_w, ref_l, 100.0, 0.0)
        worse, _ = P.pair_objective(torch.tensor(0.07), torch.tensor(0.05), ref_w, ref_l, 100.0, 0.0)
        self.assertLess(float(better), float(worse))

    def test_winner_weight_adds_flow_matching_anchor(self):
        a, _ = P.pair_objective(torch.tensor(0.05), torch.tensor(0.05), torch.tensor(0.05), torch.tensor(0.05), 10.0, 0.0)
        b, info = P.pair_objective(torch.tensor(0.05), torch.tensor(0.05), torch.tensor(0.05), torch.tensor(0.05), 10.0, 2.0)
        self.assertAlmostEqual(float(b - a), 0.1, places=6)
        self.assertEqual(set(info), {"pref", "delta", "winner_fm", "loser_fm"})

    def test_extreme_beta_is_numerically_stable(self):
        lw = torch.tensor(0.0, requires_grad=True)
        pref, _ = P.preference_loss(lw, torch.tensor(5.0), torch.tensor(5.0), torch.tensor(0.0), beta=1e4)
        pref.backward()
        self.assertTrue(torch.isfinite(pref)); self.assertTrue(torch.isfinite(lw.grad))

    def test_load_and_schedule(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "pairs.jsonl"
            p.write_text("".join(json.dumps({"token": f"t{i}", "split": "train", "winner": 1, "loser": 2}) + "\n" for i in range(6)))
            pairs = P.load_pairs(p, {f"t{i}" for i in range(4)})
            self.assertEqual(len(pairs), 4)
            s1 = P.build_pair_schedule(pairs, 2, seed=3); s2 = P.build_pair_schedule(pairs, 2, seed=3)
            self.assertEqual([(a["token"], e) for a, e in s1], [(a["token"], e) for a, e in s2])
            self.assertEqual(len(s1), 8)
            self.assertEqual(sorted(a["token"] for a, e in s1 if e == 0), sorted(a["token"] for a in pairs))
            with self.assertRaises(ValueError):
                P.load_pairs(p, {"nope"})
            bad = Path(t) / "bad.jsonl"; bad.write_text(json.dumps({"token": "x", "winner": 3, "loser": 3}) + "\n")
            with self.assertRaises(ValueError):
                P.load_pairs(bad, {"x"})


if __name__ == "__main__":
    unittest.main()


class PairTrainingTest(unittest.TestCase):
    def test_training_moves_winner_loss_below_loser_loss(self):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.zeros(()))
        opt = torch.optim.SGD([w], lr=0.05)
        # the "flow loss" of a target t is (w - t)^2; winner target 1.0 and loser target -1.0
        def loss_fn(sample, target, seed):
            return (w - float(target)) ** 2
        pairs = [{"token": f"t{i}", "winner": 0, "loser": 1} for i in range(8)]
        schedule = P.build_pair_schedule(pairs, 3, seed=1)
        def clip(params):
            return torch.nn.utils.clip_grad_norm_(params, 10.0)
        seen = {"seeds": []}
        def seed_fn(epoch, token):
            seen["seeds"].append((epoch, token)); return hash((epoch, token)) % 1000
        before_gap = float(loss_fn(None, 1.0, 0) - loss_fn(None, -1.0, 0))
        updates, examples, losses, infos = P.run_pair_training(
            schedule=schedule, accumulation=4, max_updates=0, beta=2.0, winner_weight=0.0, expert_weight=0.0,
            get_pair_targets=lambda p: (1.0, -1.0), get_sample=lambda t: None, loss_fn=loss_fn, seed_fn=seed_fn,
            expert_loss_fn=None, optimizer=opt, trainable=[w], clip_fn=clip)
        after_gap = float(loss_fn(None, 1.0, 0) - loss_fn(None, -1.0, 0))
        self.assertEqual(updates, 6); self.assertEqual(examples, 24)
        self.assertLess(after_gap, before_gap)                       # winner loss fell relative to loser loss
        self.assertGreater(float(w), 0.0)                            # parameter moved toward the winner target
        self.assertLess(infos[-1]["delta"], infos[0]["delta"])       # delta (policy vs reference) became more negative
        # reference pass and training pass used the same seeds for the same (epoch, token)
        n = len(schedule)
        self.assertEqual(seen["seeds"][:n], seen["seeds"][n:2 * n])

    def test_max_updates_and_anchor_are_respected(self):
        w = torch.nn.Parameter(torch.zeros(()))
        opt = torch.optim.SGD([w], lr=0.01)
        calls = []
        updates, examples, losses, _ = P.run_pair_training(
            schedule=P.build_pair_schedule([{"token": f"t{i}", "winner": 0, "loser": 1} for i in range(8)], 5, 0),
            accumulation=2, max_updates=3, beta=1.0, winner_weight=1.0, expert_weight=0.5,
            get_pair_targets=lambda p: (1.0, -1.0), get_sample=lambda t: None,
            loss_fn=lambda s, t, seed: (w - float(t)) ** 2, seed_fn=lambda e, t: 0,
            expert_loss_fn=lambda s, u: (calls.append(u) or (w - 0.5) ** 2),
            optimizer=opt, trainable=[w], clip_fn=lambda ps: torch.nn.utils.clip_grad_norm_(ps, 10.0))
        self.assertEqual(updates, 3); self.assertEqual(examples, 6); self.assertEqual(calls, [0, 1, 2])
