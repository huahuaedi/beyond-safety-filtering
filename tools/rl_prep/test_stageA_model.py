import unittest

import numpy as np
import torch

import stageA_model as M


def batch(B=3, C=7, N=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.stack([torch.zeros(N, dtype=torch.long), (torch.arange(N) // 4) % 18, torch.arange(N) % 32], dim=1)
    return dict(traj=torch.randn(B, C, 40, generator=g), tokens=torch.randn(B, N, 128, generator=g), grid=grid, geom=torch.randn(B, C, 15, generator=g))


class ModelTest(unittest.TestCase):
    def test_output_shapes_and_loss_for_every_variant(self):
        b = batch()
        for v in "abcde":
            m = M.Evaluator(v)
            out = m(b["traj"], tokens=b["tokens"], grid=b["grid"], geom=b["geom"])
            self.assertEqual(out["rank"].shape, (3, 7)); self.assertEqual(out["gates"].shape, (3, 7, 3)); self.assertEqual(out["cont"].shape, (3, 7, 5))
            loss, parts = M.loss_fn(out, torch.zeros(3, 7), torch.ones(3, 7, 3), torch.zeros(3, 7, 5))
            self.assertTrue(torch.isfinite(loss)); loss.backward()

    def test_tokens_change_c_but_not_a(self):
        b = batch(); other = torch.randn_like(b["tokens"])
        torch.manual_seed(0); a = M.Evaluator("a").eval(); c = M.Evaluator("c").eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(a(b["traj"])["rank"], a(b["traj"])["rank"]))
            r1 = c(b["traj"], tokens=b["tokens"], grid=b["grid"])["rank"]; r2 = c(b["traj"], tokens=other, grid=b["grid"])["rank"]
        self.assertFalse(torch.allclose(r1, r2))

    def test_candidates_of_one_scene_see_the_same_tokens_and_ranking_loss_ignores_scene_offset(self):
        b = batch(); c = M.Evaluator("c").eval()
        with torch.no_grad():
            out = c(b["traj"], tokens=b["tokens"], grid=b["grid"])
        shifted = {"rank": out["rank"] + 5.0, "gates": out["gates"], "cont": out["cont"]}
        t = torch.randn(3, 7); t = t - t.mean(1, keepdim=True)
        l1, _ = M.loss_fn(out, t, torch.ones(3, 7, 3), torch.zeros(3, 7, 5)); l2, _ = M.loss_fn(shifted, t, torch.ones(3, 7, 3), torch.zeros(3, 7, 5))
        self.assertAlmostEqual(float(l1), float(l2), places=5)

    def test_parameter_matching_between_b_and_c(self):
        target = M.count_params(M.Evaluator("c"))
        h, n = M.match_hidden("b", target)
        self.assertLess(abs(n - target) / target, 0.05)

    def test_c_learns_a_planted_dependence_on_token_position(self):
        # target depends on the token at a fixed position and on the candidate's first coordinate: c should fit it, a cannot
        torch.manual_seed(0); B, C, N = 64, 6, 10
        grid = torch.stack([torch.zeros(N, dtype=torch.long), torch.arange(N) // 5, torch.arange(N) % 5], dim=1)
        tokens = torch.randn(B, N, 128); traj = torch.randn(B, C, 40)
        y = tokens[:, 3, 0].unsqueeze(1) * traj[:, :, 0]; y = y - y.mean(1, keepdim=True)
        losses = {}
        for v in ("a", "c"):
            m = M.Evaluator(v, drop=0.0); opt = torch.optim.AdamW(m.parameters(), 3e-3)
            for _ in range(250):
                out = m(traj, tokens=tokens, grid=grid)
                l, _ = M.loss_fn(out, y, torch.ones(B, C, 3), torch.zeros(B, C, 5), aux_weight=0.0)
                opt.zero_grad(); l.backward(); opt.step()
            losses[v] = float(l)
        self.assertLess(losses["c"], 0.5 * losses["a"])


if __name__ == "__main__":
    unittest.main()
