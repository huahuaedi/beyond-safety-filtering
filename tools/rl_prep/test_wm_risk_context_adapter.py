import unittest

try:
    import torch
    from wm_risk_context_adapter import WorldRiskContextAdapter
except ModuleNotFoundError:
    torch = None
    WorldRiskContextAdapter = None


@unittest.skipIf(torch is None, "PyTorch is only installed in the drivevla environment")
class WorldRiskContextAdapterTest(unittest.TestCase):
    def test_shapes_attention_and_near_zero_initial_effect(self):
        adapter = WorldRiskContextAdapter(16, 32, condition_dim=11, bottleneck=8)
        latent = torch.randn(3, 4, 16)
        condition = torch.randn(3, 11)
        token, attention = adapter(latent, condition)
        self.assertEqual(token.shape, (3, 1, 32))
        self.assertEqual(attention.shape, (3, 4))
        torch.testing.assert_close(attention.sum(1), torch.ones(3))
        self.assertLess(float(token.abs().max()), 1e-6)

    def test_gate_can_open_and_receive_gradients(self):
        adapter = WorldRiskContextAdapter(16, 32, condition_dim=11, bottleneck=8)
        adapter.gate_logit.data.fill_(0.0)
        token, _ = adapter(torch.randn(2, 4, 16), torch.randn(2, 11))
        token.square().mean().backward()
        self.assertIsNotNone(adapter.gate_logit.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in adapter.parameters()))

    def test_rejects_bad_shapes(self):
        adapter = WorldRiskContextAdapter(16, 32)
        with self.assertRaises(ValueError):
            adapter(torch.randn(4, 16), torch.randn(4, 11))


if __name__ == "__main__":
    unittest.main()
