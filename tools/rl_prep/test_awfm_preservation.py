import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from awfm_preservation import preservation_loss, teacher_mode_velocity


@unittest.skipUnless(HAS_TORCH, 'torch unavailable; required on HPC')
class PreservationTest(unittest.TestCase):
    def test_identity_and_reference_detached(self):
        v = torch.tensor([1., 2.], requires_grad=True)
        ref = v.detach().clone().requires_grad_(True)
        loss = preservation_loss(v, ref)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(v.grad.abs().sum().item(), 0.)
        self.assertIsNone(ref.grad)

    def test_nonzero_gradient_points_toward_teacher(self):
        v = torch.tensor([2., -2.], requires_grad=True)
        ref = torch.zeros(2, requires_grad=True)
        loss = preservation_loss(v, ref)
        loss.backward()
        self.assertTrue(torch.allclose(v.grad, torch.tensor([2., -2.])))
        self.assertIsNone(ref.grad)
        self.assertLess(preservation_loss(v - .1*v.grad, ref).item(), loss.item())

    def test_eval_probes_keep_gradients_and_restore_mixed_modes(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.9))
        model.train(); model[0].eval()
        before = [m.training for m in model.modules()]
        x = torch.ones(2)
        a = teacher_mode_velocity(model, model, x)
        b = teacher_mode_velocity(model, model, x)
        self.assertTrue(torch.equal(a, b))
        a.sum().backward()
        self.assertIsNotNone(model[0].weight.grad)
        self.assertEqual(before, [m.training for m in model.modules()])
        with self.assertRaises(RuntimeError):
            teacher_mode_velocity(model, lambda: (_ for _ in ()).throw(RuntimeError()))
        self.assertEqual(before, [m.training for m in model.modules()])

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            preservation_loss(torch.zeros(2), torch.zeros(3))
        with self.assertRaises(FloatingPointError):
            preservation_loss(torch.tensor([float('nan')]), torch.zeros(1))


if __name__ == '__main__':
    unittest.main()
