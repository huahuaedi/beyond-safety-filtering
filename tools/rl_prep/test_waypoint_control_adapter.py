import unittest

try:
    import torch
    from waypoint_control_adapter import WaypointControlAdapter
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class WaypointControlAdapterTest(unittest.TestCase):
    def test_shapes_and_bounds(self):
        model = WaypointControlAdapter(dim=32, layers=1, heads=4).eval()
        waypoints = torch.randn(3, 8, 3)
        initial = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]]).expand(3, -1)
        output = model(waypoints, initial)
        self.assertEqual(output["controls"].shape, (3, 8, 2))
        self.assertEqual(output["states"].shape, (3, 8, 5))
        self.assertEqual(output["relative_trajectory"].shape, (3, 8, 3))
        self.assertGreaterEqual(float(output["controls"][..., 0].min().detach()), -6.0)
        self.assertLessEqual(float(output["controls"][..., 0].max().detach()), 4.0)

    def test_gradient_reaches_waypoint_input(self):
        model = WaypointControlAdapter(dim=32, layers=1, heads=4)
        waypoints = torch.randn(2, 8, 3, requires_grad=True)
        initial = torch.tensor([[0.0, 0.0, 0.0, 4.0, 0.0]]).expand(2, -1)
        model(waypoints, initial)["absolute_trajectory"].square().mean().backward()
        self.assertIsNotNone(waypoints.grad)
        self.assertTrue(torch.isfinite(waypoints.grad).all())
        self.assertGreater(float(waypoints.grad.abs().sum()), 0.0)

    def test_scene_candidate_axes_are_preserved(self):
        model = WaypointControlAdapter(dim=32, layers=1, heads=4).eval()
        output = model(torch.randn(2, 5, 8, 3), torch.randn(2, 5, 5))
        self.assertEqual(output["controls"].shape, (2, 5, 8, 2))


if __name__ == "__main__":
    unittest.main()
