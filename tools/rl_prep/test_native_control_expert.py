import unittest

import torch

from native_control_expert import (
    controls_from_absolute_trajectory,
    controls_to_model_action,
    initial_state_from_relative_history,
    model_action_to_controls,
)


class NativeControlRepresentationTest(unittest.TestCase):
    def test_round_trip(self):
        controls = torch.tensor([[[-6.0, -0.2], [0.0, 0.0], [4.0, 0.2]]])
        action = controls_to_model_action(controls)
        self.assertEqual(tuple(action.shape), (1, 3, 3))
        self.assertTrue(torch.equal(action[..., 2], torch.zeros_like(action[..., 2])))
        torch.testing.assert_close(model_action_to_controls(action), controls, atol=0.01, rtol=0.0)

    def test_asymmetric_acceleration_scaling(self):
        large = torch.atanh(torch.tensor(0.999))
        action = torch.tensor([[[-large, 0.0, 0.0], [large, 0.0, 0.0]]])
        controls = model_action_to_controls(action)
        torch.testing.assert_close(
            controls[..., 0], torch.tensor([[-5.994, 3.996]]), atol=1e-4, rtol=0.0
        )

    def test_history_and_inverse_controls_share_state_definition(self):
        history = torch.tensor([[[1.0, 0.0, 0.0]]])
        initial = initial_state_from_relative_history(history)
        torch.testing.assert_close(initial[..., 3], torch.tensor([2.0]))
        absolute = torch.zeros(1, 8, 3)
        absolute[0, :, 0] = torch.arange(1, 9, dtype=torch.float32)
        controls = controls_from_absolute_trajectory(absolute, initial)
        torch.testing.assert_close(controls, torch.zeros_like(controls), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
