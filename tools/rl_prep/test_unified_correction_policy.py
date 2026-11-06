import unittest

import torch

from differentiable_vehicle_model import VehicleModelConfig, relative_to_absolute_se2
from unified_correction_policy import (
    UnifiedCorrectionPolicy,
    infer_reference_controls,
    physics_condition,
    physics_violation,
    project_to_feasible_reference,
)


class UnifiedCorrectionPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.config = VehicleModelConfig()
        self.policy = UnifiedCorrectionPolicy(
            dim=48, layers=1, heads=4, wm_context_dim=32, config=self.config
        ).eval()
        self.proposal = torch.zeros(2, 8, 3)
        self.proposal[..., 0] = 1.0
        self.risk = torch.zeros(2, 4)
        self.context = torch.zeros(2, 32)

    def test_inverse_controls_have_expected_shape_and_gradients(self):
        proposal = self.proposal.clone().requires_grad_(True)
        initial, controls = infer_reference_controls(proposal, self.config)
        self.assertEqual(initial.shape, (2, 5))
        self.assertEqual(controls.shape, (2, 8, 2))
        controls.sum().backward()
        self.assertIsNotNone(proposal.grad)
        self.assertTrue(torch.isfinite(proposal.grad).all())

    def test_zero_initialized_policy_is_close_to_identity(self):
        with torch.no_grad():
            output = self.policy(self.proposal, self.risk, self.context)
        self.assertTrue(torch.all(output["gate_probability"] < 0.02))
        self.assertTrue(torch.allclose(output["residual_controls"], torch.zeros_like(output["residual_controls"])))
        self.assertEqual(output["corrected_relative"].shape, (2, 8, 3))

    def test_risk_and_physics_are_conditions_on_one_policy(self):
        vlm = torch.randn(2, 32)
        policy = UnifiedCorrectionPolicy(
            dim=48, layers=1, heads=4, wm_context_dim=32,
            vlm_context_dim=32, config=self.config,
        )
        output = self.policy(
            self.proposal,
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            self.context,
            correction_trace=self.proposal[:, None],
        )
        loss = output["corrected_relative"].square().mean() + output["gate_probability"].mean()
        loss.backward()
        self.assertIsNotNone(self.policy.local_projection[1].weight.grad)
        self.assertTrue(torch.isfinite(self.policy.local_projection[1].weight.grad).all())
        conditioned = policy(
            self.proposal, self.risk, self.context, vlm_context=vlm
        )
        conditioned["gate_probability"].mean().backward()
        self.assertIsNotNone(policy.vlm_projection[1].weight.grad)

    def test_physics_violation_detects_excessive_control(self):
        states = torch.zeros(1, 8, 5)
        states[..., 3] = 30.0
        states[..., 4] = 0.25
        controls = torch.zeros(1, 8, 2)
        condition = physics_condition(states, controls, self.config)
        self.assertEqual(condition.shape, (1, 8, 7))
        self.assertGreater(float(physics_violation(condition)), 0.0)

    def test_projection_recovers_an_already_feasible_proposal(self):
        # A slow, near-straight proposal is already well within every bound,
        # so the closest feasible realization should coincide with it almost
        # exactly -- this is the case the old finite-difference round trip
        # got wrong (nonzero reconstruction error even with nothing to fix).
        proposal = torch.zeros(2, 8, 3)
        proposal[..., 0] = 0.5
        projection = project_to_feasible_reference(
            proposal, self.config, num_iterations=50,
        )
        self.assertEqual(projection["reference_controls"].shape, (2, 8, 2))
        self.assertEqual(projection["reference_relative"].shape, (2, 8, 3))
        self.assertTrue(torch.isfinite(projection["projection_error"]).all())
        self.assertTrue(torch.all(projection["projection_error"] < 0.05))
        self.assertTrue(
            torch.allclose(projection["reference_relative"], proposal, atol=0.05)
        )

    def test_projection_output_is_detached_from_the_proposal_graph(self):
        # No trainable parameter influences tau^0 through this projection --
        # the Action Expert's gradient comes from the separate log-prob path,
        # not through the reference. Keeping this detached is what makes an
        # internal LBFGS solve safe to run from inside a no_grad() call site.
        proposal = torch.zeros(1, 8, 3).requires_grad_(True)
        proposal_leaf = proposal
        with torch.no_grad():
            projection = project_to_feasible_reference(
                proposal_leaf, self.config, num_iterations=10,
            )
        self.assertFalse(projection["reference_controls"].requires_grad)
        self.assertFalse(projection["reference_states"].requires_grad)
        self.assertFalse(projection["reference_relative"].requires_grad)

    def test_projection_respects_the_physical_envelope_for_an_extreme_proposal(self):
        # A wildly aggressive proposal (far beyond max speed/curvature) must
        # still land on a trajectory that respects every bound the vehicle
        # model actually enforces *independently*: speed and curvature are
        # each smooth-clamped to their own limit inside the RK4 step no
        # matter how infeasible the target is.
        #
        # The full 2D friction circle (lateral = speed^2*curvature vs.
        # longitudinal accel) is deliberately NOT asserted as a hard
        # guarantee here: the model only clamps longitudinal control
        # authority given the state's *current* lateral acceleration --
        # curvature's own bound does not shrink as speed grows, so a target
        # this extreme can still land on a state where speed^2*curvature
        # alone exceeds the friction budget. That is exactly the case
        # physics_condition's lateral_acceleration_ratio/friction_ratio
        # channels exist to detect and physics_violation() exists to
        # penalize in the reward, so this test checks that detection path
        # instead of a false hard guarantee (see the docstring note on
        # project_to_feasible_reference).
        proposal = torch.zeros(1, 8, 3)
        proposal[..., 0] = 20.0  # far beyond any single-step feasible advance
        proposal[..., 2] = 1.0   # large heading swing every step
        projection = project_to_feasible_reference(
            proposal, self.config, num_iterations=50,
        )
        states = projection["reference_states"]
        self.assertTrue(torch.isfinite(states).all())
        speed = states[..., 3]
        curvature = states[..., 4]
        self.assertTrue(torch.all(speed <= self.config.max_speed + 1e-3))
        self.assertTrue(torch.all(speed >= self.config.min_speed - 1e-3))
        self.assertTrue(torch.all(curvature.abs() <= self.config.max_abs_curvature + 1e-3))
        condition = physics_condition(
            states, projection["reference_controls"], self.config,
        )
        self.assertEqual(condition.shape, (1, 8, 7))
        self.assertTrue(torch.isfinite(condition).all())
        self.assertGreater(float(physics_violation(condition)), 0.0)

    def test_policy_forward_uses_projected_reference(self):
        with torch.no_grad():
            output = self.policy(self.proposal, self.risk, self.context)
        self.assertIn("projection_error", output)
        self.assertEqual(output["projection_error"].shape, (2,))
        self.assertTrue(torch.isfinite(output["projection_error"]).all())
        self.assertIn("projection_valid", output)
        self.assertTrue(bool(output["projection_valid"].all()))
        self.assertEqual(output["reference_relative"].shape, (2, 8, 3))
        self.assertEqual(output["reference_absolute"].shape, (2, 8, 3))

    def test_project_then_correct_projected_matches_forward(self):
        # forward() is now defined as project() + correct_projected(); this
        # locks in that the split didn't change behavior for callers who
        # still use the one-shot convenience wrapper, while also being the
        # shape/API contract new callers (e.g. apply_verified_correction)
        # rely on when they compute risk_condition on tau^ref instead of
        # tau^0 between the two calls.
        with torch.no_grad():
            projected = self.policy.project(self.proposal)
            self.assertEqual(projected["reference_relative"].shape, (2, 8, 3))
            self.assertIn("projection_valid", projected)
            split_output = self.policy.correct_projected(
                projected, self.risk, self.context
            )
            direct_output = self.policy(self.proposal, self.risk, self.context)
        self.assertTrue(
            torch.allclose(
                split_output["corrected_relative"], direct_output["corrected_relative"]
            )
        )
        self.assertTrue(
            torch.allclose(
                split_output["reference_relative"], direct_output["reference_relative"]
            )
        )

    def test_deterministic_gate_is_binary(self):
        with torch.no_grad():
            output = self.policy(
                self.proposal, self.risk, self.context, deterministic_gate=True
            )
        self.assertTrue(torch.all((output["gate"] == 0) | (output["gate"] == 1)))


if __name__ == "__main__":
    unittest.main()
