"""Apply one WM/physics-conditioned correction policy to DriveVLA outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from continuous_scene_outcome_critic import ContinuousSceneOutcomeCritic
from differentiable_vehicle_model import relative_to_absolute_se2
from trajectory_dynamics import analyze_pose_trajectories
from unified_correction_policy import UnifiedCorrectionPolicy, apply_verified_correction
from world_action_latent_encoder import WorldActionLatentEncoder


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--outcome-checkpoint", type=Path, required=True)
    parser.add_argument("--tokens-json", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--wm-scene-pkl", type=Path, required=True)
    parser.add_argument("--wm-fast-tokenizer", type=Path, required=True)
    parser.add_argument("--wm-norm-stats", type=Path, required=True)
    parser.add_argument("--wm-vq-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--context-scales", default="0.8,0.9,1.0,1.1",
        help="Proposal perturbations used only to form an action-robust WM scene context.",
    )
    parser.add_argument("--max-tokens", type=int, default=0)
    return parser.parse_args()


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location="cpu")


def main():
    args = arguments()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    policy_payload = load(args.policy_checkpoint)
    outcome_payload = load(args.outcome_checkpoint)
    policy = UnifiedCorrectionPolicy(
        dim=int(policy_payload.get("dim", 192)),
        layers=int(policy_payload.get("layers", 3)),
        heads=int(policy_payload.get("heads", 6)),
        wm_context_dim=int(policy_payload.get("wm_context_dim", 4096)),
        vlm_context_dim=int(policy_payload.get("vlm_context_dim", 4096)),
        max_trace_length=int(policy_payload.get("max_trace_length", 2)),
        projection_iterations=int(policy_payload.get("projection_iterations", 30)),
    ).to(device)
    policy.load_state_dict(policy_payload["model"]); policy.eval()
    outcome = ContinuousSceneOutcomeCritic(
        latent_dim=int(outcome_payload.get("latent_dim", 4096)),
        dim=int(outcome_payload.get("dim", 256)),
    ).to(device)
    outcome.load_state_dict(outcome_payload["model"]); outcome.eval()
    encoder = WorldActionLatentEncoder(
        args.wm_checkpoint, scene_pkl=args.wm_scene_pkl,
        fast_tokenizer=args.wm_fast_tokenizer, norm_stats=args.wm_norm_stats,
        vq_root=args.wm_vq_root, device=device,
    )
    tokens = json.loads(args.tokens_json.read_text())
    context_scales = [float(value) for value in args.context_scales.split(",")]
    if not context_scales:
        raise ValueError("context-scales cannot be empty")
    if args.max_tokens > 0: tokens = tokens[:args.max_tokens]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempted = accepted = gate_selected_count = semantic_rejections = physics_rejections = 0
    before_risk, after_risk, before_progress, after_progress = [], [], [], []
    reference_risk_log, reference_progress_log = [], []
    original_absolute, corrected_absolute = [], []
    # no_grad(), not inference_mode(): apply_verified_correction ->
    # policy.project() runs an internal LBFGS solve under
    # torch.enable_grad() (see project_to_feasible_reference in
    # unified_correction_policy.py), and enable_grad() cannot re-enable
    # autograd for tensors created inside an outer inference_mode() block --
    # it silently produces a loss with no grad_fn and crashes on backward().
    with torch.no_grad():
        for index, token in enumerate(tokens):
            source = args.input_dir / f"{token}.json"
            record = json.loads(source.read_text())
            proposal_np = np.asarray(record["action"], dtype=np.float32)
            if proposal_np.shape != (8, 3) or not np.isfinite(proposal_np).all():
                raise ValueError(f"{source}: invalid action {proposal_np.shape}")
            if "vlm_context" not in record:
                raise ValueError(
                    f"{source}: unified inference requires exported vlm_context"
                )
            vlm_context = torch.tensor(
                record["vlm_context"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            context_candidates = np.stack([
                proposal_np * np.float32(scale) for scale in context_scales
            ])
            latent = encoder.encode(str(token), context_candidates).to(device).unsqueeze(0)
            context = outcome.scene_context(latent)
            proposal = torch.from_numpy(proposal_np).to(device).unsqueeze(0)
            # proposal_outcome describes the raw proposal tau^0 -- reported
            # separately below for the three-way (tau^0/tau^ref/tau^1)
            # breakdown, but it is *not* what conditions or verifies the
            # correction; apply_verified_correction computes and uses
            # tau^ref's risk for that internally (see its docstring / the
            # tex's Step 3 implementation note).
            proposal_outcome = outcome.outcomes(proposal.flatten(-2), context)
            applied = apply_verified_correction(
                policy, outcome, proposal, context, vlm_context,
                deterministic=True,
            )
            output = applied["output"]
            reference_risk = applied["reference_risk"]
            corrected_risk = applied["corrected_risk"]
            reference_progress = outcome.outcomes(
                output["reference_relative"].flatten(-2), context
            )["progress"]
            corrected_progress = outcome.outcomes(
                output["corrected_relative"].flatten(-2), context
            )["progress"]
            projection_succeeded = bool(applied["projection_valid"][0].item())
            gate_selected = bool(applied["gate_selected"][0].item())
            physics_ok = bool(applied["physics_ok"][0].item())
            semantic_ok = bool(applied["semantic_ok"][0].item())
            correction_passes_verification = bool(applied["correction_mask"][0].item())
            selected = applied["final_relative"][0]
            attempted += 1
            accepted += correction_passes_verification
            gate_selected_count += gate_selected
            semantic_rejections += gate_selected and physics_ok and not semantic_ok
            physics_rejections += gate_selected and not physics_ok
            transformed = dict(record)
            transformed["action"] = selected.cpu().numpy().astype(float).tolist()
            transformed["unified_correction"] = {
                "policy_checkpoint": str(args.policy_checkpoint),
                "outcome_checkpoint": str(args.outcome_checkpoint),
                "gate_probability": float(output["gate_probability"][0]),
                "gate_selected": gate_selected,
                "projection_succeeded": projection_succeeded,
                "physics_ok": physics_ok,
                "semantic_ok": semantic_ok,
                "applied": correction_passes_verification,
                "proposal_risk": proposal_outcome["risk"][0].cpu().tolist(),
                "reference_risk": reference_risk[0].cpu().tolist(),
                "corrected_risk": corrected_risk[0].cpu().tolist(),
                "proposal_progress": float(proposal_outcome["progress"][0]),
                "reference_progress": float(reference_progress[0]),
                "corrected_progress": float(corrected_progress[0]),
                "reference_physics_violation": float(output["reference_physics_violation"][0]),
                "corrected_physics_violation": float(output["corrected_physics_violation"][0]),
                "projection_error": float(output["projection_error"][0]),
                "controls": output["corrected_controls"][0].cpu().tolist(),
            }
            (args.output_dir / f"{token}.json").write_text(
                json.dumps(transformed, indent=2) + "\n"
            )
            before_risk.append(float(proposal_outcome["risk"].mean()))
            after_risk.append(float(corrected_risk.mean()))
            before_progress.append(float(proposal_outcome["progress"][0]))
            after_progress.append(float(corrected_progress[0]))
            reference_risk_log.append(float(reference_risk.mean()))
            reference_progress_log.append(float(reference_progress[0]))
            original_absolute.append(relative_to_absolute_se2(proposal[0]).cpu().numpy())
            corrected_absolute.append(relative_to_absolute_se2(selected).cpu().numpy())
            if index % 50 == 0 or index + 1 == len(tokens):
                print(f"corrected {index + 1}/{len(tokens)}", flush=True)
    original_dynamics = analyze_pose_trajectories(np.stack(original_absolute), dt=0.5)
    corrected_dynamics = analyze_pose_trajectories(np.stack(corrected_absolute), dt=0.5)
    summary = {
        "schema_version": 1,
        "architecture": "unified_constrained_correction_rl_v1",
        "tokens": len(tokens), "correction_attempted": attempted,
        "correction_accepted": accepted,
        "gate_selected": gate_selected_count,
        "rejected_for_physics": physics_rejections,
        "rejected_for_semantic_regression": semantic_rejections,
        "original_infeasible": int((~original_dynamics["dynamics_feasible"]).sum()),
        "output_infeasible": int((~corrected_dynamics["dynamics_feasible"]).sum()),
        # Three trajectories, reported separately per the method's own
        # summary (tau^0 raw proposal, tau^ref projected reference, tau^1
        # learned correction) so an improvement cannot be silently
        # attributed to the wrong stage.
        "predicted_risk_raw_proposal": float(np.mean(before_risk)),
        "predicted_risk_projected_reference": float(np.mean(reference_risk_log)),
        "predicted_risk_after_correction": float(np.mean(after_risk)),
        "predicted_progress_raw_proposal": float(np.mean(before_progress)),
        "predicted_progress_projected_reference": float(np.mean(reference_progress_log)),
        "predicted_progress_after_correction": float(np.mean(after_progress)),
    }
    (args.output_dir / "unified_correction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
