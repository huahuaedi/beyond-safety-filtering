"""Velocity regularisation on re-noised candidates, not an ODE-path or KL bound."""


def preservation_loss(velocity, reference):
    import torch
    if velocity.shape != reference.shape:
        raise ValueError('teacher/student velocity shape mismatch')
    if not torch.isfinite(velocity).all() or not torch.isfinite(reference).all():
        raise FloatingPointError('nonfinite preservation velocity')
    # The target must never acquire gradients, even if a caller passes a live tensor.
    return (velocity.float() - reference.detach().to(velocity.device).float()).square().mean()


def teacher_mode_velocity(policy, forward, *args, **kwargs):
    """Disable dropout for both cached teacher and differentiable student probes.

    eval() does not disable autograd. Restore every module's previous mode.
    """
    modes = [(module, module.training) for module in policy.modules()]
    try:
        policy.eval()
        return forward(*args, **kwargs)
    finally:
        for module, mode in modes:
            module.training = mode
