"""Reference regularization at stored flow-chain states, not endpoint KL."""
import math


def equal_variance_transition_kl(policy_mean, reference_mean, sigma):
    """Mean per-coordinate conditional Gaussian KL on rollout states.

    Both transitions must use the same fixed scalar standard deviation.
    Averaging across batch, denoising steps, horizon and action coordinates
    makes this a normalized regularizer, not the summed trajectory KL.
    Reference gradients are always stopped. No safety guarantee is implied.
    """
    if not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if policy_mean.shape != reference_mean.shape or policy_mean.numel() == 0:
        raise ValueError("means must have matching, nonempty shapes")
    delta = policy_mean.float() - reference_mean.detach().float()
    return (delta.square() / (2.0 * sigma ** 2)).mean()
