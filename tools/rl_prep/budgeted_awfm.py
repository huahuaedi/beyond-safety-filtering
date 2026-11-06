"""Finite-pool KL-budgeted sampling for gate-aware flow matching.

The uniform measure on N cached candidates is an empirical measure, not the
continuous SFT policy. KL(w || uniform) bounds only the candidate reweighting.
The weighted passing share is a target statistic, NOT a promised trained-policy
passing probability. Flow fitting, finite samples and distribution shift can
all change the realised result. Preservation is a soft velocity regulariser,
not a policy-KL constraint or a deployment safety guarantee.
"""
from __future__ import annotations

import numpy as np

GATES = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")
EPS_PASS = 1e-6


def gate_pass(scores):
    """Boolean vector: candidate passes NC, DAC and TTC (the simulator scores are 0/1 per gate)."""
    ok = None
    for name in GATES:
        v = np.asarray(scores[name], dtype=np.float64)
        if v.ndim != 1 or not v.size or not np.isfinite(v).all():
            raise ValueError(f"invalid gate score {name}")
        if ok is not None and v.shape != ok.shape:
            raise ValueError("gate shape mismatch")
        ok = v >= 1 - EPS_PASS if ok is None else ok & (v >= 1 - EPS_PASS)
    return ok


def scene_reward(scores, gate_weight=1.0, utility_weight=0.0):
    """Reward = gate_weight * pass + utility_weight * centred progress among passers.

    utility_weight defaults to 0: progress-ranked labels changed the observed safety/progress trade-off.
    """
    g = gate_pass(scores).astype(np.float64)
    r = gate_weight * g
    if utility_weight:
        prog = np.asarray(scores["ego_progress"], dtype=np.float64)
        if prog.shape != g.shape:
            raise ValueError("progress/gate shape mismatch")
        if g.sum() >= 2:
            p = prog[g > 0]
            r[g > 0] += utility_weight * (p - p.mean()) / (p.std() + 1e-6) * 0.5
    return r


def kl_to_uniform(w):
    w = np.asarray(w, dtype=np.float64)
    pos = w > 0
    return float(np.sum(w[pos] * np.log(w[pos] * len(w))))


def tilt_weights(reward, eps, iters=80):
    """Softmax tilt of the uniform distribution with KL(w||uniform) == eps (or the maximum reachable below eps).

    Returns (weights, tau, kl). If the reward is constant, or eps == 0, the weights are uniform. The KL is monotone in
    1/tau, so bisection on log(1/tau) is exact up to tolerance. The reachable KL is bounded by log(N / #argmax); if eps
    exceeds it the tilt concentrates on the argmax set (tau -> 0) and the realised KL is reported.
    """
    r = np.asarray(reward, dtype=np.float64)
    if r.ndim != 1 or not r.size or not np.isfinite(r).all():
        raise ValueError("reward must be a finite non-empty vector")
    if eps < 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and non-negative")
    n = len(r)
    uniform = np.full(n, 1.0 / n)
    if eps == 0 or np.ptp(r) < 1e-12:
        return uniform, np.inf, 0.0

    def weights(beta):
        z = beta * (r - r.max())
        e = np.exp(z)
        return e / e.sum()

    max_kl = float(np.log(n / np.sum(r >= r.max() - 1e-12)))
    if eps >= max_kl - 1e-9:
        w = (r >= r.max() - 1e-12).astype(np.float64)
        w /= w.sum()
        return w, 0.0, kl_to_uniform(w)
    lo, hi = 0.0, 1.0
    while kl_to_uniform(weights(hi)) < eps:
        hi *= 2.0
        if hi > 1e9:
            break
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if kl_to_uniform(weights(mid)) < eps:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    w = weights(beta)
    return w, 1.0 / beta, kl_to_uniform(w)


def promised_pass(scores, eps, **reward_kwargs):
    """(p_before, q_after, kl, tau): passing probability of the sampling distribution before / after the tilt."""
    g = gate_pass(scores).astype(np.float64)
    w, tau, kl = tilt_weights(scene_reward(scores, **reward_kwargs), eps)
    return float(g.mean()), float((w * g).sum()), kl, tau


def build_tilted_schedule(records, tilt_fn, *, candidates_per_scene, epochs, seed):
    """Deterministic (record, candidate, epoch) schedule drawn from per-scene tilted distributions.

    ``tilt_fn(record)`` returns the weight vector of the scene. Scenes with no gate-passing candidate are skipped and
    counted by the caller (no positive target for this binary-reward recipe; not proof of absent policy support). Candidates are drawn with replacement
    from w, so the expected gradient equals the w-weighted flow-matching gradient.
    """
    rng = np.random.default_rng(seed)
    out = []
    for epoch in range(epochs):
        for j in rng.permutation(len(records)):
            r = records[int(j)]
            w = tilt_fn(r)
            if w is None:
                continue
            w = np.asarray(w, dtype=np.float64)
            if w.ndim != 1 or abs(w.sum() - 1) > 1e-6 or (w < 0).any():
                raise ValueError("tilt weights must be a probability vector")
            chosen = rng.choice(len(w), size=candidates_per_scene, replace=True, p=w)
            out.extend((r, int(c), epoch) for c in chosen)
    return out


def budget_for_scene(p_before, eps_unsafe, eps_safe, safe_threshold=0.999):
    """No tilt for empirically all-passing pools by default.

    Zero tilt still permits FM/expert gradients and shared-parameter drift.
    Finite-pool all-pass status does not establish true policy safety.
    """
    return eps_safe if p_before >= safe_threshold else eps_unsafe


def transfer_table(p_before, target_gain, realised_fix, realised_new,
                   edges=(0.0, 0.02, 0.1, 0.3, 0.7, 1.0001)):
    """Descriptive stratification; no ratio between unmatched sampling protocols."""
    p, tg, fx, nw = [np.asarray(x, dtype=float) for x in
                     (p_before, target_gain, realised_fix, realised_new)]
    if not (p.shape == tg.shape == fx.shape == nw.shape) or p.ndim != 1:
        raise ValueError('per-scene arrays must share a one-dimensional shape')
    if not all(np.isfinite(x).all() for x in (p, tg, fx, nw)):
        raise ValueError('nonfinite transfer data')
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        row = {'p_lo': lo, 'p_hi': hi, 'n': int(m.sum())}
        if m.any():
            row.update(target_pool_pass_gain=float(tg[m].mean()),
                       fixed=float(fx[m].mean()), new=float(nw[m].mean()),
                       realised=float((fx[m]-nw[m]).mean()))
        rows.append(row)
    return rows


def preserve_candidate(scores, kind, seed):
    """Candidate whose teacher velocity field the update must keep (None = nothing to preserve in this scene).

    kind='passers': uniform over observed joint gate-passing candidates.
    kind='all': uniform over all candidates of the retained scene pool.
    The trainer re-noises an endpoint with independent noise. These states are
    NOT asserted to be actual teacher ODE states or safety-certified states.
    Shared parameters can still couple preserved and repaired behaviour.
    """
    g = gate_pass(scores)
    if kind == "passers":
        pool = np.flatnonzero(g)
    elif kind == "all":
        pool = np.arange(len(g))
    else:
        raise ValueError(f"unknown preserve kind {kind}")
    if not len(pool):
        return None
    return int(np.random.default_rng(seed).choice(pool))
