"""Pairwise preference objective on flow-matching losses (Diffusion-DPO style), independent of the policy code.

For a pair (winner w, loser l) of trajectories of one scene, the flow-matching loss of the current policy and of a
fixed reference (here: the same policy before any update, evaluated with the same noise and flow time) give

    delta = (L(w) - L_ref(w)) - (L(l) - L_ref(l))          (lower loss = more likely under the policy)
    pref  = softplus(beta * delta) = -log sigmoid(-beta * delta)

so minimising pref lowers the loss of the winner relative to the loser. A weighted flow-matching term on the winner
keeps the policy anchored on safe trajectories. Reference losses are cached once, before training, so no second copy
of the action expert is needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_pairs(path, allowed_tokens):
    pairs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    kept = [p for p in pairs if p["token"] in allowed_tokens]
    if not kept:
        raise ValueError("no preference pair belongs to the allowed (training) scenes")
    for p in kept:
        if p["winner"] == p["loser"]:
            raise ValueError(f"winner equals loser in {p['token']}")
    return kept


def build_pair_schedule(pairs, epochs, seed):
    """Deterministic list of (pair, epoch); a different shuffle in every epoch."""
    rng = np.random.default_rng(seed)
    out = []
    for epoch in range(epochs):
        for j in rng.permutation(len(pairs)):
            out.append((pairs[int(j)], epoch))
    return out


def preference_loss(loss_w, loss_l, ref_w, ref_l, beta):
    if beta <= 0:
        raise ValueError("beta must be positive")
    delta = (loss_w - ref_w) - (loss_l - ref_l)
    return F.softplus(beta * delta), delta


def pair_objective(loss_w, loss_l, ref_w, ref_l, beta, winner_weight):
    pref, delta = preference_loss(loss_w, loss_l, ref_w, ref_l, beta)
    total = pref + winner_weight * loss_w
    return total, {"pref": float(pref.detach()), "delta": float(delta.detach()),
                   "winner_fm": float(loss_w.detach()), "loser_fm": float(loss_l.detach())}


def run_pair_training(*, schedule, accumulation, max_updates, beta, winner_weight, expert_weight,
                      get_pair_targets, get_sample, loss_fn, seed_fn, expert_loss_fn,
                      optimizer, trainable, clip_fn, log=None):
    """Train on (pair, epoch) schedule items.

    ``loss_fn(sample, target, seed)`` is the flow-matching loss of the policy. Reference losses are computed once for
    every schedule item before any update, under no_grad, with the seed that training will reuse, so the same noise
    and flow time are used for winner, loser, and reference.
    """
    ref = {}
    with torch.no_grad():
        for pair, epoch in schedule:
            tw, tl = get_pair_targets(pair)
            sample = get_sample(pair["token"])
            seed = seed_fn(epoch, pair["token"])
            ref[(pair["token"], epoch)] = (float(loss_fn(sample, tw, seed)), float(loss_fn(sample, tl, seed)))
    updates = examples = 0
    losses, infos = [], []
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, len(schedule), accumulation):
        if max_updates and updates >= max_updates:
            break
        group = schedule[start:start + accumulation]
        group_loss, sample = 0.0, None
        for pair, epoch in group:
            tw, tl = get_pair_targets(pair)
            sample = get_sample(pair["token"])
            seed = seed_fn(epoch, pair["token"])
            lw, ll = loss_fn(sample, tw, seed), loss_fn(sample, tl, seed)
            rw, rl = ref[(pair["token"], epoch)]
            total, info = pair_objective(lw, ll, lw.new_tensor(rw), lw.new_tensor(rl), beta, winner_weight)
            (total / len(group)).backward()
            group_loss += float(total.detach()) / len(group)
            infos.append(info)
            examples += 1
        if expert_weight and expert_loss_fn is not None:
            anchor = expert_loss_fn(sample, updates)
            (expert_weight * anchor).backward()
        grad = clip_fn(trainable)
        if float(grad) == 0:
            raise RuntimeError("zero policy gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        losses.append(group_loss)
        if log is not None:
            log(updates, group_loss, float(grad), examples, infos[-1])
    return updates, examples, losses, infos
