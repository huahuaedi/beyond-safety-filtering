"""Candidate evaluators of Stage A (PyTorch). All variants share the trajectory encoder, trunk, output heads and loss; only the information
given about the scene differs (a: none, b: one pooled-context token, c: candidate queries over all visual tokens, d: as c with tokens of another
scene (data-level change), e: privileged geometry). See docs/stageA_protocol.md."""
import torch
from torch import nn


def mlp(i, h, o, drop):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, o))


class Evaluator(nn.Module):
    def __init__(self, variant, traj_dim=40, tok_dim=128, geom_dim=15, d=128, heads=4, drop=0.1, hidden=256, n_cont=5,
                 frames=2, rows=18, cols=32):
        super().__init__()
        if variant not in ("a", "b", "c", "d", "e"):
            raise ValueError(variant)
        self.variant = variant
        self.traj_enc = mlp(traj_dim, d, d, drop)
        if variant in ("b", "c", "d"):
            self.tok_proj = nn.Linear(tok_dim, d)
        if variant in ("c", "d"):
            self.frame_emb, self.row_emb, self.col_emb = nn.Embedding(frames, d), nn.Embedding(rows, d), nn.Embedding(cols, d)
            self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
            self.ln = nn.LayerNorm(d)
        if variant == "e":
            self.geom_enc = mlp(geom_dim, d, d, drop)
        in_dim = 2 * d if variant in ("b", "e") else d
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, hidden), nn.GELU())
        self.rank = nn.Linear(hidden, 1)
        self.gates = nn.Linear(hidden, 3)
        self.cont = nn.Linear(hidden, n_cont)

    def forward(self, traj, tokens=None, grid=None, geom=None):
        q = self.traj_enc(traj)                                       # [B, C, d]
        v = self.variant
        if v == "b":
            ctx = self.tok_proj(tokens).mean(dim=1, keepdim=True).expand(-1, q.shape[1], -1)
            z = torch.cat([q, ctx], dim=-1)
        elif v in ("c", "d"):
            pos = self.frame_emb(grid[:, 0]) + self.row_emb(grid[:, 1]) + self.col_emb(grid[:, 2])      # [N, d]
            kv = self.tok_proj(tokens) + pos.unsqueeze(0)
            o, _ = self.attn(q, kv, kv, need_weights=False)
            z = self.ln(q + o)
        elif v == "e":
            z = torch.cat([q, self.geom_enc(geom)], dim=-1)
        else:
            z = q
        h = self.trunk(z)
        return {"rank": self.rank(h).squeeze(-1), "gates": self.gates(h), "cont": self.cont(h)}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_hidden(variant, target, **kwargs):
    """Trunk width of `variant` whose parameter count is closest to `target` (so b is not smaller than c by construction)."""
    best = None
    for h in range(64, 2049, 16):
        n = count_params(Evaluator(variant, hidden=h, **kwargs))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (h, n)
    return best


def loss_fn(out, rank_t, gates_t, cont_t, aux_weight=0.5):
    """Within-scene ranking regression (scene mean of the prediction removed) + auxiliary gate and geometry supervision."""
    rank = out["rank"] - out["rank"].mean(dim=1, keepdim=True)
    l_rank = nn.functional.mse_loss(rank, rank_t)
    l_gate = nn.functional.binary_cross_entropy_with_logits(out["gates"], gates_t)
    l_cont = nn.functional.mse_loss(out["cont"], cont_t)
    return l_rank + aux_weight * (l_gate + l_cont), {"rank": float(l_rank.detach()), "gate": float(l_gate.detach()), "cont": float(l_cont.detach())}
