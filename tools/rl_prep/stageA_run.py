"""Train and evaluate ONE Stage A evaluator (variant a-e, one seed) following docs/stageA_protocol.md.

Training scenes: manifest `train`. Development scenes: the pool scenes not in the manifest (log-disjoint from training), split BY LOG into
`tune` and `decision`. `--eval` lists the sets to score: tune, decision, navtest (the last only after the scheme is locked). Fixed budget:
`--epochs` over the training scenes, no early stopping, no checkpoint selection. Writes per-scene selections to <out>/<set>.npz.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import stageA_data as D
import stageA_model as M

BATCH_SCENES = 16


def scene_files(root):
    return [p for split in ("train", "val", "test") for p in sorted((Path(root) / split).glob("*.npz"))]


def load_scenes(root, geom_root, keep=None):
    out = []
    for p in scene_files(root):
        if keep is None or p.stem in keep:
            out.append(D.load_scene_data(p, geom_root))
    if not out:
        raise ValueError(f"no scenes loaded from {root}")
    return out


class Stats:
    def __init__(self, scenes, tok_h, rows):
        self.traj = D.standardizer([s["traj"] for s in scenes])
        self.geom = D.standardizer([s["geom"] for s in scenes])
        self.cont = D.standardizer([s["cont"] for s in scenes])
        centred = np.concatenate([s["pdms"] - s["pdms"].mean() for s in scenes])
        self.pdms_sd = float(centred.std() + 1e-6)
        sample = tok_h[np.asarray(rows[:: max(1, len(rows) // 400)])].astype(np.float32)
        self.tok = (sample.mean(axis=(0, 1)).astype(np.float32), sample.std(axis=(0, 1)).astype(np.float32) + 1e-6)


def tensors(scenes, rows, tok_h, st, device, need_tokens):
    z = lambda a, ms: (a - ms[0]) / ms[1]
    traj = torch.as_tensor(np.stack([z(s["traj"], st.traj) for s in scenes]), device=device)
    geom = torch.as_tensor(np.stack([z(s["geom"], st.geom) for s in scenes]), device=device)
    rank = np.stack([(s["pdms"] - s["pdms"].mean()) / st.pdms_sd for s in scenes])
    gates = torch.as_tensor(np.stack([s["gates"] for s in scenes]), device=device)
    cont = torch.as_tensor(np.stack([z(s["cont"], st.cont) for s in scenes]), device=device)
    tokens = None
    if need_tokens:
        tokens = torch.as_tensor(z(tok_h[np.asarray(rows)].astype(np.float32), st.tok), device=device)
    return traj, tokens, geom, torch.as_tensor(rank, dtype=torch.float32, device=device), gates, cont


@torch.no_grad()
def evaluate(model, scenes, rows, tok_h, st, grid, device, need_tokens):
    model.eval()
    sel, scores = [], []
    for i in range(0, len(scenes), 32):
        chunk = scenes[i:i + 32]
        traj, tokens, geom, *_ = tensors(chunk, rows[i:i + 32], tok_h, st, device, need_tokens)
        r = model(traj, tokens=tokens, grid=grid, geom=geom)["rank"]
        scores.append(r.cpu().numpy())
        sel += r.argmax(dim=1).cpu().tolist()
    sel = np.asarray(sel)
    take = lambda key: np.stack([s[key][j] for s, j in zip(scenes, sel)])
    return {
        "tokens": np.array([s["token"] for s in scenes]), "selected": sel, "selected_pdms": take("pdms"), "selected_gates": take("gates"),
        "ref_pdms": np.array([s["ref_pdms"] for s in scenes]), "ref_gate_fail": np.stack([s["ref_gate_fail"] for s in scenes]),
        "oracle_pdms": np.array([s["pdms"].max() for s in scenes]), "mean_pdms": np.array([s["pdms"].mean() for s in scenes]),
        "scores": np.concatenate(scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list("abcde"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-root", type=Path, required=True)
    ap.add_argument("--pool-geom", type=Path, required=True)
    ap.add_argument("--pool-tokens", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pool-log-map", type=Path, required=True, help="json {token: 'log_name:<log>'} for the pool scenes")
    ap.add_argument("--apply-root", type=Path, default=None)
    ap.add_argument("--apply-geom", type=Path, default=None)
    ap.add_argument("--apply-tokens", type=Path, default=None)
    ap.add_argument("--apply-log-map", type=Path, default=None)
    ap.add_argument("--eval", nargs="+", default=["tune"], choices=["tune", "decision", "navtest"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--match-to-c", action="store_true", help="choose the trunk width of this variant to match the parameter count of c")
    ap.add_argument("--shuffle-check", action="store_true", help="also score the trained model with tokens of another log (c only)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--max-train-scenes", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "navtest" in a.eval and not (a.apply_root and a.apply_geom and a.apply_tokens and a.apply_log_map):
        raise SystemExit("--eval navtest needs --apply-root/--apply-geom/--apply-tokens/--apply-log-map")
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    manifest = json.loads(a.manifest.read_text())
    train_tokens = {r["token"] for r in manifest["train"]}
    pool_log = {t: l.split("log_name:")[-1] for t, l in json.loads(a.pool_log_map.read_text()).items()}
    tok_index, tok_h, grid_np, fp = D.load_token_shards(a.pool_tokens)
    grid = torch.as_tensor(grid_np, device=device)
    all_scenes = load_scenes(a.pool_root, a.pool_geom)
    train = [s for s in all_scenes if s["token"] in train_tokens]
    dev = [s for s in all_scenes if s["token"] not in train_tokens]
    if a.max_train_scenes:
        train = train[:a.max_train_scenes]
    dev_logs = [pool_log[s["token"]] for s in dev]
    tune_logs, decision_logs = D.split_logs(dev_logs, 0.7, a.split_seed)
    sets = {"tune": [s for s, l in zip(dev, dev_logs) if l in tune_logs], "decision": [s for s, l in zip(dev, dev_logs) if l in decision_logs]}
    need_tokens = a.variant in ("b", "c", "d")
    shuffled = a.variant == "d"

    def rows_for(scenes, index, logs):
        base = [index[s["token"]] for s in scenes] if need_tokens else [0] * len(scenes)
        if shuffled:
            j = D.cross_log_derangement(logs, a.seed + 17)
            return [base[k] for k in j]
        return base

    train_logs = [pool_log[s["token"]] for s in train]
    train_rows = rows_for(train, tok_index, train_logs)
    st = Stats(train, tok_h, train_rows if need_tokens else list(range(min(len(tok_h), 400))))
    hidden = a.hidden
    if a.match_to_c and a.variant != "c":
        hidden, _ = M.match_hidden(a.variant, M.count_params(M.Evaluator("c", drop=a.dropout, hidden=a.hidden)), drop=a.dropout)
    model = M.Evaluator(a.variant, drop=a.dropout, hidden=hidden).to(device)
    n_params = M.count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = a.epochs * int(np.ceil(len(train) / BATCH_SCENES))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    rng = np.random.default_rng(a.seed)
    log = {"variant": a.variant, "seed": a.seed, "params": n_params, "hidden": hidden, "train_scenes": len(train), "tune_scenes": len(sets["tune"]),
           "decision_scenes": len(sets["decision"]), "projection_fingerprint": fp, "args": {k: str(v) for k, v in vars(a).items()}, "loss": []}
    t0 = time.time()
    for epoch in range(a.epochs):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for i in range(0, len(order), BATCH_SCENES):
            idx = order[i:i + BATCH_SCENES]
            traj, tokens, geom, rank_t, gates_t, cont_t = tensors([train[k] for k in idx], [train_rows[k] for k in idx], tok_h, st, device, need_tokens)
            out = model(traj, tokens=tokens, grid=grid, geom=geom)
            loss, _ = M.loss_fn(out, rank_t, gates_t, cont_t)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            total += float(loss.detach()) * len(idx)
        log["loss"].append(total / len(train))
        if epoch % 5 == 0 or epoch == a.epochs - 1:
            print(f"epoch {epoch} loss {log['loss'][-1]:.4f} elapsed {time.time() - t0:.0f}s", flush=True)
    for name in a.eval:
        if name == "navtest":
            sc = load_scenes(a.apply_root, a.apply_geom)
            idx2, h2, g2, fp2 = D.load_token_shards(a.apply_tokens)
            if fp2 != fp or not np.array_equal(g2, grid_np):
                raise ValueError("navtest tokens use a different projection or grid from the pool tokens")
            lm = {t: l.split("log_name:")[-1] for t, l in json.loads(a.apply_log_map.read_text()).items()}
            logs = [lm[s["token"]] for s in sc]
            rows = rows_for(sc, idx2, logs); table = h2
        else:
            sc = sets[name]; logs = [pool_log[s["token"]] for s in sc]
            rows = rows_for(sc, tok_index, logs); table = tok_h
        res = evaluate(model, sc, rows, table, st, grid, device, need_tokens)
        np.savez(a.out / f"{name}.npz", logs=np.array(logs), **res)
        print(f"{name}: scenes {len(sc)} selected PDMS {res['selected_pdms'].mean() * 100:.3f}  reference {res['ref_pdms'].mean() * 100:.3f}  oracle {res['oracle_pdms'].mean() * 100:.3f}", flush=True)
        if a.shuffle_check and a.variant == "c":
            j = D.cross_log_derangement(logs, a.seed + 99)
            base = [(idx2 if name == "navtest" else tok_index)[s["token"]] for s in sc]
            res2 = evaluate(model, sc, [base[k] for k in j], table, st, grid, device, need_tokens)
            np.savez(a.out / f"{name}_shuffled_tokens.npz", logs=np.array(logs), **res2)
            print(f"{name} (tokens of another log at inference): selected PDMS {res2['selected_pdms'].mean() * 100:.3f}", flush=True)
    (a.out / "run.json").write_text(json.dumps(log, indent=1) + "\n")
    print(f"COMPLETE {a.out}", flush=True)


if __name__ == "__main__":
    main()
