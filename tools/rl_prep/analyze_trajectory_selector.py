"""Can trajectory shape alone pick the better candidate? (CPU, numpy only.)

Features use only the candidate's own waypoints and its relation to the two
reference probes: no images, map or other agents. Training scenes come from a
log-disjoint manifest (build_log_disjoint_manifest.py); every other scene in
the data root is held-out, so train and held-out never share a driving log.

The target is within-scene centred PDMS, so the model only has to rank
candidates inside a scene. Selection quality is the mean PDMS of the picked
candidate minus the reference PDMS, next to cheap baselines and the oracle.
With ``--geom-root`` the 12 non-learned geometric features (extract_geometric_features.py)
are added and three selectors are compared: trajectory-only, geometry-only, both.
The geometric features use only t=0 observed agents and the static drivable area.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DT = 0.5
SAFETY = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")


def pdms(nc, dac, ep, ttc, comfort):
    return nc * dac * (5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0


def load_scene(path):
    with np.load(path, allow_pickle=False) as z:
        scores = {k: z["score__" + k].astype(np.float64)
                  for k in (*SAFETY, "ego_progress", "comfort", "driving_direction_compliance")}
        ref = {k: z["reference_probe__" + k].astype(np.float64)
               for k in (*SAFETY, "ego_progress", "comfort")}
        traj = z["absolute_trajectories"].astype(np.float64)
        ref_traj = z["reference_probe_absolute_trajectories"].astype(np.float64)
    return traj, ref_traj, scores, ref


def step_geometry(traj):
    """traj [..., T, 3] -> path length, step speed, curvature-like yaw rate."""
    xy = traj[..., :2]
    origin = np.zeros_like(xy[..., :1, :])
    steps = np.diff(np.concatenate([origin, xy], axis=-2), axis=-2)
    seg = np.linalg.norm(steps, axis=-1)
    yaw = np.unwrap(traj[..., 2], axis=-1)
    dyaw = np.diff(np.concatenate([np.zeros_like(yaw[..., :1]), yaw], axis=-1), axis=-1)
    return seg, dyaw


def set_level_features(traj, ref_traj):
    """How a candidate sits among the other candidates of its scene: [N, 4].

    Distance of its end point to the median end point, its mean end-point distance to the others,
    and the within-scene ranks of its path length and of its deviation from the reference probes."""
    n = len(traj)
    end = traj[:, -1, :2]
    d_med = np.linalg.norm(end - np.median(end, axis=0), axis=1)
    pair = np.linalg.norm(end[:, None] - end[None], axis=-1).sum(axis=1) / max(n - 1, 1)
    seg, _ = step_geometry(traj)
    path_rank = np.argsort(np.argsort(seg.sum(-1))) / max(n - 1, 1)
    dev = np.linalg.norm(traj[..., :2] - ref_traj.mean(axis=0)[None, :, :2], axis=-1).mean(-1)
    dev_rank = np.argsort(np.argsort(dev)) / max(n - 1, 1)
    return np.stack([d_med, pair, path_rank, dev_rank], axis=1)


def scene_features(traj, ref_traj, set_features=False):
    """Candidate features [N, F] built from one scene."""
    ref = ref_traj.mean(axis=0)
    seg, dyaw = step_geometry(traj)
    rseg, rdyaw = step_geometry(ref)
    speed = seg / DT
    accel = np.abs(np.diff(speed, axis=-1)) / DT
    path = seg.sum(axis=-1)
    rpath = rseg.sum()
    dev = np.linalg.norm(traj[..., :2] - ref[None, :, :2], axis=-1)
    end = traj[:, -1, :]
    rend = ref[-1]
    yaw_dev = np.abs(np.unwrap(traj[..., 2], axis=-1) - np.unwrap(ref[:, 2])[None])
    base = np.stack([
        path - rpath, path / max(rpath, 1e-3),
        speed.mean(-1), speed.max(-1), accel.max(-1), accel.mean(-1),
        np.abs(dyaw).max(-1), np.abs(dyaw).sum(-1),
        dev.mean(-1), dev.max(-1), yaw_dev.max(-1),
        end[:, 0] - rend[0], end[:, 1] - rend[1], end[:, 2] - rend[2],
        np.abs(end[:, 1] - rend[1]),
    ], axis=1)
    ctx = np.array([rseg.sum() / (DT * len(rseg)), np.abs(rdyaw).sum()])
    inter = np.concatenate([base[:, [0, 8, 9]] * ctx[0], base[:, [0, 8, 9]] * ctx[1]], axis=1)
    sq = base[:, [0, 8, 9, 11, 12]] ** 2
    out = np.concatenate([base, inter, sq], axis=1)
    return np.concatenate([out, set_level_features(traj, ref_traj)], axis=1) if set_features else out


def scene_target(scores):
    return pdms(scores["no_at_fault_collisions"], scores["drivable_area_compliance"],
                scores["ego_progress"], scores["time_to_collision_within_bound"], scores["comfort"])


def scene_ref_pdms(ref):
    m = {k: float(v.mean()) for k, v in ref.items()}
    return float(pdms(m["no_at_fault_collisions"], m["drivable_area_compliance"], m["ego_progress"],
                      m["time_to_collision_within_bound"], m["comfort"]))


def fully_safe(scores):
    ok = np.ones(len(scores["ego_progress"]), bool)
    for k in (*SAFETY, "driving_direction_compliance"):
        ok &= scores[k] >= 1 - 1e-6
    return ok


class Ridge:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, x, y):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        z = (x - self.mu) / self.sd
        self.w = np.linalg.solve(z.T @ z + self.lam * len(z) * np.eye(z.shape[1]), z.T @ (y - y.mean()))
        return self

    def predict(self, x):
        return ((x - self.mu) / self.sd) @ self.w

    def predict_scene(self, f, cols=None):
        fc = f - f.mean(0, keepdims=True)
        return self.predict(fc if cols is None else fc[:, cols])


class HGB:
    """Gradient-boosted regression of scene-centred PDMS; raw and centred features both given."""

    def __init__(self, iters=300, lr=0.05, leaves=31, seed=0):
        self.iters, self.lr, self.leaves, self.seed = iters, lr, leaves, seed

    @staticmethod
    def design(f, cols):
        fc = f - f.mean(0, keepdims=True)
        return np.concatenate([f[:, cols], fc[:, cols]], axis=1)

    def fit_scenes(self, meta, cols):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = np.concatenate([self.design(m["feat"], cols) for m in meta])
        y = np.concatenate([m["pdms"] - m["pdms"].mean() for m in meta])
        # no sklearn early stopping: its random row split would leak candidates of one scene
        self.model = HistGradientBoostingRegressor(
            max_iter=self.iters, learning_rate=self.lr, max_leaf_nodes=self.leaves,
            l2_regularization=1.0, early_stopping=False, random_state=self.seed).fit(X, y)
        return self

    def predict_scene(self, f, cols=None):
        return self.model.predict(self.design(f, list(range(f.shape[1])) if cols is None else cols))


class StructHGB:
    """Predict each PDMS component separately and compose them with the PDMS formula.

    PDMS = NC * DAC * (5 EP + 5 TTC + 2 comfort) / 12 is a product of gates and a weighted sum; regressing it
    directly asks one model to learn the interaction. Here one boosted regressor per component predicts the raw
    component (0/1 gates become probabilities) and the composed value ranks the candidates of a scene."""
    COMPONENTS = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress")

    def __init__(self, iters=200, lr=0.05, leaves=31, seed=0):
        self.iters, self.lr, self.leaves, self.seed = iters, lr, leaves, seed

    def fit_scenes(self, meta, cols):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = np.concatenate([HGB.design(m["feat"], cols) for m in meta])
        self.models = {}
        for k in self.COMPONENTS:
            y = np.concatenate([m["metrics"][k] for m in meta])
            self.models[k] = HistGradientBoostingRegressor(
                max_iter=self.iters, learning_rate=self.lr, max_leaf_nodes=self.leaves,
                l2_regularization=1.0, early_stopping=False, random_state=self.seed).fit(X, y)
        return self

    def predict_scene(self, f, cols=None):
        X = HGB.design(f, list(range(f.shape[1])) if cols is None else cols)
        p = {k: np.clip(self.models[k].predict(X), 0.0, 1.0) for k in self.COMPONENTS}
        return (p["no_at_fault_collisions"] * p["drivable_area_compliance"] *
                (5 * p["ego_progress"] + 5 * p["time_to_collision_within_bound"] + 2.0) / 12.0)


GEOM_DIM = 12


def geom_block(geom_root, path):
    g = Path(geom_root) / path.parent.name / path.name
    with np.load(g, allow_pickle=False) as z:
        x = z["geom_features"].astype(np.float64)
    if x.ndim != 2 or x.shape[1] != GEOM_DIM:
        raise ValueError(f"unexpected geom_features shape {x.shape} in {g}")
    x = np.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
    # explicit threshold indicators: PDMS is multiplicative, so failures are step functions
    ind = np.stack([(x[:, 8] > 0).astype(float), (x[:, 10] > 0).astype(float),
                    (x[:, 0] < 0.5).astype(float)], axis=1)
    return np.concatenate([x, ind], axis=1)


def extra_block(root, path):
    """Generic per-candidate feature block: key 'features' (or legacy 'geom_features')."""
    g = Path(root) / path.parent.name / path.name
    with np.load(g, allow_pickle=False) as z:
        x = z["features"] if "features" in z.files else z["geom_features"]
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=1e3, neginf=-1e3)
    if x.ndim != 2:
        raise ValueError(f"unexpected feature shape {x.shape} in {g}")
    return x


class Extra:
    """A named extra feature block. ``shuffle`` takes each scene's block from another scene:
    same dimension and candidate statistics, no scene-specific information (a noise control)."""

    def __init__(self, name, root, shuffle=False):
        self.name, self.root, self.shuffle = name, Path(root), shuffle


def build(paths, geom_root=None, extras=(), set_features=False):
    feats, targets, groups, meta = [], [], [], []
    perm = np.random.default_rng(0).permutation(len(paths))
    for i, p in enumerate(paths):
        traj, ref_traj, scores, ref = load_scene(p)
        f = scene_features(traj, ref_traj, set_features)
        n_traj = f.shape[1]
        spans = {"traj": (0, n_traj)}
        veto = np.zeros(len(f), bool)
        if geom_root is not None:
            gb = geom_block(geom_root, p)
            if len(gb) != len(f):
                raise ValueError(f"candidate count mismatch for {p.name}")
            f = np.concatenate([f, gb], axis=1)
            spans["geom"] = (n_traj, f.shape[1])
            veto |= (gb[:, 8] > 0) | (gb[:, 0] < 0)           # off drivable area / overlapping an agent
        for ex in extras:
            src = paths[perm[i]] if ex.shuffle else p
            eb = extra_block(ex.root, src)
            if len(eb) != len(f):
                raise ValueError(f"candidate count mismatch for {src.name} in extra '{ex.name}'")
            spans[ex.name] = (f.shape[1], f.shape[1] + eb.shape[1])
            if ex.name == "ttc" and not ex.shuffle:
                veto |= eb[:, 0] > 0                           # ttc_violation_frac
            f = np.concatenate([f, eb], axis=1)
        y = scene_target(scores)
        feats.append(f - f.mean(0, keepdims=True))      # within-scene centring
        targets.append(y - y.mean())
        groups.append(np.full(len(y), i))
        meta.append({"token": p.stem, "pdms": y, "ref": scene_ref_pdms(ref), "safe": fully_safe(scores),
                     "ref_unsafe": any(ref[k].min() < 1 - 1e-6 for k in SAFETY), "feat": f,
                     "n_traj": n_traj, "spans": spans, "veto": veto,
                     "metrics": {k: scores[k] for k in (*SAFETY, "ego_progress")},
                     "ref_metrics": {k: float(ref[k].mean()) for k in (*SAFETY, "ego_progress")},
                     "fail": {k: scores[k] < 1 - 1e-6 for k in SAFETY},
                     "ref_fail": {k: bool(ref[k].min() < 1 - 1e-6) for k in SAFETY}})
    return np.concatenate(feats), np.concatenate(targets), np.concatenate(groups), meta


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def boot(delta, rng, n=3000):
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    m = delta[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cluster_boot(delta, logs, rng, n=3000):
    """Bootstrap over driving logs (scenes of one log are resampled together)."""
    delta = np.asarray(delta, dtype=np.float64)
    uniq = sorted(set(logs))
    index = {l: i for i, l in enumerate(uniq)}
    lid = np.array([index[l] for l in logs])
    sums = np.bincount(lid, weights=delta, minlength=len(uniq))
    cnt = np.bincount(lid, minlength=len(uniq)).astype(np.float64)
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    m = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired_report(fs, a, b, token_log, rng):
    """Per-scene gain of set ``b`` minus set ``a`` on the same held-out scenes."""
    ta, tb = fs[a]["heldout"]["per_scene"], fs[b]["heldout"]["per_scene"]
    if ta["token"] != tb["token"]:
        raise ValueError("held-out scene order differs between feature sets")
    d = np.asarray(tb["gain"]) - np.asarray(ta["gain"])
    out = {"mean_diff_x100": float(d.mean() * 100), "scenes": len(d),
           "scene_ci95_x100": [x * 100 for x in boot(d, rng)],
           "share_b_better": float((d > 1e-9).mean()), "share_a_better": float((d < -1e-9).mean())}
    if token_log is not None:
        out["log_cluster_ci95_x100"] = [x * 100 for x in cluster_boot(d, [token_log[t] for t in ta["token"]], rng)]
        out["logs"] = len(set(token_log[t] for t in ta["token"]))
    return out


def auc_pairs(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


KS_CURVE = (2, 4, 8, 16, 32, 64)


def evaluate(model, meta, rng, cols=None):
    veto_pick, veto_frac = [], []
    veto_fail = {k: [] for k in SAFETY}
    veto_metric = {k: [] for k in (*SAFETY, "ego_progress")}
    sel = {k: [] for k in ("selector", "selector_veto", "random", "closest_to_ref", "max_path", "min_path", "oracle")}
    rho, repair = [], {"selector": [], "random": []}
    pick_fail = {k: [] for k in SAFETY}
    rand_fail = {k: [] for k in SAFETY}
    ref_fail = {k: [] for k in SAFETY}
    within_auc = {k: [] for k in SAFETY}
    sub_rng = np.random.default_rng(1234)                 # fixed: K-subsets identical across feature sets
    metric_delta = {k: [] for k in (*SAFETY, "ego_progress")}
    kcurve = {K: {"gain": [], "dac_fail": [], "random_gain": []} for K in KS_CURVE}
    for m in meta:
        f = m["feat"]
        pred = model.predict_scene(f, cols)
        y, ref = m["pdms"], m["ref"]
        for k in SAFETY:
            pick_fail[k].append(float(m["fail"][k][np.argmax(pred)]))
            rand_fail[k].append(float(m["fail"][k].mean()))
            ref_fail[k].append(float(m["ref_fail"][k]))
            if m["fail"][k].any() and (~m["fail"][k]).any():
                within_auc[k].append(auc_pairs(-pred, m["fail"][k].astype(int)))
        top = int(np.argmax(pred))
        for k in metric_delta:
            metric_delta[k].append(float(m["metrics"][k][top] - m["ref_metrics"][k]))
        for K in KS_CURVE:
            if K > len(y):
                continue
            idx = sub_rng.choice(len(y), K, replace=False)
            j = idx[int(np.argmax(pred[idx]))]
            kcurve[K]["gain"].append(float(y[j] - ref))
            kcurve[K]["random_gain"].append(float(y[idx].mean() - ref))
            kcurve[K]["dac_fail"].append(float(m["fail"]["drivable_area_compliance"][j]))
        sel["selector"].append(y[np.argmax(pred)] - ref)
        veto = m["veto"]
        vtop = int(np.argmax(np.where(veto, -np.inf, pred))) if not veto.all() else int(np.argmax(pred))
        sel["selector_veto"].append(y[vtop] - ref)
        veto_pick.append(vtop)
        for k in SAFETY:
            veto_fail[k].append(float(m["fail"][k][vtop]))
        for k in veto_metric:
            veto_metric[k].append(float(m["metrics"][k][vtop] - m["ref_metrics"][k]))
        veto_frac.append(float(veto.mean()))
        sel["random"].append(y.mean() - ref)
        sel["closest_to_ref"].append(y[np.argmin(f[:, 8])] - ref)
        sel["max_path"].append(y[np.argmax(f[:, 0])] - ref)
        sel["min_path"].append(y[np.argmin(f[:, 0])] - ref)
        sel["oracle"].append(y.max() - ref)
        if y.std() > 0:
            rho.append(spearman(pred, y))
        if m["ref_unsafe"]:
            repair["selector"].append(float(m["safe"][np.argmax(pred)]))
            repair["random"].append(float(m["safe"].mean()))
    out = {"scenes": len(meta), "ref_pdms_x100": float(np.mean([m["ref"] for m in meta]) * 100)}
    for k, v in sel.items():
        d = np.asarray(v)
        out[k] = {"gain_vs_ref_x100": float(d.mean() * 100), "ci95_x100": [x * 100 for x in boot(d, rng)]}
    gap = out["oracle"]["gain_vs_ref_x100"] - out["random"]["gain_vs_ref_x100"]
    out["selector_share_of_oracle_gap"] = float(
        (out["selector"]["gain_vs_ref_x100"] - out["random"]["gain_vs_ref_x100"]) / gap) if gap > 0 else None
    out["veto"] = {"mean_fraction_of_candidates_vetoed": float(np.mean(veto_frac)),
                   "failure_rate_of_pick": {k: float(np.mean(v)) for k, v in veto_fail.items()},
                   "pick_metric_delta_vs_ref_x100": {k: float(np.mean(v) * 100) for k, v in veto_metric.items()}}
    out["mean_within_scene_spearman"] = float(np.nanmean(rho)) if rho else None
    out["failure_rate_of_pick"] = {k: float(np.mean(pick_fail[k])) for k in SAFETY}
    out["failure_rate_random"] = {k: float(np.mean(rand_fail[k])) for k in SAFETY}
    out["failure_rate_reference"] = {k: float(np.mean(ref_fail[k])) for k in SAFETY}
    out["within_scene_auc_of_predicted_pdms_vs_failure"] = {
        k: ({"mean": float(np.mean(v)), "scenes": len(v)} if v else None) for k, v in within_auc.items()}
    out["pick_metric_delta_vs_ref_x100"] = {k: float(np.mean(v) * 100) for k, v in metric_delta.items()}
    out["k_curve"] = {str(K): {"n": len(v["gain"]),
                               "selector_gain_x100": float(np.mean(v["gain"]) * 100),
                               "selector_gain_ci95_x100": [x * 100 for x in boot(np.asarray(v["gain"]), rng)],
                               "random_gain_x100": float(np.mean(v["random_gain"]) * 100),
                               "pick_dac_fail_rate": float(np.mean(v["dac_fail"]))}
                      for K, v in kcurve.items() if v["gain"]}
    pick_arr = {k: np.asarray(v) > 0.5 for k, v in pick_fail.items()}
    ref_arr = {k: np.asarray(v) > 0.5 for k, v in ref_fail.items()}
    out["fix_break"] = {k: {"ref_failures": int(ref_arr[k].sum()), "fixed": int((ref_arr[k] & ~pick_arr[k]).sum()),
                            "broken": int((~ref_arr[k] & pick_arr[k]).sum()),
                            "p_fix": float((ref_arr[k] & ~pick_arr[k]).sum() / max(ref_arr[k].sum(), 1)),
                            "p_break": float((~ref_arr[k] & pick_arr[k]).sum() / max((~ref_arr[k]).sum(), 1)),
                            "net_passing_scenes": int((ref_arr[k] & ~pick_arr[k]).sum() - (~ref_arr[k] & pick_arr[k]).sum())}
                        for k in SAFETY}
    out["per_scene"] = {"token": [m["token"] for m in meta], "gain": [float(v) for v in sel["selector"]]}
    out["ref_unsafe_scenes"] = len(repair["selector"])
    if repair["selector"]:
        out["p_pick_fully_safe_when_ref_unsafe"] = {"selector": float(np.mean(repair["selector"])),
                                                    "random": float(np.mean(repair["random"]))}
    return out


def split_paths(root, manifest):
    train_tokens = {r["token"] for r in manifest["train"]}
    train, held = [], []
    for split in ("train", "val", "test"):
        for p in sorted((root / split).glob("*.npz")):
            (train if p.stem in train_tokens else held).append(p)
    if not train or not held:
        raise ValueError("empty train or held-out set")
    return train, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--geom-root", type=Path, default=None, help="dir with geom_features npz per split")
    ap.add_argument("--hgb", action="store_true", help="also fit gradient boosting (needs scikit-learn)")
    ap.add_argument("--struct", action="store_true", help="also fit the component-wise structured scorer (needs scikit-learn)")
    ap.add_argument("--set-features", action="store_true", help="add features that compare a candidate with the others of its scene")
    ap.add_argument("--hgb-sets", default="traj+geom", help="comma list of feature sets to run HGB on")
    ap.add_argument("--hgb-iters", default="100,300", help="candidate iteration counts, chosen on the inner validation scenes")
    ap.add_argument("--extra-root", action="append", default=[], metavar="NAME=DIR",
                    help="extra per-candidate feature block (npz key 'features'); repeatable")
    ap.add_argument("--shuffle-control", action="append", default=[], metavar="NAME",
                    help="also run NAME with its blocks taken from other scenes (noise control)")
    ap.add_argument("--metric-cache-dirs", type=Path, nargs="*", default=[],
                    help="NAVSIM metric caches (<cache>/<log>/<type>/<token>/): enables log-cluster intervals")
    ap.add_argument("--max-train-scenes", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    manifest = json.loads(args.manifest.read_text())
    train, held = split_paths(args.root, manifest)
    if args.max_train_scenes:
        train = [train[i] for i in sorted(rng.choice(len(train), min(args.max_train_scenes, len(train)), replace=False))]
    print(f"train scenes {len(train)}  held-out scenes {len(held)}", flush=True)
    extras = []
    for spec in args.extra_root:
        name, _, root = spec.partition("=")
        if not root:
            raise SystemExit(f"--extra-root expects NAME=DIR, got {spec!r}")
        extras.append(Extra(name, root))
    for name in args.shuffle_control:
        match = [e for e in extras if e.name == name]
        if not match:
            raise SystemExit(f"--shuffle-control {name!r} has no --extra-root")
        extras.append(Extra(name + "_shuf", match[0].root, shuffle=True))
    x, y, g, tmeta = build(train, args.geom_root, extras, args.set_features)
    n_traj = tmeta[0]["n_traj"]
    spans = tmeta[0]["spans"]

    def cols_of(*names):
        return [c for n in names for c in range(*spans[n])]

    sets = {"traj": cols_of("traj")}
    if args.geom_root is not None:
        sets["geom"] = cols_of("geom")
        sets["traj+geom"] = cols_of("traj", "geom")
    for e in extras:
        if args.geom_root is not None:
            sets[f"traj+geom+{e.name}"] = cols_of("traj", "geom", e.name)
        if e.name != "ttc":
            sets[f"traj+{e.name}"] = cols_of("traj", e.name)
    real = [e.name for e in extras if not e.shuffle]
    if args.geom_root is not None and len(real) > 1:
        sets["traj+geom+" + "+".join(real)] = cols_of("traj", "geom", *real)
    order = rng.permutation(len(train))
    val_ids = sorted(order[: len(train) // 5].tolist())          # 20% of training scenes pick the penalty
    mask = np.array([gi in set(val_ids) for gi in g])
    _, _, _, hmeta = build(held, args.geom_root, extras, args.set_features)
    report = {"train_scenes": len(train), "heldout_scenes": len(held), "feature_sets": {}}
    for name, cols in sets.items():
        best, tried = None, {}
        for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            model = Ridge(lam).fit(x[~mask][:, cols], y[~mask])
            sc = evaluate(model, [tmeta[i] for i in val_ids], rng, cols)["selector"]["gain_vs_ref_x100"]
            tried[str(lam)] = sc
            if best is None or sc > best[0]:
                best = (sc, lam)
        model = Ridge(best[1]).fit(x[:, cols], y)
        report["feature_sets"][name] = {"chosen_lambda": best[1], "lambda_validation_gain_x100": tried,
                                        "heldout": evaluate(model, hmeta, rng, cols),
                                        "train_fit": evaluate(model, tmeta, rng, cols)}
        h = report["feature_sets"][name]["heldout"]
        print(f"[{name}] heldout selector gain {h['selector']['gain_vs_ref_x100']:.2f} "
              f"(random {h['random']['gain_vs_ref_x100']:.2f}, oracle {h['oracle']['gain_vs_ref_x100']:.2f}) "
              f"veto-gain {h['selector_veto']['gain_vs_ref_x100']:.2f} "
              f"share_of_gap {h['selector_share_of_oracle_gap']}", flush=True)
    if args.hgb or args.struct:
        iter_list = [int(v) for v in args.hgb_iters.split(",")]
        val_set = set(val_ids)
        tr_meta = [tmeta[i] for i in range(len(train)) if i not in val_set]
        va_meta = [tmeta[i] for i in val_ids]
        families = ([("hgb", HGB)] if args.hgb else []) + ([("struct", StructHGB)] if args.struct else [])
        for name in [v for v in args.hgb_sets.split(",") if v in sets]:
            cols = sets[name]
            for fam_name, Model in families:
                best, tried = None, {}
                for iters in iter_list:
                    m = Model(iters, seed=args.seed).fit_scenes(tr_meta, cols)
                    sc = evaluate(m, va_meta, rng, cols)["selector"]["gain_vs_ref_x100"]
                    tried[str(iters)] = sc
                    print(f"[{name}|{fam_name}] iters {iters} inner-val gain {sc:.3f}", flush=True)
                    if best is None or sc > best[0]:
                        best = (sc, iters)
                model = Model(best[1], seed=args.seed).fit_scenes(tmeta, cols)
                report["feature_sets"][f"{name}|{fam_name}"] = {
                    "chosen_iters": best[1], "iters_validation_gain_x100": tried,
                    "heldout": evaluate(model, hmeta, rng, cols), "train_fit": evaluate(model, tmeta, rng, cols)}
                h = report["feature_sets"][f"{name}|{fam_name}"]["heldout"]
                print(f"[{name}|{fam_name}] heldout selector gain {h['selector']['gain_vs_ref_x100']:.2f} "
                      f"CI {[round(x, 2) for x in h['selector']['ci95_x100']]} veto-gain {h['selector_veto']['gain_vs_ref_x100']:.2f} "
                      f"(random {h['random']['gain_vs_ref_x100']:.2f}, oracle {h['oracle']['gain_vs_ref_x100']:.2f})", flush=True)
    token_log = None
    if args.metric_cache_dirs:
        from build_log_disjoint_manifest import token_to_log
        token_log = token_to_log(args.metric_cache_dirs)
        missing = [m["token"] for m in hmeta if m["token"] not in token_log]
        if missing:
            raise SystemExit(f"{len(missing)} held-out scenes have no metric cache, e.g. {missing[0]}")
        for name, v in report["feature_sets"].items():
            h = v["heldout"]
            gains = np.asarray(h["per_scene"]["gain"])
            h["selector"]["log_cluster_ci95_x100"] = [x * 100 for x in cluster_boot(
                gains, [token_log[t] for t in h["per_scene"]["token"]], rng)]
    fs = report["feature_sets"]
    paired = {}
    for model in ("", "|hgb"):
        base = "traj+geom" + model
        if base not in fs:
            continue
        for name in fs:
            if name != base and name.endswith(model) and (model or "|" not in name) and name.startswith("traj+geom"):
                paired[f"{name} minus {base}"] = paired_report(fs, base, name, token_log, rng)
        wm, shuf = "traj+geom+wm" + model, "traj+geom+wm_shuf" + model
        if wm in fs and shuf in fs:
            paired[f"{wm} minus {shuf}"] = paired_report(fs, shuf, wm, token_log, rng)
    for name in fs:
        if name.endswith("|struct") and name[:-len("|struct")] + "|hgb" in fs:
            paired[f"{name} minus {name[:-len('|struct')]}|hgb"] = paired_report(fs, name[:-len("|struct")] + "|hgb", name, token_log, rng)
    report["paired"] = paired
    for k, v in paired.items():
        ci = v.get("log_cluster_ci95_x100", v["scene_ci95_x100"])
        print(f"[paired] {k}: {v['mean_diff_x100']:+.2f} {'log' if 'log_cluster_ci95_x100' in v else 'scene'}-CI [{ci[0]:.2f}, {ci[1]:.2f}]", flush=True)
    report["heldout"] = report["feature_sets"]["traj"]["heldout"]       # kept for older readers
    report["train_fit"] = report["feature_sets"]["traj"]["train_fit"]
    report["chosen_lambda"] = report["feature_sets"]["traj"]["chosen_lambda"]
    report["note"] = "Scene-centred ridge on PDMS; held-out logs are disjoint from training logs."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {"heldout_gain_x100": v["heldout"]["selector"]["gain_vs_ref_x100"], "veto_gain_x100": v["heldout"]["selector_veto"]["gain_vs_ref_x100"], "veto_pick_metric_delta_x100": v["heldout"]["veto"]["pick_metric_delta_vs_ref_x100"], "fail_rate_pick": v["heldout"]["failure_rate_of_pick"], "fail_rate_random": v["heldout"]["failure_rate_random"], "p_safe_when_ref_unsafe": v["heldout"].get("p_pick_fully_safe_when_ref_unsafe"), "gain_ci95_x100": v["heldout"]["selector"]["ci95_x100"], "pick_metric_delta_vs_ref_x100": v["heldout"]["pick_metric_delta_vs_ref_x100"], "failure_rate_reference": v["heldout"]["failure_rate_reference"], "k_curve": {K: [round(c["selector_gain_x100"], 2), round(c["random_gain_x100"], 2), round(c["pick_dac_fail_rate"], 4)] for K, c in v["heldout"]["k_curve"].items()}} for k, v in report["feature_sets"].items()}, indent=1))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
