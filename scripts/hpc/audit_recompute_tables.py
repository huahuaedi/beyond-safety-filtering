"""Read-only traceability audit: hash the 12 per-scene score tables (+SFT) and recompute the paper's table entries.
Usage: python scripts/hpc/audit_recompute_tables.py $M [--out audit.json]
For every run: sha256 of scores.csv, number of scenes, mean PDMS, difference to SFT, NC/DAC/TTC repaired / newly failed, net events, progress change.
Writes nothing except the optional --out file."""
import argparse, csv, hashlib, json
from pathlib import Path
CRIT = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")
def read(p):
    rows = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["token"] in ("", "average") or r.get("valid", "True") != "True": continue
            rows[r["token"]] = {c: float(r[c]) for c in CRIT + ("score", "ego_progress")}
    return rows
ap = argparse.ArgumentParser(); ap.add_argument("merged", type=Path); ap.add_argument("--out", type=Path); a = ap.parse_args()
base = read(a.merged / "sft" / "scores.csv")
res = {"sft": {"sha256": hashlib.sha256((a.merged / "sft" / "scores.csv").read_bytes()).hexdigest(), "scenes": len(base),
               "pdms": 100 * sum(r["score"] for r in base.values()) / len(base)}}
print(f"{'run':18s} {'scenes':>6s} {'PDMS':>8s} {'dPDMS':>7s} {'dProg':>7s}  NC r/n   DAC r/n   TTC r/n   net  sha256[:12]")
for rule in ("pdmspref", "nearestpref", "marginpref"):
    for s in (20, 21, 22, 23):
        name = f"{rule}_s{s}"; p = a.merged / name / "scores.csv"
        if not p.exists(): print(f"{name:18s} missing"); continue
        arm = read(p); toks = sorted(set(arm) & set(base)); n = len(toks)
        d = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "scenes": n,
             "pdms": 100 * sum(arm[t]["score"] for t in toks) / n,
             "dpdms": 100 * sum(arm[t]["score"] - base[t]["score"] for t in toks) / n,
             "dprogress": 100 * sum(arm[t]["ego_progress"] - base[t]["ego_progress"] for t in toks) / n}
        net = 0; cells = []
        for c in CRIT:
            rep = sum(1 for t in toks if base[t][c] < 1 - 1e-6 and arm[t][c] >= 1 - 1e-6)
            new = sum(1 for t in toks if base[t][c] >= 1 - 1e-6 and arm[t][c] < 1 - 1e-6)
            d[c] = [rep, new]; net += rep - new; cells.append(f"{rep}/{new}")
        d["net"] = net; res[name] = d
        print(f"{name:18s} {n:6d} {d['pdms']:8.3f} {d['dpdms']:+7.3f} {d['dprogress']:+7.3f}  " + "  ".join(cells) + f"  {net:+5d}  {d['sha256'][:12]}")
if a.out: a.out.write_text(json.dumps(res, indent=2) + "\n")
