"""Print, per guidance strength, how the candidates and the frozen selector change (inputs: selector_dev.json of each run)."""
import json
import sys
from pathlib import Path


def main():
    rows = []
    for spec in sys.argv[1:]:
        label, _, path = spec.partition("=")
        r = json.loads(Path(path).read_text())
        rows.append((label, r))
    print(f"{'run':10s} {'random':>7s} {'oracle':>7s} | {'ridge sel':>22s} | {'hgb sel':>22s} | fail rate of a random candidate (NC DAC TTC)")
    for label, r in rows:
        m = r["models"]
        any_m = next(iter(m.values()))
        f = lambda x: f"{x['gain_vs_reference_probes_x100']:+.2f} [{x['gain_log_ci95_x100'][0]:+.2f},{x['gain_log_ci95_x100'][1]:+.2f}]"
        fr = any_m["failure_rate_random"]
        print(f"{label:10s} {any_m['random_gain_x100']:+7.2f} {any_m['oracle_gain_x100']:+7.2f} | {f(m['ridge']):>22s} | {f(m['hgb']):>22s} | "
              f"{fr['no_at_fault_collisions']:.4f} {fr['drivable_area_compliance']:.4f} {fr['time_to_collision_within_bound']:.4f}")
    base = rows[0][1]
    for label, r in rows[1:]:
        print(f"{label}: components of the hgb pick vs probes", {k: round(v['delta_x100'], 2) for k, v in r['models']['hgb']['detail']['component_vs_reference_probes'].items()})


if __name__ == "__main__":
    main()
