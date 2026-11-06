"""Table of the selector's gain against the independent SFT evaluation as the perception errors grow."""
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    rows = [("0 (clean)", "clean-trained", root / "selector_navtest_v3.json")]
    for level in (1, 2, 3):
        rows.append((str(level), "clean-trained", root / f"selector_noise{level}_clean.json"))
        rows.append((str(level), "matched", root / f"selector_noise{level}_matched.json"))
    print(f"{'level':10s} {'regime':14s} {'ridge vs SFT':>24s} {'hgb vs SFT':>24s} | hgb TTC fix/new  DAC fix/new")
    for level, regime, path in rows:
        if not path.exists():
            print(f"{level:10s} {regime:14s} (missing {path.name})")
            continue
        m = json.loads(path.read_text())["models"]
        cell = lambda d: "%+.2f [%+.2f,%+.2f]" % (d["detail"]["vs_independent_sft"]["gain_x100"], *d["detail"]["vs_independent_sft"]["log_ci95_x100"])
        fb = m["hgb"]["detail"]["vs_independent_sft"]["fix_break"]
        print(f"{level:10s} {regime:14s} {cell(m['ridge']):>24s} {cell(m['hgb']):>24s} | "
              f"{fb['time_to_collision_within_bound']['fixed']}/{fb['time_to_collision_within_bound']['broken']}  "
              f"{fb['drivable_area_compliance']['fixed']}/{fb['drivable_area_compliance']['broken']}")


if __name__ == "__main__":
    main()
