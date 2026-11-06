"""Single source of truth for the evaluated arms (used by job.sh, prepare.py and report.py).

Primary arms are the five checkpoints the original result selected (per project record, on a
150-scene validation cohort). Everything else is descriptive only and never used to choose a
checkpoint: other saved steps of the same runs, and two SFT re-evaluations with different
evaluation seeds to measure evaluation noise.
"""
import sys

SFT_REL = "Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
SELECTED = {20: 25, 21: 25, 22: 50, 23: 10, 24: 40}
STEPS = (10, 25, 40, 50)


def arms():
    out = [("sft", SFT_REL, 0)]
    out += [(f"sel_seed{s}_step{st:06d}", f"direct_rl_seed{s}/step_{st:06d}", 0) for s, st in SELECTED.items()]
    out += [("sft_evalseed1", SFT_REL, 1), ("sft_evalseed2", SFT_REL, 2)]
    for s in SELECTED:
        for st in STEPS:
            if st != SELECTED[s]:
                out.append((f"desc_seed{s}_step{st:06d}", f"direct_rl_seed{s}/step_{st:06d}", 0))
    names = [a[0] for a in out]
    if len(set(names)) != len(names):
        raise ValueError("duplicate arm names")
    return out


def selected_names():
    return [f"sel_seed{s}_step{st:06d}" for s, st in SELECTED.items()]


def step_arm(seed, step):
    prefix = "sel" if SELECTED[seed] == step else "desc"
    return f"{prefix}_seed{seed}_step{step:06d}"


if __name__ == "__main__":
    # usage: python3 arms.py <model_root> <array_index>   ->  name|checkpoint_path|eval_seed
    root, idx = sys.argv[1], int(sys.argv[2])
    name, rel, seed = arms()[idx]
    print(f"{name}|{root}/{rel}|{seed}")
