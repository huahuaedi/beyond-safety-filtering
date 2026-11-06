"""Optional absolute pass gate on positive reference-relative advantages."""
import numpy as np
KEYS=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')


def guard_positive_advantage(advantage, scores, *, enabled=False, ttc_threshold=0.5):
    """Cap positive weights at zero for executed failures; never recenter.

    Scores are real NAVSIM executed outcomes. Keep negative learning signals,
    safe candidates and reference-relative feasibility untouched. This is a
    reward preference, not a guarantee about the updated policy. In all-fail
    groups it removes progress-based positive reinforcement, which may slow
    recovery; report the activation and evaluate that tradeoff.
    """
    adv=np.asarray(advantage)
    if adv.ndim!=1 or not len(adv) or not np.issubdtype(adv.dtype,np.floating) or not np.isfinite(adv).all():
        raise ValueError('advantage must be a finite nonempty floating vector')
    if not np.isfinite(ttc_threshold) or not 0<=ttc_threshold<=1: raise ValueError('Invalid TTC threshold')
    arrays=[np.asarray(scores[k],dtype=float) for k in KEYS]
    if any(a.shape!=adv.shape or not np.isfinite(a).all() or (a<0).any() or (a>1).any() for a in arrays):
        raise ValueError('Invalid NAVSIM score vectors')
    nc,dac,ttc=arrays
    safe=(nc>=1.0)&(dac>=1.0)&(ttc>=ttc_threshold)
    blocked=(adv>0)&~safe
    result=adv.copy()
    if enabled: result[blocked]=0
    return result, {'positive_unsafe_candidates':int(blocked.sum()),
                    'changed_candidates':int(blocked.sum()) if enabled else 0,
                    'all_candidates_fail_absolute_gate':bool((~safe).all()),
                    'removed_positive_mass':float(adv[blocked].sum()) if enabled else 0.0}
