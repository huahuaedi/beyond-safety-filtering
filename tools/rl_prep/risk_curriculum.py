"""Pure-Python helpers for learning-aware risk-memory sampling."""

import math


RISK_TYPES = ("unsafe", "ttc_failure", "dac_failure", "nc_failure")


def parse_source_mix(spec):
    """Parse ``manifest:0.5,coverage:0.25,online:0.25`` into probabilities."""
    weights = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            name, value = item.split(":", 1)
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid risk curriculum source mix item: {item!r}") from exc
        name = name.strip().lower()
        if name not in {"manifest", "coverage", "online"}:
            raise ValueError(f"unknown risk curriculum source: {name!r}")
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"risk curriculum weight must be finite and >= 0: {item!r}")
        weights[name] = weights.get(name, 0.0) + value
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("risk curriculum source weights must sum to > 0")
    return {name: value / total for name, value in weights.items() if value > 0}


def memory_sources(memory):
    sources = memory.get("sources")
    if sources is not None:
        return set(sources)
    return set(str(memory.get("source", "online")).split("+"))


def _choose_weighted(rng, items, weights):
    if not items:
        raise ValueError("cannot sample from an empty risk-memory pool")
    if not any(weight > 0 for weight in weights):
        return rng.choice(items)
    return rng.choices(items, weights=weights, k=1)[0]


def tempered_priority_weights(priorities, power):
    """Return positive priority weights; power 0 is uniform and 1 is legacy."""
    if power < 0 or not math.isfinite(power):
        raise ValueError("priority power must be finite and >= 0")
    return [max(float(priority), 1e-8) ** power for priority in priorities]


def select_curriculum_memory(
    risk_buffer,
    rng,
    source_mix,
    *,
    type_balance=True,
    redundancy_coef=0.25,
    priority_power=1.0,
):
    """Select a memory with source quotas, type balance, and replay novelty.

    Source selection is performed first. If a requested source has no eligible
    memories, its probability is redistributed over the sources that do.
    Candidate weights then combine risk priority, inverse-frequency risk-type
    coverage, and a decreasing function of prior replay count.
    """
    tokens = sorted(risk_buffer)
    if not tokens:
        raise ValueError("risk buffer is empty")

    available = {
        source: [token for token in tokens if source in memory_sources(risk_buffer[token])]
        for source in source_mix
    }
    available = {source: pool for source, pool in available.items() if pool}
    if not available:
        available = {"fallback": tokens}
        source_weights = [1.0]
    else:
        source_weights = [source_mix[source] for source in available]
    selected_source = _choose_weighted(rng, list(available), source_weights)
    pool = available[selected_source]

    totals = {risk: 0.0 for risk in RISK_TYPES}
    if type_balance:
        for token in tokens:
            counts = risk_buffer[token].get("label_counts", {})
            for risk in RISK_TYPES:
                totals[risk] += max(0.0, float(counts.get(risk, 0)))

    weights = []
    for token in pool:
        memory = risk_buffer[token]
        priority = max(float(memory.get("priority", 0.0)), 1e-8) ** priority_power
        type_factor = 1.0
        counts = memory.get("label_counts", {})
        represented = [risk for risk in RISK_TYPES if float(counts.get(risk, 0)) > 0]
        if type_balance and represented:
            inverse = [1.0 / max(totals[risk], 1.0) for risk in represented]
            mean_inverse = sum(inverse) / len(inverse)
            nonzero_totals = [value for value in totals.values() if value > 0]
            reference_inverse = 1.0 / max(sum(nonzero_totals) / len(nonzero_totals), 1.0)
            type_factor = min(3.0, max(0.5, mean_inverse / reference_inverse))
        replay_count = max(0, int(memory.get("replay_count", 0)))
        novelty = 1.0 / (1.0 + redundancy_coef * replay_count)
        weights.append(priority * type_factor * novelty)

    return _choose_weighted(rng, pool, weights), selected_source
