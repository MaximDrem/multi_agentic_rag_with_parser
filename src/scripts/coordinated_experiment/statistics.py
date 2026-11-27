import math
from collections.abc import Mapping

import numpy as np

DEFAULT_SEED = 20261127


def _unique_keys(group_keys):
    keys = list(group_keys)
    if not keys:
        raise ValueError("At least one query group is required")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate query group keys")
    return keys


def make_bootstrap_indices(group_keys, n_samples=10000, seed=DEFAULT_SEED):
    keys = _unique_keys(group_keys)
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(keys), size=(n_samples, len(keys)))


def _metric_values(rows, group_keys, metric, side):
    values = []
    missing = [key for key in group_keys if key not in rows]
    if missing:
        raise ValueError(f"Missing {side} groups: {missing[:5]}")
    for key in group_keys:
        row = rows[key]
        value = row[metric] if isinstance(row, Mapping) else row
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"Non-numeric {side} metric for group {key}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {side} metric for group {key}")
        values.append(value)
    return np.asarray(values, dtype=float)


def _paired_keys(left_by_group, right_by_group, group_keys):
    if group_keys is not None:
        return _unique_keys(group_keys)
    left_keys = set(left_by_group)
    right_keys = set(right_by_group)
    if left_keys != right_keys:
        missing_left = sorted(right_keys - left_keys, key=str)
        missing_right = sorted(left_keys - right_keys, key=str)
        raise ValueError(
            f"Unpaired groups; missing left={missing_left[:5]}, missing right={missing_right[:5]}"
        )
    return _unique_keys(sorted(left_keys, key=str))


def _validated_bootstrap_indices(bootstrap_indices, n_groups):
    indices = np.asarray(bootstrap_indices)
    if indices.ndim != 2 or indices.shape[0] == 0 or indices.shape[1] != n_groups:
        raise ValueError(
            f"bootstrap_indices must have shape (n_samples, {n_groups}) with n_samples > 0"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        if not np.all(np.isfinite(indices)) or not np.all(indices == np.floor(indices)):
            raise ValueError("bootstrap_indices must contain integers")
        indices = indices.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= n_groups):
        raise ValueError("bootstrap_indices contain an out-of-range group index")
    return indices


def paired_contrast(
    left_by_group,
    right_by_group,
    metric="recall@10",
    group_keys=None,
    bootstrap_indices=None,
    n_bootstrap=10000,
    n_randomization=10000,
    seed=DEFAULT_SEED,
):
    group_keys = _paired_keys(left_by_group, right_by_group, group_keys)
    left = _metric_values(left_by_group, group_keys, metric, "left")
    right = _metric_values(right_by_group, group_keys, metric, "right")
    paired_deltas = left - right
    observed = float(np.mean(paired_deltas))

    if bootstrap_indices is None:
        bootstrap_indices = make_bootstrap_indices(group_keys, n_samples=n_bootstrap, seed=seed)
    else:
        bootstrap_indices = _validated_bootstrap_indices(bootstrap_indices, len(group_keys))
    bootstrap_statistics = np.mean(paired_deltas[bootstrap_indices], axis=1)
    ci_low, ci_high = np.quantile(bootstrap_statistics, [0.025, 0.975])

    if not isinstance(n_randomization, int) or isinstance(n_randomization, bool) or n_randomization <= 0:
        raise ValueError("n_randomization must be a positive integer")
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_randomization, len(group_keys)), dtype=np.int8) * 2 - 1
    randomized_statistics = np.mean(signs * paired_deltas, axis=1)
    extreme = int(np.count_nonzero(np.abs(randomized_statistics) >= abs(observed)))
    p_randomization = (1 + extreme) / (n_randomization + 1)

    return {
        "delta": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_randomization": float(p_randomization),
        "observed_statistic": observed,
        "n_groups": len(group_keys),
    }


def apply_holm(contrast_rows, primary_families):
    rows = [dict(row) for row in contrast_rows]
    primary_by_family = {
        family: set(contrast_ids) for family, contrast_ids in primary_families.items()
    }
    grouped = {}
    for index, row in enumerate(rows):
        row["p_holm"] = None
        family = row.get("family")
        contrast_id = row.get("contrast_id")
        if contrast_id in primary_by_family.get(family, set()):
            grouped.setdefault(family, []).append(index)

    for indices in grouped.values():
        p_values = {}
        for index in indices:
            p_value = rows[index]["p_randomization"]
            if p_value is None or not isinstance(p_value, (int, float, np.number)):
                raise ValueError("Primary contrasts require numeric p_randomization")
            p_value = float(p_value)
            if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
                raise ValueError("p_randomization must be finite and between 0 and 1")
            p_values[index] = p_value

        ordered = sorted(indices, key=lambda index: (p_values[index], index))
        family_size = len(ordered)
        running_max = 0.0
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (family_size - rank) * p_values[index])
            running_max = max(running_max, adjusted)
            rows[index]["p_holm"] = running_max
    return rows


def classify_effect(row, margin_pp=1.0):
    if not isinstance(margin_pp, (int, float)) or isinstance(margin_pp, bool):
        raise ValueError("margin_pp must be numeric")
    if not math.isfinite(margin_pp) or margin_pp < 0:
        raise ValueError("margin_pp must be finite and non-negative")
    try:
        delta_pp = float(row["delta"]) * 100.0
        ci95_low_pp = float(row["ci95_low"]) * 100.0
        ci95_high_pp = float(row["ci95_high"]) * 100.0
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Effect row requires numeric delta, ci95_low, and ci95_high") from exc
    if not all(math.isfinite(value) for value in (delta_pp, ci95_low_pp, ci95_high_pp)):
        raise ValueError("Effect values must be finite")
    if ci95_high_pp < 0.0:
        return "HARM"
    if delta_pp >= margin_pp and ci95_low_pp > 0.0:
        return "POS"
    if ci95_high_pp < margin_pp:
        return "BELOW_MARGIN"
    return "UNCERTAIN"
