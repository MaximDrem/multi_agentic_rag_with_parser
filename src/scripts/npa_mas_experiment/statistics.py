from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import METRIC_NAMES, average_metrics
from .schema import CONTRASTS, DEFAULT_SEED, PRACTICAL_CONTRASTS, QUERY_LABELS


def _rng(seed: int, stream: str) -> np.random.Generator:
    digest = hashlib.sha256(stream.encode("utf-8")).digest()
    entropy = [int(seed), int.from_bytes(digest[:4], "big"), int.from_bytes(digest[4:8], "big")]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _deltas(values: Any) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Paired analysis requires a non-empty one-dimensional sample")
    if not np.all(np.isfinite(array)):
        raise ValueError("Paired analysis requires finite values")
    return array


def paired_bootstrap(
    differences: Iterable[float] | np.ndarray,
    *,
    iterations: int = 10000,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_SEED,
    stream: str = "paired_bootstrap",
) -> dict[str, Any]:
    delta = _deltas(differences)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    rng = _rng(seed, stream)
    estimates = np.empty(iterations, dtype=np.float64)
    block_size = max(1, min(iterations, 1_000_000 // delta.size))
    offset = 0
    while offset < iterations:
        size = min(block_size, iterations - offset)
        indices = rng.integers(0, delta.size, size=(size, delta.size))
        estimates[offset : offset + size] = delta[indices].mean(axis=1)
        offset += size
    alpha = 1.0 - confidence_level
    return {
        "n_groups": int(delta.size),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci_low": float(np.quantile(estimates, alpha / 2)),
        "ci_high": float(np.quantile(estimates, 1 - alpha / 2)),
        "bootstrap_iterations": int(iterations),
        "confidence_level": float(confidence_level),
    }


def sign_flip_randomization(
    differences: Iterable[float] | np.ndarray,
    *,
    iterations: int = 10000,
    seed: int = DEFAULT_SEED,
    stream: str = "sign_flip_randomization",
) -> dict[str, Any]:
    delta = _deltas(differences)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = _rng(seed, stream)
    observed = abs(float(delta.mean()))
    estimates = np.empty(iterations, dtype=np.float64)
    block_size = max(1, min(iterations, 1_000_000 // delta.size))
    offset = 0
    while offset < iterations:
        size = min(block_size, iterations - offset)
        signs = rng.integers(0, 2, size=(size, delta.size), dtype=np.int8) * 2 - 1
        estimates[offset : offset + size] = np.abs((delta * signs).mean(axis=1))
        offset += size
    extreme = int(np.count_nonzero(estimates >= observed - 1e-15))
    return {
        "p_randomization": (extreme + 1) / (iterations + 1),
        "randomization_iterations": int(iterations),
    }


def _keys_and_values(
    left: Mapping[str, float],
    right: Mapping[str, float],
    groups: Iterable[str] | None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if groups is None:
        if set(left) != set(right):
            raise ValueError("Paired inputs must contain identical query groups")
        keys = sorted(left)
    else:
        keys = sorted(set(groups))
        missing = [key for key in keys if key not in left or key not in right]
        if missing:
            raise ValueError(f"Paired inputs are missing groups: {missing[:5]}")
    if not keys:
        raise ValueError("Paired analysis requires at least one group")
    return keys, np.asarray([left[key] for key in keys], dtype=np.float64), np.asarray([right[key] for key in keys], dtype=np.float64)


def _fallback_flags(values: Any, keys: Sequence[str]) -> dict[str, bool]:
    if values is None:
        return {key: False for key in keys}
    if isinstance(values, Mapping) and "all_groups" in values:
        values = {row["query_group"]: row for row in values["all_groups"]}
    if not isinstance(values, Mapping):
        values = {row["query_group"]: row for row in values}
    result = {}
    for key in keys:
        value = values.get(key, False)
        if isinstance(value, Mapping):
            status = str(value.get("status", value.get("group_status", "success")))
            result[key] = bool(value.get("fallback_used")) or bool(value.get("strict_failed")) or status in {"fallback", "failed", "error"}
        else:
            result[key] = bool(value)
    return result


def paired_test(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    contrast: str = "contrast",
    left_variant: str = "left",
    right_variant: str = "right",
    groups: Iterable[str] | None = None,
    bootstrap_iterations: int = 10000,
    randomization_iterations: int = 10000,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_SEED,
    fallback_left: Any = None,
    fallback_right: Any = None,
) -> dict[str, Any]:
    keys, left_values, right_values = _keys_and_values(left, right, groups)
    differences = left_values - right_values
    bootstrap = paired_bootstrap(
        differences,
        iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        seed=seed,
        stream=f"{contrast}:bootstrap",
    )
    randomization = sign_flip_randomization(
        differences,
        iterations=randomization_iterations,
        seed=seed,
        stream=f"{contrast}:randomization",
    )
    left_fallbacks = _fallback_flags(fallback_left, keys)
    right_fallbacks = _fallback_flags(fallback_right, keys)
    return {
        "contrast": contrast,
        "left_variant": left_variant,
        "right_variant": right_variant,
        **bootstrap,
        **randomization,
        "mean_left": float(left_values.mean()),
        "mean_right": float(right_values.mean()),
        "wins": int(np.count_nonzero(differences > 0)),
        "losses": int(np.count_nonzero(differences < 0)),
        "ties": int(np.count_nonzero(differences == 0)),
        "fallback_groups_left": sum(left_fallbacks.values()),
        "fallback_groups_right": sum(right_fallbacks.values()),
        "seed": int(seed),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, max(0.0, (total - rank) * float(p_values[key])))
        running = max(running, value)
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


def metric_by_group(score: Any, metric: str = "recall@10") -> dict[str, float]:
    if isinstance(score, Mapping) and "per_group" in score:
        rows = score["per_group"]
        return {str(row["query_group"]): float(row[metric]) for row in rows}
    if isinstance(score, Mapping):
        result = {}
        for group, value in score.items():
            if isinstance(value, Mapping):
                result[str(group)] = float(value[metric])
            else:
                result[str(group)] = float(value)
        return result
    return {str(row["query_group"]): float(row[metric]) for row in score}


def contrast_table(
    scores: Mapping[str, Any],
    *,
    contrasts: Mapping[str, tuple[str, str]] = CONTRASTS,
    metric: str = "recall@10",
    bootstrap_iterations: int = 10000,
    randomization_iterations: int = 10000,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_SEED,
    fallbacks: Mapping[str, Any] | None = None,
    groups: Iterable[str] | None = None,
    family: str = "confirmatory",
) -> list[dict[str, Any]]:
    rows = []
    for contrast, (left_variant, right_variant) in contrasts.items():
        rows.append(
            paired_test(
                metric_by_group(scores[left_variant], metric),
                metric_by_group(scores[right_variant], metric),
                contrast=contrast,
                left_variant=left_variant,
                right_variant=right_variant,
                groups=groups,
                bootstrap_iterations=bootstrap_iterations,
                randomization_iterations=randomization_iterations,
                confidence_level=confidence_level,
                seed=seed,
                fallback_left=(fallbacks or {}).get(left_variant, scores[left_variant]),
                fallback_right=(fallbacks or {}).get(right_variant, scores[right_variant]),
            )
        )
    adjusted = holm_adjust({row["contrast"]: row["p_randomization"] for row in rows})
    for row in rows:
        row["p_holm"] = adjusted[row["contrast"]]
        row["metric"] = metric
        row["family"] = family
    return rows


def confirmatory_contrasts(
    scores: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    statistics = dict((config or {}).get("statistics", {}))
    options = {
        "metric": statistics.get("primary_metric", "recall@10"),
        "bootstrap_iterations": int(statistics.get("bootstrap_iterations", 10000)),
        "randomization_iterations": int(statistics.get("randomization_iterations", 10000)),
        "confidence_level": float(statistics.get("confidence_level", 0.95)),
        "seed": int((config or {}).get("seed", DEFAULT_SEED)),
        **kwargs,
    }
    return contrast_table(scores, contrasts=CONTRASTS, family="confirmatory", **options)


def practical_comparisons(
    scores: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    statistics = dict((config or {}).get("statistics", {}))
    options = {
        "metric": statistics.get("primary_metric", "recall@10"),
        "bootstrap_iterations": int(statistics.get("bootstrap_iterations", 10000)),
        "randomization_iterations": int(statistics.get("randomization_iterations", 10000)),
        "confidence_level": float(statistics.get("confidence_level", 0.95)),
        "seed": int((config or {}).get("seed", DEFAULT_SEED)),
        **kwargs,
    }
    return contrast_table(scores, contrasts=PRACTICAL_CONTRASTS, family="secondary_practical", **options)


def annotation_map(annotations: Any) -> dict[str, dict[str, bool]]:
    if isinstance(annotations, Mapping) and "query_group" in annotations:
        rows = [annotations]
    elif isinstance(annotations, Mapping):
        rows = []
        for group, value in annotations.items():
            if "query_group" in value:
                rows.append(value)
            elif "labels" in value:
                rows.append({"query_group": group, **value})
            else:
                rows.append({"query_group": group, "labels": value})
    else:
        rows = annotations
    result = {}
    for row in rows:
        group = str(row["query_group"])
        labels = row.get("labels", row)
        unknown = set(labels) - set(QUERY_LABELS)
        missing = set(QUERY_LABELS) - set(labels)
        if unknown or missing:
            raise ValueError(f"Frozen labels must exactly match the six-label schema: unknown={sorted(unknown)}, missing={sorted(missing)}")
        if any(not isinstance(labels.get(label), bool) for label in QUERY_LABELS):
            raise ValueError("Frozen annotation labels must be boolean")
        result[group] = {label: labels[label] for label in QUERY_LABELS}
    return result


def interaction_test(
    differences_true: Iterable[float],
    differences_false: Iterable[float],
    *,
    contrast: str,
    label: str,
    bootstrap_iterations: int = 10000,
    randomization_iterations: int = 10000,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    true_values = _deltas(differences_true)
    false_values = _deltas(differences_false)
    if bootstrap_iterations <= 0 or randomization_iterations <= 0:
        raise ValueError("Interaction iterations must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    observed = float(true_values.mean() - false_values.mean())
    rng_bootstrap = _rng(seed, f"{contrast}:{label}:interaction_bootstrap")
    estimates = np.empty(bootstrap_iterations, dtype=np.float64)
    total_size = true_values.size + false_values.size
    block_size = max(1, min(bootstrap_iterations, 1_000_000 // total_size))
    offset = 0
    while offset < bootstrap_iterations:
        size = min(block_size, bootstrap_iterations - offset)
        true_indices = rng_bootstrap.integers(0, true_values.size, size=(size, true_values.size))
        false_indices = rng_bootstrap.integers(0, false_values.size, size=(size, false_values.size))
        estimates[offset : offset + size] = (
            true_values[true_indices].mean(axis=1) - false_values[false_indices].mean(axis=1)
        )
        offset += size
    alpha = 1 - confidence_level
    rng_randomization = _rng(seed, f"{contrast}:{label}:interaction_randomization")
    combined = np.concatenate([true_values, false_values])
    randomized = np.empty(randomization_iterations, dtype=np.float64)
    for index in range(randomization_iterations):
        permuted = rng_randomization.permutation(combined)
        randomized[index] = abs(permuted[: true_values.size].mean() - permuted[true_values.size :].mean())
    extreme = int(np.count_nonzero(randomized >= abs(observed) - 1e-15))
    return {
        "interaction_delta": observed,
        "interaction_ci_low": float(np.quantile(estimates, alpha / 2)),
        "interaction_ci_high": float(np.quantile(estimates, 1 - alpha / 2)),
        "interaction_p_randomization": (extreme + 1) / (randomization_iterations + 1),
    }


def subgroup_metric_summary(
    scores: Mapping[str, Any],
    annotations: Any,
) -> list[dict[str, Any]]:
    labels = annotation_map(annotations)
    rows = []
    for variant, score in scores.items():
        if isinstance(score, Mapping) and "per_group" in score:
            per_group = {row["query_group"]: row for row in score["per_group"]}
        elif isinstance(score, Mapping):
            per_group = {
                str(group): dict(value) if isinstance(value, Mapping) else {"recall@10": value}
                for group, value in score.items()
            }
        else:
            per_group = {row["query_group"]: row for row in score}
        available = sorted(set(per_group) & set(labels))
        for label in QUERY_LABELS:
            for label_value in (True, False):
                groups = [group for group in available if labels[group][label] is label_value]
                metrics = {}
                for metric in METRIC_NAMES:
                    values = [float(per_group[group][metric]) for group in groups if per_group[group].get(metric) is not None]
                    metrics[metric] = math.fsum(values) / len(values) if values else None
                rows.append(
                    {
                        "variant": variant,
                        "label": label,
                        "label_value": label_value,
                        "n_groups": len(groups),
                        **metrics,
                    }
                )
    return rows


def subgroup_effects(
    scores: Mapping[str, Any],
    annotations: Any,
    *,
    contrasts: Mapping[str, tuple[str, str]] = CONTRASTS,
    metric: str = "recall@10",
    bootstrap_iterations: int = 10000,
    randomization_iterations: int = 10000,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    labels = annotation_map(annotations)
    rows = []
    for label in QUERY_LABELS:
        for contrast, (left_variant, right_variant) in contrasts.items():
            left = metric_by_group(scores[left_variant], metric)
            right = metric_by_group(scores[right_variant], metric)
            common = sorted(set(left) & set(right) & set(labels))
            true_groups = [group for group in common if labels[group][label]]
            false_groups = [group for group in common if not labels[group][label]]
            if not true_groups or not false_groups:
                rows.append(
                    {
                        "label": label,
                        "contrast": contrast,
                        "left_variant": left_variant,
                        "right_variant": right_variant,
                        "metric": metric,
                        "n_label_true": len(true_groups),
                        "n_label_false": len(false_groups),
                        "mean_delta_label_true": None,
                        "mean_delta_label_false": None,
                        "interaction_delta": None,
                        "descriptive_only": True,
                        "status": "insufficient_nonempty_subgroups",
                    }
                )
                continue
            true_result = paired_test(
                left,
                right,
                contrast=f"{contrast}:{label}:true",
                left_variant=left_variant,
                right_variant=right_variant,
                groups=true_groups,
                bootstrap_iterations=bootstrap_iterations,
                randomization_iterations=randomization_iterations,
                confidence_level=confidence_level,
                seed=seed,
                fallback_left=scores[left_variant],
                fallback_right=scores[right_variant],
            )
            false_result = paired_test(
                left,
                right,
                contrast=f"{contrast}:{label}:false",
                left_variant=left_variant,
                right_variant=right_variant,
                groups=false_groups,
                bootstrap_iterations=bootstrap_iterations,
                randomization_iterations=randomization_iterations,
                confidence_level=confidence_level,
                seed=seed,
                fallback_left=scores[left_variant],
                fallback_right=scores[right_variant],
            )
            true_differences = [left[group] - right[group] for group in true_groups]
            false_differences = [left[group] - right[group] for group in false_groups]
            interaction = interaction_test(
                true_differences,
                false_differences,
                contrast=contrast,
                label=label,
                bootstrap_iterations=bootstrap_iterations,
                randomization_iterations=randomization_iterations,
                confidence_level=confidence_level,
                seed=seed,
            )
            rows.append(
                {
                    "label": label,
                    "contrast": contrast,
                    "left_variant": left_variant,
                    "right_variant": right_variant,
                    "metric": metric,
                    "n_label_true": len(true_groups),
                    "n_label_false": len(false_groups),
                    "mean_delta_label_true": true_result["mean_delta"],
                    "ci_low_label_true": true_result["ci_low"],
                    "ci_high_label_true": true_result["ci_high"],
                    "p_randomization_label_true": true_result["p_randomization"],
                    "wins_label_true": true_result["wins"],
                    "losses_label_true": true_result["losses"],
                    "ties_label_true": true_result["ties"],
                    "mean_delta_label_false": false_result["mean_delta"],
                    "ci_low_label_false": false_result["ci_low"],
                    "ci_high_label_false": false_result["ci_high"],
                    "p_randomization_label_false": false_result["p_randomization"],
                    "wins_label_false": false_result["wins"],
                    "losses_label_false": false_result["losses"],
                    "ties_label_false": false_result["ties"],
                    **interaction,
                    "descriptive_only": len(true_groups) < 30 or len(false_groups) < 30,
                    "family": "exploratory_subgroup",
                }
            )
    return rows


def frozen_label_summary(annotations: Any) -> dict[str, Any]:
    labels = annotation_map(annotations)
    combinations = Counter(
        "+".join(label for label in QUERY_LABELS if values[label]) or "none"
        for values in labels.values()
    )
    return {
        "groups": len(labels),
        "label_counts": {label: sum(values[label] for values in labels.values()) for label in QUERY_LABELS},
        "multi_label_groups": sum(sum(values.values()) > 1 for values in labels.values()),
        "combinations": dict(sorted(combinations.items())),
        "labels_are_mutually_exclusive": False,
    }
