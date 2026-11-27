import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

LABELS = (
    "exact_reference",
    "lay_paraphrase",
    "multi_issue",
    "applicability_conditions",
    "cross_reference_multi_document",
    "other_unclear",
)


def _group_id(group: Mapping[str, Any]) -> str:
    value = group.get("query_group", group.get("group_id"))
    if not value:
        raise ValueError("Missing query_group")
    return str(value)


def _representative_id(group: Mapping[str, Any]) -> str:
    value = group.get("representative_question_id")
    if value:
        return str(value)
    question_ids = group.get("question_ids") or group.get("questions")
    if not question_ids:
        raise ValueError("Missing representative question ID")
    first = question_ids[0]
    return str(first.get("question_id") if isinstance(first, Mapping) else first)


def annotation_template(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        rows.append({
            "query_group": _group_id(group),
            "representative_question_id": _representative_id(group),
            "labels": {label: None for label in LABELS},
            "annotation_confidence": None,
            "annotator": "unassigned",
            "notes": "",
            "review_status": "draft_requires_human_review",
        })
    return rows


def annotation_review_pack(groups: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {_group_id(group): group for group in groups}
    pack = []
    for row in rows:
        group = by_id.get(_group_id(row))
        if group is None:
            raise ValueError(f"Unknown annotation group: {_group_id(row)}")
        questions = group.get("questions") or []
        query = group.get("query") or group.get("representative_query")
        if query is None and questions:
            query = questions[0].get("query")
        pack.append({**dict(row), "representative_query": query})
    return pack


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_double_annotation(groups: Sequence[Mapping[str, Any]], seed: int) -> list[str]:
    count = max(100, math.ceil(len(groups) * 0.2))
    if count > len(groups):
        raise ValueError("Not enough groups for independent double annotation")
    ordered = sorted(
        (_group_id(group) for group in groups),
        key=lambda group_id: hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest(),
    )
    return ordered[:count]


def cohen_kappa(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    right_by_id = {_group_id(row): row for row in right}
    pairs = [(_group_id(row), row, right_by_id.get(_group_id(row))) for row in left]
    pairs = [(group_id, a, b) for group_id, a, b in pairs if b is not None]
    if not pairs:
        raise ValueError("No overlapping annotations")
    result = {}
    for label in LABELS:
        observed = sum(a["labels"][label] == b["labels"][label] for _, a, b in pairs) / len(pairs)
        left_true = sum(bool(a["labels"][label]) for _, a, _ in pairs) / len(pairs)
        right_true = sum(bool(b["labels"][label]) for _, _, b in pairs) / len(pairs)
        expected = left_true * right_true + (1 - left_true) * (1 - right_true)
        kappa = None if expected == 1 else (observed - expected) / (1 - expected)
        result[label] = {
            "n": len(pairs),
            "observed_agreement": observed,
            "expected_agreement": expected,
            "cohen_kappa": kappa,
            "undefined_reason": "zero_variance_in_both_annotations" if expected == 1 else None,
            "primary_prevalence": left_true,
            "secondary_prevalence": right_true,
            "confusion": {
                "both_true": sum(bool(a["labels"][label]) and bool(b["labels"][label]) for _, a, b in pairs),
                "both_false": sum(not a["labels"][label] and not b["labels"][label] for _, a, b in pairs),
                "primary_true_secondary_false": sum(bool(a["labels"][label]) and not b["labels"][label] for _, a, b in pairs),
                "primary_false_secondary_true": sum(not a["labels"][label] and bool(b["labels"][label]) for _, a, b in pairs),
            },
            "disagreements": [group_id for group_id, a, b in pairs if a["labels"][label] != b["labels"][label]],
        }
    return result


def validate_annotations(
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    require_human_review: bool = True,
) -> dict[str, Any]:
    expected = {_group_id(group): _representative_id(group) for group in groups}
    seen = set()
    errors = []
    for index, row in enumerate(rows):
        group_id = _group_id(row)
        if group_id not in expected:
            errors.append(f"row {index}: unknown group {group_id}")
        if group_id in seen:
            errors.append(f"row {index}: duplicate group {group_id}")
        seen.add(group_id)
        if row.get("representative_question_id") != expected.get(group_id):
            errors.append(f"row {index}: representative mismatch")
        labels = row.get("labels")
        if not isinstance(labels, Mapping) or set(labels) != set(LABELS):
            errors.append(f"row {index}: labels must exactly match codebook")
            continue
        for label, value in labels.items():
            if not isinstance(value, bool):
                errors.append(f"row {index}: {label} must be boolean")
        if row.get("annotation_confidence") not in {"high", "medium", "low"}:
            errors.append(f"row {index}: invalid confidence")
        annotator = str(row.get("annotator", ""))
        if not annotator or annotator in {"unassigned", "llm_draft"}:
            errors.append(f"row {index}: final annotator required")
        if require_human_review and row.get("review_status") != "human_reviewed":
            errors.append(f"row {index}: human review not recorded")
        for forbidden in ("retrieved_ids", "system_outputs", "metrics", "qrels"):
            if forbidden in row:
                errors.append(f"row {index}: forbidden field {forbidden}")
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"missing {len(missing)} groups: {missing[:5]}")
    if len(rows) != len(groups):
        errors.append(f"row count {len(rows)} != {len(groups)}")
    return {
        "valid": not errors,
        "groups": len(rows),
        "expected_groups": len(groups),
        "errors": errors,
        "label_counts": {
            label: sum(bool(row.get("labels", {}).get(label)) for row in rows)
            for label in LABELS
        },
    }


def validate_independent_annotation_protocol(
    final_rows: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    secondary_rows: Sequence[Mapping[str, Any]],
    adjudication_rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    selected_group_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    final_report = validate_annotations(final_rows, groups)
    primary_report = validate_annotations(primary_rows, groups)
    group_by_id = {_group_id(group): group for group in groups}
    selected = list(selected_group_ids)
    selected_groups = [group_by_id[group_id] for group_id in selected if group_id in group_by_id]
    secondary_report = validate_annotations(secondary_rows, selected_groups)
    errors = final_report["errors"] + primary_report["errors"] + secondary_report["errors"]
    if len(selected_groups) != len(selected) or set(selected) != set(select_double_annotation(groups, seed)):
        errors.append("Control sample does not match the frozen deterministic selection")
    primary_by_id = {_group_id(row): row for row in primary_rows}
    secondary_by_id = {_group_id(row): row for row in secondary_rows}
    final_by_id = {_group_id(row): row for row in final_rows}
    if set(secondary_by_id) != set(selected):
        errors.append("Secondary annotations must cover exactly the control sample")
    same_annotator = []
    expected_changes = {}
    control_disagreements = 0
    for group_id in selected:
        primary = primary_by_id.get(group_id)
        secondary = secondary_by_id.get(group_id)
        if primary is None or secondary is None:
            continue
        if primary.get("annotator") == secondary.get("annotator"):
            same_annotator.append(group_id)
        for label in LABELS:
            left = primary["labels"].get(label)
            right = secondary["labels"].get(label)
            if isinstance(left, bool) and isinstance(right, bool) and left != right:
                expected_changes[f"{group_id}:{label}"] = (group_id, label, left, right)
                control_disagreements += 1
    for group_id, primary in primary_by_id.items():
        final = final_by_id.get(group_id)
        if final is None:
            continue
        secondary = secondary_by_id.get(group_id)
        for label in LABELS:
            left = primary["labels"].get(label)
            final_value = final["labels"].get(label)
            if isinstance(left, bool) and isinstance(final_value, bool) and left != final_value:
                right = secondary["labels"].get(label) if secondary is not None else None
                expected_changes.setdefault(f"{group_id}:{label}", (group_id, label, left, right))
    adjudication_by_id = {str(row.get("adjudication_id", "")): row for row in adjudication_rows}
    if len(adjudication_by_id) != len(adjudication_rows):
        errors.append("Duplicate adjudication_id")
    for change_id, (group_id, label, left, right) in expected_changes.items():
        row = adjudication_by_id.get(change_id)
        if row is None:
            errors.append(f"Missing adjudication {change_id}")
            continue
        final = final_by_id.get(group_id)
        if row.get("query_group") != group_id or row.get("label") != label:
            errors.append(f"Adjudication identity mismatch {change_id}")
        if row.get("primary_label") is not left or row.get("secondary_label") is not right:
            errors.append(f"Original labels altered in {change_id}")
        if final is None or row.get("final_label") is not final["labels"].get(label):
            errors.append(f"Final label mismatch in {change_id}")
        if not str(row.get("reason", "")).strip() or not str(row.get("adjudicator", "")).strip():
            errors.append(f"Human adjudicator and reason required in {change_id}")
        if row.get("review_status") != "human_reviewed":
            errors.append(f"Adjudication is not human reviewed: {change_id}")
    unexpected = sorted(set(adjudication_by_id) - set(expected_changes))
    if unexpected:
        errors.append(f"Unexpected adjudications: {unexpected[:5]}")
    try:
        agreement = cohen_kappa(primary_rows, secondary_rows)
    except (ValueError, KeyError, TypeError) as exc:
        agreement = {}
        errors.append(f"Kappa failed: {exc}")
    return {
        "valid": not errors,
        "groups": len(groups),
        "control_groups": len(selected),
        "agreement_type": "audit_agreement" if same_annotator else "independent_annotators",
        "same_annotator_groups": same_annotator,
        "per_label": agreement,
        "control_disagreements": control_disagreements,
        "adjudications_expected": len(expected_changes),
        "adjudications_recorded": len(adjudication_rows),
        "errors": errors,
        "final": final_report,
        "primary": primary_report,
        "secondary": secondary_report,
    }


def draft_rows(
    groups: Sequence[Mapping[str, Any]],
    labeler: Callable[[str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for group, template in zip(groups, annotation_template(groups), strict=True):
        query = group.get("query") or group.get("representative_query")
        if query is None:
            query = group["questions"][0]["query"]
        value = dict(labeler(str(query)))
        if set(value) != set(LABELS) or any(not isinstance(item, bool) for item in value.values()):
            raise ValueError(f"Invalid draft labels for {_group_id(group)}")
        rows.append({
            **template,
            "labels": value,
            "annotation_confidence": "medium",
            "annotator": "llm_draft",
            "review_status": "draft_requires_human_review",
        })
    return rows
