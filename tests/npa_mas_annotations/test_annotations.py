import pytest

from scripts.npa_mas_experiment.annotations import (
    LABELS,
    annotation_review_pack,
    annotation_template,
    cohen_kappa,
    draft_rows,
    select_double_annotation,
    validate_annotations,
    validate_independent_annotation_protocol,
)


def groups(count=120):
    return [
        {
            "query_group": f"group_{index:04d}",
            "question_ids": [f"row_{index:04d}"],
            "query": f"Вопрос {index}",
        }
        for index in range(count)
    ]


def test_template_requires_human_review_and_full_coverage():
    fixture = groups(3)
    rows = annotation_template(fixture)
    assert len(rows) == 3
    assert all(row["review_status"] == "draft_requires_human_review" for row in rows)
    report = validate_annotations(rows, fixture)
    assert not report["valid"]
    assert len(report["errors"]) == 27
    for row in rows:
        row["labels"] = {label: False for label in LABELS}
        row["annotation_confidence"] = "high"
        row["annotator"] = "human_a"
        row["review_status"] = "human_reviewed"
    assert validate_annotations(rows, fixture)["valid"]


def test_review_pack_contains_query_but_final_rows_do_not():
    fixture = groups(1)
    rows = annotation_template(fixture)
    assert annotation_review_pack(fixture, rows)[0]["representative_query"] == "Вопрос 0"
    assert "representative_query" not in rows[0]


def test_double_annotation_uses_twenty_percent_and_deterministic_hash():
    fixture = groups(120)
    selected = select_double_annotation(fixture, 20261127)
    assert len(selected) == 100
    assert selected == select_double_annotation(fixture, 20261127)
    assert len(select_double_annotation(groups(556), 20261127)) == 112


def test_cohen_kappa_and_disagreements():
    rows = annotation_template(groups(2))
    for row in rows:
        row["labels"] = {label: False for label in LABELS}
    other = annotation_template(groups(2))
    for row in other:
        row["labels"] = {label: False for label in LABELS}
    other[0]["labels"]["multi_issue"] = True
    result = cohen_kappa(rows, other)
    assert result["multi_issue"]["observed_agreement"] == 0.5
    assert result["multi_issue"]["disagreements"] == ["group_0000"]
    assert result["exact_reference"]["cohen_kappa"] is None


def test_draft_labeler_is_strict_and_never_final():
    fixture = groups(2)
    rows = draft_rows(fixture, lambda query: {label: False for label in LABELS})
    assert all(row["annotator"] == "llm_draft" for row in rows)
    assert not validate_annotations(rows, fixture, require_human_review=False)["valid"]
    with pytest.raises(ValueError):
        draft_rows(fixture, lambda query: {label: True for label in LABELS[:-1]})


def reviewed(rows, annotator):
    result = []
    for row in rows:
        value = dict(row)
        value["labels"] = {label: False for label in LABELS}
        value["annotation_confidence"] = "high"
        value["annotator"] = annotator
        value["review_status"] = "human_reviewed"
        result.append(value)
    return result


def test_original_pairs_and_all_disagreements_are_required():
    fixture = groups(100)
    selected = select_double_annotation(fixture, 20261127)
    by_id = {row["query_group"]: row for row in annotation_template(fixture)}
    primary = reviewed(annotation_template(fixture), "human_a")
    secondary = reviewed([by_id[group_id] for group_id in selected], "human_b")
    final = reviewed(annotation_template(fixture), "human_adjudicator")
    report = validate_independent_annotation_protocol(final, primary, secondary, [], fixture, selected, 20261127)
    assert report["valid"]
    assert report["agreement_type"] == "independent_annotators"
    secondary[0]["labels"]["multi_issue"] = True
    report = validate_independent_annotation_protocol(final, primary, secondary, [], fixture, selected, 20261127)
    assert not report["valid"]
    assert report["adjudications_expected"] == 1
    assert any("Missing adjudication" in error for error in report["errors"])


@pytest.mark.parametrize("outside_control", [False, True])
def test_every_final_override_requires_adjudication(outside_control):
    fixture = groups(120)
    selected = select_double_annotation(fixture, 20261127)
    by_id = {row["query_group"]: row for row in annotation_template(fixture)}
    primary = reviewed(annotation_template(fixture), "human_a")
    secondary = reviewed([by_id[group_id] for group_id in selected], "human_b")
    final = reviewed(annotation_template(fixture), "human_adjudicator")
    group_id = sorted(set(by_id) - set(selected))[0] if outside_control else selected[0]
    final_by_id = {row["query_group"]: row for row in final}
    final_by_id[group_id]["labels"]["multi_issue"] = True
    report = validate_independent_annotation_protocol(final, primary, secondary, [], fixture, selected, 20261127)
    assert not report["valid"]
    assert report["adjudications_expected"] == 1
    secondary_value = next((row["labels"]["multi_issue"] for row in secondary if row["query_group"] == group_id), None)
    adjudication = {
        "adjudication_id": f"{group_id}:multi_issue",
        "query_group": group_id,
        "label": "multi_issue",
        "primary_label": False,
        "secondary_label": secondary_value,
        "final_label": True,
        "adjudicator": "human_c",
        "reason": "Codebook review",
        "review_status": "human_reviewed",
    }
    report = validate_independent_annotation_protocol(final, primary, secondary, [adjudication], fixture, selected, 20261127)
    assert report["valid"], report["errors"]
