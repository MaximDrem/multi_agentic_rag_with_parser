import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.coordinated_experiment.analysis import CONTRAST_REGISTRY, PRIMARY_METRIC, STAGE_CONFIGS
from scripts.coordinated_experiment.schema import atomic_write_json, atomic_write_text

BRANCHES = (
    "coordinated_communication",
    "coordinated_candidate_generation",
    "iterative_selection_without_exchange",
    "generic_llm_reranking",
    "simpler_candidate_generation",
    "simple_hybrid_is_sufficient",
    "inconclusive",
)


def _all_rows(results: Any) -> list[dict[str, Any]]:
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        return [dict(row) for row in results]
    if not isinstance(results, Mapping):
        return []
    rows = results.get("contrasts", results.get("paired_contrasts", []))
    if isinstance(rows, Mapping):
        rows = rows.get("contrasts", [])
    if not isinstance(rows, Sequence):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _contrast_row(results: Any, contrast_id: str) -> dict[str, Any] | None:
    if isinstance(results, Mapping) and "delta" in results and contrast_id in {"", "row"}:
        return dict(results)
    rows = _all_rows(results)
    candidates = [
        row for row in rows
        if row.get("contrast_id") == contrast_id
        and row.get("metric", PRIMARY_METRIC) == PRIMARY_METRIC
        and row.get("strict_or_backfill", "strict") == "strict"
        and row.get("fallback_policy", "all_groups") in {"all_groups", "all"}
    ]
    if not candidates:
        return None
    return next((row for row in candidates if row.get("analysis_role") == "primary"), candidates[0])


def _row(value: Any, contrast_id: str | None) -> dict[str, Any] | None:
    if contrast_id is None and isinstance(value, Mapping) and "delta" in value:
        return dict(value)
    if contrast_id is None:
        return None
    return _contrast_row(value, contrast_id)


def _finite(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    try:
        values = [float(row[key]) for key in ("delta", "ci95_low", "ci95_high")]
    except (KeyError, TypeError, ValueError):
        return False
    return all(value == value and abs(value) != float("inf") for value in values)


def POS(results: Any, contrast_id: str | None = None, margin_pp: float = 1.0) -> bool:
    row = _row(results, contrast_id)
    return bool(_finite(row) and float(row["delta"]) * 100 >= margin_pp and float(row["ci95_low"]) > 0)


def CONF(results: Any, contrast_id: str | None = None) -> bool:
    if contrast_id in CONTRAST_REGISTRY and not CONTRAST_REGISTRY[contrast_id]["primary"]:
        return False
    row = _row(results, contrast_id)
    if not POS(row) or (contrast_id is None and row.get("analysis_role") not in {None, "primary"}):
        return False
    p_holm = row.get("p_holm")
    return isinstance(p_holm, (int, float)) and not isinstance(p_holm, bool) and 0 <= p_holm < 0.05


def BELOW_MARGIN(results: Any, contrast_id: str | None = None, margin_pp: float = 1.0) -> bool:
    row = _row(results, contrast_id)
    return bool(_finite(row) and float(row["ci95_high"]) * 100 < margin_pp)


def HARM(results: Any, contrast_id: str | None = None) -> bool:
    row = _row(results, contrast_id)
    return bool(_finite(row) and float(row["ci95_high"]) < 0)


def UNCERTAIN(results: Any, contrast_id: str | None = None) -> bool:
    row = _row(results, contrast_id)
    return bool(_finite(row) and not POS(row) and not BELOW_MARGIN(row) and not HARM(row))


def _gate_result(gate: str, run: bool, reason: str, rows: Mapping[str, Any], expected_stage: str) -> dict[str, Any]:
    return {
        "gate": gate,
        "status": "evaluated",
        "run": run,
        "passed": run,
        "reason": reason,
        "contrast_ids": list(rows),
        "contrasts": {key: dict(value) for key, value in rows.items()},
        "next_stage": expected_stage if run else None,
        "expected_configs": list(STAGE_CONFIGS.get(expected_stage, ())) if run else [],
        "skipped_stages": [] if run else [expected_stage],
        "recorded_before_branch_execution": True,
    }


def _missing_gate(gate: str, missing: Sequence[str], stage: str) -> dict[str, Any]:
    return {
        "gate": gate,
        "status": "not_applicable",
        "run": False,
        "passed": False,
        "reason": "Контрасты недоступны: " + ", ".join(missing) + "; отсутствие результата не трактуется как отрицательный эффект",
        "missing_contrasts": list(missing),
        "expected_configs": [],
        "skipped_stages": [stage],
        "recorded_before_branch_execution": True,
    }


def gate_a(results: Any) -> dict[str, Any]:
    row = _contrast_row(results, "SCREEN1")
    if not _finite(row):
        return _missing_gate("A", ["SCREEN1"], "selector_core")
    run = float(row["ci95_high"]) * 100 >= 1.0
    reason = "SCREEN1 upper CI >= +1 п.п.; разрешён selector replay" if run else "SCREEN1 upper CI < +1 п.п.; дорогие communication ablations не запускаются"
    return _gate_result("A", run, reason, {"SCREEN1": row}, "selector_core")


def gate_b(results: Any) -> dict[str, Any]:
    rows = {key: _contrast_row(results, key) for key in ("SEL1", "SEL2")}
    missing = [key for key, row in rows.items() if not _finite(row)]
    if missing:
        return _missing_gate("B", missing, "communication")
    sel1_positive = POS(rows["SEL1"])
    sel2_possible = float(rows["SEL2"]["ci95_high"]) * 100 >= 1.0
    run = sel1_positive and sel2_possible
    reason = (
        "SEL1 point >= +1 п.п. и lower CI > 0; SEL2 upper CI >= +1 п.п.; Holm для gate не требуется"
        if run else "Gate B не выполнен: требуется POS(SEL1) и SEL2 upper CI >= +1 п.п."
    )
    return _gate_result("B", run, reason, rows, "communication")


def gate_c(results: Any) -> dict[str, Any]:
    row = _contrast_row(results, "CG2")
    if not _finite(row):
        return _missing_gate("C", ["CG2"], "candidate_mechanism")
    run = float(row["ci95_high"]) * 100 >= 1.0
    reason = "CG2 upper CI >= +1 п.п.; S/NP controls разрешены" if run else "CG2 upper CI < +1 п.п.; S/NP пропускаются"
    return _gate_result("C", run, reason, {"CG2": row}, "candidate_mechanism")


Gate_A = gate_a
Gate_B = gate_b
Gate_C = gate_c
evaluate_gate_a = gate_a
evaluate_gate_b = gate_b
evaluate_gate_c = gate_c


def save_gate(experiment_dir: str | os.PathLike[str], gate: str | Mapping[str, Any], decision: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if decision is None and isinstance(gate, Mapping):
        decision = gate
        gate_name = str(decision.get("gate", ""))
    else:
        gate_name = str(gate).removeprefix("gate_")
    if gate_name not in {"A", "B", "C"} or decision is None:
        raise ValueError("gate must be A, B, or C and decision must be supplied")
    root = Path(experiment_dir)
    path = root / "analysis" / "gates.json"
    existing = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            existing = dict(loaded)
    gates = dict(existing.get("gates", {}))
    value = dict(decision)
    value["gate"] = gate_name
    value["saved_at"] = datetime.now(timezone.utc).isoformat()
    gates[gate_name] = value
    document = {"protocol_version": "coordinated_mechanism_v1", "gates": gates}
    atomic_write_json(path, document)
    atomic_write_json(root / "analysis" / f"gate_{gate_name}.json", value)
    return value


write_gate = save_gate
save_gate_decision = save_gate


def _stability_values(stability: Any, contrast_id: str) -> list[float]:
    if stability is None:
        return []
    if isinstance(stability, Sequence) and not isinstance(stability, (str, bytes)):
        values = [row for row in stability if isinstance(row, Mapping) and row.get("contrast_id") == contrast_id]
        return [float(row.get("delta", row.get("recall_delta"))) for row in values if isinstance(row.get("delta", row.get("recall_delta")), (int, float))]
    if not isinstance(stability, Mapping):
        return []
    direct = stability.get(contrast_id)
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        result = []
        for item in direct:
            value = item.get("delta", item.get("recall_delta")) if isinstance(item, Mapping) else item
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result.append(float(value))
        return result
    contrasts = stability.get("contrasts", stability.get("deltas", {}))
    if isinstance(contrasts, Mapping) and contrast_id in contrasts:
        return _stability_values({contrast_id: contrasts[contrast_id]}, contrast_id)
    runs = stability.get("runs", stability.get("executions", []))
    if isinstance(runs, Sequence):
        result = []
        for run in runs:
            if not isinstance(run, Mapping):
                continue
            row = _contrast_row(run, contrast_id)
            if row is None and contrast_id in run and isinstance(run[contrast_id], (int, float)):
                result.append(float(run[contrast_id]))
            elif row is not None and isinstance(row.get("delta"), (int, float)):
                result.append(float(row["delta"]))
        return result
    return []


def STABLE(stability: Any, contrast_id: str) -> bool:
    values = _stability_values(stability, contrast_id)
    return len(values) == 3 and all(value > 0 for value in values)


def _sensitivity_rows(results: Any, contrast_id: str) -> dict[str, dict[str, Any]]:
    output = {}
    for row in _all_rows(results):
        if row.get("contrast_id") != contrast_id or row.get("metric", PRIMARY_METRIC) != PRIMARY_METRIC:
            continue
        if row.get("strict_or_backfill") == "common_backfill":
            output["common_backfill"] = row
        if row.get("fallback_policy") in {"exclude_fallback_pairs", "fallback_excluded"}:
            output["fallback_excluded"] = row
    if isinstance(results, Mapping):
        for row in results.get("failures_and_sensitivity", []):
            if not isinstance(row, Mapping) or row.get("contrast_id") != contrast_id:
                continue
            for key, field in (("common_backfill", "common_backfill_delta"), ("fallback_excluded", "fallback_excluded_delta")):
                if row.get(field) is not None:
                    output.setdefault(key, {"delta": row[field]})
    return output


def sensitivity_sign_reversals(results: Any, contrast_id: str) -> list[str]:
    strict = _contrast_row(results, contrast_id)
    if not _finite(strict):
        return []
    reversals = []
    for policy, row in _sensitivity_rows(results, contrast_id).items():
        delta = row.get("delta")
        if isinstance(delta, (int, float)) and float(strict["delta"]) * float(delta) < 0:
            reversals.append(policy)
    return reversals


def _uncertainties(results: Any) -> list[dict[str, Any]]:
    output = []
    for contrast_id in CONTRAST_REGISTRY:
        row = _contrast_row(results, contrast_id)
        if row is not None and UNCERTAIN(row):
            output.append({"contrast_id": contrast_id, "reason": "UNCERTAIN", "ci95": [row["ci95_low"], row["ci95_high"]]})
    return output


def _not_applicable(results: Any) -> list[dict[str, str]]:
    checks = (
        ("selector_core", ("SEL1", "SEL2", "SEL3")),
        ("communication", ("MECH1", "MECH2", "MECH3", "MECH4", "MECH5")),
        ("candidate_mechanism", ("CG4", "CG5", "CG6", "CG7")),
        ("generic_arbiter", ("GEN1", "GEN2", "GEN3", "GEN4")),
    )
    output = []
    for stage, contrasts in checks:
        if not any(_contrast_row(results, contrast) is not None for contrast in contrasts):
            output.append({"stage": stage, "status": "not_applicable", "reason": "stage skipped, not completed, or contrasts unavailable; no effect is inferred"})
    if isinstance(results, Mapping):
        gates = results.get("gates", {})
        if isinstance(gates, Mapping):
            gates = gates.get("gates", gates)
            for gate in ("A", "B", "C"):
                value = gates.get(gate, gates.get(f"gate_{gate}"))
                if isinstance(value, Mapping):
                    for stage in value.get("skipped_stages", []):
                        item = {"stage": str(stage), "status": "skipped_by_gate", "reason": f"Gate {gate}: {value.get('reason', '')}"}
                        if item not in output:
                            output.append(item)
    return output


def _cost_summary(results: Any, configs: Sequence[str]) -> dict[str, Any]:
    costs = results.get("costs", []) if isinstance(results, Mapping) else []
    lookup = {row.get("config_id"): row for row in costs if isinstance(row, Mapping)}
    return {config_id: dict(lookup[config_id]) for config_id in configs if config_id in lookup}


def _base_decision(results: Any, pre_stability: bool) -> dict[str, Any]:
    return {
        "protocol_version": "coordinated_mechanism_v1",
        "branch": "inconclusive",
        "status": "pre_stability" if pre_stability else "evaluated",
        "pre_stability": pre_stability,
        "route": None,
        "required_contrasts": [],
        "stable_contrasts": {},
        "stability": {"status": "not_required_pre_stability" if pre_stability else "not_applicable", "not_repeated": []},
        "sensitivity": {},
        "uncertainties": _uncertainties(results),
        "not_applicable": _not_applicable(results),
        "strongest_same_pool_control": None,
        "simplest_sufficient_system": None,
        "cost": {},
        "caveat": "Результат pre-specified development-эксперимента не заменяет последующий независимый frozen test.",
    }


def _downgrade(decision: dict[str, Any], source: str, reason: str, required: Sequence[str], stability: Any = None, pre: bool = False) -> dict[str, Any]:
    decision.update({"branch": "inconclusive", "downgraded_from": source, "downgrade_reason": reason, "route": None, "required_contrasts": list(required)})
    decision["stable_contrasts"] = {key: STABLE(stability, key) for key in required} if not pre else {}
    decision["stability"] = {"status": "not_required_pre_stability" if pre else "failed_or_unavailable", "not_repeated": [key for key in required if not STABLE(stability, key)] if not pre else []}
    decision["uncertainties"].append({"reason": reason, "downgraded_from": source})
    return decision


def _finalize(decision: dict[str, Any], results: Any, branch: str, route: str | None, required: Sequence[str], system: str, control: str | None, controls: Sequence[str], stability: Any, pre: bool) -> dict[str, Any]:
    decision.update({
        "branch": branch, "route": route, "required_contrasts": list(required),
        "simplest_sufficient_system": system, "strongest_same_pool_control": control,
        "cost": _cost_summary(results, [system, *controls]),
    })
    if branch == "simpler_candidate_generation":
        decision["simplest_system"] = system
    if branch == "simple_hybrid_is_sufficient":
        positive_stability = []
    elif branch in {"generic_llm_reranking", "simpler_candidate_generation"}:
        positive_stability = [required[-1]]
    else:
        positive_stability = [key for key in required if POS(results, key)]
    decision["stable_contrasts"] = {key: STABLE(stability, key) for key in positive_stability} if not pre else {}
    stability_status = "not_required_pre_stability" if pre else ("not_applicable" if not positive_stability else ("passed" if all(STABLE(stability, key) for key in positive_stability) else "failed_or_unavailable"))
    decision["stability"] = {
        "status": stability_status,
        "not_repeated": [] if pre else [key for key in positive_stability if not STABLE(stability, key)],
    }
    decision["sensitivity"] = {key: sensitivity_sign_reversals(results, key) for key in required}
    return decision


def decide(results: Any, stability: Any = None, pre_stability: bool = False) -> dict[str, Any]:
    decision = _base_decision(results, pre_stability)
    comm_route = None
    comm_required: tuple[str, ...] = ()
    if all(CONF(results, key) for key in ("MECH2", "SEL4", "MECH3")):
        comm_route, comm_required = "ARB", ("MECH2", "SEL4", "MECH3")
    elif CONF(results, "MECH5") and CONF(results, "MECH4") and BELOW_MARGIN(results, "MECH3"):
        comm_route, comm_required = "WRRF", ("MECH5", "MECH4")
    if comm_route:
        reversals = {key: sensitivity_sign_reversals(results, key) for key in comm_required}
        if any(reversals.values()):
            return _downgrade(decision, "coordinated_communication", f"sensitivity sign reversal: {reversals}", comm_required, stability, pre_stability)
        if not pre_stability and not all(STABLE(stability, key) for key in comm_required):
            return _downgrade(decision, "coordinated_communication", "required STABLE predicate failed or repeats are unavailable", comm_required, stability)
        system = "E__R2P__ARB_CLEAN__strict" if comm_route == "ARB" else "E__R2P__WRRF"
        controls = ["E__JINA", "E__R2NP__ARB__strict", "E__R2P__WRRF"] if comm_route == "ARB" else ["E__JINA", "E__R2P__ARB_CLEAN__strict"]
        return _finalize(decision, results, "coordinated_communication", comm_route, comm_required, system, "E__JINA", controls, stability, pre_stability)

    if all(CONF(results, key) for key in ("CG2", "CG4", "CG5")):
        required = ("CG2", "CG4", "CG5")
        reversals = {key: sensitivity_sign_reversals(results, key) for key in required}
        if any(reversals.values()):
            return _downgrade(decision, "coordinated_candidate_generation", f"sensitivity sign reversal: {reversals}", required, stability, pre_stability)
        if not pre_stability and not all(STABLE(stability, key) for key in required):
            return _downgrade(decision, "coordinated_candidate_generation", "required STABLE predicate failed or repeats are unavailable", required, stability)
        return _finalize(decision, results, "coordinated_candidate_generation", "planner_workers", required, "E__JINA", "W__JINA", ["W__JINA", "S__JINA", "NP__JINA"], stability, pre_stability)

    if CONF(results, "MECH1") and CONF(results, "SEL5") and BELOW_MARGIN(results, "MECH2") and BELOW_MARGIN(results, "MECH5"):
        required = ("MECH1", "SEL5")
        reversals = {key: sensitivity_sign_reversals(results, key) for key in required}
        if any(reversals.values()):
            return _downgrade(decision, "iterative_selection_without_exchange", f"sensitivity sign reversal: {reversals}", required, stability, pre_stability)
        if not pre_stability and not all(STABLE(stability, key) for key in required):
            return _downgrade(decision, "iterative_selection_without_exchange", "required STABLE predicate failed or repeats are unavailable", required, stability)
        return _finalize(decision, results, "iterative_selection_without_exchange", "R2NP", required, "E__R2NP__ARB__strict", "E__JINA", ["E__JINA", "E__R1__ARB__strict"], stability, pre_stability)

    if CONF(results, "SEL3") and BELOW_MARGIN(results, "SEL2"):
        routes = []
        if CONF(results, "GEN1") and BELOW_MARGIN(results, "GEN3"):
            routes.append(("B", "GEN1", "B__R0__ARB__strict", "B__JINA"))
        if CONF(results, "GEN2") and BELOW_MARGIN(results, "GEN4"):
            routes.append(("W", "GEN2", "W__R0__ARB__strict", "W__JINA"))
        available = [route for route in routes if pre_stability or STABLE(stability, route[1])]
        if available:
            route, gen, system, control = available[0]
            return _finalize(decision, results, "generic_llm_reranking", route, ("SEL3", gen), system, control, [control], stability, pre_stability)
        if routes:
            return _downgrade(decision, "generic_llm_reranking", "no simple-pool route has a stable positive GEN contrast", ("SEL3", routes[0][1]), stability)

    s_route = CONF(results, "CG2") and BELOW_MARGIN(results, "CG4") and CONF(results, "CG6")
    np_route = CONF(results, "CG2") and CONF(results, "CG4") and BELOW_MARGIN(results, "CG5") and CONF(results, "CG7")
    routes = []
    if s_route:
        routes.append(("S", "CG6", "S__JINA"))
    if np_route:
        routes.append(("NP", "CG7", "NP__JINA"))
    available = [route for route in routes if pre_stability or STABLE(stability, route[1])]
    if available:
        route, required_positive, system = available[0]
        required = ("CG2", required_positive)
        return _finalize(decision, results, "simpler_candidate_generation", route, required, system, "W__JINA", ["E__JINA", "W__JINA"], stability, pre_stability)
    if routes:
        return _downgrade(decision, "simpler_candidate_generation", "required STABLE predicate failed", ("CG2", routes[0][1]), stability)

    if BELOW_MARGIN(results, "CG2"):
        selector_available = _contrast_row(results, "SEL1") is not None or _contrast_row(results, "SEL3") is not None
        selector_sufficient = BELOW_MARGIN(results, "SEL1") and BELOW_MARGIN(results, "SEL3") if selector_available else BELOW_MARGIN(results, "SCREEN1")
        if selector_sufficient:
            system = "B__JINA" if BELOW_MARGIN(results, "CG0") else ("W__JINA" if POS(results, "CG0") else "B__JINA_or_W__JINA_uncertain")
            return _finalize(decision, results, "simple_hybrid_is_sufficient", None, ("CG2",), system, "E__JINA", ["E__JINA"], stability, pre_stability)
    return decision


def _stability_target(decision: Mapping[str, Any], results: Any) -> dict[str, Any]:
    branch = decision.get("branch")
    route = decision.get("route")
    targets = {
        "coordinated_communication": ("R2NP/R2P", ["E__R2NP__ARB", "E__R2P__ARB_CLEAN", "E__R2NP__WRRF", "E__R2P__WRRF"]),
        "iterative_selection_without_exchange": ("R1/R2NP", ["E__R1__ARB", "E__R2NP__ARB"]),
        "coordinated_candidate_generation": ("E/S/NP", ["E__JINA", "S__JINA", "NP__JINA"]),
    }
    if branch in targets:
        contrast, configs = targets[branch]
    elif branch == "simpler_candidate_generation":
        other = "S" if route == "S" else "NP"
        contrast, configs = f"E/{other}", ["E__JINA", f"{other}__JINA"]
    elif branch == "generic_llm_reranking":
        simple = route if route in {"B", "W"} else "B"
        contrast, configs = f"E+R0/{simple}+R0", ["E__R0__ARB", f"{simple}__R0__ARB"]
    elif branch == "simple_hybrid_is_sufficient":
        return {"status": "not_applicable", "reason": "simple_hybrid_is_sufficient has no LLM stability target", "contrast": None, "configs": [], "total_executions": 3}
    elif branch == "inconclusive":
        if any(_contrast_row(results, key) is not None for key in ("MECH2", "MECH5", "SEL4")):
            contrast, configs = "clean R2NP/R2P", ["E__R2NP__ARB", "E__R2P__ARB_CLEAN"]
        elif any(_contrast_row(results, key) is not None for key in ("CG4", "CG6")):
            contrast, configs = "E/S", ["E__JINA", "S__JINA"]
        elif any(_contrast_row(results, key) is not None for key in ("SEL1", "SEL2", "SEL3")):
            contrast, configs = "full cached R2P/R0", ["E__LEGACY_R2P__ARB_REPLAY", "E__R0__ARB"]
        else:
            return {"status": "not_applicable", "reason": "no completed LLM stage is available for stability replay", "contrast": None, "configs": [], "total_executions": 3}
    else:
        return {"status": "not_applicable", "reason": "no stability target", "contrast": None, "configs": [], "total_executions": 3}
    return {
        "status": "selected",
        "contrast": contrast,
        "configs": configs,
        "total_executions": 3,
        "predeclared_execution": 1,
        "repeat_generation_and_selector_together": True,
        "selection_rule": "STABLE requires strictly positive primary Recall@10 delta in every execution; no best-repeat selection",
    }


def decision_pre_stability(results: Any, experiment_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    decision = decide(results, stability=None, pre_stability=True)
    target = _stability_target(decision, results)
    decision["stability_target"] = target
    if experiment_dir is not None:
        root = Path(experiment_dir) / "analysis"
        atomic_write_json(root / "decision_pre_stability.json", decision)
        atomic_write_json(root / "stability_target.json", target)
    return decision


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown(decision: Mapping[str, Any], results: Any) -> str:
    rows = [row for row in results.get("configs", []) if row.get("aggregation") == "unique_query_macro"] if isinstance(results, Mapping) else []
    lines = [
        "# Решение coordinated_mechanism_v1", "",
        "## 1. Целостность данных", "",
        f"complete: `{_fmt(results.get('complete') if isinstance(results, Mapping) else None)}`. Дубликаты, неизвестные группы и same-pool mismatch являются ошибками анализа, а не исключаемыми наблюдениями.", "",
        "## 2. Качество на unique query groups", "",
        "| config | policy | Recall@10 | nDCG@10 | MRR@10 | fallback | repair |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row.get('config_id')} | {row.get('output_policy')} | {_fmt(row.get('recall@10'))} | {_fmt(row.get('ndcg@10'))} | {_fmt(row.get('mrr@10'))} | {_fmt(row.get('fallback_groups'))} | {_fmt(row.get('repair_groups'))} |")
    lines.extend(["", "## 3. Paired deltas, CI и Holm", "", "| contrast | delta, п.п. | 95% CI, п.п. | p Holm | interpretation |", "|---|---:|---:|---:|---|"])
    for row in _all_rows(results):
        if row.get("metric") == PRIMARY_METRIC and row.get("strict_or_backfill") == "strict" and row.get("fallback_policy") == "all_groups":
            lines.append(f"| {row['contrast_id']} | {_fmt(row.get('delta_pp'))} | [{_fmt(row.get('ci95_low_pp'))}; {_fmt(row.get('ci95_high_pp'))}] | {_fmt(row.get('p_holm'))} | {row.get('effect_interpretation')} |")
    lines.extend(["", "## 4. Candidate pool decomposition", "", "| pool | size | raw hits | candidate recall | E−B new | E−B relevant |", "|---|---:|---:|---:|---:|---:|"])
    pools = results.get("pool_metrics", []) if isinstance(results, Mapping) else []
    for row in pools:
        lines.append(f"| {row.get('pool_id')} | {_fmt(row.get('unique_candidates_mean'))} | {_fmt(row.get('raw_hits_mean'))} | {_fmt(row.get('candidate_recall_all'))} | {_fmt(row.get('new_ids_vs_base'))} | {_fmt(row.get('new_relevant_vs_base'))} |")
    lines.extend(["", "## 5. Selector и communication decomposition", "", "Выводы основаны только на предуказанных same-pool контрактах; отсутствие этапа обозначено `not_applicable` и не считается нулевым эффектом.", "", "## 6. Стоимость и fallback", "", "| config | cached tokens | new selector tokens | estimated system tokens | search hits |", "|---|---:|---:|---:|---:|"])
    costs = results.get("costs", []) if isinstance(results, Mapping) else []
    for row in costs:
        lines.append(f"| {row.get('config_id')} | {_fmt(row.get('cached_generation_tokens'))} | {_fmt(row.get('new_selector_tokens'))} | {_fmt(row.get('total_estimated_system_tokens'))} | {_fmt(row.get('raw_search_hits'))} |")
    lines.extend(["", "## 7. Stability", "", _json_for_md(decision.get("stability", {})), "", "## 8. Выбранная ветка", "", f"**{decision['branch']}**" + (f", route **{decision['route']}**" if decision.get("route") else "") + ".", ""])
    if decision.get("downgrade_reason"):
        lines.extend([f"Downgrade: {decision['downgrade_reason']}", ""])
    lines.extend(["## 9. Simplest sufficient system", "", _fmt(decision.get("simplest_sufficient_system")), "", f"Strongest same-pool control: {_fmt(decision.get('strongest_same_pool_control'))}", "", "## 10. Что проверить перед рукописью", "", "- Независимый frozen confirmation для заранее выбранного winning/simplest варианта.", "- Standalone end-to-end latency; replay и legacy wall time нельзя складывать как измеренный end-to-end run.", "- Оставшиеся UNCERTAIN контрасты, фактические токены, fallback/repair и protocol deviations.", "", decision["caveat"], ""])
    return "\n".join(lines)


def _json_for_md(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n```"


def write_decision(
    experiment_dir: str | os.PathLike[str],
    results: Any,
    stability: Any = None,
    pre_stability: bool = False,
) -> dict[str, Any]:
    if isinstance(results, Mapping) and results.get("branch") in BRANCHES and not _all_rows(results):
        decision = dict(results)
    else:
        decision = decide(results, stability=stability, pre_stability=pre_stability)
    root = Path(experiment_dir) / "analysis"
    if pre_stability:
        target = _stability_target(decision, results)
        decision["stability_target"] = target
        atomic_write_json(root / "decision_pre_stability.json", decision)
        atomic_write_json(root / "stability_target.json", target)
        return decision
    atomic_write_json(root / "decision.json", decision)
    atomic_write_text(root / "decision.md", _markdown(decision, results))
    return decision


__all__ = [
    "BELOW_MARGIN", "BRANCHES", "CONF", "Gate_A", "Gate_B", "Gate_C", "HARM", "POS", "STABLE",
    "UNCERTAIN", "decide", "decision_pre_stability", "evaluate_gate_a", "evaluate_gate_b", "evaluate_gate_c",
    "gate_a", "gate_b", "gate_c", "save_gate", "save_gate_decision", "sensitivity_sign_reversals", "write_decision", "write_gate",
]
