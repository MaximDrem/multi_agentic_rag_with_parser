import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path

import httpx

from scripts.coordinated_experiment.analysis import STAGE_CONFIGS, analyze_experiment
from scripts.coordinated_experiment.decision import (
    CONF,
    BELOW_MARGIN,
    decision_pre_stability,
    gate_a,
    gate_b,
    gate_c,
    save_gate,
    write_decision,
)
from scripts.coordinated_experiment.legacy_reuse import audit_legacy_run
from scripts.coordinated_experiment.packaging import pack_experiment, validate_run_integrity
from scripts.coordinated_experiment.runner import ExperimentRunner
from scripts.coordinated_experiment.schema import atomic_write_json, canonical_sha256, load_config
from scripts.coordinated_experiment.stability import analyze_stability


def _prepare_experiment(config_path: str, experiment_dir: str) -> None:
    root = Path(experiment_dir)
    root.mkdir(parents=True, exist_ok=True)
    protocol = root / "protocol.yaml"
    if protocol.exists() and protocol.read_bytes() != Path(config_path).read_bytes():
        raise ValueError("Experiment protocol.yaml differs from the supplied config")
    if not protocol.exists():
        shutil.copyfile(config_path, protocol)
    deviations = root / "validation" / "protocol_deviations.jsonl"
    deviations.parent.mkdir(parents=True, exist_ok=True)
    if not deviations.exists():
        deviations.write_text("", encoding="utf-8")


def _record_stage(experiment_dir: str, result: dict) -> None:
    root = Path(experiment_dir)
    path = root / "stage_status.json"
    document = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "protocol_version": "coordinated_mechanism_v1",
        "stages": {},
        "completed_stages": [],
    }
    stage = result["stage"]
    completed = result.get("status") in {None, "ok", "complete", "completed", "not_applicable"}
    state = {**result, "status": "ok" if completed else result["status"]}
    document.setdefault("stages", {})[stage] = state
    document["completed_stages"] = sorted(
        set(document.get("completed_stages", [])) | ({stage} if completed and stage in STAGE_CONFIGS else set())
    )
    atomic_write_json(path, document)


def _existing_gate(experiment_dir: str, name: str, computed: dict) -> dict:
    path = Path(experiment_dir) / "analysis" / f"gate_{name}.json"
    if not path.exists():
        return save_gate(experiment_dir, name, computed)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if bool(frozen.get("run")) != bool(computed.get("run")):
        raise ValueError(f"Frozen Gate {name} decision differs from current inputs")
    return frozen


def _generic_gate(results: dict) -> dict:
    rows = {row["contrast_id"]: row for row in results["contrasts"] if row.get("metric") == "recall@10" and row.get("strict_or_backfill") == "strict" and row.get("fallback_policy") == "all_groups"}
    run = CONF(results, "SEL3") and BELOW_MARGIN(results, "SEL2")
    return {
        "gate": "generic_arbiter_simplification",
        "run": run,
        "reason": "CONF(SEL3) and BELOW_MARGIN(SEL2)",
        "contrasts": {key: rows[key] for key in ("SEL2", "SEL3") if key in rows},
        "recorded_before_branch_execution": True,
    }


def _frozen_generic_gate(experiment_dir: str, computed: dict) -> dict:
    path = Path(experiment_dir) / "analysis" / "generic_gate.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if bool(frozen.get("run")) != bool(computed.get("run")):
            raise ValueError("Frozen generic-arbiter gate differs from current inputs")
        return frozen
    atomic_write_json(path, computed)
    return computed


async def _check_models(runner: ExperimentRunner) -> None:
    if not runner.config.get("execution", {}).get("stop_on_model_change", True):
        return
    from common.config import (
        RETRIEVER_BM25_LLM_BASE_URL,
        RETRIEVER_BM25_LLM_MODEL,
        RETRIEVER_EMB_LLM_BASE_URL,
        RETRIEVER_EMB_LLM_MODEL,
    )
    expected = {
        "dense": RETRIEVER_EMB_LLM_MODEL,
        "bm25": RETRIEVER_BM25_LLM_MODEL,
        "planner": RETRIEVER_EMB_LLM_MODEL,
        "arbiter": RETRIEVER_EMB_LLM_MODEL,
    }
    observed = {}
    async with httpx.AsyncClient(timeout=20) as client:
        for role, base_url in (("dense", RETRIEVER_EMB_LLM_BASE_URL), ("bm25", RETRIEVER_BM25_LLM_BASE_URL)):
            if role in observed:
                continue
            response = await client.get(base_url.rstrip("/") + "/models")
            response.raise_for_status()
            identifiers = {item.get("id") for item in response.json().get("data", [])}
            if expected[role] not in identifiers:
                raise RuntimeError(f"Configured model {expected[role]!r} is not served for {role}")
            observed[role] = sorted(str(item) for item in identifiers if item is not None)
    document = {
        "expected_models": expected,
        "served_models": observed,
        "model_config_sha256": canonical_sha256(expected),
    }
    path = runner.experiment_dir / "validation" / "model_config.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("model_config_sha256") != document["model_config_sha256"]:
            raise RuntimeError("Model configuration changed after the experiment started")
    atomic_write_json(path, document)


def _dry_run(stage: str, runner: ExperimentRunner) -> dict:
    groups = runner.snapshots
    active = sum(bool(snapshot["expansion_active"]) for snapshot in groups)
    smoke_path = runner.experiment_dir / "smoke_groups.json"
    smoke_groups = len(json.loads(smoke_path.read_text(encoding="utf-8")).get("groups", [])) if smoke_path.exists() else 12
    plans = {
        "candidate_core": {"groups": len(groups), "max_new_llm_calls": 0, "configs": list(STAGE_CONFIGS["candidate_core"])},
        "selector_core": {"groups": len(groups), "max_new_llm_calls": 2 * len(groups), "configs": list(STAGE_CONFIGS["selector_core"])},
        "communication": {"groups": len(groups), "max_new_llm_calls": len(groups) + 6 * active, "configs": list(STAGE_CONFIGS["communication"])},
        "candidate_mechanism": {"groups": len(groups), "max_new_llm_calls": 3 * len(groups), "configs": list(STAGE_CONFIGS["candidate_mechanism"])},
        "generic_arbiter": {"groups": len(groups), "max_new_llm_calls": 2 * len(groups), "configs": list(STAGE_CONFIGS["generic_arbiter"])},
        "smoke": {"groups": smoke_groups, "max_new_llm_calls": 2 * smoke_groups},
        "stability": {"groups": 40, "max_new_llm_calls": None, "reason": "determined by frozen stability_target.json"},
    }
    return {"dry_run": True, "stage": stage, "protocol_version": runner.protocol_version, **plans.get(stage, {"groups": len(groups)})}


async def _run_candidate(runner: ExperimentRunner) -> dict:
    result = await runner.run_candidate_core()
    _record_stage(str(runner.experiment_dir), result)
    return analyze_experiment(str(runner.config_path), str(runner.experiment_dir))


async def _run_all_gated(runner: ExperimentRunner) -> dict:
    await _check_models(runner)
    results = await _run_candidate(runner)
    gate_a_value = _existing_gate(str(runner.experiment_dir), "A", gate_a(results))
    gate_c_value = _existing_gate(str(runner.experiment_dir), "C", gate_c(results))
    if gate_a_value["run"]:
        _record_stage(str(runner.experiment_dir), await runner.run_selector_core())
        results = analyze_experiment(str(runner.config_path), str(runner.experiment_dir))
        gate_b_value = _existing_gate(str(runner.experiment_dir), "B", gate_b(results))
        generic_value = _frozen_generic_gate(str(runner.experiment_dir), _generic_gate(results))
        if gate_b_value["run"]:
            _record_stage(str(runner.experiment_dir), await runner.run_communication())
        if generic_value["run"]:
            _record_stage(str(runner.experiment_dir), await runner.run_simplification(candidate=False, generic=True))
    if gate_c_value["run"]:
        _record_stage(str(runner.experiment_dir), await runner.run_simplification(candidate=True, generic=False))
    results = analyze_experiment(str(runner.config_path), str(runner.experiment_dir))
    decision_pre_stability(results, str(runner.experiment_dir))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", required=True)
    audit.add_argument("--benchmark-dir", required=True)
    audit.add_argument("--legacy-run", required=True)
    audit.add_argument("--chunks-pkl", required=True)
    audit.add_argument("--output", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--experiment-dir", required=True)
    smoke.add_argument("--concurrency", type=int, default=2)
    smoke.add_argument("--dry-run", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--stage", required=True, choices=("candidate_core", "all_gated", "stability"))
    run.add_argument("--config", required=True)
    run.add_argument("--experiment-dir", required=True)
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--dry-run", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--experiment-dir", required=True)
    analyze.add_argument("--bootstrap", type=int, default=10000)
    analyze.add_argument("--randomization", type=int, default=10000)
    analyze.add_argument("--seed", type=int, default=20261127)
    pack = subparsers.add_parser("pack")
    pack.add_argument("--experiment-dir", required=True)
    pack.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "audit":
        _prepare_experiment(args.config, args.output)
        report = audit_legacy_run(args.benchmark_dir, args.legacy_run, args.chunks_pkl, args.output, args.config)
        print(json.dumps({"status": report["status"], "summary": report["summary"]}, ensure_ascii=False, indent=2))
        return
    if args.command == "pack":
        print(json.dumps(pack_experiment(args.experiment_dir, args.output), ensure_ascii=False, indent=2))
        return
    _prepare_experiment(args.config, args.experiment_dir)
    if args.command == "analyze":
        results = analyze_experiment(args.config, args.experiment_dir, args.bootstrap, args.randomization, args.seed, require_complete=True)
        stability_path = Path(args.experiment_dir) / "stability" / "execution_manifest.json"
        stability = analyze_stability(args.config, args.experiment_dir) if stability_path.exists() else None
        decision = write_decision(args.experiment_dir, results, stability=stability, pre_stability=stability is None)
        integrity = validate_run_integrity(args.experiment_dir)
        print(json.dumps({"complete": results["complete"], "branch": decision["branch"], "integrity": integrity["status"]}, ensure_ascii=False, indent=2))
        return
    runner = ExperimentRunner(args.config, args.experiment_dir, concurrency=args.concurrency, smoke_only=args.command == "smoke")
    if args.dry_run:
        print(json.dumps(_dry_run(args.command if args.command == "smoke" else args.stage, runner), ensure_ascii=False, indent=2))
        return
    if args.command == "smoke":
        async def smoke_run():
            await _check_models(runner)
            return await runner.run_smoke()
        report = asyncio.run(smoke_run())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.stage == "candidate_core":
        async def candidate_run():
            results = await _run_candidate(runner)
            save_gate(args.experiment_dir, "A", gate_a(results))
            save_gate(args.experiment_dir, "C", gate_c(results))
            return {"complete": results["complete"], "configs": len(results["configs"])}
        print(json.dumps(asyncio.run(candidate_run()), ensure_ascii=False, indent=2))
        return
    if args.stage == "all_gated":
        results = asyncio.run(_run_all_gated(runner))
        print(json.dumps({"complete": results["complete"], "configs": len(results["configs"]), "stages": results["stage_status"]}, ensure_ascii=False, indent=2))
        return
    target_path = Path(args.experiment_dir) / "analysis" / "stability_target.json"
    if not target_path.exists():
        results = analyze_experiment(args.config, args.experiment_dir)
        decision_pre_stability(results, args.experiment_dir)
    async def stability_run():
        await _check_models(runner)
        return await runner.run_stability(target_path)
    report = asyncio.run(stability_run())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
