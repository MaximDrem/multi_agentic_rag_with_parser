import argparse
import asyncio
import json
from pathlib import Path

from scripts.npa_mas_experiment.runner import ExperimentRunner

DEFAULT_CONFIG = "experiments/npa_mas_component_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Воспроизводимый эксперимент npa_mas_component_v1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def command(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--config", default=DEFAULT_CONFIG)
        return child

    audit = command("audit-inputs")
    audit.add_argument("--allow-missing-annotations", action="store_true")

    template = command("annotations-template")
    template.add_argument("--output-dir", default="var/audit/npa_mas_component_v1")

    annotations = command("annotations-validate")
    annotations.add_argument("--second")

    for name in ("pilot", "freeze", "run-core", "score", "analyze", "pack"):
        child = command(name)
        child.add_argument("--run-id", required=True)
        if name == "pilot":
            child.add_argument("--no-scoring", action="store_true")
            child.add_argument("--group-ids")
            child.add_argument("--concurrency", type=int)
        if name == "run-core":
            child.add_argument("--concurrency", type=int)

    validate = command("validate")
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--require-complete", action="store_true")

    stability = command("stability")
    stability.add_argument("--run-id", required=True)
    stability.add_argument("--core-run-id", default="core_001")
    stability.add_argument("--concurrency", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runner = ExperimentRunner(Path(args.config))
    if args.command == "audit-inputs":
        result = runner.audit_inputs(require_annotations=not args.allow_missing_annotations)
    elif args.command == "annotations-template":
        result = runner.prepare_annotation_template(Path(args.output_dir))
    elif args.command == "annotations-validate":
        result = runner.validate_annotation_inputs(Path(args.second) if args.second else None)
    elif args.command == "freeze":
        result = runner.freeze(args.run_id)
    elif args.command == "validate":
        result = runner.validate_run(args.run_id, require_complete=args.require_complete)
    elif args.command == "score":
        result = runner.score(args.run_id)
    elif args.command == "analyze":
        result = runner.analyze(args.run_id)
    elif args.command == "pack":
        result = runner.pack(args.run_id)
    elif args.command == "pilot":
        result = asyncio.run(runner.run_pilot(
            args.run_id,
            group_ids=args.group_ids.split(",") if args.group_ids else None,
            concurrency=args.concurrency,
        ))
    elif args.command == "run-core":
        result = asyncio.run(runner.run_core(args.run_id, concurrency=args.concurrency))
    elif args.command == "stability":
        result = asyncio.run(runner.run_stability(
            args.run_id,
            core_run_id=args.core_run_id,
            concurrency=args.concurrency,
        ))
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if isinstance(result, dict) and result.get("valid") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
