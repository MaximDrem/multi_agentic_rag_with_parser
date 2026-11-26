import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.score_retrieval import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--chunks-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result", action="append", required=True, help="name=run_directory")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    systems = {}
    for specification in args.result:
        name, directory = specification.split("=", 1)
        run_dir = Path(directory)
        command = [
            sys.executable,
            str(Path(__file__).with_name("score_retrieval.py")),
            "--run", str(run_dir / "run.jsonl"),
            "--qrels", args.qrels,
            "--questions", args.questions,
            "--chunks-pkl", args.chunks_pkl,
            "--output-dir", str(run_dir / "scores"),
            "--allow-subset",
        ]
        subprocess.run(command, check=True)
        scores = json.loads((run_dir / "scores/summary.json").read_text())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        detailed = read_jsonl(run_dir / "detailed.jsonl")
        token_total = sum(group["result"].get("token_usage", {}).get("total_tokens", 0) or 0 for group in detailed)
        expected = sum(len(group["questions"]) for group in detailed)
        manifest["llm_tokens_total"] = token_total
        manifest["llm_tokens_mean_per_group"] = token_total / len(detailed)
        manifest["llm_tokens_mean_per_question"] = token_total / expected
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        systems[name] = {"scores": scores, "run_manifest": manifest}
    rows = []
    for name, data in systems.items():
        summary = data["scores"]
        manifest = data["run_manifest"]
        rows.append({
            "system": name,
            "questions": summary["questions_total"],
            "answerable_rows": summary["answerable_rows"],
            "fallback_groups": manifest.get("fallback_groups", 0),
            "wall_time_s": manifest.get("wall_time_s"),
            "llm_tokens_total": manifest["llm_tokens_total"],
            "llm_tokens_mean_per_group": manifest["llm_tokens_mean_per_group"],
            "llm_tokens_mean_per_question": manifest["llm_tokens_mean_per_question"],
            **summary["row_macro"],
        })
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(json.dumps({"systems": systems}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "systems": list(systems), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
