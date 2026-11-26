import argparse
import asyncio
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

from common.a2a_client import A2AClientManager
from common.config import RETRIEVER_URL
from scripts.score_retrieval import normalize_query, read_jsonl, write_jsonl

RETRIEVER_CARDS = {
    "base": "base_retriever",
    "agentic": "agentic_retriever",
    "agentic_alpha": "agentic_retriever_alpha",
    "multiagent": "multiagent_retriever",
    "multiagent_v2": "multiagent_retriever_v2",
    "multiagent_coordinated": "multiagent_coordinated",
    "baseline_reranker": "baseline_reranker",
}


def serialize_result(data):
    return {
        "chunks": [{key: chunk.get(key) for key in ("chunk_id", "rank", "score", "relevance", "answer_aspect")} for chunk in data["chunks"]],
        "retriever": data["retriever"],
        "fallback_used": data["fallback_used"],
        "trace": data.get("trace", {}),
        "token_usage": data.get("token_usage", {"total_tokens": 0}),
        "latency_ms": data.get("latency_ms", 0),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", required=True, choices=sorted(RETRIEVER_CARDS))
    parser.add_argument("--url", default=RETRIEVER_URL)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--chunks-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-ids")
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    grouped = defaultdict(list)
    for row in questions:
        grouped[normalize_query(row["query"])].append(row)
    groups = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row["source_row"])
        groups.append({"query_group": key, "questions": rows, "query": rows[0]["query"]})
    groups.sort(key=lambda group: min(row["source_row"] for row in group["questions"]))
    if args.question_ids:
        selected = set(args.question_ids.split(","))
        groups = [group for group in groups if selected & {row["question_id"] for row in group["questions"]}]
    if args.limit is not None:
        groups = groups[:args.limit]
    if not groups:
        raise ValueError("No query groups selected")

    database = pickle.load(open(args.chunks_pkl, "rb"))
    valid_chunk_ids = {chunk.id for chunk in database.chunks}
    output = Path(args.output_dir)
    cache = output / "groups"
    cache.mkdir(parents=True, exist_ok=True)
    manager = A2AClientManager()
    card = await manager.fetch_agent_card(args.url)
    if card.name != RETRIEVER_CARDS[args.retriever]:
        raise ValueError(f"Expected AgentCard {RETRIEVER_CARDS[args.retriever]}, got {card.name}")
    semaphore = asyncio.Semaphore(args.concurrency)
    run_started = time.perf_counter()

    async def process_group(index, group):
        path = cache / f"group_{index:04d}.json"
        if path.exists():
            print(f"cached {index + 1}/{len(groups)}", flush=True)
            return
        async with semaphore:
            started = time.perf_counter()
            try:
                _, _, artifacts = await manager.send_message_and_wait(
                    args.url,
                    group["query"],
                    data={"query": group["query"]},
                    timeout=args.timeout,
                )
                data_list = manager.extract_artifact_data(artifacts)
                data = next((item for item in data_list if "chunks" in item), None)
                if data is None:
                    raise RuntimeError("A2A response has no chunks artifact")
                ids = [chunk["chunk_id"] for chunk in data["chunks"]]
                if len(ids) > 10 or len(ids) != len(set(ids)) or not set(ids) <= valid_chunk_ids:
                    raise RuntimeError("Invalid chunk IDs in A2A response")
                result = {
                    **group,
                    "result": serialize_result(data),
                    "wall_latency_ms": int((time.perf_counter() - started) * 1000),
                    "agent_card": card.name,
                    "url": args.url,
                }
            except Exception as exc:
                error = {"query_group": group["query_group"], "question_ids": [row["question_id"] for row in group["questions"]], "error": str(exc)}
                with (output / "errors.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                raise
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n")
            os.replace(temporary, path)
            print(
                f"{args.retriever} {index + 1}/{len(groups)} chunks={len(ids)} "
                f"fallback={data['fallback_used']} latency={result['wall_latency_ms']}ms",
                flush=True,
            )

    try:
        await asyncio.gather(*(process_group(index, group) for index, group in enumerate(groups)))
    finally:
        await manager.close()

    detailed = [json.loads((cache / f"group_{index:04d}.json").read_text()) for index in range(len(groups))]
    write_jsonl(output / "detailed.jsonl", detailed)
    run_rows = []
    for group in detailed:
        ids = [chunk["chunk_id"] for chunk in group["result"]["chunks"]]
        for row in group["questions"]:
            run_rows.append({"question_id": row["question_id"], "query": row["query"], "query_group": group["query_group"], "ranked_chunk_ids": ids})
    write_jsonl(output / "run.jsonl", run_rows)
    expected = sum(len(group["questions"]) for group in detailed)
    wall_time = time.perf_counter() - run_started
    llm_tokens_total = sum(group["result"]["token_usage"].get("total_tokens", 0) or 0 for group in detailed)
    manifest = {
        "system": args.retriever,
        "agent_card": card.name,
        "url": args.url,
        "protocol": "A2A v1.0",
        "query_groups": len(detailed),
        "run_rows": len(run_rows),
        "expected_questions": expected,
        "complete": len(run_rows) == expected,
        "fallback_groups": sum(group["result"]["fallback_used"] for group in detailed),
        "empty_groups": sum(not group["result"]["chunks"] for group in detailed),
        "wall_time_s": round(wall_time, 2),
        "llm_tokens_total": llm_tokens_total,
        "llm_tokens_mean_per_group": llm_tokens_total / len(detailed),
        "llm_tokens_mean_per_question": llm_tokens_total / expected,
        "concurrency": args.concurrency,
        "query_group_policy": "one representative query per normalized group",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
