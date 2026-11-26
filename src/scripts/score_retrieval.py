import argparse
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

CUTOFFS = (5, 10)


def normalize_query(text):
    return re.sub(r"[^a-zа-яё0-9]+", " ", text.casefold()).strip()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_questions(path):
    rows = read_jsonl(path)
    questions = {row["question_id"]: row["query"] for row in rows}
    if len(questions) != len(rows):
        raise ValueError("Duplicate question_id")
    return questions


def load_qrels(path):
    qrels = defaultdict(dict)
    for row in read_jsonl(path):
        relevance = int(row["relevance"])
        if relevance <= 0:
            continue
        qid, cid = row["question_id"], row["chunk_id"]
        previous = qrels[qid].get(cid)
        if previous is not None and previous != relevance:
            raise ValueError(f"Conflicting relevance for {(qid, cid)}")
        qrels[qid][cid] = relevance
    return qrels


def load_run(path, questions, valid_chunk_ids):
    run = {}
    for row in read_jsonl(path):
        qid = row["question_id"]
        if qid not in questions:
            raise ValueError(f"Unknown question_id {qid}")
        if qid in run:
            raise ValueError(f"Duplicate run question_id {qid}")
        if "query" in row and row["query"] != questions[qid]:
            raise ValueError(f"Query mismatch for {qid}")
        raw = row.get("ranked_chunk_ids", row.get("chunks", []))
        ranking = []
        seen = set()
        for item in raw:
            cid = item if isinstance(item, str) else item["chunk_id"]
            if cid not in valid_chunk_ids:
                raise ValueError(f"Unknown chunk_id {cid}")
            if cid not in seen:
                ranking.append(cid)
                seen.add(cid)
        run[qid] = ranking
    return run


def query_metrics(ranking, relevant):
    if not relevant:
        raise ValueError("Empty qrel set")
    ideal = sorted(relevant.values(), reverse=True)
    result = {}
    for cutoff in CUTOFFS:
        top = ranking[:cutoff]
        retrieved = set(top)
        hits = [cid for cid in top if cid in relevant]
        dcg = sum((2 ** relevant[cid] - 1) / math.log2(rank + 1) for rank, cid in enumerate(top, 1) if cid in relevant)
        idcg = sum((2 ** relevance - 1) / math.log2(rank + 1) for rank, relevance in enumerate(ideal[:cutoff], 1))
        first = next((rank for rank, cid in enumerate(top, 1) if cid in relevant), None)
        result[f"hit_rate@{cutoff}"] = float(bool(hits))
        result[f"recall@{cutoff}"] = len(retrieved & set(relevant)) / len(relevant)
        result[f"ndcg@{cutoff}"] = dcg / idcg if idcg else 0.0
        result[f"mrr@{cutoff}"] = 1.0 / first if first else 0.0
    return result


def average_metrics(rows):
    if not rows:
        return {}
    keys = [f"{metric}@{cutoff}" for cutoff in CUTOFFS for metric in ("hit_rate", "recall", "ndcg", "mrr")]
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def grouped_inputs(questions, qrels, run):
    groups = defaultdict(list)
    for qid, query in questions.items():
        groups[normalize_query(query)].append(qid)
    grouped = []
    for group_key, qids in sorted(groups.items()):
        present = [qid for qid in qids if qid in run]
        if not present:
            continue
        signatures = {tuple(run[qid]) for qid in present}
        if len(signatures) != 1:
            raise ValueError(f"Rankings differ within duplicate query group {qids}")
        relevant = {}
        for qid in qids:
            for cid, relevance in qrels.get(qid, {}).items():
                relevant[cid] = max(relevant.get(cid, 0), relevance)
        if not relevant:
            continue
        grouped.append({
            "query_group": group_key,
            "question_ids": qids,
            "query": questions[present[0]],
            "ranking": run[present[0]],
            "relevant": relevant,
        })
    return grouped


def score(questions, qrels, run):
    missing = sorted(set(qrels) - set(run))
    if missing:
        raise ValueError(f"Missing {len(missing)} answerable run rows: {missing[:5]}")
    per_query = []
    for qid in sorted(set(questions) & set(qrels)):
        per_query.append({
            "question_id": qid,
            "query": questions[qid],
            "qrels": len(qrels[qid]),
            "retrieved": len(run[qid]),
            **query_metrics(run[qid], qrels[qid]),
        })
    per_group = []
    for item in grouped_inputs(questions, qrels, run):
        per_group.append({
            "query_group": item["query_group"],
            "question_ids": item["question_ids"],
            "query": item["query"],
            "qrels": len(item["relevant"]),
            "retrieved": len(item["ranking"]),
            **query_metrics(item["ranking"], item["relevant"]),
        })
    return per_query, per_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--chunks-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    questions = load_questions(args.questions)
    qrels = load_qrels(args.qrels)
    if set(qrels) - set(questions):
        raise ValueError("Unknown question IDs in qrels")
    database = pickle.load(open(args.chunks_pkl, "rb"))
    valid_chunk_ids = {chunk.id for chunk in database.chunks}
    run = load_run(args.run, questions, valid_chunk_ids)
    benchmark_questions_total = len(questions)
    if args.allow_subset:
        questions = {qid: query for qid, query in questions.items() if qid in run}
        qrels = {qid: values for qid, values in qrels.items() if qid in run}
    per_query, per_group = score(questions, qrels, run)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_query.jsonl", per_query)
    write_jsonl(output / "per_unique_query.jsonl", per_group)
    summary = {
        "run": str(Path(args.run)),
        "qrels": str(Path(args.qrels)),
        "questions_total": len(questions),
        "benchmark_questions_total": benchmark_questions_total,
        "subset": args.allow_subset,
        "run_rows": len(run),
        "answerable_rows": len(per_query),
        "answerable_unique_queries": len(per_group),
        "rows_without_qrels": len(questions) - len(set(questions) & set(qrels)),
        "row_macro": average_metrics(per_query),
        "unique_query_macro": average_metrics(per_group),
        "ndcg_gain": "2^relevance-1",
        "duplicate_policy": "identical ranking per normalized query; union qrels with maximum relevance",
        "empty_qrel_policy": "excluded from metric denominators and reported separately",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
