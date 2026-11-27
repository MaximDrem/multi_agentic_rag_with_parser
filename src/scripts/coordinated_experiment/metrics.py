import math
from collections.abc import Mapping

from scripts.score_retrieval import average_metrics, normalize_query, score

POOL_CUTOFFS = (10, 20, 30)


def _ranking(record):
    raw = record.get("ranked_chunk_ids", record.get("chunks", []))
    if raw is None:
        raise ValueError("ranked_chunk_ids must not be null")
    ranking = []
    seen = set()
    for item in raw:
        chunk_id = item if isinstance(item, str) else item["chunk_id"]
        if chunk_id not in seen:
            ranking.append(chunk_id)
            seen.add(chunk_id)
    return ranking


def _question_ids(record):
    question_ids = record.get("question_ids")
    if question_ids is None:
        raise ValueError("Selector record is missing question_ids")
    if isinstance(question_ids, (str, bytes)):
        raise ValueError("question_ids must be a sequence of question IDs")
    question_ids = list(question_ids)
    if not question_ids:
        raise ValueError("question_ids must not be empty")
    return question_ids


def expand_group_run(selector_records):
    run = {}
    for record in selector_records:
        question_ids = _question_ids(record)
        ranking = _ranking(record)
        for question_id in question_ids:
            if question_id in run:
                raise ValueError(f"Duplicate run question_id {question_id}")
            run[question_id] = list(ranking)
    return run


def score_group_records(selector_records, questions, qrels, valid_chunk_ids):
    records = list(selector_records)
    questions = dict(questions)
    qrels = {question_id: dict(values) for question_id, values in qrels.items()}
    unknown_qrels = sorted(set(qrels) - set(questions))
    if unknown_qrels:
        raise ValueError(f"Unknown question IDs in qrels: {unknown_qrels[:5]}")

    run = expand_group_run(records)
    unknown_questions = sorted(set(run) - set(questions))
    if unknown_questions:
        raise ValueError(f"Unknown question_id {unknown_questions[0]}")

    valid_chunk_ids = set(valid_chunk_ids)
    for record in records:
        question_ids = _question_ids(record)
        if "query_group" in record:
            actual_groups = {normalize_query(questions[qid]) for qid in question_ids}
            if actual_groups != {record["query_group"]}:
                raise ValueError(f"Query group mismatch for {question_ids}")
        if "query" in record and record["query"] not in {questions[qid] for qid in question_ids}:
            raise ValueError(f"Query mismatch for {question_ids}")
        for chunk_id in run[question_ids[0]]:
            if chunk_id not in valid_chunk_ids:
                raise ValueError(f"Unknown chunk_id {chunk_id}")

    per_query, per_group = score(questions, qrels, run)
    summary = {
        "questions_total": len(questions),
        "benchmark_questions_total": len(questions),
        "subset": False,
        "run_rows": len(run),
        "answerable_rows": len(per_query),
        "answerable_unique_queries": len(per_group),
        "rows_without_qrels": len(questions) - len(set(questions) & set(qrels)),
        "row_macro": average_metrics(per_query),
        "unique_query_macro": average_metrics(per_group),
        "ndcg_gain": "2^relevance-1",
        "duplicate_policy": "identical ranking per normalized query; union qrels with maximum relevance",
        "empty_qrel_policy": "excluded from metric denominators and reported separately",
        "per_query": per_query,
        "per_group": per_group,
        "run": run,
    }
    return summary


def _relevant_ids(relevant):
    if isinstance(relevant, Mapping):
        relevant_ids = {chunk_id for chunk_id, value in relevant.items() if int(value) > 0}
    else:
        relevant_ids = set(relevant)
    if not relevant_ids:
        raise ValueError("Empty qrel set")
    return relevant_ids


def pool_quality_metrics(pool_ids, relevant, rrf_ranking=None):
    pool_ids = list(pool_ids)
    pool = set(pool_ids)
    relevant_ids = _relevant_ids(relevant)
    result = {
        "unique_candidates": len(pool),
        "candidate_recall_all": len(pool & relevant_ids) / len(relevant_ids),
    }
    if rrf_ranking is None:
        for cutoff in POOL_CUTOFFS:
            result[f"candidate_recall_rrf@{cutoff}"] = None
        return result

    ranking = _ranking({"ranked_chunk_ids": rrf_ranking})
    unknown = set(ranking) - pool
    if unknown:
        raise ValueError(f"RRF ranking contains IDs outside the pool: {sorted(unknown)[:5]}")
    for cutoff in POOL_CUTOFFS:
        retrieved = set(ranking[:cutoff])
        result[f"candidate_recall_rrf@{cutoff}"] = len(retrieved & relevant_ids) / len(relevant_ids)
    return result


def rank_overlap_and_correlation(left, right, k=10):
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    left_top = _ranking({"ranked_chunk_ids": left})[:k]
    right_top = _ranking({"ranked_chunk_ids": right})[:k]
    left_set = set(left_top)
    right_set = set(right_top)
    union = list(dict.fromkeys(left_top + right_top))
    overlap_count = len(left_set & right_set)
    result = {
        f"overlap@{k}": overlap_count / k,
        "overlap_count": overlap_count,
        "rank_correlation": None,
    }
    if len(union) < 2:
        return result

    left_rank = {chunk_id: rank for rank, chunk_id in enumerate(left_top, 1)}
    right_rank = {chunk_id: rank for rank, chunk_id in enumerate(right_top, 1)}
    left_values = [float(left_rank.get(chunk_id, k + 1)) for chunk_id in union]
    right_values = [float(right_rank.get(chunk_id, k + 1)) for chunk_id in union]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    left_centered = [value - left_mean for value in left_values]
    right_centered = [value - right_mean for value in right_values]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss == 0.0 or right_ss == 0.0:
        return result
    covariance = sum(a * b for a, b in zip(left_centered, right_centered))
    correlation = covariance / math.sqrt(left_ss * right_ss)
    result["rank_correlation"] = max(-1.0, min(1.0, correlation))
    return result
