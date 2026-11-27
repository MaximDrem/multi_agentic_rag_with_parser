from collections.abc import Mapping, Sequence
from typing import Any

from scripts.coordinated_experiment.schema import canonical_sha256, text_sha256


def _chunk_value(chunk: Any, key: str) -> Any:
    if isinstance(chunk, Mapping):
        return chunk[key]
    return getattr(chunk, key)


def build_pool_record(chunk_ids: Sequence[str], chunk_map: Mapping[str, Any]) -> dict[str, Any]:
    ids = list(chunk_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Pool chunk_ids must be unique")
    aliases = [f"C{index:03d}" for index in range(1, len(ids) + 1)]
    text_hashes: dict[str, str] = {}
    for chunk_id in ids:
        if chunk_id not in chunk_map:
            raise KeyError(f"Missing chunk: {chunk_id}")
        text_hashes[chunk_id] = text_sha256(str(_chunk_value(chunk_map[chunk_id], "text")))
    ordered_identity = [
        [alias, chunk_id, text_hashes[chunk_id]]
        for alias, chunk_id in zip(aliases, ids, strict=True)
    ]
    return {
        "chunk_ids": ids,
        "aliases": aliases,
        "alias_to_id": dict(zip(aliases, ids, strict=True)),
        "text_sha256": text_hashes,
        "pool_sha256": canonical_sha256(ordered_identity),
    }


def _event_order(event: Mapping[str, Any], index: int) -> tuple[int, int, int]:
    stage = str(event.get("stage", "initial"))
    channel = str(event.get("channel", ""))
    stage_order = {"initial": 0, "expansion": 1}.get(stage, 2)
    channel_order = {"dense": 0, "bm25": 1}.get(channel, 2)
    return stage_order, channel_order, index


def multi_event_rrf(
    events: Sequence[Mapping[str, Any]],
    rrf_k: int,
    limit: int = 30,
) -> list[dict[str, Any]]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    indexed_events = sorted(enumerate(events), key=lambda item: _event_order(item[1], item[0]))
    nonempty: list[tuple[int, Mapping[str, Any], list[Any]]] = []
    channel_counts: dict[str, int] = {}
    for index, event in indexed_events:
        hits = list(event.get("hits", []))
        if not hits:
            continue
        channel = str(event["channel"])
        nonempty.append((index, event, hits))
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
    scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    first_registration: dict[str, int] = {}
    used_sources: set[str] = set()
    for index, event, hits in nonempty:
        channel = str(event["channel"])
        explicit_source = event.get("event_id")
        source = str(explicit_source) if explicit_source else (
            channel if channel_counts[channel] == 1 else f"{channel}:{index}"
        )
        if source in used_sources:
            raise ValueError(f"Duplicate RRF event_id: {source}")
        used_sources.add(source)
        weight = 0.5 / channel_counts[channel]
        seen: set[str] = set()
        implicit_rank = 0
        for hit in hits:
            if isinstance(hit, Mapping):
                chunk_id = str(hit["chunk_id"])
                raw_rank = hit.get("rank")
            else:
                chunk_id = str(hit)
                raw_rank = None
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            implicit_rank += 1
            rank = int(raw_rank) if raw_rank is not None else implicit_rank
            if rank < 1:
                raise ValueError("Hit ranks must be positive")
            if chunk_id not in first_registration:
                first_registration[chunk_id] = len(first_registration)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
            source_ranks.setdefault(chunk_id, {})[source] = rank
    ordered = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], first_registration[chunk_id], chunk_id),
    )[:limit]
    return [
        {
            "chunk_id": chunk_id,
            "score": scores[chunk_id],
            "rank": rank,
            "source_ranks": source_ranks[chunk_id],
        }
        for rank, chunk_id in enumerate(ordered, 1)
    ]


def worker_rrf(
    reports: Mapping[str, Any],
    alias_to_id: Mapping[str, str],
    rrf_k: int,
    limit: int = 30,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for channel in ("dense", "bm25"):
        report = reports.get(channel) or {}
        assessments = report.get("assessments", [])
        ordered = sorted(
            enumerate(assessments),
            key=lambda item: (-item[1]["relevance"], item[0]),
        )
        hits: list[str] = []
        for _, assessment in ordered:
            chunk_id = alias_to_id.get(str(assessment.get("alias", "")))
            if chunk_id is not None:
                hits.append(chunk_id)
        events.append({
            "event_id": channel,
            "channel": channel,
            "stage": "initial",
            "hits": hits,
        })
    return multi_event_rrf(events, rrf_k, limit)


def validate_ranking(
    ids: Sequence[str],
    pool_record: Mapping[str, Any],
    max_k: int = 10,
) -> list[str]:
    ranking = list(ids)
    if len(ranking) > max_k:
        raise ValueError(f"Ranking contains {len(ranking)} IDs; maximum is {max_k}")
    pool_ids = set(pool_record["chunk_ids"])
    unknown = [chunk_id for chunk_id in ranking if chunk_id not in pool_ids]
    if unknown:
        raise ValueError(f"Ranking contains IDs outside the pool: {unknown}")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Ranking contains duplicate IDs")
    return ranking


def apply_backfill(
    strict_ids: Sequence[str],
    pool_record: Mapping[str, Any],
    rrf_rows: Sequence[Mapping[str, Any]],
    output_k: int = 10,
    reports: Mapping[str, Any] | None = None,
    policy: str = "common",
) -> list[str]:
    if policy not in {"common", "legacy"}:
        raise ValueError("policy must be 'common' or 'legacy'")
    selected = validate_ranking(strict_ids, pool_record, max_k=output_k)
    if len(selected) >= output_k:
        return selected
    seen = set(selected)
    if policy == "common":
        backfill = [str(row["chunk_id"]) for row in rrf_rows]
    else:
        if reports is None:
            raise ValueError("Legacy backfill requires worker reports")
        from common.config import RRF_K

        backfill = [
            row["chunk_id"]
            for row in worker_rrf(reports, pool_record["alias_to_id"], RRF_K, limit=30)
        ]
        backfill.extend(pool_record["chunk_ids"])
    pool_ids = set(pool_record["chunk_ids"])
    for chunk_id in backfill:
        if len(selected) >= output_k:
            break
        if chunk_id in seen or chunk_id not in pool_ids:
            continue
        selected.append(chunk_id)
        seen.add(chunk_id)
    return selected


def _pool_details(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if "chunk_ids" not in value or "text_sha256" not in value:
        return None
    ids = list(value["chunk_ids"])
    aliases = list(value.get("aliases", []))
    alias_to_id = dict(value.get("alias_to_id", {}))
    text_hashes = dict(value["text_sha256"])
    if not aliases:
        aliases = [f"C{index:03d}" for index in range(1, len(ids) + 1)]
    if len(aliases) != len(ids):
        raise ValueError("Pool aliases and chunk_ids have different lengths")
    expected_alias_to_id = dict(zip(aliases, ids, strict=True))
    if alias_to_id and alias_to_id != expected_alias_to_id:
        raise ValueError("Pool alias_to_id is inconsistent with ordered aliases")
    alias_to_id = expected_alias_to_id
    ordered_identity = [
        [alias, chunk_id, text_hashes[chunk_id]]
        for alias, chunk_id in zip(aliases, ids, strict=True)
    ]
    computed_hash = canonical_sha256(ordered_identity)
    stored_hash = value.get("pool_sha256")
    if stored_hash is not None and stored_hash != computed_hash:
        raise ValueError("pool_sha256 does not match ordered alias/chunk/text identity")
    return {
        "pool_sha256": computed_hash,
        "chunk_ids": ids,
        "aliases": aliases,
        "alias_to_id": alias_to_id,
        "text_sha256": text_hashes,
    }


def assert_same_pool(*selector_inputs: Any) -> str:
    if len(selector_inputs) == 1 and isinstance(selector_inputs[0], Sequence) and not isinstance(selector_inputs[0], Mapping):
        selector_inputs = tuple(selector_inputs[0])
    if not selector_inputs:
        raise ValueError("At least one pool input is required")
    reference_hash: str | None = None
    reference_details: dict[str, Any] | None = None
    for selector_input in selector_inputs:
        if not isinstance(selector_input, Mapping):
            raise TypeError("Pool inputs must be mappings")
        value = selector_input.get("pool_record", selector_input)
        if not isinstance(value, Mapping):
            raise TypeError("pool_record must be a mapping")
        details = _pool_details(value)
        pool_hash = details["pool_sha256"] if details else value.get("pool_sha256")
        if not isinstance(pool_hash, str) or not pool_hash:
            raise ValueError("Pool input is missing pool_sha256")
        if reference_hash is None:
            reference_hash = pool_hash
            reference_details = details
            continue
        if pool_hash != reference_hash:
            raise ValueError(f"Pool hash mismatch: {pool_hash} != {reference_hash}")
        if details is not None and reference_details is not None and details != reference_details:
            raise ValueError("Pool aliases, chunk IDs, or text hashes differ")
    if reference_hash is None:
        raise ValueError("No pool supplied")
    return reference_hash
