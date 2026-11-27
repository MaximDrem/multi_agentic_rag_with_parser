from __future__ import annotations

import hashlib
from contextlib import nullcontext
from typing import Any, Protocol, Sequence

from common.models import Chunk
from retrieval.npa_mas.budget import BudgetLedger
from retrieval.npa_mas.schemas import Candidate, CandidatePool, Channel, RetrievalSettings
from retrieval.rrf import rrf_merge, weighted_rrf


class RawSearchHit:
    def __init__(self, chunk: Chunk, score: float, rank: int):
        self.chunk = chunk
        self.score = float(score)
        self.rank = int(rank)


class SearchBackend(Protocol):
    def search(self, channel: Channel, query: str, requested_hits: int) -> Sequence[Any]: ...


class EngineSearchAdapter:
    def __init__(self, engine: Any, score_threshold: float | None = None):
        self.engine = engine
        self.score_threshold = score_threshold

    def search(self, channel: Channel, query: str, requested_hits: int) -> list[RawSearchHit]:
        if requested_hits <= 0:
            return []
        threshold = self.score_threshold
        if threshold is None:
            from common.config import SCORE_THRESHOLD

            threshold = SCORE_THRESHOLD
        if channel == "dense":
            lock = getattr(self.engine, "_dense_lock", None)
            with lock if lock is not None else nullcontext():
                raw = self.engine.embedding.search(query, top_k=requested_hits, score_threshold=threshold)
            chunks = self.engine.db.chunks
        elif channel == "bm25":
            lock = getattr(self.engine, "_bm25_lock", None)
            with lock if lock is not None else nullcontext():
                raw = self.engine.bm25.search(query, top_k=requested_hits, score_threshold=threshold)
            chunks = self.engine.bm25.chunks
        else:
            raise ValueError(f"unsupported retrieval channel: {channel}")
        hits = [RawSearchHit(chunks[index], float(score), rank) for rank, (index, score) in enumerate(raw, 1)]
        if len(hits) > requested_hits:
            raise ValueError("search backend returned more hits than requested")
        return hits


class CandidateBuilder:
    def __init__(self, search_backend: SearchBackend | Any, settings: RetrievalSettings | None = None):
        self.search_backend = search_backend
        self.settings = settings or RetrievalSettings()

    def build(
        self,
        *,
        query_group: str,
        query: str,
        channels: Sequence[Channel],
        generated_queries: dict[Channel, list[str]] | None = None,
        wide: bool = False,
    ) -> CandidatePool:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if len(set(channels)) != len(channels):
            raise ValueError("channels must be unique")
        expanded = generated_queries is not None
        generated_queries = generated_queries or {}
        targets = {
            channel: (
                self.settings.expanded_target_raw_hits_per_channel
                if expanded
                else self.settings.wide_target_raw_hits_per_channel
                if wide
                else self.settings.baseline_target_raw_hits_per_channel
            )
            for channel in channels
        }
        ledger = BudgetLedger(targets)
        normalized: dict[Channel, list[str]] = {}
        raw_generated_queries: dict[Channel, list[str]] = {}
        query_keys: dict[str, list[str]] = {}
        raw_provenance: dict[str, list[dict[str, Any]]] = {}
        channel_rankings: dict[str, list[str]] = {}
        within_channel_details: dict[str, list[dict[str, Any]]] = {}
        observations: dict[str, list[dict[str, Any]]] = {}
        truncated_queries: dict[str, list[str]] = {}
        duplicate_queries: dict[str, list[str]] = {}
        overlong_queries: dict[str, list[str]] = {}
        chunks: dict[str, Chunk] = {}

        for channel in channels:
            raw_generated_queries[channel] = list(generated_queries.get(channel, []))
            cleaned, truncated, duplicates, overlong = self._normalize_generated(
                query,
                generated_queries.get(channel, []),
            )
            normalized[channel] = cleaned
            truncated_queries[channel] = truncated
            duplicate_queries[channel] = duplicates
            overlong_queries[channel] = overlong
            allocations = [("original", query, self._original_target(expanded=expanded, wide=wide))]
            allocations.extend(
                (f"generated_{index:03d}", generated, self.settings.expanded_each_generated_query_target_raw_hits)
                for index, generated in enumerate(cleaned, 1)
            )
            query_keys[channel] = [item[0] for item in allocations]
            rankings: dict[str, list[str]] = {}
            raw_provenance[channel] = []
            observed_ids = set()
            observations[channel] = []

            for query_key, allocated_query, target in allocations:
                try:
                    hits = list(self.search_backend.search(channel, allocated_query, target))
                    if len(hits) > target:
                        raise ValueError("search backend returned more hits than requested")
                    ledger.record(
                        channel=channel,
                        query=allocated_query,
                        query_key=query_key,
                        target=target,
                        requested=target,
                        actual=len(hits),
                    )
                    error = None
                except Exception as exc:
                    hits = []
                    error = type(exc).__name__
                    ledger.record(
                        channel=channel,
                        query=allocated_query,
                        query_key=query_key,
                        target=target,
                        requested=target,
                        actual=0,
                        error=error,
                    )
                ranking: list[str] = []
                seen_in_ranking = set()
                raw_hits = []
                for rank, hit in enumerate(hits, 1):
                    chunk = hit.chunk
                    chunks.setdefault(chunk.id, chunk)
                    if chunk.id not in seen_in_ranking:
                        seen_in_ranking.add(chunk.id)
                        ranking.append(chunk.id)
                    raw_hits.append({"chunk_id": chunk.id, "rank": rank, "score": float(hit.score)})
                    if chunk.id not in observed_ids:
                        observed_ids.add(chunk.id)
                        observations[channel].append(
                            {
                                "chunk_id": chunk.id,
                                "text": chunk.text,
                                "doc_path": chunk.doc_path,
                                "title": chunk.title,
                                "query_key": query_key,
                            }
                        )
                rankings[query_key] = ranking
                raw_provenance[channel].append(
                    {
                        "query_key": query_key,
                        "query": allocated_query,
                        "target_raw_hits": target,
                        "requested_raw_hits": target,
                        "actual_raw_hits": len(hits),
                        "error": error,
                        "hits": raw_hits,
                    }
                )

            within = rrf_merge(
                rankings,
                k=self.settings.rrf_k,
                limit=targets[channel],
                source_order=query_keys[channel],
            )
            channel_rankings[channel] = [item["chunk_id"] for item in within]
            within_channel_details[channel] = within

        if tuple(channels) == ("dense", "bm25") or set(channels) == {"dense", "bm25"}:
            full_merged = weighted_rrf(
                channel_rankings.get("dense", []),
                channel_rankings.get("bm25", []),
                alpha=self.settings.final_alpha_dense,
                k=self.settings.rrf_k,
                limit=None,
                source_order=("dense", "bm25"),
            )
        else:
            channel = channels[0]
            full_merged = rrf_merge(
                {channel: channel_rankings[channel]},
                k=self.settings.rrf_k,
                limit=None,
                source_order=(channel,),
            )

        merged = full_merged[: self.settings.pool_size]

        candidates = []
        for item in merged:
            chunk = chunks[item["chunk_id"]]
            selector_text = chunk.text[: self.settings.selector_text_max_chars]
            candidates.append(
                Candidate(
                    chunk_id=chunk.id,
                    pool_rank=item["rank"],
                    selector_text=selector_text,
                    text_sha256=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                    doc_path=chunk.doc_path,
                    title=chunk.title,
                    source_ranks=item["source_ranks"],
                    rrf_score=item["score"],
                    text=chunk.text,
                )
            )

        provenance = {
            "query_keys": query_keys,
            "raw_hits": raw_provenance,
            "channel_rankings": channel_rankings,
            "within_channel_fusion": within_channel_details,
            "final_fusion": full_merged,
            "final_fusion_after_cap": merged,
            "pool_size_before_cap": len(full_merged),
            "pool_size_after_cap": len(merged),
            "raw_generated_queries": raw_generated_queries,
            "observations": observations,
            "truncated_queries": truncated_queries,
            "duplicate_queries": duplicate_queries,
            "overlong_queries": overlong_queries,
        }
        return CandidatePool.create(
            query_group=query_group,
            candidates=candidates,
            budget=ledger.snapshot(),
            generated_queries={channel: normalized[channel] for channel in channels},
            provenance=provenance,
        )

    def _original_target(self, *, expanded: bool, wide: bool) -> int:
        if expanded:
            return self.settings.expanded_original_query_target_raw_hits
        if wide:
            return self.settings.wide_target_raw_hits_per_channel
        return self.settings.baseline_target_raw_hits_per_channel

    def _normalize_generated(
        self,
        original_query: str,
        queries: Sequence[str],
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        cleaned = []
        truncated = []
        duplicates = []
        overlong = []
        seen = {original_query.strip()}
        for raw_query in queries:
            query = raw_query.strip()
            if not query:
                continue
            if len(query) > self.settings.max_generated_query_chars:
                overlong.append(query)
                continue
            if query in seen:
                duplicates.append(query)
                continue
            seen.add(query)
            if len(cleaned) >= self.settings.max_generated_queries_per_channel:
                truncated.append(query)
                continue
            cleaned.append(query)
        return cleaned, truncated, duplicates, overlong
