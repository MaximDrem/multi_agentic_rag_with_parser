from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from retrieval.npa_mas.schemas import Channel


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    channels: tuple[Channel, ...]
    query_generation: Literal["original", "single", "specialized", "frozen"]
    budget: Literal["baseline", "wide", "expanded", "frozen"]
    planner: bool
    selector: Literal["raw", "rrf", "jina", "llm"]
    reports: bool = False
    requires_m2_pool: bool = False


CORE_VARIANT_ORDER = (
    "B0_BM25",
    "B0_DENSE",
    "B0_RRF",
    "B1",
    "B1W",
    "S",
    "M1",
    "M2",
    "M3",
    "M4",
)

CORE_VARIANTS = {
    "B0_BM25": VariantSpec("B0_BM25", ("bm25",), "original", "baseline", False, "raw"),
    "B0_DENSE": VariantSpec("B0_DENSE", ("dense",), "original", "baseline", False, "raw"),
    "B0_RRF": VariantSpec("B0_RRF", ("dense", "bm25"), "original", "baseline", False, "rrf"),
    "B1": VariantSpec("B1", ("dense", "bm25"), "original", "baseline", False, "jina"),
    "B1W": VariantSpec("B1W", ("dense", "bm25"), "original", "wide", False, "jina"),
    "S": VariantSpec("S", ("dense", "bm25"), "single", "expanded", False, "jina"),
    "M1": VariantSpec("M1", ("dense", "bm25"), "specialized", "expanded", False, "jina"),
    "M2": VariantSpec("M2", ("dense", "bm25"), "specialized", "expanded", True, "jina"),
    "M3": VariantSpec("M3", (), "frozen", "frozen", False, "llm", requires_m2_pool=True),
    "M4": VariantSpec("M4", (), "frozen", "frozen", False, "llm", reports=True, requires_m2_pool=True),
}


def get_variant(variant: str) -> VariantSpec:
    try:
        return CORE_VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(f"unknown npa_mas_component_v1 variant: {variant}") from exc
