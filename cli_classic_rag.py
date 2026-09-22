#!/usr/bin/env python3
"""CLI interface for the classic (baseline) RAG system.

Отличие от cli.py: вместо A2A-агентов — локальный пайплайн
FAISS + BM25 → RRF → реранк → генерация (src/baseline_rag/).

Индексы полностью совпадают с мультиагентной системой: parse/index
используют те же пути (var/markdown_output_clean → var/data), а запросы
читают те же chunks.pkl / faiss_index / bm25_index.pkl.
Каждый запрос логируется через BenchmarkLogger (pipeline_type="baseline")
в var/logs_benchmark/experiment_*_baseline/.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from common.config import SUMMARIZER_LLM_MODEL

INDEX_FILES = [
    Path(os.getenv("CHUNKS_PATH", "./var/data/chunks.pkl")),
    Path(os.getenv("BM25_INDEX_PATH", "./var/data/bm25_index.pkl")),
    Path(os.getenv("FAISS_INDEX_PATH", "./var/data/faiss_index/index.faiss")),
]

PROJECT_ROOT = Path(__file__).parent
PARSER_OUTPUT_DIR = Path(os.getenv("PARSER_OUTPUT_DIR", str(PROJECT_ROOT / "var" / "markdown_output")))
PARSER_CLEAN_OUTPUT_DIR = Path(os.getenv("PARSER_CLEAN_OUTPUT_DIR", str(PROJECT_ROOT / "var" / "markdown_output_clean")))

_pipeline = None


def indices_exist() -> bool:
    return all(f.exists() for f in INDEX_FILES)


def parsed_output_exists() -> bool:
    return PARSER_OUTPUT_DIR.is_dir() and len(list(PARSER_OUTPUT_DIR.glob("*.md"))) > 0


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from baseline_rag.pipeline import BaselineRAG
        print("\nLoading baseline RAG (chunks, FAISS, BM25, reranker, LLM)...")
        t0 = time.time()
        _pipeline = BaselineRAG()
        print(f"Ready in {time.time() - t0:.0f}s")
    return _pipeline


def clear_screen():
    print("\033[2J\033[H", end="")


def menu():
    clear_screen()
    print("=" * 60)
    print("  Classic RAG (FAISS + BM25 -> RRF -> Rerank -> LLM)")
    print("=" * 60)

    indexed = indices_exist()
    parsed = parsed_output_exists()

    status_indexed = "[INDEXED]" if indexed else "[no index]"
    status_parsed = "[PARSED]" if parsed else "[no parse]"
    print(f"  Index: {status_indexed}  |  Parse: {status_parsed}  (shared with agent system)")
    print("=" * 60)
    print(f"  1. {'Re-parse documents' if parsed else 'Parse documents'}")
    print(f"  2. {'Re-index documents' if indexed else 'Index documents'}")
    print("  3. Query the classic RAG")
    print("  0. Exit")
    print("=" * 60)


def do_parsing():
    """Parse documents: НПА для НИР -> markdown_output -> markdown_output_clean."""
    from parser.config import INPUT_DIR, OUTPUT_DIR, CLEAN_OUTPUT_DIR, TEMP_DIR

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    clean_output_dir = Path(CLEAN_OUTPUT_DIR)
    temp_dir = Path(TEMP_DIR)

    if not input_dir.is_dir():
        print(f"\nInput directory not found: {input_dir}")
        input("\nPress Enter...")
        return

    files = list(input_dir.glob("*"))
    if not files:
        print(f"\nNo files found in {input_dir}")
        input("\nPress Enter...")
        return

    if parsed_output_exists():
        md_files = list(output_dir.glob("*.md"))
        print(f"\nParsed Markdown files already exist: {len(md_files)}")
        for f in sorted(md_files)[:5]:
            sz = f.stat().st_size
            print(f"  {f.name} ({sz / 1024:.1f} KB)")
        if len(md_files) > 5:
            print(f"  ... and {len(md_files) - 5} more")
        answer = input("\nRe-parse all documents? [y/N]: ").strip().lower()
        if answer != "y":
            print("Skipped.")
            return

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nParsing documents...")
    print(f"  Input:  {input_dir} ({len(files)} files)")
    print(f"  Output: {output_dir}")
    print()

    from parser.parser import run as parser_run
    stats = parser_run(input_dir=str(input_dir), output_dir=str(output_dir))

    print("=" * 60)
    print(f"Parsing complete. Success: {stats['success']}, Errors: {stats['errors']}, Total: {stats['total']}")

    if stats["success"] > 0:
        print(f"\nCleaning Markdown: {output_dir} -> {clean_output_dir}")
        clean_output_dir.mkdir(parents=True, exist_ok=True)

        from parser.clean_markdown import process_files as clean_process_files
        clean_process_files(str(output_dir), str(clean_output_dir))
        print(f"\nCleaning complete. Results in {clean_output_dir}/")


def do_indexing():
    """Build the SAME indices as the agent system: var/data/ (chunks + FAISS + BM25)."""
    indexed = indices_exist()
    if indexed:
        print("\nIndex files already exist:")
        for f in INDEX_FILES:
            sz = f.stat().st_size if f.exists() else 0
            print(f"  {f} ({sz / 1024 / 1024:.1f} MB)")
        answer = input("\nRebuild indices? [y/N]: ").strip().lower()
        if answer != "y":
            print("Skipped.")
            return

    if not parsed_output_exists():
        print("\nNo parsed documents found. Run parsing first (option 1).")
        return

    print("\nIndexing documents (same indices as the agent system)...")
    from common.chunk_store import ChunkDatabase
    from indexing.chunker import chunk_directory
    from indexing.faiss_indexer import FaissIndexer
    from indexing.bm25_indexer import BM25Indexer

    markdown_dir = Path(os.getenv("MARKDOWN_DIR", str(PARSER_CLEAN_OUTPUT_DIR)))
    chunks_path = INDEX_FILES[0]
    faiss_path = INDEX_FILES[2]
    if faiss_path.is_dir() or str(faiss_path).endswith("/"):
        faiss_path = faiss_path / "index.faiss"
    bm25_path = INDEX_FILES[1]

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Chunking: {markdown_dir}")
    chunks = chunk_directory(markdown_dir)
    print(f"  Chunks: {len(chunks)}")

    db = ChunkDatabase(chunks)
    db.save(chunks_path)
    print(f"  Saved chunks: {chunks_path}")

    print("  Building FAISS index...")
    faiss = FaissIndexer()
    faiss.load_model()
    faiss.build(chunks)
    faiss.save(faiss_path)
    print(f"  Saved FAISS: {faiss_path}")

    print("  Building BM25 index...")
    bm25 = BM25Indexer()
    bm25.build(chunks)
    bm25.save(bm25_path)
    print(f"  Saved BM25: {bm25_path}")

    print("\nIndexing complete.")

    # индексы пересобраны — перезагрузить пайплайн, если он уже был загружен
    global _pipeline
    _pipeline = None


async def do_query():
    if not indices_exist():
        print("\nNo indices found. Run indexing first (option 2).")
        return

    query = input("\nEnter your question: ").strip()
    if not query:
        print("Empty query.")
        return

    pipeline = get_pipeline()

    from common.benchmark_logger import BenchmarkLogger, next_request_id

    print(f"\nRetrieving (FAISS + BM25 -> RRF -> rerank on {os.getenv('RERANKER_DEVICE', 'cpu')})...")
    bench = BenchmarkLogger(
        request_id=next_request_id(),
        question_id="",
        question=query,
        pipeline_type="baseline",
    )
    result = await pipeline.answer(query)

    retrieval_chunks = result["chunks"]
    bench.log_chunks(iteration=1, source_agent="hybrid_rrf_rerank", chunks=result["candidates"])
    bench.log_final_selection(retrieval_chunks)
    bench.finalize_request(
        iterations_done=1,
        total_chunks_collected=len(result["candidates"]),
        final_answer=result["answer"],
        total_latency_ms=result["latency_ms"],
        total_tokens=result["token_usage"]["total_tokens"],
        generator_model=SUMMARIZER_LLM_MODEL,
    )

    print(f"\nTop chunks ({len(retrieval_chunks)}):")
    for c in retrieval_chunks[:5]:
        print(f"  [{c.get('rank', '?')}] {c['rerank_score']:.4f}  {c['title'][:70]}")
    if len(retrieval_chunks) > 5:
        print(f"  ... and {len(retrieval_chunks) - 5} more")

    print(f"\nResponse ({result['latency_ms'] / 1000:.1f}s, {result['token_usage']['total_tokens']} tokens):")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    print(f"Benchmark log: {bench.dir}")


def main():
    try:
        while True:
            menu()
            choice = input("\n> ").strip()

            if choice == "1":
                do_parsing()
                input("\nPress Enter...")

            elif choice == "2":
                do_indexing()
                input("\nPress Enter...")

            elif choice == "3":
                asyncio.run(do_query())
                input("\nPress Enter...")

            elif choice == "0":
                print("\nGoodbye.")
                break

            else:
                print(f"\nUnknown option: {choice}")
                input("Press Enter...")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
