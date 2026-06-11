#!/usr/bin/env python3
"""CLI interface for the combined Document Parser + Multi-Agent RAG system."""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from common.config import (
    ORCHESTRATOR_PORT,
    ORCHESTRATOR_URL,
    RETRIEVER_EMB_PORT,
    RETRIEVER_BM25_PORT,
    SUMMARIZER_PORT,
)

AGENT_PORTS = {
    "orchestrator": ORCHESTRATOR_PORT,
    "retriever_embedding": RETRIEVER_EMB_PORT,
    "retriever_bm25": RETRIEVER_BM25_PORT,
    "summarizer": SUMMARIZER_PORT,
}

INDEX_FILES = [
    Path(os.getenv("CHUNKS_PATH", "./data/chunks.pkl")),
    Path(os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")),
    Path(os.getenv("FAISS_INDEX_PATH", "./data/faiss_index/index.faiss")),
]

VENV_PYTHON = str(Path(__file__).parent / "venv" / "bin" / "python")
LOG_DIR = Path(__file__).parent / "logs"

PROJECT_ROOT = Path(__file__).parent
PARSER_OUTPUT_DIR = Path(os.getenv("PARSER_OUTPUT_DIR", str(PROJECT_ROOT / "markdown_output")))
PARSER_CLEAN_OUTPUT_DIR = Path(os.getenv("PARSER_CLEAN_OUTPUT_DIR", str(PROJECT_ROOT / "markdown_output_clean")))

_agent_procs: dict[str, subprocess.Popen] = {}


def is_port_open(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.3)
    try:
        result = sock.connect_ex(("127.0.0.1", port))
        return result == 0
    finally:
        sock.close()


def any_agent_running() -> bool:
    return any(is_port_open(p) for p in AGENT_PORTS.values())


def all_agents_running() -> bool:
    return all(is_port_open(p) for p in AGENT_PORTS.values())


def indices_exist() -> bool:
    return all(f.exists() for f in INDEX_FILES)


def parsed_output_exists() -> bool:
    return PARSER_OUTPUT_DIR.is_dir() and len(list(PARSER_OUTPUT_DIR.glob("*.md"))) > 0


def clear_screen():
    print("\033[2J\033[H", end="")


def menu():
    clear_screen()
    print("=" * 60)
    print("  Document Parser + A2A Multi-Agent RAG System")
    print("=" * 60)

    running = any_agent_running()
    indexed = indices_exist()
    parsed = parsed_output_exists()

    status_running = "[RUNNING]" if running else "[stopped]"
    status_indexed = "[INDEXED]" if indexed else "[no index]"
    status_parsed = "[PARSED]" if parsed else "[no parse]"
    print(f"  Agents: {status_running}  |  Index: {status_indexed}  |  Parse: {status_parsed}")
    print("=" * 60)
    print(f"  1. {'Re-parse documents' if parsed else 'Parse documents'}")
    print(f"  2. {'Re-index documents' if indexed else 'Index documents'}")
    if running:
        print("  3. Stop agents")
    else:
        print("  3. Start agents")
    print("  4. Query the system")
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

    parsed = parsed_output_exists()
    if parsed:
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

    # Ensure temp/output dirs exist
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nParsing documents...")
    print(f"  Input:  {input_dir} ({len(files)} files)")
    print(f"  Output: {output_dir}")
    print()

    from parser.parser import run as parser_run
    stats = parser_run(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )

    print("=" * 60)
    print(f"Parsing complete. Success: {stats['success']}, Errors: {stats['errors']}, Total: {stats['total']}")

    # Step 2: Clean markdown_output -> markdown_output_clean
    if stats["success"] > 0:
        print(f"\nCleaning Markdown: {output_dir} -> {clean_output_dir}")
        clean_output_dir.mkdir(parents=True, exist_ok=True)

        from parser.clean_markdown import process_files as clean_process_files
        clean_process_files(str(output_dir), str(clean_output_dir))
        print(f"\nCleaning complete. Results in {clean_output_dir}/")


def do_indexing():
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

    print("\nIndexing documents...")
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


def start_agents():
    if any_agent_running():
        print("\nSome agents are already running.")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    procs_def = [
        ("retriever_embedding", RETRIEVER_EMB_PORT, "agents.retriever_embedding"),
        ("retriever_bm25", RETRIEVER_BM25_PORT, "agents.retriever_bm25"),
        ("summarizer", SUMMARIZER_PORT, "agents.summarizer"),
        ("orchestrator", ORCHESTRATOR_PORT, "main"),
    ]

    print("\nStarting agents...")
    for name, port, module in procs_def:
        log_path = LOG_DIR / f"{name}.log"
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            [VENV_PYTHON, "-m", module],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        _agent_procs[name] = proc
        print(f"  {name}: PID={proc.pid}  port={port}  log={log_path}")

    print("\nWaiting for readiness...")
    for attempt in range(1, 6):
        all_ok = True
        for name, port in AGENT_PORTS.items():
            ok = is_port_open(port)
            if not ok:
                all_ok = False
        if all_ok:
            break
        time.sleep(10)
    for name, port in AGENT_PORTS.items():
        ok = "OK" if is_port_open(port) else "NOT READY"
        print(f"  {name}:{port} [{ok}]")

    print("\nAll agents started.")


def stop_agents():
    for name, proc in list(_agent_procs.items()):
        if proc.poll() is None:
            proc.terminate()
    time.sleep(1)
    for name, proc in list(_agent_procs.items()):
        if proc.poll() is None:
            proc.kill()
    _agent_procs.clear()

    for port in AGENT_PORTS.values():
        pid = _pid_on_port(port)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    time.sleep(1)
    remaining = sum(1 for p in AGENT_PORTS.values() if is_port_open(p))
    if remaining:
        print(f"\nWarning: {remaining} port(s) still occupied.")
    else:
        print("\nAll agents stopped.")


def _pid_on_port(port: int) -> int | None:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=3,
        )
        if result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


async def do_query():
    if not all_agents_running():
        print("\nAgents are not running. Start them first (option 3).")
        return

    query = input("\nEnter your question: ").strip()
    if not query:
        print("Empty query.")
        return

    from a2a.client import create_client, ClientConfig
    from a2a.types import (
        Message, Part, Role,
        SendMessageRequest, SendMessageConfiguration,
    )
    from common.config import A2A_VERSION
    from common.a2a_client import stream_to_dicts, A2AClientManager
    import httpx

    orch_url = ORCHESTRATOR_URL
    print(f"\nSending query to orchestrator ({orch_url})...")
    print(f"Query: {query}\n")

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0),
        headers={"A2A-Version": A2A_VERSION},
    )
    config = ClientConfig(httpx_client=http_client)
    client = await create_client(orch_url, client_config=config)
    try:
        start = time.time()
        request = SendMessageRequest(
            message=Message(
                role=Role.ROLE_USER,
                parts=[Part(text=query, media_type="text/plain")],
                message_id=uuid.uuid4().hex,
            ),
            configuration=SendMessageConfiguration(return_immediately=False),
        )
        responses: list[dict] = []
        print(f"Waiting for response...")
        part_num = 0
        async for resp in stream_to_dicts(client.send_message(request)):
            responses.append(resp)
            if "artifactUpdate" in resp:
                art = resp["artifactUpdate"].get("artifact", {})
                ans = A2AClientManager.extract_answer_from_artifacts([art])
                if ans:
                    part_num += 1
                    t = time.time() - start
                    print(f"\n--- part {part_num} ({t:.1f}s) ---")
                    print(ans)

        elapsed = time.time() - start

        all_artifact_dicts: list[dict] = []
        if part_num == 0:
            for resp in responses:
                if "task" in resp:
                    task = resp["task"]
                    all_artifact_dicts.extend(task.get("artifacts", []))
                elif "message" in resp:
                    all_artifact_dicts.append(resp["message"])

            answer = A2AClientManager.extract_answer_from_artifacts(all_artifact_dicts)
            if not answer:
                for art in all_artifact_dicts:
                    for part in art.get("parts", []):
                        if part.get("text"):
                            answer = part["text"]
                            break

            print(f"Response ({elapsed:.1f}s):")
            print("-" * 60)
            print(answer if answer else "[no answer]")
            print("-" * 60)
        else:
            print(f"\nDone ({elapsed:.1f}s, {part_num} parts).")
    finally:
        await client.close()
        await http_client.aclose()


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
                if any_agent_running():
                    stop_agents()
                else:
                    start_agents()
                input("\nPress Enter...")

            elif choice == "4":
                asyncio.run(do_query())
                input("\nPress Enter...")

            elif choice == "0":
                if any_agent_running():
                    answer = input("\nAgents are running. Stop them before exit? [Y/n]: ").strip().lower()
                    if answer != "n":
                        stop_agents()
                print("\nGoodbye.")
                break

            else:
                print(f"\nUnknown option: {choice}")
                input("Press Enter...")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        if any_agent_running():
            stop_agents()
        sys.exit(0)


if __name__ == "__main__":
    main()
