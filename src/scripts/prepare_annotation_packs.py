import hashlib
import json
import shutil
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.chunk_store import ChunkDatabase


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    root = Path.cwd()
    benchmark = root / "var" / "benchmark"
    input_dir = benchmark / "input"
    agents_dir = benchmark / "agents"
    input_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    xlsx = root / "vopros-otvety (1).xlsx"
    workbook = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    question_index = header.index("Вопрос")
    questions = []
    for source_row, row in enumerate(rows, 2):
        value = row[question_index]
        if value is None or not str(value).strip():
            continue
        questions.append({
            "question_id": f"row_{source_row:04d}",
            "source_row": source_row,
            "query": str(value).strip(),
        })
    write_jsonl(input_dir / "questions.jsonl", questions)
    shutil.copy2(xlsx, input_dir / xlsx.name)

    all_chunks = ChunkDatabase.load(root / "var" / "data" / "chunks.pkl").chunks
    invalid_titles = [chunk.id for chunk in all_chunks if chunk.title != Path(chunk.doc_path).name]
    if invalid_titles:
        raise ValueError(f"Title не совпадает с именем файла: {invalid_titles[:5]}")
    by_document = {}
    for chunk in all_chunks:
        by_document.setdefault(chunk.doc_path, []).append(chunk)
    documents = sorted(by_document.items(), key=lambda item: (-len(item[1]), item[0]))
    packs = [{"documents": [], "chunk_count": 0} for _ in range(4)]
    for doc_path, chunks in documents:
        available = [index for index, pack in enumerate(packs) if len(pack["documents"]) < 20]
        target = min(available, key=lambda index: (packs[index]["chunk_count"], len(packs[index]["documents"]), index))
        packs[target]["documents"].append((doc_path, chunks))
        packs[target]["chunk_count"] += len(chunks)

    manifest_packs = []
    for index, pack in enumerate(packs, 1):
        agent_id = f"agent_{index:02d}"
        agent_dir = agents_dir / agent_id
        documents_dir = agent_dir / "documents"
        output_dir = agent_dir / "output"
        documents_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_rows = []
        document_rows = []
        for doc_path, chunks in sorted(pack["documents"], key=lambda item: item[0]):
            source = root / doc_path
            destination = documents_dir / source.name
            shutil.copy2(source, destination)
            document_rows.append({
                "doc_path": doc_path,
                "document_file": f"documents/{source.name}",
                "chunk_count": len(chunks),
            })
            for chunk in chunks:
                chunk_rows.append({
                    "chunk_id": chunk.id,
                    "doc_path": chunk.doc_path,
                    "document_file": f"documents/{source.name}",
                    "title": chunk.title,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "prev_chunk_id": chunk.prev_chunk_id,
                    "next_chunk_id": chunk.next_chunk_id,
                })
        write_jsonl(agent_dir / "chunks.jsonl", chunk_rows)
        manifest = {
            "agent_id": agent_id,
            "document_count": len(document_rows),
            "chunk_count": len(chunk_rows),
            "questions_xlsx": "input/vopros-otvety (1).xlsx",
            "questions_jsonl": "input/questions.jsonl",
            "chunks": f"agents/{agent_id}/chunks.jsonl",
            "output_dir": f"agents/{agent_id}/output",
            "documents": document_rows,
        }
        write_json(agent_dir / "manifest.json", manifest)
        manifest_packs.append({
            "agent_id": agent_id,
            "document_count": len(document_rows),
            "chunk_count": len(chunk_rows),
        })

    corpus_payload = json.dumps([[chunk.id, chunk.text] for chunk in all_chunks], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    write_json(benchmark / "manifest.json", {
        "question_count": len(questions),
        "document_count": len(by_document),
        "chunk_count": len(all_chunks),
        "chunker": "structural-hybrid-jina-token-v3",
        "corpus_sha256": hashlib.sha256(corpus_payload).hexdigest(),
        "packs": manifest_packs,
    })
    print(json.dumps({"questions": len(questions), "packs": manifest_packs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
