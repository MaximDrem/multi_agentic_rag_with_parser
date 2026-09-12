import re
from pathlib import Path
from typing import Optional

from common.models import Chunk


MAX_TOKENS = 512
OVERLAP_TOKENS = 128
TOKENS_PER_CHAR = 0.4


def estimate_tokens(text: str) -> int:
    return int(len(text) * TOKENS_PER_CHAR)


def split_by_tokens(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    max_chars = int(max_tokens / TOKENS_PER_CHAR)
    overlap_chars = int(overlap / TOKENS_PER_CHAR)
    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        chunks.append(text[start:end].strip())
        if end >= text_len:
            break
        start = end - overlap_chars
        if start <= 0:
            break
    return chunks


def parse_markdown_by_headers(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"^(#{1,3})\s+(.*?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body))
    return sections


def chunk_markdown_file(doc_path: Path, text: str) -> list[Chunk]:
    source_name = doc_path.name
    sections = parse_markdown_by_headers(text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for title, body in sections:
        if not body.strip():
            continue
        if estimate_tokens(body) <= MAX_TOKENS:
            chunk_id = f"{source_name}::chunk-{chunk_index:05d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_path=str(doc_path),
                    title=title or source_name,
                    text=body,
                    token_count=estimate_tokens(body),
                )
            )
            chunk_index += 1
        else:
            sub_chunks = split_by_tokens(body, MAX_TOKENS, OVERLAP_TOKENS)
            for j, sub in enumerate(sub_chunks):
                if not sub.strip():
                    continue
                chunk_id = f"{source_name}::chunk-{chunk_index:05d}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_path=str(doc_path),
                        title=title or source_name,
                        text=sub,
                        token_count=estimate_tokens(sub),
                    )
                )
                chunk_index += 1

    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk.prev_chunk_id = chunks[i - 1].id
        if i < len(chunks) - 1:
            chunk.next_chunk_id = chunks[i + 1].id

    return chunks


def chunk_directory(markdown_dir: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(markdown_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown_file(path, text))
    return all_chunks
