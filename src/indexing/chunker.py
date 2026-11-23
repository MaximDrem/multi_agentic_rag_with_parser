import bisect
import re
from functools import lru_cache
from pathlib import Path

from common.models import Chunk


MAX_TOKENS = 512
MAX_ATOMIC_TOKENS = 4096
MAX_STRUCTURAL_TOKENS = int(MAX_TOKENS * 1.25)
OVERLAP_TOKENS = 128
MIN_TOKENS = 60
TOKENS_PER_CHAR_CYR = 0.32
TOKENS_PER_CHAR_OTHER = 0.40

_CYR_RE = re.compile(r"[а-яА-ЯёЁ]")
_WS_RE = re.compile(r"\s")
_SENT_END_RE = re.compile(r"[.!?;:]")
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*$", re.MULTILINE)
_NON_STRUCTURAL_HEADER_RE = re.compile(r"^(?:[-•–]\s+|\d+\)\s+|[а-яА-Яa-zA-Z]\)\s+|№\s*\d+)")
_CLAUSE_RE = re.compile(
    r"^[ \t]*(?:#{1,4}[ \t]+)?\d+(?:\.\d+)*\.[ \t]+",
    re.MULTILINE,
)
_SENTENCE_BOUNDARY = r";[\"')\]»]*[ \t\n]+|(?<!\d)[.!?][\"')\]»]*[ \t\n]+(?=[A-ZА-ЯЁ0-9#|])"
_FUNC_WORDS = {
    "и", "в", "с", "по", "не", "на", "или", "из", "за", "к", "у", "о", "от",
    "до", "для", "при", "что", "которые", "который", "а", "но", "да", "также",
    "их", "его", "ее", "об", "со", "во", "без", "под", "над", "между",
}


def _tokens_per_char(text: str) -> float:
    if not text:
        return TOKENS_PER_CHAR_OTHER
    cyr = len(_CYR_RE.findall(text))
    share = cyr / len(text)
    return TOKENS_PER_CHAR_CYR * share + TOKENS_PER_CHAR_OTHER * (1 - share)


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v5-text-small", trust_remote_code=True)


@lru_cache(maxsize=8192)
def estimate_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


@lru_cache(maxsize=256)
def _token_offsets(text: str) -> tuple[tuple[int, int], ...]:
    encoded = _tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    return tuple((start, end) for start, end in encoded["offset_mapping"] if end > start)


def _table_blocks(text: str) -> list[tuple[int, int, int]]:
    blocks: list[tuple[int, int, int]] = []
    matches = list(_TABLE_ROW_RE.finditer(text))
    if not matches:
        return blocks
    run: list[re.Match] = []
    for m in matches:
        if run and not text[run[-1].end():m.start()].strip():
            run.append(m)
        else:
            if len(run) >= 2:
                blocks.append(_block_range(text, run))
            run = [m]
    if len(run) >= 2:
        blocks.append(_block_range(text, run))
    return blocks


def _block_range(text: str, run: list[re.Match]) -> tuple[int, int, int]:
    start = run[0].start()
    end = run[-1].end()
    header_end = end
    if len(run) >= 2 and re.match(r"^\s*\|[\s\-:|]+\|?\s*$", run[1].group()):
        header_end = run[1].end()
    return start, end, header_end


def _clause_blocks(text: str) -> list[tuple[int, int]]:
    matches = list(_CLAUSE_RE.finditer(text))
    if not matches:
        return []
    blocks = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if text[start:end].strip() and estimate_tokens(text[start:end]) <= MAX_STRUCTURAL_TOKENS:
            blocks.append((start, end))
    return blocks


def _clause_at(blocks: list[tuple[int, int]], position: int) -> tuple[int, int] | None:
    for start, end in blocks:
        if start <= position < end:
            return start, end
    return None


def _find_cut(
    text: str,
    start: int,
    max_chars: int,
    tables: list[tuple[int, int, int]],
    clauses: list[tuple[int, int]],
) -> int:
    hard_end = min(start + max_chars, len(text))
    if hard_end >= len(text):
        return len(text)

    clause = _clause_at(clauses, hard_end - 1)
    if clause:
        clause_start, clause_end = clause
        if start <= clause_start and estimate_tokens(text[start:clause_end]) <= MAX_STRUCTURAL_TOKENS:
            return clause_end
        if clause_start > start:
            return clause_start

    zone_start = start + int(max_chars * 0.65)
    best = -1
    for m in re.finditer(_SENTENCE_BOUNDARY + r"|\n{2,}", text[zone_start:hard_end]):
        best = m.end()
    if best >= 0 and zone_start + best > start + MIN_TOKENS and not _inside_table(tables, zone_start + best):
        return zone_start + best

    best = -1
    for m in re.finditer(r"\n(?=\s*\d+(?:\.\d+)*[.)][ \t])", text[zone_start:hard_end]):
        best = m.end()
    if best >= 0 and zone_start + best > start + MIN_TOKENS:
        return zone_start + best

    best = -1
    for m in re.finditer(r"\n(?=\s*\|)", text[zone_start:hard_end]):
        best = m.end()
    if best >= 0 and zone_start + best > start + MIN_TOKENS:
        return zone_start + best

    in_table = _inside_table(tables, start) or _inside_table(tables, hard_end)
    atomic_pattern = r"\n(?=\s*\|)" if in_table else _SENTENCE_BOUNDARY
    atomic_match = re.search(atomic_pattern, text[hard_end:])
    if atomic_match:
        atomic_end = hard_end + atomic_match.end()
        if estimate_tokens(text[start:atomic_end]) <= MAX_ATOMIC_TOKENS:
            return atomic_end
    elif estimate_tokens(text[start:]) <= MAX_ATOMIC_TOKENS:
        return len(text)

    tail15 = text[start + int(max_chars * 0.85):hard_end]
    comma = tail15.rfind(", ")
    if comma >= 0 and start + int(max_chars * 0.85) + comma > start + MIN_TOKENS:
        return start + int(max_chars * 0.85) + comma + 2

    tail = text[zone_start:hard_end]
    ws = max(tail.rfind(" "), tail.rfind("\n"))
    if ws >= 0 and zone_start + ws > start + MIN_TOKENS:
        pos = zone_start + ws + 1
        for _ in range(3):
            seg = text[start:pos].rstrip()
            m = re.search(r"([а-яА-ЯёЁa-zA-Z]+)[ \t\n]*$", seg)
            if m and m.group(1).lower() in _FUNC_WORDS and seg.rfind(" ") > 0:
                pos = start + seg.rfind(" ")
            else:
                break
        return pos

    return hard_end


def _inside_table(tables: list[tuple[int, int, int]], pos: int) -> bool:
    return any(s < pos < e for s, e, _ in tables)


def _overlap_start(
    text: str,
    current_start: int,
    target: int,
    cut: int,
    tables: list[tuple[int, int, int]],
    clauses: list[tuple[int, int]],
) -> int:
    if any(cut == start or start < cut <= end for start, end in clauses):
        return cut

    segment = text[current_start:cut]
    candidates = []
    in_table = _inside_table(tables, target) or _inside_table(tables, cut - 1)
    patterns = [r"\n+(?=\s*\|)"] if in_table else [
        _SENTENCE_BOUNDARY,
        r"\n{2,}",
        r"\n+(?=\s*\|)",
        r"\n+(?=\s*(?:[-•]|\d+(?:\.\d+)*[.)])\s)",
        r"\n+(?=\s*#{1,4}\s)",
    ]
    for pattern in patterns:
        candidates.extend(current_start + m.end() for m in re.finditer(pattern, segment))
    candidates = sorted({p for p in candidates if current_start < p < cut})
    after = [p for p in candidates if p >= target]
    if after:
        return after[0]
    before = [p for p in candidates if p < target]
    if before:
        return before[-1]
    if not candidates:
        return cut
    return target


def split_by_tokens(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    if not text.strip():
        return []
    offsets = _token_offsets(text)
    if len(offsets) <= max_tokens:
        return [text.strip()]

    tables = _table_blocks(text)
    clauses = _clause_blocks(text)
    starts = [start for start, _ in offsets]
    ends = [end for _, end in offsets]
    pieces: list[tuple[int, str]] = []
    start = 0

    while start < len(text):
        start_token = bisect.bisect_left(starts, start)
        if start_token >= len(offsets):
            break
        end_token = min(start_token + max_tokens, len(offsets))
        hard_end = ends[end_token - 1]
        cut = _find_cut(text, start, hard_end - start, tables, clauses)
        if cut <= start:
            cut = hard_end
        piece = text[start:cut].strip()
        if piece:
            pieces.append((start, piece))
        if cut >= len(text):
            break

        cut_token = bisect.bisect_right(ends, cut)
        overlap_token = max(cut_token - overlap, start_token + 1)
        if overlap_token >= len(offsets):
            break
        target = offsets[overlap_token][0]
        next_start = _overlap_start(text, start, target, cut, tables, clauses)
        start = max(min(next_start, cut), start + 1)

    fixed: list[str] = []
    for offset, piece in pieces:
        for table_start, table_end, header_end in tables:
            if table_start < offset < table_end and header_end <= offset:
                candidate = text[table_start:header_end].strip() + "\n" + piece
                if estimate_tokens(candidate) <= max_tokens or estimate_tokens(piece) > max_tokens:
                    piece = candidate
                break
        fixed.append(piece)
    return fixed


def _is_structural_header(title: str) -> bool:
    title = title.strip()
    if not title or _NON_STRUCTURAL_HEADER_RE.match(title):
        return False
    first_letter = re.search(r"[A-Za-zА-Яа-яЁё]", title)
    return not first_letter or not first_letter.group().islower()


def parse_markdown_by_headers(text: str) -> list[tuple[str, str]]:
    matches = [match for match in _HEADER_RE.finditer(text) if _is_structural_header(match.group(2))]
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        header = text[match.start():match.end()].strip()
        sections.append((title, f"{header}\n\n{body}" if body else header))
    return sections


def _make_chunk(doc_path: Path, chunk_index: int, title: str, text: str) -> tuple[Chunk, int]:
    source_name = doc_path.name
    chunk = Chunk(
        id=f"{source_name}::chunk-{chunk_index:05d}",
        doc_path=str(doc_path),
        title=source_name,
        text=text,
        token_count=estimate_tokens(text),
    )
    return chunk, chunk_index + 1


def chunk_markdown_file(doc_path: Path, text: str) -> list[Chunk]:
    sections = parse_markdown_by_headers(text)
    chunks: list[Chunk] = []
    idx = 0

    for title, body in sections:
        if not body.strip():
            continue
        if estimate_tokens(body) <= MAX_TOKENS:
            chunk, idx = _make_chunk(doc_path, idx, title, body)
            chunks.append(chunk)
        else:
            for sub in split_by_tokens(body, MAX_TOKENS, OVERLAP_TOKENS):
                if not sub.strip():
                    continue
                chunk, idx = _make_chunk(doc_path, idx, title, sub)
                chunks.append(chunk)

    merged: list[Chunk] = []
    for chunk in chunks:
        if merged and chunk.token_count < MIN_TOKENS:
            prev = merged[-1]
            candidate = prev.text + "\n\n" + chunk.text
            if estimate_tokens(candidate) <= int(MAX_TOKENS * 1.25):
                prev.text = candidate
                prev.token_count = estimate_tokens(candidate)
                continue
        merged.append(chunk)

    forward: list[Chunk] = []
    for i, chunk in enumerate(merged):
        if i + 1 < len(merged) and chunk.token_count < MIN_TOKENS:
            next_chunk = merged[i + 1]
            candidate = chunk.text + "\n\n" + next_chunk.text
            if estimate_tokens(candidate) <= int(MAX_TOKENS * 1.25):
                next_chunk.text = candidate
                next_chunk.token_count = estimate_tokens(candidate)
                continue
        forward.append(chunk)

    for i, chunk in enumerate(forward):
        chunk.id = f"{doc_path.name}::chunk-{i:05d}"
    for i, chunk in enumerate(forward):
        chunk.prev_chunk_id = forward[i - 1].id if i > 0 else None
        chunk.next_chunk_id = forward[i + 1].id if i < len(forward) - 1 else None

    return forward


def chunk_directory(markdown_dir: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(markdown_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown_file(path, text))
    return all_chunks
