import bisect
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from common.models import Chunk


MAX_TOKENS = 512
MAX_ATOMIC_TOKENS = 4096
OVERLAP_TOKENS = 128
MIN_TOKENS = 60          # мельче — склеиваем с соседом
TOKENS_PER_CHAR_CYR = 0.32   # кириллица: ~3.1 симв/токен (рус. юртекст)
TOKENS_PER_CHAR_OTHER = 0.40 # латиница/цифры/код

_CYR_RE = re.compile(r"[а-яА-ЯёЁ]")
_WS_RE = re.compile(r"\s")
_SENT_END_RE = re.compile(r"[.!?;:]")
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*$", re.MULTILINE)
_LIST_HEADING_RE = re.compile(r"^\s*#{1,4}\s+")
_LIST_BULLET_RE = re.compile(r"[-•–]\s+")
_LIST_ALPHA_RE = re.compile(r"[а-яА-Яa-zA-Z]\)\s+")
_LIST_NUMBER_PAREN_RE = re.compile(r"\d+\)\s+")
_LIST_NUMBER_DOT_RE = re.compile(r"\d+\.\s+")
_LIST_LOWER_RE = re.compile(r"[а-яёa-z]")
_LIST_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")
_LIST_PARAGRAPH_RE = re.compile(r"(?:^|\n[ \t]*\n)(.*?)(?=\n[ \t]*\n|$)", re.DOTALL)
_LIST_SENTENCE_END_RE = re.compile(r"(?<!\d)[.!?](?:\s|$)")
_MAX_CONTINUATION_TOKENS = int(MAX_TOKENS * 1.25)
_FUNC_WORDS = {
    "и", "в", "с", "по", "не", "на", "или", "из", "за", "к", "у", "о", "от",
    "до", "для", "при", "что", "которые", "который", "а", "но", "да", "также",
    "их", "его", "ее", "об", "со", "во", "без", "под", "над", "между",
}


def _tokens_per_char(text: str) -> float:
    """Калиброванная оценка: доля кириллицы задаёт коэффициент."""
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
    """Диапазоны markdown-таблиц: (start, end, header_end).

    header_end — конец строки-разделителя (2-я строка таблицы), которую
    тащим в начало чанков-продолжений.
    """
    blocks: list[tuple[int, int, int]] = []
    matches = list(_TABLE_ROW_RE.finditer(text))
    if not matches:
        return blocks
    run: list[re.Match] = []
    for m in matches:
        if run and not text[run[-1].end():m.start()].strip():
            run.append(m)  # соседние строки (между ними только перевод строки)
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


def _find_cut(text: str, start: int, max_chars: int, tables: list[tuple[int, int, int]]) -> int:
    """Граница реза окна: конец предложения > строка таблицы > граница слова.

    Никогда не режем строго внутри строки таблицы.
    """
    hard_end = min(start + max_chars, len(text))
    if hard_end >= len(text):
        return len(text)

    zone_start = start + int(max_chars * 0.65)  # зона поиска — последние 35% окна

    # 1) последний конец предложения в зоне: точка/!?/; + любой пробел.
    # Точка после цифры не считается («1. », «31. » — нумерация, не конец).
    best = -1
    for m in re.finditer(r"(?<!\d)[.!?;][ \t\n]+|\n{2,}", text[zone_start:hard_end]):
        best = m.end()
    if best >= 0 and zone_start + best > start + MIN_TOKENS and not _inside_table(tables, zone_start + best):
        return zone_start + best

    # 1.5) граница нумерованного пункта: начало строки «12. » / «12.1) »
    best = -1
    for m in re.finditer(r"\n(?=\s*\d+(?:\.\d+)*[.)][ \t])", text[zone_start:hard_end]):
        best = m.end()
    if best >= 0 and zone_start + best > start + MIN_TOKENS:
        return zone_start + best

    # 2) граница строки таблицы (после \n перед |)
    best = -1
    for m in re.finditer(r"\n(?=\s*\|)", text[zone_start:hard_end]):
        best = m.end()
    if best >= 0 and zone_start + best > start + MIN_TOKENS:
        return zone_start + best

    in_table = _inside_table(tables, start) or _inside_table(tables, hard_end)
    atomic_pattern = r"\n(?=\s*\|)" if in_table else r"(?<!\d)[.!?;][\"')\]»]*[ \t\n]+"
    atomic_match = re.search(atomic_pattern, text[hard_end:])
    if atomic_match:
        atomic_end = hard_end + atomic_match.end()
        if estimate_tokens(text[start:atomic_end]) <= MAX_ATOMIC_TOKENS:
            return atomic_end
    elif estimate_tokens(text[start:]) <= MAX_ATOMIC_TOKENS:
        return len(text)

    # 3) последняя запятая в последних 15% окна — лучше, чем оборвать на «не»/«с»
    tail15 = text[start + int(max_chars * 0.85):hard_end]
    comma = tail15.rfind(", ")
    if comma >= 0 and start + int(max_chars * 0.85) + comma > start + MIN_TOKENS:
        return start + int(max_chars * 0.85) + comma + 2

    # 4) последняя граница слова (пробел/перенос); если режем на предлоге/союзе
    # («...обязанность по|»), откатываемся ещё на слово назад
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


def _overlap_start(text: str, current_start: int, target: int, cut: int, tables: list[tuple[int, int, int]]) -> int:
    segment = text[current_start:cut]
    candidates = []
    in_table = _inside_table(tables, target) or _inside_table(tables, cut - 1)
    patterns = [r"\n+(?=\s*\|)"] if in_table else [
        r"(?<!\d)[.!?;][\"')\]»]*[ \t\n]+",
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
        cut = _find_cut(text, start, hard_end - start, tables)
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
        next_start = _overlap_start(text, start, target, cut, tables)
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


def parse_markdown_by_headers(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        header = text[m.start():m.end()].strip()
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


@dataclass(frozen=True)
class _ListContext:
    prefix: str
    style: str


def _list_item_style(text: str, allow_numeric_dot: bool = False) -> str | None:
    stripped = _LIST_HEADING_RE.sub("", text, count=1).lstrip()
    if _LIST_BULLET_RE.match(stripped):
        return "bullet"
    if _LIST_ALPHA_RE.match(stripped):
        return "alpha"
    if _LIST_NUMBER_PAREN_RE.match(stripped):
        return "numeric_paren"
    if allow_numeric_dot and _LIST_NUMBER_DOT_RE.match(stripped):
        return "numeric_dot"
    if _LIST_LOWER_RE.match(stripped):
        return "lower"
    return None


def _list_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in _LIST_BLANK_LINE_RE.split(text) if part.strip()]


def _list_opener(text: str) -> tuple[str, str] | None:
    paragraphs = list(_LIST_PARAGRAPH_RE.finditer(text))
    candidates = [match for match in paragraphs if match.group(1).strip().endswith(":")]
    if not candidates:
        return None
    match = candidates[-1]
    prefix = match.group(1).strip()
    complete = _list_item_style(prefix, allow_numeric_dot=True) != "lower"
    if not complete:
        preceding = _list_paragraphs(text[:match.start(1)])
        if preceding and preceding[-1].lstrip().startswith("#"):
            prefix = preceding[-1] + "\n\n" + prefix
            complete = True
    if not complete:
        return None
    return prefix, text[match.end():]


def _independent_numeric_opener(text: str) -> bool:
    paragraphs = _list_paragraphs(text)
    return bool(
        paragraphs
        and paragraphs[0].endswith(":")
        and _list_item_style(text, allow_numeric_dot=True) == "numeric_dot"
    )


def _list_body_style(text: str) -> str | None:
    styled = []
    for paragraph in _list_paragraphs(text):
        style = _list_item_style(paragraph)
        if style:
            styled.append((paragraph, style))
    if not styled:
        return None
    first_style = styled[0][1]
    if first_style == "lower":
        lower_items = sum(
            style == "lower" and paragraph.rstrip().endswith(";")
            for paragraph, style in styled
        )
        if lower_items < 2:
            return None
    return first_style


def _last_list_item_style(text: str) -> str | None:
    explicit_styles = []
    for line in text.splitlines():
        style = _list_item_style(line, allow_numeric_dot=True)
        if style and style != "lower":
            explicit_styles.append(style)
    if explicit_styles:
        return explicit_styles[-1]
    for paragraph in reversed(_list_paragraphs(text)):
        if _list_item_style(paragraph) == "lower":
            return "lower"
    return None


def _list_is_unfinished(text: str) -> bool:
    semicolon = text.rfind(";")
    return semicolon >= 0 and not _LIST_SENTENCE_END_RE.search(text[semicolon + 1:])


def _make_list_continuations_standalone(chunks: list[Chunk]) -> int:
    snapshot = tuple(
        (chunk.id, chunk.prev_chunk_id, chunk.next_chunk_id, chunk.text)
        for chunk in chunks
    )
    contexts: dict[int, _ListContext] = {}
    changed = 0

    for i, chunk in enumerate(chunks):
        if i == 0:
            continue
        current_id, current_prev, _, current_text = snapshot[i]
        previous_id, _, previous_next, previous_text = snapshot[i - 1]
        if current_prev != previous_id or previous_next != current_id:
            continue

        opener = _list_opener(previous_text)
        previous_context = contexts.get(i - 1)
        existing_prefixes = []
        if previous_context:
            existing_prefixes.append(previous_context.prefix)
        if opener:
            existing_prefixes.append(opener[0])

        already_prefixed = False
        for prefix in dict.fromkeys(existing_prefixes):
            marker = prefix + "\n\n"
            if current_text.startswith(marker):
                remainder = current_text[len(marker):]
                style = _list_item_style(remainder) or _list_item_style(
                    remainder, allow_numeric_dot=True
                )
                if style:
                    contexts[i] = _ListContext(prefix, style)
                already_prefixed = True
                break
        if already_prefixed or _independent_numeric_opener(current_text):
            continue

        prefix = None
        style = _list_item_style(current_text)
        if opener and previous_text.rstrip().endswith(":"):
            colon_style = _list_item_style(current_text, allow_numeric_dot=True)
            if colon_style:
                prefix = opener[0]
                style = colon_style
        elif (
            opener
            and style
            and previous_text.rstrip().endswith(";")
            and _list_body_style(opener[1]) == style
            and _last_list_item_style(previous_text) == style
        ):
            prefix = opener[0]
        elif (
            previous_context
            and style
            and previous_context.style == style
            and _last_list_item_style(previous_text) == style
            and _list_is_unfinished(previous_text)
        ):
            prefix = previous_context.prefix

        if prefix is None or style is None:
            continue
        contexts[i] = _ListContext(prefix, style)
        if prefix in current_text:
            continue
        candidate = prefix + "\n\n" + current_text
        token_count = estimate_tokens(candidate)
        if token_count <= _MAX_CONTINUATION_TOKENS:
            chunk.text = candidate
            chunk.token_count = token_count
            changed += 1

    return changed


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

    # склейка мелочи: чанк < MIN_TOKENS присоединяем к предыдущему (тот же документ),
    # заголовок секции сохраняем в тексте, чтобы не терять его термы
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

    _make_list_continuations_standalone(forward)
    return forward


def chunk_directory(markdown_dir: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(markdown_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown_file(path, text))
    return all_chunks
