import re
from pathlib import Path

from common.models import Chunk


MAX_TOKENS = 512
OVERLAP_TOKENS = 128
MIN_TOKENS = 60          # мельче — склеиваем с соседом
TOKENS_PER_CHAR_CYR = 0.32   # кириллица: ~3.1 симв/токен (рус. юртекст)
TOKENS_PER_CHAR_OTHER = 0.40 # латиница/цифры/код

_CYR_RE = re.compile(r"[а-яА-ЯёЁ]")
_WS_RE = re.compile(r"\s")
_SENT_END_RE = re.compile(r"[.!?;:]")
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*$", re.MULTILINE)
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


def estimate_tokens(text: str) -> int:
    return int(len(text) * _tokens_per_char(text))


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


def split_by_tokens(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Скользящее окно с привязкой к границам предложений/строк/слов.

    Окончание окна никогда не проходит внутри слова или внутри строки
    таблицы; чанк-продолжение внутри длинной таблицы получает её заголовок.
    """
    coef = _tokens_per_char(text)
    max_chars = int(max_tokens / coef)
    overlap_chars = int(overlap / coef)
    tables = _table_blocks(text)

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        cut = _find_cut(text, start, max_chars, tables)
        if cut <= start:  # защита от вырождения
            cut = min(start + max_chars, text_len)
        piece = text[start:cut].strip()
        if piece:
            chunks.append(piece)
        if cut >= text_len:
            break
        start = max(cut - overlap_chars, start + 1)

    # чанк, начинающийся внутри длинной таблицы, снабжаем её заголовком
    fixed: list[str] = []
    for piece in chunks:
        offset = text.find(piece)
        if offset < 0:
            fixed.append(piece)
            continue
        for s, e, he in tables:
            if s < offset < e and he <= offset:
                header = text[s:he].strip()
                fixed.append(header + "\n" + piece)
                break
        else:
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
        sections.append((title, body))
    return sections


def _make_chunk(doc_path: Path, chunk_index: int, title: str, text: str) -> tuple[Chunk, int]:
    source_name = doc_path.name
    chunk = Chunk(
        id=f"{source_name}::chunk-{chunk_index:05d}",
        doc_path=str(doc_path),
        title=title or source_name,
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

    # склейка мелочи: чанк < MIN_TOKENS присоединяем к предыдущему (тот же документ),
    # заголовок секции сохраняем в тексте, чтобы не терять его термы
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and chunk.token_count < MIN_TOKENS
            and estimate_tokens(merged[-1].text) + chunk.token_count <= int(MAX_TOKENS * 1.25)
        ):
            prev = merged[-1]
            prev.text = prev.text + "\n\n" + chunk.title + "\n\n" + chunk.text
            prev.token_count = estimate_tokens(prev.text)
        else:
            merged.append(chunk)

    for i, chunk in enumerate(merged):
        chunk.prev_chunk_id = merged[i - 1].id if i > 0 else None
        chunk.next_chunk_id = merged[i + 1].id if i < len(merged) - 1 else None

    return merged


def chunk_directory(markdown_dir: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(markdown_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown_file(path, text))
    return all_chunks
