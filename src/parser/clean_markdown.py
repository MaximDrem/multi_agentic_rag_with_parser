#!/usr/bin/env python3
"""
Очистка markdown-файлов из markdown_output/ и запись в markdown_output_clean/.
Преобразования:
1. Удаление многоточий-отбивок в оглавлении (сохраняем текст и номера страниц)
2. Сокращение длинных последовательностей подчёркиваний
3. Склейка разорванных строк в плотные абзацы для лучших векторов в RAG
"""

import re
from pathlib import Path


def clean_toc_dots(text: str) -> str:
    """
    Убирает точечные отбивки в строках оглавления.
    "3.1 Текст . . . . . . . . . . 42" -> "3.1 Текст 42"
    """
    # Последовательности пробелов и точек (3+ точек)
    text = re.sub(r'(?:\s\.){3,}\s*', '  ', text)
    return text


def shorten_underscores(text: str) -> str:
    """Заменяет длинные подчёркивания (5+) на '__'."""
    text = re.sub(r'_{5,}', '__', text)
    return text


def collapse_multispace(text: str) -> str:
    """Схлопывает множественные пробелы в один."""
    text = re.sub(r'  +', ' ', text)
    return text


def is_boundary_start(stripped: str) -> bool:
    """Проверяет, является ли строка началом нового структурного элемента."""
    if not stripped:
        return True
    if stripped.startswith('#'):
        return True
    if stripped.startswith('|'):
        return True
    if re.match(r'^[-•*•]\s', stripped):
        return True
    # Нумерованные пункты: "1)", "а)", "1.2.3)", "3.1.1)"
    if re.match(r'^(?:\d+(?:\.\d+)*|[а-яё])\)', stripped):
        return True
    # Нумерация разделов: "3.1 ", "3.1.1 " — номер с точками
    if re.match(r'^\d+\.\d+', stripped):
        return True
    # Приложения и Библиография (в т.ч. после TOC-разбивки)
    if re.match(r'^Приложение\s+[А-ЯA-Z\d]', stripped):
        return True
    if stripped.startswith('Библиография'):
        return True
    return False


def is_sentence_end(text: str) -> bool:
    """Проверяет, заканчивается ли строка знаком конца предложения."""
    return bool(re.search(r'[.!?;:)"»—]$', text))


def consolidate_paragraphs(text: str) -> str:
    """
    Склеивает разорванные строки в плотные абзацы.
    Правила:
    - Строки, заканчивающиеся на .!?:;)»—  — конец предложения, сброс буфера.
    - Строки с ­ (soft hyphen) в конце — склейка без пробела.
    - Строки с дефисом в конце — склейка без пробела (перенос слова).
    - Остальное — склейка через пробел (продолжение предложения).
    - Структурные маркеры (# | - • 1) 4.1) — всегда сброс буфера.
    """
    lines = text.split('\n')
    result = []
    para_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if para_lines:
                result.append(' '.join(para_lines))
                para_lines = []
            result.append('')
            continue

        if is_boundary_start(stripped):
            if para_lines:
                result.append(' '.join(para_lines))
                para_lines = []
            result.append(stripped)
            continue

        if not para_lines:
            para_lines.append(stripped)
            continue

        prev = para_lines[-1]

        # Soft hyphen (U+00AD) or regular hyphen at end of previous line = word break
        if prev.rstrip()[-1:] in ('\xad',):
            para_lines[-1] = prev.rstrip('\xad') + stripped
            continue

        if prev.endswith('-'):
            para_lines[-1] = prev + stripped
            continue

        # If prev ends with sentence-ending punctuation, flush and start new
        if is_sentence_end(prev.rstrip()):
            result.append(' '.join(para_lines))
            para_lines = [stripped]
            continue

        # Continue current paragraph
        para_lines.append(stripped)

    if para_lines:
        result.append(' '.join(para_lines))

    # Collapse multiple empty lines
    final = []
    prev_empty = False
    for l in result:
        if l == '':
            if not prev_empty:
                final.append(l)
            prev_empty = True
        else:
            final.append(l)
            prev_empty = False

    return '\n'.join(final).strip()


TOC_ENTRY_RE = re.compile(
    r'((?:\d+\.\d+(?:\.\d+)?|'
    r'(?<!\.\d)\b\d+\s+(?=[А-ЯA-Z])|'
    r'Приложение\s+[А-ЯA-Z]|'
    r'Библиография)'
    r'\b.*?)\s{2,}(\d+)'
)


def _split_toc_block(block_text: str) -> str:
    """Разбить блок оглавления на строки по одной записи на строку."""
    entries = TOC_ENTRY_RE.findall(block_text)
    if not entries:
        return block_text
    return '\n'.join(f'{title.strip()}  {page}' for title, page in entries)


def split_toc_entries(text: str) -> str:
    """
    Собирает строки оглавления и разбивает каждую запись на отдельную строку.
    Определяет TOC-блок: от «Содержание» до следующего markdown-заголовка.
    """
    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        if stripped == 'Содержание':
            # Собираем все строки до маркдаун-заголовка (## или ###)
            result.append(stripped)
            i += 1
            toc_parts = []
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith('##') or s.startswith('###'):
                    break
                if s:
                    toc_parts.append(s)
                i += 1
            if toc_parts:
                result.append(_split_toc_block(' '.join(toc_parts)))
                result.append('')
            continue

        result.append(stripped)
        i += 1

    return '\n'.join(result)


def clean_text(text: str) -> str:
    """Применить все очищающие преобразования."""
    if not text:
        return ''

    text = clean_toc_dots(text)
    text = shorten_underscores(text)
    text = consolidate_paragraphs(text)
    return text


def process_files(input_dir: str, output_dir: str):
    """Обработать все .md файлы из input_dir в output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f'Директория {input_dir} не найдена')
        return

    output_path.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_path.glob('*.md'))
    print(f'Найдено файлов: {len(md_files)}')
    print()

    for file in md_files:
        try:
            content = file.read_text(encoding='utf-8')

            # Пропускаем пустые файлы
            if not content.strip():
                print(f'  Пропущен (пустой): {file.name}')
                continue

            cleaned = clean_text(content)

            out_file = output_path / file.name
            out_file.write_text(cleaned, encoding='utf-8')

            orig_lines = content.count('\n')
            new_lines = cleaned.count('\n')
            reduction = (1 - new_lines / orig_lines) * 100 if orig_lines > 0 else 0
            print(f'  Очищен: {file.name}  |  строк: {orig_lines} -> {new_lines}  ({reduction:.0f}% сжато)')

        except Exception as e:
            print(f'  Ошибка {file.name}: {e}')

    print()
    print(f'Готово. Результаты в {output_dir}/')


if __name__ == '__main__':
    from parser.config import OUTPUT_DIR, CLEAN_OUTPUT_DIR
    process_files(OUTPUT_DIR, CLEAN_OUTPUT_DIR)
