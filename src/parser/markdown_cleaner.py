"""Пост-обработка Markdown: удаление мусора, многоточий, ссылок, пустых строк."""

import re


def clean_markdown(text: str) -> str:
    """Очистка Markdown от паразитных элементов."""

    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Пропускаем невидимые/мусорные строки
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        # Пропускаем строки, состоящие ТОЛЬКО из спецсимволов/точек/пробелов
        if not stripped.startswith("|") and re.fullmatch(r"[\.\-\—\–\s_\*#\|\+\=~\^]+", stripped):
            continue

        # Пропускаем строки-разделители из точек (типа "..........")
        if re.fullmatch(r"\.{3,}", stripped):
            continue

        # Удаляем строки с большим количеством подчеркиваний (отбивки)
        if re.fullmatch(r"_{5,}", stripped):
            continue

        # Удаляем строки из дефисов/тире
        if re.fullmatch(r"[\-\—\–]{5,}", stripped):
            continue

        # Очистка самой строки
        cleaned = _clean_line(stripped)

        if cleaned:
            cleaned_lines.append(cleaned)
        else:
            cleaned_lines.append("")

    # Схлопываем множественные пустые строки
    result = []
    prev_empty = False
    for line in cleaned_lines:
        if line == "":
            if not prev_empty:
                result.append(line)
            prev_empty = True
        else:
            result.append(line)
            prev_empty = False

    return "\n".join(result).strip()


def _clean_line(line: str) -> str:
    """Очистка одной строки."""

    # Удаляем многоточия и неразрывные пробелы
    line = line.replace("\xa0", " ")
    line = line.replace("\u00a0", " ")

    # Заменяем комбинации ……… → …
    line = re.sub(r"…{2,}", "…", line)

    # Удаляем бессмысленные линии из ...... (оглавление)
    # Но если это часть нормального текста — оставляем
    # Удаляем длинные последовательности точек между словами
    line = re.sub(r"\.{5,}", "", line)

    # Удаляем заставочные номера страниц (одиноко стоящее число)
    # Не трогаем строки, которые являются частью заголовков

    # Схлопываем множественные пробелы
    line = re.sub(r" {2,}", " ", line)

    return line.strip()


def normalize_headings(text: str) -> str:
    """Нормализация заголовков: убираем лишние #, чистим отбивки."""

    lines = text.split("\n")
    result = []

    for line in lines:
        stripped = line.strip()

        # Если строка начинается с # — это заголовок
        if stripped.startswith("#"):
            # Удаляем лишние пробелы после #
            mo = re.match(r"^(#{1,6})\s*(.+)$", stripped)
            if mo:
                level = min(len(mo.group(1)), 6)
                title = mo.group(2).strip()
                # Нормализуем заголовок
                title = re.sub(r"\s{2,}", " ", title)
                result.append(f"{'#' * level} {title}")
            else:
                result.append(stripped)
        else:
            result.append(stripped)

    return "\n".join(result)


def remove_redundant_linebreaks(text: str) -> str:
    """Удаление избыточных переносов строк внутри предложений."""

    lines = text.split("\n")
    result = []
    buffer = []

    for line in lines:
        stripped = line.strip()

        # Если строка пустая или заголовок/таблица — сбрасываем буфер
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            if buffer:
                result.append(" ".join(buffer))
                buffer = []
            result.append(stripped)
            continue

        # Если строка заканчивается на точку/вопросительный/восклицательный знак — конец предложения
        if stripped.endswith((".", "!", "?", ":", ";", ")")):
            buffer.append(stripped)
            result.append(" ".join(buffer))
            buffer = []
        else:
            buffer.append(stripped)

    if buffer:
        result.append(" ".join(buffer))

    return "\n".join(result)
