"""Обработчик XLSX → Markdown с таблицами."""

import os
import re
from pathlib import Path
import openpyxl
from openpyxl.utils import get_column_letter

from parser.config import TEMP_DIR
from parser.markdown_cleaner import clean_markdown


def process_xlsx(filepath: str, ocr_client=None) -> str:
    """
    Конвертирует XLSX в структурированный Markdown.

    Каждый лист → раздел ##.
    Каждая таблица на листе → markdown-таблица.
    Изображения (если есть) → через OCR.

    Args:
        filepath: путь к .xlsx файлу
        ocr_client: опциональный экземпляр OCRCient для распознавания изображений

    Returns:
        строка Markdown
    """

    filename = os.path.basename(filepath)
    md_lines = [f"# {Path(filepath).stem}", ""]

    wb = openpyxl.load_workbook(filepath, data_only=True)

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        if ws.max_row == 0:
            continue

        md_lines.append(f"## {sheet_name}")
        md_lines.append("")

        # Находим таблицы на листе (области с данными)
        table_ranges = _find_data_regions(ws)

        for trange in table_ranges:
            table_md = _range_to_markdown_table(ws, trange)
            if table_md:
                md_lines.append(table_md)
                md_lines.append("")

        # Проверяем наличие изображений на листе
        images_md = _extract_sheet_images(ws, filepath, sheet_name)
        if images_md:
            md_lines.append(images_md)

    wb.close()

    md_text = "\n".join(md_lines)
    md_text = clean_markdown(md_text)

    return md_text


def _find_data_regions(ws) -> list:
    """
    Находит непрерывные области данных на листе.
    Возвращает список кортежей (min_row, min_col, max_row, max_col).
    """

    non_empty = []
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            if cell.value is not None and str(cell.value).strip():
                non_empty.append((cell.row, cell.column))

    if not non_empty:
        return []

    # Простой алгоритм: группируем строки с промежутками не более 2 пустых
    rows_with_data = sorted(set(r for r, c in non_empty))
    cols_with_data = sorted(set(c for r, c in non_empty))

    if not rows_with_data:
        return []

    # Находим границы
    regions = []
    start_row = rows_with_data[0]
    end_row = start_row

    for i in range(1, len(rows_with_data)):
        if rows_with_data[i] - end_row <= 3:
            end_row = rows_with_data[i]
        else:
            regions.append((start_row, min(cols_with_data), end_row, max(cols_with_data)))
            start_row = rows_with_data[i]
            end_row = start_row

    regions.append((start_row, min(cols_with_data), end_row, max(cols_with_data)))

    return regions


def _range_to_markdown_table(ws, trange: tuple) -> str:
    """Конвертирует диапазон ячеек в Markdown-таблицу."""

    min_row, min_col, max_row, max_col = trange

    if max_row - min_row < 1:
        return ""

    rows = []
    for row_idx in range(min_row, max_row + 1):
        cells = []
        for col_idx in range(min_col, max_col + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            value = cell.value
            if value is None:
                cell_text = ""
            else:
                cell_text = str(value).strip().replace("\n", " ").replace("|", "\\|")
            cells.append(cell_text)
        rows.append(cells)

    # Убираем строки где все ячейки пустые
    rows = [r for r in rows if any(c.strip() for c in r)]

    if not rows:
        return ""

    # Определяем header (первая непустая строка, которая похожа на заголовок)
    header_idx = 0
    for i, row in enumerate(rows):
        if any(c.strip() for c in row):
            header_idx = i
            break

    # Если перед данными были пустые строки — убираем их
    rows = rows[header_idx:]

    # Проверяем, есть ли строки перед header которые являются заголовком таблицы
    # (например "Таблица 1 - ...")

    # Строим MD
    lines = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(row)) + " |")

    return "\n" + "\n".join(lines) + "\n"


def _extract_sheet_images(ws, filepath: str, sheet_name: str) -> str:
    """Извлекает изображения из листа Excel."""

    images = ws._images if hasattr(ws, '_images') else []
    if not images:
        return ""

    results = []
    for idx, img in enumerate(images):
        try:
            from PIL import Image
            import io

            img_data = img._data()
            pil_img = Image.open(io.BytesIO(img_data))

            os.makedirs(TEMP_DIR, exist_ok=True)
            img_name = f"xlsx_{Path(filepath).stem}_{sheet_name}_{idx}.png"
            img_path = os.path.join(TEMP_DIR, img_name)
            pil_img.save(img_path, "PNG")

            results.append(f"[Изображение: {img_path}]")

        except Exception:
            pass

    if results:
        return "\n" + "\n".join(results) + "\n"
    return ""
