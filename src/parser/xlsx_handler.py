"""Обработчик XLSX → Markdown с таблицами."""

import os
import re
from pathlib import Path

import openpyxl

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

    md_lines = [f"# {Path(filepath).stem}", ""]
    wb_values = openpyxl.load_workbook(filepath, data_only=True)
    wb_formulas = None

    try:
        wb_formulas = openpyxl.load_workbook(filepath, data_only=False)

        for sheet_name in wb_values.sheetnames:
            ws = wb_values[sheet_name]
            formula_ws = wb_formulas[sheet_name]

            if ws.max_row == 0:
                continue

            md_lines.append(f"## {sheet_name}")
            md_lines.append("")

            table_ranges = _find_data_regions(ws, formula_ws)

            for trange in table_ranges:
                table_md = _range_to_markdown_table(ws, trange, formula_ws)
                if table_md:
                    md_lines.append(table_md)
                    md_lines.append("")

            images_md = _extract_sheet_images(ws, filepath, sheet_name, ocr_client)
            if images_md:
                md_lines.append(images_md)

        return clean_markdown("\n".join(md_lines))
    finally:
        if wb_formulas is not None:
            wb_formulas.close()
        wb_values.close()


def _formula_text(cell) -> str:
    if cell is None:
        return ""

    value = cell.value
    data_type = getattr(cell, "data_type", None)
    if data_type is not None and data_type != "f":
        return ""
    if data_type is None and not (isinstance(value, str) and value.startswith("=")):
        return ""

    value = getattr(value, "text", value)
    if value is None:
        return ""
    return str(value).strip()


def _hyperlink_target(*cells) -> str:
    for cell in cells:
        if cell is None:
            continue
        hyperlink = getattr(cell, "hyperlink", None)
        if not hyperlink:
            continue
        if isinstance(hyperlink, str):
            return hyperlink.strip()
        target = getattr(hyperlink, "target", None) or getattr(hyperlink, "location", None)
        if target:
            return str(target).strip()
    return ""


def _cell_text(value_cell, formula_cell=None) -> str:
    formula_source = formula_cell if formula_cell is not None else value_cell
    formula = _formula_text(formula_source)
    cached_value = value_cell.value

    if formula:
        if cached_value is not None and str(cached_value).strip():
            text = f"{cached_value} ({formula})"
        else:
            text = formula
    else:
        text = "" if cached_value is None else str(cached_value)

    text = re.sub(r"[\r\n]+", " ", text).strip().replace("|", "\\|")
    target = _hyperlink_target(value_cell, formula_source)
    if target:
        label = text or target.replace("|", "\\|")
        target = re.sub(r"[\r\n]+", " ", target).replace("|", "\\|")
        return f"[{label}]({target})"
    return text


def _find_data_regions(ws, formula_ws=None) -> list:
    """
    Находит непрерывные области данных на листе.
    Возвращает список кортежей (min_row, min_col, max_row, max_col).
    """

    non_empty = []
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            formula_cell = formula_ws.cell(row=cell.row, column=cell.column) if formula_ws else None
            if _cell_text(cell, formula_cell).strip():
                non_empty.append((cell.row, cell.column))

    if not non_empty:
        return []

    rows_with_data = sorted(set(r for r, c in non_empty))
    regions = []
    start_row = rows_with_data[0]
    end_row = start_row

    for row_idx in rows_with_data[1:]:
        if row_idx - end_row <= 3:
            end_row = row_idx
            continue

        region_cols = [c for r, c in non_empty if start_row <= r <= end_row]
        regions.append((start_row, min(region_cols), end_row, max(region_cols)))
        start_row = row_idx
        end_row = row_idx

    region_cols = [c for r, c in non_empty if start_row <= r <= end_row]
    regions.append((start_row, min(region_cols), end_row, max(region_cols)))
    return regions


def _range_to_markdown_table(ws, trange: tuple, formula_ws=None) -> str:
    """Конвертирует диапазон ячеек в Markdown-таблицу."""

    min_row, min_col, max_row, max_col = trange
    rows = []

    for row_idx in range(min_row, max_row + 1):
        cells = []
        for col_idx in range(min_col, max_col + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            formula_cell = formula_ws.cell(row=row_idx, column=col_idx) if formula_ws else None
            cells.append(_cell_text(cell, formula_cell))
        rows.append(cells)

    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return ""

    lines = []
    for idx, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if idx == 0:
            lines.append("| " + " | ".join(["---"] * len(row)) + " |")

    return "\n" + "\n".join(lines) + "\n"


def _extract_sheet_images(ws, filepath: str, sheet_name: str, ocr_client=None) -> str:
    """Извлекает изображения из листа Excel."""

    images = ws._images if hasattr(ws, "_images") else []
    if not images:
        return ""

    results = []
    for idx, img in enumerate(images):
        from PIL import Image
        import io

        img_data = img._data()
        with Image.open(io.BytesIO(img_data)) as pil_img:
            os.makedirs(TEMP_DIR, exist_ok=True)
            img_name = f"xlsx_{Path(filepath).stem}_{sheet_name}_{idx}.png"
            img_path = os.path.join(TEMP_DIR, img_name)
            pil_img.save(img_path, "PNG")

        if ocr_client is not None:
            ocr_result = ocr_client.ocr_image(img_path)
            if ocr_result:
                results.append(ocr_result)
        else:
            results.append(f"[Изображение: {img_path}]")

    if results:
        return "\n" + "\n".join(results) + "\n"
    return ""
