"""Обработчик PDF → Markdown с OCR для изображений и формул."""

import os
import re
import time
from pathlib import Path
import fitz

from parser.config import TEMP_DIR
from parser.markdown_cleaner import clean_markdown, normalize_headings, remove_redundant_linebreaks


def process_pdf(filepath: str, ocr_client) -> str:
    """
    Конвертирует PDF в структурированный Markdown.

    Использует pymupdf для извлечения текста и изображений.
    Все изображения/формулы распознаются через внешний OCR-клиент (VLM).
    """

    filename = os.path.basename(filepath)
    md_lines = [f"# {Path(filepath).stem}", ""]

    doc = fitz.open(filepath)
    os.makedirs(TEMP_DIR, exist_ok=True)

    for page_num in range(doc.page_count):
        page = doc[page_num]

        blocks = page.get_text("blocks")

        image_regions = _extract_page_images(page, page_num, filepath)

        # Обнаруживаем пропуски между текстовыми блоками с рисунками (формулы)
        formula_regions = _find_formula_gaps(page, blocks, page_num, filepath)

        elements = _merge_text_and_images(blocks, image_regions, formula_regions)

        for elem in elements:
            result = _process_pdf_element(elem, ocr_client)
            if result:
                md_lines.append(result)

        md_lines.append("")

    doc.close()

    md_text = "\n".join(md_lines)
    md_text = remove_redundant_linebreaks(md_text)
    md_text = clean_markdown(md_text)
    md_text = normalize_headings(md_text)

    return md_text


def _extract_page_images(page, page_num: int, filepath: str) -> list:
    """Извлекает изображения со страницы PDF, возвращает список регионов."""

    regions = []
    image_list = page.get_images(full=True)

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        base_image = page.parent.extract_image(xref)

        if not base_image:
            continue

        img_bytes = base_image["image"]
        ext = base_image["ext"]

        img_name = f"pdf_{Path(filepath).stem}_p{page_num}_i{img_index}.{ext}"
        img_path = os.path.join(TEMP_DIR, img_name)

        with open(img_path, "wb") as f:
            f.write(img_bytes)

        rects = page.get_image_rects(xref)
        if rects:
            y0 = rects[0].y0
            y1 = rects[0].y1
        else:
            y0 = 0
            y1 = 0

        regions.append({
            "type": "image",
            "path": img_path,
            "y0": y0,
            "y1": y1,
        })

    return regions


def _find_formula_gaps(page, blocks: list, page_num: int, filepath: str) -> list:
    """
    Находит области с формулами (векторная графика + фрагментированный текст).
    Группирует текст и рисунки в кластеры, рендерит как PNG, возвращает для OCR.
    """

    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return []

    drawings = page.get_drawings()
    page_height = page.rect.height
    page_width = page.rect.width
    margin = 50

    text_blocks.sort(key=lambda b: b[1])

    # === Шаг 1: Ищем текстовые блоки-фрагменты формул ===
    # Признаки: изолированные "=", подстрочные символы (YSt–1, αi, и т.д.),
    # одиночные скобки, короткие блоки вблизи рисунков

    formula_fragment_indices = set()

    for i, b in enumerate(text_blocks):
        text = b[4].strip()
        block_height = b[3] - b[1]
        block_x0 = b[0]
        block_width = b[2] - b[0]

        # Очень короткий текст (< 8 символов) и высота блока маленькая → формула
        if len(text) < 8 and block_height < 20:
            formula_fragment_indices.add(i)

        # Изолированный знак равенства, скобка, запятая
        if text in ("=", "(", ")", ",", "–", "—", "×", "+", "−"):
            formula_fragment_indices.add(i)

        # Текст содержит подстрочные символы (Unicode subscript digits)
        if any(0x2080 <= ord(c) <= 0x2089 for c in text):
            formula_fragment_indices.add(i)

        # Блок находится посреди страницы (не у левого края) и короткий
        if block_x0 > 200 and len(text) < 20 and block_height < 25:
            formula_fragment_indices.add(i)

    # === Шаг 2: Добавляем блоки рядом с рисунками ===
    for di, d in enumerate(drawings):
        dr = d["rect"]
        if dr.y0 < margin or dr.y1 > page_height - margin:
            continue

        items = d.get("items", [])
        # Пропускаем простые линии/прямоугольники (разделители страниц)
        if len(items) <= 1 and all(it[0] in ("l", "re") for it in items):
            continue

        # Находим текстовые блоки, перекрывающиеся с рисунком
        for i, b in enumerate(text_blocks):
            if (max(b[0], dr.x0) < min(b[2], dr.x1) and
                    max(b[1], dr.y0) < min(b[3], dr.y1)):
                formula_fragment_indices.add(i)
            # Или рядом по вертикали (в пределах 10pt)
            if (abs(b[1] - dr.y0) < 10 or abs(b[3] - dr.y1) < 10):
                if abs(b[0] - dr.x0) < 100:
                    formula_fragment_indices.add(i)

    # === Шаг 3: Группируем индексы в кластеры ===
    if not formula_fragment_indices:
        return []

    sorted_indices = sorted(formula_fragment_indices)

    clusters = []
    current_cluster = [sorted_indices[0]]

    for idx in sorted_indices[1:]:
        prev = current_cluster[-1]
        prev_bottom = text_blocks[prev][3]
        curr_top = text_blocks[idx][1]

        # Если между блоками не более 20pt — это один кластер
        if curr_top - prev_bottom < 20:
            current_cluster.append(idx)
        else:
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            elif current_cluster:
                # Одиночный блок-фрагмент: проверяем, есть ли рисунки рядом
                bi = current_cluster[0]
                b = text_blocks[bi]
                has_nearby_drawing = False
                for d in drawings:
                    dr = d["rect"]
                    items = d.get("items", [])
                    if len(items) > 2 and not all(it[0] in ("l", "re") for it in items):
                        if abs(dr.y0 - b[1]) < 15 or abs(dr.y1 - b[3]) < 15:
                            has_nearby_drawing = True
                            break
                if has_nearby_drawing:
                    clusters.append(current_cluster)
            current_cluster = [idx]

    # Последний кластер
    if len(current_cluster) >= 2:
        clusters.append(current_cluster)
    elif current_cluster:
        bi = current_cluster[0]
        b = text_blocks[bi]
        has_nearby_drawing = False
        for d in drawings:
            dr = d["rect"]
            items = d.get("items", [])
            if len(items) > 2 and not all(it[0] in ("l", "re") for it in items):
                if abs(dr.y0 - b[1]) < 15 or abs(dr.y1 - b[3]) < 15:
                    has_nearby_drawing = True
                    break
        if has_nearby_drawing:
            clusters.append(current_cluster)

    # === Шаг 4: Рендерим каждый кластер как изображение для OCR ===
    regions = []
    formula_counter = 0

    for cluster in clusters:
        # Находим границы кластера
        min_y = min(text_blocks[i][1] for i in cluster)
        max_y = max(text_blocks[i][3] for i in cluster)

        # Расширяем границы, чтобы включить все рисунки в области
        for d in drawings:
            dr = d["rect"]
            if dr.y0 > margin and dr.y1 < page_height - margin:
                if min_y - 10 <= dr.y1 and max_y + 10 >= dr.y0:
                    min_y = min(min_y, dr.y0)
                    max_y = max(max_y, dr.y1)

        # Добавляем отступ
        min_y -= 8
        max_y += 8

        clip_rect = fitz.Rect(margin, min_y, page_width - margin, max_y)

        if clip_rect.height < 10 or clip_rect.width < 10:
            continue

        mat = fitz.Matrix(3.0, 3.0)  # x3 разрешение для формул

        formula_name = f"pdf_{Path(filepath).stem}_p{page_num}_f{formula_counter}.png"
        formula_path = os.path.join(TEMP_DIR, formula_name)

        try:
            pix = page.get_pixmap(clip=clip_rect, matrix=mat)
            pix.save(formula_path)

            regions.append({
                "type": "formula",
                "path": formula_path,
                "y0": min_y,
                "y1": max_y,
            })
            formula_counter += 1
        except Exception:
            pass

    return regions


def _merge_text_and_images(blocks: list, image_regions: list, formula_regions: list) -> list:
    """Сливает текстовые блоки, изображения и формулы в единый список.
    Текстовые блоки, перекрывающиеся с формульными регионами, удаляются."""

    elements = []

    for block in blocks:
        if block[6] == 0:
            text = block[4].strip()
            block_y0 = block[1]
            block_y1 = block[3]

            # Проверяем, пересекается ли текстовый блок с формульным регионом
            overlaps_formula = False
            for fr in formula_regions:
                if max(block_y0, fr["y0"]) < min(block_y1, fr["y1"]):
                    overlaps_formula = True
                    break

            if overlaps_formula:
                continue

            elements.append({
                "type": "text",
                "y0": block_y0,
                "y1": block_y1,
                "text": text,
                "x0": block[0],
                "x1": block[2],
                "size": _get_block_font_size(block),
            })

    elements.extend(image_regions)
    elements.extend(formula_regions)

    elements.sort(key=lambda e: (e["y0"], e.get("x0", 0)))

    return elements


def _get_block_font_size(block) -> float:
    height = block[3] - block[1]
    if height > 0:
        return height
    return 10.0


def _process_pdf_element(elem: dict, ocr_client) -> str:
    """Обрабатывает один элемент (текст, изображение или формула)."""

    if elem["type"] == "text":
        text = elem["text"]

        if not text:
            return ""

        heading = _detect_pdf_heading(text, elem)
        if heading:
            return heading

        return f"\n{text}\n"

    elif elem["type"] in ("image", "formula"):
        ocr_result = ocr_client.ocr_image(elem["path"])

        if not ocr_result:
            ocr_result = ocr_client.ocr_formula_region(elem["path"])

        if not ocr_result:
            ocr_result = ocr_client.ocr_table_image(elem["path"])

        if ocr_result:
            return f"\n{ocr_result}\n"

    return ""


def _detect_pdf_heading(text: str, elem: dict) -> str:
    """Определяет, является ли текст заголовком в PDF."""

    if not text or len(text) > 150:
        return ""

    size = elem.get("size", 10)

    if re.fullmatch(r"\d{1,4}", text):
        return ""
    if re.fullmatch(r"[IVX]{1,5}", text):
        return ""

    if size > 14:
        return f"\n## {text}\n"
    if size > 12 and len(text) < 80:
        return f"\n### {text}\n"

    if len(text) < 120:
        heading_patterns = [
            r"^(Глава|ГЛАВА|Раздел|РАЗДЕЛ|Статья|СТАТЬЯ)\s+\d+",
            r"^\d+\.\d+\.?\s+\w",
            r"^\d+\.\s+[А-ЯЁA-Z]",
            r"^(Предисловие|Введение|Заключение|Содержание|Приложение)",
            r"^[А-ЯЁ][А-ЯЁ\s\-]{4,}$",
        ]
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return f"\n### {text}\n"

    if re.match(r"^(ГОСТ|ГОСТ Р|ГОСТ Р ИСО)\s", text) and len(text) < 80:
        return f"\n## {text}\n"

    return ""
