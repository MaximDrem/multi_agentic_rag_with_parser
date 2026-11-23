"""Обработчик PDF → Markdown с OCR для изображений и формул."""

import io
import os
import re
import time
from pathlib import Path
import fitz
from PIL import Image, ImageStat

from parser.config import TEMP_DIR
from parser.markdown_cleaner import clean_markdown, normalize_headings


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
        detected_regions = _find_formula_gaps(page, blocks, page_num, filepath)
        formula_regions = [region for region in detected_regions if region["type"] == "formula"]
        table_regions = _find_table_regions(page, page_num, filepath, formula_regions)

        elements = _merge_text_and_images(blocks, image_regions, formula_regions + table_regions)
        elements = _merge_link_elements(page, elements)

        for elem in elements:
            result = _process_pdf_element(elem, ocr_client)
            if result:
                md_lines.append(result)

        md_lines.append("")

    doc.close()

    md_text = "\n".join(md_lines)
    md_text = clean_markdown(md_text)
    md_text = normalize_headings(md_text)

    return md_text


def _extract_page_images(page, page_num: int, filepath: str) -> list:
    """Извлекает изображения со страницы PDF, возвращает список регионов."""

    regions = []

    for img_index, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        for rect_index, image_rect in enumerate(page.get_image_rects(xref)):
            clip_rect = image_rect & page.rect
            if clip_rect.width < 5 or clip_rect.height < 5:
                continue

            pix = page.get_pixmap(clip=clip_rect, matrix=fitz.Matrix(3.0, 3.0), alpha=False)
            pil_image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
            if pil_image.width < 80 and pil_image.height < 80:
                continue
            if ImageStat.Stat(pil_image).stddev[0] < 3:
                continue

            img_name = f"pdf_{Path(filepath).stem}_p{page_num}_i{img_index}_r{rect_index}.png"
            img_path = os.path.join(TEMP_DIR, img_name)
            pix.save(img_path)
            regions.append({
                "type": "image",
                "path": img_path,
                "x0": clip_rect.x0,
                "x1": clip_rect.x1,
                "y0": clip_rect.y0,
                "y1": clip_rect.y1,
            })

    return regions


def _markdown_table(rows: list) -> str:
    normalized = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", "\\|") for cell in row]
        if any(cells):
            normalized.append(cells)
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    if width < 2:
        return ""
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    lines = ["| " + " | ".join(row) + " |" for row in normalized]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _has_raster_content(page, rect: fitz.Rect) -> bool:
    area = rect.width * rect.height
    if area <= 0:
        return False
    covered = 0.0
    for image in page.get_images(full=True):
        for image_rect in page.get_image_rects(image[0]):
            intersection = image_rect & rect
            if intersection.width > 0 and intersection.height > 0:
                covered += intersection.width * intersection.height
    return covered >= 0.25 * area


def _find_table_regions(page, page_num: int, filepath: str, formula_regions: list) -> list:
    regions = []
    try:
        tables = page.find_tables().tables
    except Exception:
        return regions
    for table_index, table in enumerate(tables):
        clip_rect = fitz.Rect(table.bbox) & page.rect
        if clip_rect.width < 10 or clip_rect.height < 10:
            continue
        native_rows = table.extract()
        native_has_content = any(str(cell or "").strip() for row in native_rows for cell in row)
        native_text = page.get_textbox(clip_rect)
        if not native_has_content and not re.search(r"[A-Za-zА-Яа-яЁё]", native_text) and not _has_raster_content(page, clip_rect):
            continue
        needs_ocr = False
        for formula in formula_regions:
            formula_rect = fitz.Rect(formula["x0"], formula["y0"], formula["x1"], formula["y1"])
            intersection = clip_rect & formula_rect
            if intersection.width > 0 and intersection.height > 0:
                needs_ocr = True
                break
        markdown = "" if needs_ocr else _markdown_table(native_rows)
        if markdown:
            regions.append({
                "type": "table_text",
                "markdown": markdown,
                "x0": clip_rect.x0,
                "x1": clip_rect.x1,
                "y0": clip_rect.y0,
                "y1": clip_rect.y1,
            })
            continue
        table_name = f"pdf_{Path(filepath).stem}_p{page_num}_t{table_index}.png"
        table_path = os.path.join(TEMP_DIR, table_name)
        pix = page.get_pixmap(clip=clip_rect, matrix=fitz.Matrix(3.0, 3.0), alpha=False)
        pix.save(table_path)
        regions.append({
            "type": "table",
            "path": table_path,
            "x0": clip_rect.x0,
            "x1": clip_rect.x1,
            "y0": clip_rect.y0,
            "y1": clip_rect.y1,
        })
    return regions


def _merge_link_elements(page, elements: list) -> list:
    result = list(elements)
    text_content = "\n".join(element.get("text", "") for element in result)
    for link in page.get_links():
        uri = link.get("uri")
        if not uri or uri in text_content:
            continue
        rect = fitz.Rect(link["from"])
        label = re.sub(r"\s+", " ", page.get_textbox(rect)).strip() or uri
        if uri.startswith("mailto:") and uri[7:] in text_content:
            continue
        result.append({
            "type": "text",
            "text": f"[{label}]({uri})",
            "x0": rect.x0,
            "x1": rect.x1,
            "y0": rect.y0,
            "y1": rect.y1,
            "size": 10,
        })
    result.sort(key=lambda element: (element["y0"], element.get("x0", 0)))
    return result


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
    clip_rects = []
    try:
        table_rects = [fitz.Rect(table.bbox) for table in page.find_tables().tables]
    except Exception:
        table_rects = []

    for cluster in clusters:
        cluster_blocks = [text_blocks[i] for i in cluster]
        min_y = min(b[1] for b in cluster_blocks)
        max_y = max(b[3] for b in cluster_blocks)
        related_drawings = []

        for d in drawings:
            dr = d["rect"]
            if dr.y0 > margin and dr.y1 < page_height - margin:
                if min_y - 10 <= dr.y1 and max_y + 10 >= dr.y0:
                    min_y = min(min_y, dr.y0)
                    max_y = max(max_y, dr.y1)
                    related_drawings.append(dr)

        min_x = min([b[0] for b in cluster_blocks] + [r.x0 for r in related_drawings])
        max_x = max([b[2] for b in cluster_blocks] + [r.x1 for r in related_drawings])
        clip_rect = fitz.Rect(min_x, min_y, max_x, max_y)
        for table_rect in table_rects:
            intersection = clip_rect & table_rect
            if intersection.width > 0 and intersection.height > 0:
                clip_rect |= table_rect
        clip_rect = fitz.Rect(clip_rect.x0 - 8, clip_rect.y0 - 8, clip_rect.x1 + 8, clip_rect.y1 + 8) & page.rect

        if clip_rect.height >= 10 and clip_rect.width >= 10:
            clip_rects.append(clip_rect)

    merged_rects = []
    for clip_rect in clip_rects:
        current = clip_rect
        i = 0
        while i < len(merged_rects):
            existing = merged_rects[i]
            intersection = existing & current
            if intersection.width <= 0 or intersection.height <= 0:
                i += 1
                continue
            vertical_overlap = intersection.height / min(existing.height, current.height)
            horizontal_overlap = intersection.width / min(existing.width, current.width)
            if vertical_overlap >= 0.8 and horizontal_overlap >= 0.5:
                current = current | merged_rects.pop(i)
                i = 0
            else:
                i += 1
        merged_rects.append(current)

    regions = []
    mat = fitz.Matrix(3.0, 3.0)
    for formula_counter, clip_rect in enumerate(sorted(merged_rects, key=lambda r: (r.y0, r.x0))):
        formula_name = f"pdf_{Path(filepath).stem}_p{page_num}_f{formula_counter}.png"
        formula_path = os.path.join(TEMP_DIR, formula_name)

        try:
            pix = page.get_pixmap(clip=clip_rect, matrix=mat)
            pix.save(formula_path)
            is_table = any(
                (clip_rect & table_rect).width > 0 and (clip_rect & table_rect).height > 0
                for table_rect in table_rects
            )
            regions.append({
                "type": "table" if is_table else "formula",
                "path": formula_path,
                "x0": clip_rect.x0,
                "x1": clip_rect.x1,
                "y0": clip_rect.y0,
                "y1": clip_rect.y1,
            })
        except Exception:
            pass

    return regions


def _merge_text_and_images(blocks: list, image_regions: list, formula_regions: list) -> list:
    """Сливает текстовые блоки, изображения и формулы в единый список.
    Текстовые блоки, перекрывающиеся с формульными регионами, удаляются."""

    elements = []
    visual_regions = [*formula_regions, *image_regions]
    for region in visual_regions:
        region_rect = fitz.Rect(region.get("x0", 0), region["y0"], region.get("x1", 0), region["y1"])
        covered_text = []
        for block in blocks:
            if block[6] != 0 or not block[4].strip():
                continue
            block_rect = fitz.Rect(block[0], block[1], block[2], block[3])
            intersection = block_rect & region_rect
            block_area = block_rect.width * block_rect.height
            if block_area > 0 and intersection.width > 0 and intersection.height > 0:
                if intersection.width * intersection.height >= 0.5 * block_area:
                    covered_text.append(block[4].strip())
        if covered_text:
            region["fallback_text"] = "\n\n".join(covered_text)

    for block in blocks:
        if block[6] == 0:
            text = block[4].strip()
            block_y0 = block[1]
            block_y1 = block[3]
            block_rect = fitz.Rect(block[0], block_y0, block[2], block_y1)
            block_area = block_rect.width * block_rect.height

            overlaps_visual_region = False
            for region in visual_regions:
                region_rect = fitz.Rect(region.get("x0", 0), region["y0"], region.get("x1", 0), region["y1"])
                intersection = block_rect & region_rect
                if block_area > 0 and intersection.width > 0 and intersection.height > 0:
                    if intersection.width * intersection.height >= 0.5 * block_area:
                        overlaps_visual_region = True
                        break

            if overlaps_visual_region:
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

    for image in image_regions:
        image_rect = fitz.Rect(image.get("x0", 0), image["y0"], image.get("x1", 0), image["y1"])
        if image_rect.width <= 0 or image_rect.height <= 0:
            elements.append(image)
            continue
        covered = False
        for region in formula_regions:
            region_rect = fitz.Rect(region.get("x0", 0), region["y0"], region.get("x1", 0), region["y1"])
            intersection = image_rect & region_rect
            if intersection.width > 0 and intersection.height > 0:
                if intersection.width * intersection.height >= 0.8 * image_rect.width * image_rect.height:
                    covered = True
                    break
        if not covered:
            elements.append(image)

    elements.extend(formula_regions)
    elements.sort(key=lambda e: (e["y0"], e.get("x0", 0)))

    return elements


def _get_block_font_size(block) -> float:
    height = block[3] - block[1]
    if height > 0:
        return height
    return 10.0


def _ocr_with_fallback(elem: dict, ocr_client) -> str:
    try:
        if elem["type"] == "table":
            result = ocr_client.ocr_table_image(elem["path"])
        elif elem["type"] == "formula":
            result = ocr_client.ocr_formula_region(elem["path"])
            if not result:
                result = ocr_client.ocr_image(elem["path"])
        else:
            result = ocr_client.ocr_image(elem["path"])
            if not result:
                result = ocr_client.ocr_table_image(elem["path"])
    except Exception:
        if elem.get("fallback_text"):
            return f"\n{elem['fallback_text']}\n"
        raise
    if result:
        return f"\n{result}\n"
    if elem.get("fallback_text"):
        return f"\n{elem['fallback_text']}\n"
    return ""


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

    elif elem["type"] == "table_text":
        return f"\n{elem['markdown']}\n"

    elif elem["type"] in ("table", "formula", "image"):
        return _ocr_with_fallback(elem, ocr_client)

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
