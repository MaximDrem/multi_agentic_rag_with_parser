"""Обработчик DOCX → Markdown с OCR для изображений и формул."""

import os
import io
import re
import zipfile
from pathlib import Path
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from lxml import etree
from PIL import Image

from parser.config import TEMP_DIR
from parser.markdown_cleaner import clean_markdown, normalize_headings, remove_redundant_linebreaks
from parser.omml_to_latex import convert_omml_to_latex

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def process_docx(filepath: str, ocr_client) -> str:
    """
    Конвертирует DOCX в структурированный Markdown.

    Args:
        filepath: путь к .docx файлу
        ocr_client: экземпляр OCRCient для распознавания изображений/формул

    Returns:
        строка Markdown
    """
    filename = os.path.basename(filepath)
    md_lines = []

    # Заголовок документа
    md_lines.append(f"# {Path(filepath).stem}")
    md_lines.append("")

    doc = Document(filepath)

    # Собираем все элементы документа в порядке их появления
    # python-docx итерирует paragraphs и tables отдельно, нам нужно смешать их
    body_elements = _get_body_elements(doc)

    # Обрабатываем элементы
    heading_level = 2

    for elem in body_elements:
        result = _process_element(elem, ocr_client, filepath, heading_level)
        if result:
            md_lines.append(result)

    md_text = "\n".join(md_lines)

    # Пост-обработка
    md_text = remove_redundant_linebreaks(md_text)
    md_text = clean_markdown(md_text)
    md_text = normalize_headings(md_text)

    return md_text


def _get_body_elements(doc: Document) -> list:
    """
    Извлекает элементы тела документа в порядке их появления.
    Возвращает список кортежей: (type, element),
    где type ∈ {'paragraph', 'table'}
    """
    elements = []

    # python-docx не гарантирует порядок paragraphs и tables,
    # используем XML для получения правильного порядка
    body = doc.element.body
    para_index = 0
    table_index = 0

    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            if para_index < len(doc.paragraphs):
                elements.append(("paragraph", doc.paragraphs[para_index]))
                para_index += 1
        elif tag == "tbl":
            if table_index < len(doc.tables):
                elements.append(("table", doc.tables[table_index]))
                table_index += 1

    return elements


def _process_element(elem, ocr_client, docx_path: str, heading_level: int) -> str:
    """Обрабатывает один элемент документа."""
    etype, obj = elem

    if etype == "paragraph":
        return _process_paragraph(obj, ocr_client)
    elif etype == "table":
        return _process_table(obj, ocr_client)
    return ""


def _process_paragraph(para, ocr_client) -> str:
    """Обрабатывает параграф: текст, изображения, OMML-формулы."""

    text = para.text.strip()
    style_name = para.style.name if para.style else ""

    # 1. OMML-формулы (Office Math) — конвертируем в LaTeX
    omml_formulas = _extract_omml_from_paragraph(para)
    if omml_formulas:
        # Если в параграфе есть ещё текст кроме формул — выводим текст + формулы
        if text and text != ",":
            return f"\n{text}\n\n" + "\n\n".join(omml_formulas) + "\n"
        return "\n" + "\n\n".join(omml_formulas) + "\n"

    # 2. Определяем, заголовок ли это
    heading_md = _detect_heading(text, style_name, para)
    if heading_md:
        return heading_md

    # 3. Проверяем наличие изображений в параграфе
    images_in_para = _extract_images_from_paragraph(para)
    if images_in_para:
        results = []
        for img_path in images_in_para:
            ocr_result = ocr_client.ocr_image(img_path)
            if ocr_result:
                results.append(ocr_result)
        if results:
            return "\n\n".join(results)

    # 4. Проверяем, есть ли в тексте потенциальные формулы (LaTeX-подобные)
    if _is_potential_formula(text):
        return f"$$\n{text}\n$$"

    # 5. Обычный текст
    if text:
        return f"\n{text}\n"

    return ""


def _extract_omml_from_paragraph(para) -> list:
    """
    Извлекает OMML-формулы (m:oMath, m:oMathPara) из параграфа,
    конвертирует в LaTeX и возвращает список LaTeX-строк.
    """
    formulas = []

    xml = para._element

    # Ищем m:oMathPara (display math paragraph)
    for math_para in xml.findall(f".//{{{MATH_NS}}}oMathPara"):
        latex = convert_omml_to_latex(math_para)
        if latex:
            formulas.append(f"$$\n{latex}\n$$")

    # Ищем m:oMath (inline math) — которые НЕ внутри m:oMathPara
    for math_elem in xml.findall(f".//{{{MATH_NS}}}oMath"):
        # Проверяем что oMath не внутри oMathPara
        parent = math_elem.getparent()
        if parent is not None:
            parent_tag = parent.tag.split("}")[-1] if "}" in parent.tag else parent.tag
            if parent_tag == "oMathPara":
                continue
        latex = convert_omml_to_latex(math_elem)
        if latex:
            formulas.append(f"$$\n{latex}\n$$")

    return formulas


def _detect_heading(text: str, style_name: str, para) -> str:
    """Определяет, является ли параграф заголовком, и возвращает MD-заголовок."""

    if not text:
        return ""

    # Прямые заголовочные стили
    heading_styles = {
        "Heading 1": "#",
        "Heading 2": "##",
        "Heading 3": "###",
        "Heading 4": "####",
        "Heading 5": "#####",
        "Heading 6": "######",
        "Заголовок 1": "#",
        "Заголовок 2": "##",
        "Заголовок 3": "###",
        "Заголовок 4": "####",
        "title": "#",
        "ConsPlusTitle": "##",
        "ConsPlusSubtitle": "###",
        "ConsPlusHeading": "###",
        "1 Heading 1": "#",
        "2 Heading 2": "##",
        "3 Heading 3": "###",
    }

    style_lower = style_name.lower()
    for key, prefix in heading_styles.items():
        if key.lower() in style_lower or style_lower == key.lower():
            return f"\n{prefix} {text}\n"

    # Эвристики: короткий текст, все буквы заглавные или полужирный
    is_bold = all(r.bold for r in para.runs if r.text.strip()) if para.runs else False
    is_all_caps = text == text.upper() and len(text) > 3

    if is_bold and len(text) < 120 and not text.endswith("."):
        # Параграф в скобках или с точкой в конце — не заголовок
        if text.startswith("(") or text.endswith(";") or text.endswith(","):
            return ""
        return f"\n### {text}\n"

    return ""


def _is_potential_formula(text: str) -> bool:
    """Проверяет, похож ли текст на математическую формулу."""
    # Много математических символов
    math_chars = sum(1 for c in text if c in "∫∑∏√∞∂∇∈∉∪∩⊂⊃≈≠≤≥±×÷⇐⇒→←↑↓∝∞∟∠∥∧∨∩∪∫∮∯∰∱∲∳∴∵∶∷∸∹∺∻∼∽∾∿≀≁≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣≤≥≦≧≨≩≪≫≬≭≮≯≰≱≲≳≴≵≶≷≸≹≺≻≼≽≾≿⊀⊁⊂⊃⊄⊅⊆⊇⊈⊉⊊⊋⊌⊍⊎⊏⊐⊑⊒⊓⊔⊕⊖⊗⊘⊙⊚⊛⊜⊝")
    total = len(text.strip())
    if total > 0 and math_chars / total > 0.3:
        return True

    # LaTeX команды
    if "\\frac" in text or "\\sqrt" in text or "\\sum" in text or "\\int" in text:
        return True

    return False


def _extract_images_from_paragraph(para) -> list:
    """Извлекает изображения из параграфа, сохраняет во временные файлы."""
    images = []
    for run in para.runs:
        for elem in run._element:
            if elem.tag.endswith("}drawing"):
                blips = elem.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
                for blip in blips:
                    embed_id = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                    if embed_id and embed_id in para.part.rels:
                        rel = para.part.rels[embed_id]
                        if "image" in rel.reltype:
                            img_blob = rel.target_part.blob
                            try:
                                img = Image.open(io.BytesIO(img_blob))
                                os.makedirs(TEMP_DIR, exist_ok=True)
                                import time
                                img_name = f"docx_img_{int(time.time()*1000000)}_{len(images)}.png"
                                img_path = os.path.join(TEMP_DIR, img_name)
                                img.save(img_path, "PNG")
                                images.append(img_path)
                            except Exception:
                                pass
    return images


def _process_table(table, ocr_client) -> str:
    """Конвертирует таблицу DOCX в Markdown-таблицу."""

    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            cell_text = cell.text.strip().replace("\n", " ").replace("|", "\\|")
            cells.append(cell_text)
        rows.append(cells)

    if not rows:
        return ""

    # Убираем полностью пустые строки
    rows = [r for r in rows if any(c.strip() for c in r)]

    if not rows:
        return ""

    # Определяем максимальное количество колонок
    max_cols = max(len(r) for r in rows)

    # Нормализуем количество колонок
    normalized = []
    for r in rows:
        while len(r) < max_cols:
            r.append("")
        normalized.append(r[:max_cols])

    # Строим MD таблицу
    lines = []
    for i, row in enumerate(normalized):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * max_cols) + " |")

    return "\n" + "\n".join(lines) + "\n"
