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
from parser.markdown_cleaner import clean_markdown, normalize_headings
from parser.omml_to_latex import convert_omml_to_latex

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
V_NS = "urn:schemas-microsoft-com:vml"


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

    text = _extract_text_from_paragraph(para)
    style_name = para.style.name if para.style else ""
    parts = []

    if text:
        list_marker = _get_list_marker(para)
        heading_md = "" if list_marker else _detect_heading(text, style_name, para)
        if list_marker:
            parts.append(f"{list_marker} {text}")
        elif heading_md:
            parts.append(heading_md.strip("\n"))
        elif _is_potential_formula(text):
            parts.append(f"$$\n{text}\n$$")
        else:
            parts.append(text)

    parts.extend(_extract_omml_from_paragraph(para))

    for img_path in _extract_images_from_paragraph(para):
        try:
            ocr_result = ocr_client.ocr_image(img_path)
        except Exception:
            continue
        if ocr_result:
            parts.append(ocr_result)

    if not parts:
        return ""

    return "\n" + "\n\n".join(part.strip("\n") for part in parts) + "\n"


def _extract_text_from_paragraph(para) -> str:
    return _extract_text_from_element(para._element, para).strip()


def _extract_text_from_element(element, para) -> str:
    parts = []
    for child in element:
        if child.tag == f"{{{W_NS}}}r":
            parts.append(_extract_text_from_run(child))
        elif child.tag == f"{{{W_NS}}}hyperlink":
            parts.append(_extract_hyperlink_text(child, para))
        elif child.tag.startswith(f"{{{MATH_NS}}}"):
            continue
        else:
            parts.append(_extract_text_from_element(child, para))
    return "".join(parts)


def _extract_text_from_run(run) -> str:
    parts = []
    for elem in run.iter():
        if elem.tag == f"{{{W_NS}}}t":
            parts.append(elem.text or "")
        elif elem.tag == f"{{{W_NS}}}tab":
            parts.append("\t")
        elif elem.tag in {f"{{{W_NS}}}br", f"{{{W_NS}}}cr"}:
            parts.append("\n")
        elif elem.tag == f"{{{W_NS}}}noBreakHyphen":
            parts.append("-")
    return "".join(parts)


def _extract_hyperlink_text(hyperlink, para) -> str:
    text = _extract_text_from_element(hyperlink, para)
    if not text:
        return ""

    rel_id = hyperlink.get(f"{{{R_NS}}}id")
    if rel_id and rel_id in para.part.rels:
        rel = para.part.rels[rel_id]
        if rel.reltype == RT.HYPERLINK:
            return f"[{text}]({rel.target_ref})"

    anchor = hyperlink.get(f"{{{W_NS}}}anchor")
    if anchor:
        return f"[{text}](#{anchor})"

    return text


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


def _get_list_marker(para) -> str:
    p_pr = para._element.find(f"{{{W_NS}}}pPr")
    if p_pr is None:
        return ""
    num_pr = p_pr.find(f"{{{W_NS}}}numPr")
    if num_pr is None:
        return ""

    num_id_elem = num_pr.find(f"{{{W_NS}}}numId")
    ilvl_elem = num_pr.find(f"{{{W_NS}}}ilvl")
    num_id = num_id_elem.get(f"{{{W_NS}}}val") if num_id_elem is not None else ""
    ilvl = ilvl_elem.get(f"{{{W_NS}}}val", "0") if ilvl_elem is not None else "0"
    num_fmt = ""

    if num_id and num_id != "0":
        try:
            numbering = para.part.numbering_part.element
        except (AttributeError, NotImplementedError):
            numbering = None
        if numbering is not None:
            num_elem = next(
                (num for num in numbering.findall(f"{{{W_NS}}}num")
                 if num.get(f"{{{W_NS}}}numId") == num_id),
                None,
            )
            if num_elem is not None:
                abstract_id_elem = num_elem.find(f"{{{W_NS}}}abstractNumId")
                abstract_id = abstract_id_elem.get(f"{{{W_NS}}}val") if abstract_id_elem is not None else ""
                abstract_elem = next(
                    (abstract for abstract in numbering.findall(f"{{{W_NS}}}abstractNum")
                     if abstract.get(f"{{{W_NS}}}abstractNumId") == abstract_id),
                    None,
                )
                if abstract_elem is not None:
                    level_elem = next(
                        (level for level in abstract_elem.findall(f"{{{W_NS}}}lvl")
                         if level.get(f"{{{W_NS}}}ilvl") == ilvl),
                        None,
                    )
                    if level_elem is not None:
                        num_fmt_elem = level_elem.find(f"{{{W_NS}}}numFmt")
                        if num_fmt_elem is not None:
                            num_fmt = num_fmt_elem.get(f"{{{W_NS}}}val", "")

    if num_fmt and num_fmt not in {"bullet", "none"}:
        return "1."
    return "-"


def _extract_images_from_paragraph(para) -> list:
    """Извлекает изображения из параграфа, сохраняет во временные файлы."""
    images = []
    seen_rel_ids = set()
    for elem in para._element.iter():
        rel_id = None
        if elem.tag == f"{{{A_NS}}}blip":
            rel_id = elem.get(f"{{{R_NS}}}embed")
        elif elem.tag == f"{{{V_NS}}}imagedata":
            rel_id = elem.get(f"{{{R_NS}}}id")
        if not rel_id or rel_id in seen_rel_ids or rel_id not in para.part.rels:
            continue
        seen_rel_ids.add(rel_id)
        rel = para.part.rels[rel_id]
        if "image" not in rel.reltype:
            continue
        try:
            with Image.open(io.BytesIO(rel.target_part.blob)) as img:
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
    processed_cells = set()
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if cell._tc in processed_cells:
                cells.append("")
                continue
            processed_cells.add(cell._tc)
            paragraph_results = [
                _process_paragraph(para, ocr_client).strip()
                for para in cell.paragraphs
            ]
            cell_text = "<br>".join(result for result in paragraph_results if result)
            cell_text = cell_text.replace("\n", " ").replace("|", "\\|")
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
