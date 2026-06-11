"""Конвертер OMML (Office Math Markup Language) → LaTeX."""

import re
from lxml import etree

MATHML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

TAG = lambda name: f"{{{M_NS}}}{name}"

# Греческие буквы (прописные)
GREEK_UPPER = {
    "Gamma": "\\Gamma", "Delta": "\\Delta", "Theta": "\\Theta",
    "Lambda": "\\Lambda", "Xi": "\\Xi", "Pi": "\\Pi",
    "Sigma": "\\Sigma", "Upsilon": "\\Upsilon", "Phi": "\\Phi",
    "Psi": "\\Psi", "Omega": "\\Omega",
}
# Греческие буквы (строчные)
GREEK_LOWER = {
    "alpha": "\\alpha", "beta": "\\beta", "gamma": "\\gamma",
    "delta": "\\delta", "epsilon": "\\epsilon", "zeta": "\\zeta",
    "eta": "\\eta", "theta": "\\theta", "iota": "\\iota",
    "kappa": "\\kappa", "lambda": "\\lambda", "mu": "\\mu",
    "nu": "\\nu", "xi": "\\xi", "omicron": "\\omicron",
    "pi": "\\pi", "rho": "\\rho", "sigma": "\\sigma",
    "tau": "\\tau", "upsilon": "\\upsilon", "phi": "\\phi",
    "chi": "\\chi", "psi": "\\psi", "omega": "\\omega",
    "varepsilon": "\\varepsilon", "vartheta": "\\vartheta",
    "varpi": "\\varpi", "varrho": "\\varrho", "varsigma": "\\varsigma",
    "varphi": "\\varphi",
}


def convert_omml_to_latex(elem) -> str:
    """
    Конвертирует OMML-элемент (m:oMath или m:oMathPara) в LaTeX-строку.

    Возвращает строку LaTeX или пустую строку при ошибке.
    """
    if elem is None:
        return ""

    tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

    if tag_local == "oMath":
        return _process_math(elem)
    elif tag_local == "oMathPara":
        parts = []
        for child in elem:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_tag == "oMath":
                parts.append(_process_math(child))
        return " \\\\\n" .join(parts)
    else:
        return _process_math_container(elem)

    return ""


def _process_math(elem) -> str:
    """Обрабатывает m:oMath (inline math container)."""
    parts = []
    for child in elem:
        part = _process_element(child)
        if part:
            parts.append(part)
    return " ".join(parts)


def _process_math_container(elem) -> str:
    """Обрабатывает любой контейнер с math-элементами."""
    parts = []
    for child in elem:
        part = _process_element(child)
        if part:
            parts.append(part)
    return " ".join(parts)


def _process_element(elem) -> str:
    """Обрабатывает один OMML-элемент."""
    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

    handlers = {
        "r": _process_run,
        "f": _process_fraction,
        "rad": _process_radical,
        "sSup": _process_superscript,
        "sSub": _process_subscript,
        "sSubSup": _process_subsuperscript,
        "d": _process_delimiter,
        "nary": _process_nary,
        "acc": _process_accent,
        "bar": _process_bar,
        "box": _process_box,
        "eqArr": _process_eqarr,
        "groupChr": _process_groupchr,
        "func": _process_func,
        "limLow": _process_limlow,
        "limUpp": _process_limupp,
        "lim": _process_limit,  # m:lim is for function names like lim, max, etc.
        "borderBox": _process_borderbox,
        "phant": _process_phantom,
        "m": _process_matrix,
        "sPre": _process_presuperscript,
        "e": _process_base,     # base of radical, etc.
        "deg": _process_base,   # degree of radical
        "den": _process_base,   # denominator
        "num": _process_base,   # numerator
        "sub": _process_base,   # subscript
        "sup": _process_base,   # superscript
    }

    handler = handlers.get(tag)
    if handler:
        return handler(elem)

    # For unknown elements, try processing children
    parts = []
    for child in elem:
        part = _process_element(child)
        if part:
            parts.append(part)
    return " ".join(parts)


def _process_base(elem) -> str:
    """Обрабатывает базовые контейнеры (num, den, e, deg, sub, sup)."""
    parts = []
    for child in elem:
        part = _process_element(child)
        if part:
            parts.append(part)
    return " ".join(parts)


def _process_run(elem) -> str:
    """Обрабатывает m:r (run) — извлекает текст."""
    texts = []
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "t":
            text = (child.text or "").strip()
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text)
            if text:
                texts.append(text)
    return " ".join(texts)


def _process_fraction(elem) -> str:
    """Обрабатывает m:f (дробь)."""
    num = ""
    den = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "num":
            num = _process_base(child)
        elif tag == "den":
            den = _process_base(child)

    if not num and not den:
        return ""

    return f"\\frac{{{num}}}{{{den}}}"


def _process_radical(elem) -> str:
    """Обрабатывает m:rad (корень)."""
    deg = ""
    base = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "deg":
            deg = _process_base(child)
        elif tag == "e":
            base = _process_base(child)

    if deg:
        if deg.strip() == "2":
            return f"\\sqrt{{{base}}}"
        return f"\\sqrt[{deg}]{{{base}}}"
    return f"\\sqrt{{{base}}}"


def _process_superscript(elem) -> str:
    """Обрабатывает m:sSup (верхний индекс)."""
    base = ""
    sup = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            base = _process_base(child)
        elif tag == "sup":
            sup = _process_base(child)
    if not base:
        return f"^{{{sup}}}"
    return f"{base}^{{{sup}}}"


def _process_subscript(elem) -> str:
    """Обрабатывает m:sSub (нижний индекс)."""
    base = ""
    sub = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            base = _process_base(child)
        elif tag == "sub":
            sub = _process_base(child)
    if not base:
        return f"_{{{sub}}}"
    return f"{base}_{{{sub}}}"


def _process_subsuperscript(elem) -> str:
    """Обрабатывает m:sSubSup (нижний + верхний индекс)."""
    base = ""
    sub = ""
    sup = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            base = _process_base(child)
        elif tag == "sub":
            sub = _process_base(child)
        elif tag == "sup":
            sup = _process_base(child)
    result = base or ""
    if sub:
        result += f"_{{{sub}}}"
    if sup:
        result += f"^{{{sup}}}"
    return result


def _process_delimiter(elem) -> str:
    """Обрабатывает m:d (скобки/разделители)."""
    left = "("
    right = ")"

    dPr = elem.find(TAG("dPr"))
    if dPr is not None:
        begChr = dPr.find(TAG("begChr"))
        endChr = dPr.find(TAG("endChr"))
        if begChr is not None and begChr.get(f"{{{M_NS}}}val"):
            left = _chr_to_latex(begChr.get(f"{{{M_NS}}}val"))
        if endChr is not None and endChr.get(f"{{{M_NS}}}val"):
            right = _chr_to_latex(endChr.get(f"{{{M_NS}}}val"))

    content = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            content = _process_base(child)

    return f"{left}{content}{right}"


def _chr_to_latex(chr_val: str) -> str:
    """Конвертирует OMML-символ в LaTeX."""
    mapping = {
        "{": "\\{",
        "}": "\\}",
        "[": "[",
        "]": "]",
        "(": "(",
        ")": ")",
        "|": "|",
        "‖": "\\|",
        "⌈": "\\lceil",
        "⌉": "\\rceil",
        "⌊": "\\lfloor",
        "⌋": "\\rfloor",
        "⟨": "\\langle",
        "⟩": "\\rangle",
        "⟦": "\\llbracket",
        "⟧": "\\rrbracket",
    }
    return mapping.get(chr_val, chr_val)


def _process_nary(elem) -> str:
    """Обрабатывает m:nary (сумма, произведение, интеграл и т.д.)."""
    nary_pr = elem.find(TAG("naryPr"))
    operator = "\\sum"

    if nary_pr is not None:
        chr_elem = nary_pr.find(TAG("chr"))
        if chr_elem is not None and chr_elem.get(f"{{{M_NS}}}val"):
            op_chr = chr_elem.get(f"{{{M_NS}}}val")
            op_map = {
                "∑": "\\sum",
                "∏": "\\prod",
                "∐": "\\coprod",
                "∫": "\\int",
                "∬": "\\iint",
                "∭": "\\iiint",
                "∮": "\\oint",
                "⋃": "\\bigcup",
                "⋂": "\\bigcap",
                "⋁": "\\bigvee",
                "⋀": "\\bigwedge",
                "⨁": "\\bigoplus",
                "⨂": "\\bigotimes",
            }
            operator = op_map.get(op_chr, operator)

    sub = ""
    sup = ""
    base = ""

    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "sub":
            sub = _process_base(child)
        elif tag == "sup":
            sup = _process_base(child)
        elif tag == "e":
            base = _process_base(child)

    parts = [operator]
    if sub:
        parts.append(f"_{{{sub}}}")
    if sup:
        parts.append(f"^{{{sup}}}")
    if base:
        parts.append(f"{{{base}}}")

    return " ".join(parts)


def _process_accent(elem) -> str:
    """Обрабатывает m:acc (акцент/диакритика)."""
    acc_pr = elem.find(TAG("accPr"))
    accent = "\\bar"
    if acc_pr is not None:
        chr_elem = acc_pr.find(TAG("chr"))
        if chr_elem is not None and chr_elem.get(f"{{{M_NS}}}val"):
            acc_map = {
                "̅": "\\bar",    # overline
                "̂": "\\hat",
                "⃗": "\\vec",
                "̇": "\\dot",
                "̈": "\\ddot",
                "̃": "\\tilde",
                "̀": "\\grave",
                "́": "\\acute",
                "̆": "\\breve",
                "̌": "\\check",
            }
            accent = acc_map.get(chr_elem.get(f"{{{M_NS}}}val"), "\\bar")

    base = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            base = _process_base(child)

    return f"{accent}{{{base}}}"


def _process_bar(elem) -> str:
    """Обрабатывает m:bar (черта)."""
    base = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            base = _process_base(child)
    return f"\\overline{{{base}}}"


def _process_box(elem) -> str:
    """Обрабатывает m:box (группировка)."""
    parts = []
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            parts.append(_process_base(child))
        else:
            part = _process_element(child)
            if part:
                parts.append(part)
    return " ".join(parts)


def _process_eqarr(elem) -> str:
    """Обрабатывает m:eqArr (массив уравнений)."""
    parts = []
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            parts.append(_process_base(child))
    return " \\\\\n ".join(parts)


def _process_groupchr(elem) -> str:
    """Обрабатывает m:groupChr (групповой символ, например дуга над выражением)."""
    group_pr = elem.find(TAG("groupChrPr"))
    above = ""
    below = ""

    if group_pr is not None:
        chr_elem = group_pr.find(TAG("chr"))
        if chr_elem is not None:
            val = chr_elem.get(f"{{{M_NS}}}val", "")
            if val == "⏞":
                above = "\\overbrace{"
                below = "}"
            elif val == "⏟":
                above = "\\underbrace{"
                below = "}"

    content = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            content = _process_base(child)

    return f"{above}{content}{below}"


def _process_func(elem) -> str:
    """Обрабатывает m:func (функция, например sin, cos)."""
    func_name = ""
    arg = ""

    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "fName":
            # Extract function name
            for sub in child:
                sub_tag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                if sub_tag == "r":
                    func_name = _process_run(sub)
        elif tag == "e":
            arg = _process_base(child)

    func_map = {
        "sin": "\\sin", "cos": "\\cos", "tan": "\\tan",
        "cot": "\\cot", "sec": "\\sec", "csc": "\\csc",
        "arcsin": "\\arcsin", "arccos": "\\arccos", "arctan": "\\arctan",
        "sinh": "\\sinh", "cosh": "\\cosh", "tanh": "\\tanh",
        "ln": "\\ln", "log": "\\log", "lg": "\\lg",
        "exp": "\\exp", "min": "\\min", "max": "\\max",
        "lim": "\\lim", "sup": "\\sup", "inf": "\\inf",
        "det": "\\det", "dim": "\\dim", "ker": "\\ker",
        "gcd": "\\gcd", "deg": "\\deg", "hom": "\\hom",
    }

    func_name_clean = func_name.strip().lower()
    latex_func = func_map.get(func_name_clean, func_name)

    return f"{latex_func}{{{arg}}}"


def _process_limlow(elem) -> str:
    """Обрабатывает m:limLow (нижний предел)."""
    base = ""
    lim = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            base = _process_base(child)
        elif tag == "lim":
            lim = _process_base(child)
    return f"{base}_{{{lim}}}"


def _process_limupp(elem) -> str:
    """Обрабатывает m:limUpp (верхний предел)."""
    base = ""
    lim = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            base = _process_base(child)
        elif tag == "lim":
            lim = _process_base(child)
    return f"{base}^{{{lim}}}"


def _process_limit(elem) -> str:
    """Обрабатывает m:lim (вложенный предел, используется для sum/prod/int с пределами)."""
    # m:lim usually appears inside nary or similar
    parts = []
    for child in elem:
        part = _process_element(child)
        if part:
            parts.append(part)
    return " ".join(parts)


def _process_borderbox(elem) -> str:
    """Обрабатывает m:borderBox (рамка)."""
    parts = []
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "e":
            parts.append(_process_base(child))
    return f"\\boxed{{{' '.join(parts)}}}"


def _process_phantom(elem) -> str:
    """Обрабатывает m:phant (фантом — невидимый символ для выравнивания)."""
    return "\\phantom{}"


def _process_matrix(elem) -> str:
    """Обрабатывает m:m (матрица)."""
    rows = []
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "mr":
            row_cells = []
            for cell in child:
                cell_tag = cell.tag.split("}")[-1] if "}" in cell.tag else cell.tag
                if cell_tag == "e":
                    row_cells.append(_process_base(cell))
            if row_cells:
                rows.append(" & ".join(row_cells))

    if not rows:
        return ""

    return "\\begin{array}{" + "c" * max(len(r.split("&")) for r in rows) + "}\n" + \
           " \\\\\n ".join(rows) + "\n\\end{array}"


def _process_presuperscript(elem) -> str:
    """Обрабатывает m:sPre (пре-верхний/нижний индекс)."""
    sub = ""
    sup = ""
    base = ""
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "sub":
            sub = _process_base(child)
        elif tag == "sup":
            sup = _process_base(child)
        elif tag == "e":
            base = _process_base(child)
    result = ""
    if sub:
        result += f"_{{{sub}}}"
    if sup:
        result += f"^{{{sup}}}"
    result += base
    return result
