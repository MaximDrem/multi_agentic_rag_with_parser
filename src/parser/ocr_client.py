"""OCR-клиент для распознавания изображений через VLM API."""

import os
import re
import base64
from pathlib import Path
from PIL import Image
import requests


class OCRCient:
    """Клиент для OCR через OpenAI-compatible API (Qwen-VL и др.)."""

    DEFAULT_PROMPT = (
        "Ты — OCR-система. Распознай ВЕСЬ текст на изображении. "
        "Если есть математические формулы — используй LaTeX внутри $...$. "
        "Если есть таблица — используй Markdown-таблицу. "
        "Если на изображении НЕТ текста, формул и таблиц — выведи ТОЛЬКО одно слово: [NO_TEXT]. "
        "Не описывай изображение, не добавляй пояснений."
    )

    FORMULA_PROMPT = (
        "Ты — OCR-система для математических формул. "
        "На изображении математическая формула. "
        "Распознай символы и запиши формулу в LaTeX формате между $...$ или $$...$$. "
        "Верни ТОЛЬКО LaTeX код, без слов [FORMULA] или [NO_TEXT] или любых других пояснений."
    )

    FORMULA_RETRY_PROMPT = (
        "Ты — OCR-система для математических формул. На изображении математическая формула. "
        "Распознай символы и запиши формулу в LaTeX формате между $$...$$. "
        "Пример: $$\\frac{a}{b} + c$$. Верни только LaTeX код, без пояснений."
    )

    TABLE_PROMPT = (
        "Ты — OCR-система. На изображении таблица с текстом и возможно формулами. "
        "Воспроизведи таблицу в Markdown-формате (pipe table). "
        "Если есть формулы — используй LaTeX внутри $...$ или $$...$$. "
        "Сохрани структуру таблицы максимально точно. "
        "Не добавляй пояснений, верни только markdown-таблицу."
    )

    def __init__(self, config: dict):
        self.url = config["url"]
        self.model = config["model"]
        self.timeout = config.get("timeout", 120)
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.05)
        self.top_p = config.get("top_p", 0.5)
        self.top_k = config.get("top_k", 10)
        self.enable_thinking = config.get("enable_thinking", False)
        self.reasoning_format = config.get("reasoning_format", "none")
        self.prompt = config.get("prompt", self.DEFAULT_PROMPT)

    def add_white_padding(
        self,
        img_path: str,
        padding_percent: float = 0.20,
        min_padding: int = 80,
        max_side: int = 1536,
    ) -> str:
        """Белая рамка вокруг изображения + downscale, если слишком большое."""
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        pad = max(min_padding, int(max(w, h) * padding_percent))
        new_w, new_h = w + 2 * pad, h + 2 * pad

        canvas = Image.new("RGB", (new_w, new_h), color="white")
        canvas.paste(img, (pad, pad))

        if max(new_w, new_h) > max_side:
            canvas.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        out = os.path.join(
            os.path.dirname(img_path), f"{Path(img_path).stem}_padded.png"
        )
        canvas.save(out, "PNG")
        return out

    # def _call_api(self, image_path: str, prompt: str) -> str:
    #     """Внутренний метод для вызова API."""
    #     prepared = self.add_white_padding(image_path)

    #     with open(prepared, "rb") as f:
    #         b64 = base64.b64encode(f.read()).decode()

    #     resp = requests.post(
    #         self.url,
    #         headers={"Content-Type": "application/json"},
    #         json={
    #             "model": self.model,
    #             "messages": [
    #                 {
    #                     "role": "user",
    #                     "content": [
    #                         {
    #                             "type": "image_url",
    #                             "image_url": {"url": f"data:image/png;base64,{b64}"},
    #                         },
    #                         {"type": "text", "text": prompt},
    #                     ],
    #                 }
    #             ],
    #             "max_tokens": self.max_tokens,
    #             "temperature": self.temperature,
    #             "top_p": self.top_p,
    #             "top_k": self.top_k,
    #             "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
    #             "reasoning_format": self.reasoning_format,
    #         },
    #         timeout=self.timeout,
    #     )

    #     if resp.status_code != 200:
    #         return f"[OCR_ERROR {resp.status_code}]"

    #     return resp.json()["choices"][0]["message"]["content"].strip()

    def _call_api(self, image_path: str, prompt: str) -> str:
        """Внутренний метод для вызова vLLM API."""
        prepared = self.add_white_padding(image_path)

        with open(prepared, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        # Формируем payload для vLLM (OpenAI-compatible)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,  # vLLM поддерживает как расширение
            # Динамическое отключение thinking для OCR
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

        resp = requests.post(
            self.url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"OCR HTTP {resp.status_code}: {resp.text[:500]}")

        payload = resp.json()
        choice = payload["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"OCR response truncated by max_tokens: {Path(image_path).name}")
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise RuntimeError("OCR response has no text content")
        return content.strip()



    def _contains_formula_marker(self, raw: str) -> bool:
        """Проверяет, содержит ли строка маркер [FORMULA] в любом виде."""
        if not raw:
            return False
        return "[FORMULA]" in raw.upper()

    def _clean_response(self, raw: str) -> str:
        """Очистка ответа OCR от служебных маркеров."""
        if not raw:
            return ""
        if raw.upper() == "[NO_TEXT]":
            return ""
        clean = raw.strip("\"'«» .")
        if clean.upper() in ("NO_TEXT", "[NO_TEXT]"):
            return ""
        if self._contains_formula_marker(raw):
            return ""

        # Удаляем <think>...</think> блоки из вывода модели
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        clean = clean.strip("\"'«» .")
        return clean

    def ocr_image(self, image_path: str) -> str:
        """OCR через VLM. Возвращает '' если на картинке нет текста."""
        raw = self._call_api(image_path, self.prompt)
        return self._clean_response(raw)

    def ocr_formula_region(self, image_path: str, retries: int = 2) -> str:
        """OCR для формул — возвращает LaTeX или пустую строку."""
        # Первичный запрос
        raw = self._call_api(image_path, self.FORMULA_PROMPT)

        # Если получили [FORMULA] или пусто — пробуем ещё раз с другим промптом
        attempt = 0
        while (self._contains_formula_marker(raw) or not raw) and attempt < retries:
            raw = self._call_api(image_path, self.FORMULA_RETRY_PROMPT)
            attempt += 1

        # Если всё ещё [FORMULA] — возвращаем пустую строку
        if self._contains_formula_marker(raw):
            return ""

        raw = self._clean_response(raw)
        if not raw:
            return ""

        # Извлекаем только LaTeX ($...$ или $$...$$ или $$$...$$$)
        # Нормализуем количество $: $$$ → $$
        raw_clean = re.sub(r"\${3,}", "$$", raw)
        display = re.search(r"\$\$(.+?)\$\$", raw_clean, re.DOTALL)
        inline = re.search(r"\$(.+?)\$", raw_clean)
        latex = ""
        if display:
            latex = f"$${display.group(1).strip()}$$"
        elif inline:
            latex = f"${inline.group(1).strip()}$"
        elif "\\" in raw:
            latex = f"${raw.strip()}$"
        # Проверяем, что извлеченный LaTeX не пустой (только $ и пробелы)
        if latex and not latex.replace("$", "").strip():
            return ""
        return latex

    def ocr_table_image(self, image_path: str) -> str:
        """OCR для таблиц — возвращает Markdown-таблицу."""
        raw = self._call_api(image_path, self.TABLE_PROMPT)
        raw = self._clean_response(raw)

        # Проверяем что это похоже на таблицу
        if "|" not in raw:
            # Попробуем добавить форматирование таблицы
            lines = raw.strip().split("\n")
            if len(lines) > 1:
                # Пытаемся превратить в markdown таблицу
                return self._convert_to_markdown_table(raw)
        return raw

    def _convert_to_markdown_table(self, text: str) -> str:
        """Пытается превратить текст в markdown таблицу."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return text

        # Определяем разделитель — обычно это пробелы или табы
        result = []
        for i, line in enumerate(lines):
            # Разбиваем по множественным пробелам
            cells = [c.strip() for c in re.split(r"\s{2,}", line) if c.strip()]
            if cells:
                result.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    result.append("| " + " | ".join(["---"] * len(cells)) + " |")

        return "\n".join(result) if result else text
