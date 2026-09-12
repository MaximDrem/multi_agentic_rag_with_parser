"""Конфигурация OCR-клиента для vLLM (Qwen3.6-27B AWQ)."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def load_ocr_config() -> dict:
    return {
        "url": os.getenv("OCR_API_URL", "http://localhost:8070/v1/chat/completions"),
        "model": os.getenv("OCR_MODEL", "Qwen3.6-27B-AWQ"),
        "timeout": int(os.getenv("OCR_TIMEOUT", "120")),
        "max_tokens": int(os.getenv("OCR_MAX_TOKENS", "4096")),
        "temperature": float(os.getenv("OCR_TEMPERATURE", "0.05")),
        "top_p": float(os.getenv("OCR_TOP_P", "0.5")),
        "top_k": int(os.getenv("OCR_TOP_K", "10")),
        "enable_thinking": False,
    }


INPUT_DIR = os.getenv("PARSER_INPUT_DIR", str(PROJECT_ROOT / "НПА для НИР"))
OUTPUT_DIR = os.getenv("PARSER_OUTPUT_DIR", str(PROJECT_ROOT / "var" / "markdown_output"))
CLEAN_OUTPUT_DIR = os.getenv("PARSER_CLEAN_OUTPUT_DIR", str(PROJECT_ROOT / "var" / "markdown_output_clean"))
TEMP_DIR = os.getenv("PARSER_TEMP_DIR", str(PROJECT_ROOT / "var" / "temp_images"))
