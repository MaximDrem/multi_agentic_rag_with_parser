"""Главный оркестратор парсинга документов в Markdown."""

import os
import sys
import time
from pathlib import Path

from parser.config import load_ocr_config, INPUT_DIR, OUTPUT_DIR, TEMP_DIR
from parser.ocr_client import OCRCient

from parser.docx_handler import process_docx
from parser.pdf_handler import process_pdf
from parser.xlsx_handler import process_xlsx


HANDLERS = {
    ".docx": process_docx,
    ".pdf": process_pdf,
    ".xlsx": process_xlsx,
}


def collect_files(input_dir: str) -> list:
    """Собирает все поддерживаемые файлы из директории."""

    files = []
    for fname in sorted(os.listdir(input_dir)):
        fpath = os.path.join(input_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in HANDLERS:
            files.append((fpath, ext))
    return files


def process_file(filepath: str, ext: str, ocr_client: OCRCient) -> str:
    """Обрабатывает один файл соответствующим обработчиком."""

    handler = HANDLERS[ext]
    t0 = time.time()

    md_content = handler(filepath, ocr_client)

    elapsed = time.time() - t0
    print(f"  [{ext}] {os.path.basename(filepath)} → {len(md_content)} chars ({elapsed:.1f}s)")

    return md_content


def save_markdown(content: str, output_dir: str, source_filename: str):
    """Сохраняет Markdown в выходную папку."""

    os.makedirs(output_dir, exist_ok=True)

    base = Path(source_filename).stem
    out_path = os.path.join(output_dir, f"{base}.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    return out_path


def run(
    input_dir: str = None,
    output_dir: str = None,
    files_filter: list = None,
    dry_run: bool = False,
) -> dict:
    """
    Запускает полный цикл парсинга.

    Args:
        input_dir: входная папка с документами (по умолчанию INPUT_DIR)
        output_dir: выходная папка для markdown (по умолчанию OUTPUT_DIR)
        files_filter: список имён файлов для обработки (None = все)
        dry_run: только показать какие файлы будут обработаны

    Returns:
        словарь со статистикой
    """

    input_dir = input_dir or INPUT_DIR
    output_dir = output_dir or OUTPUT_DIR

    if files_filter:
        files = []
        for fname in files_filter:
            fpath = os.path.join(input_dir, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fname)[1].lower()
                if ext in HANDLERS:
                    files.append((fpath, ext))
                else:
                    print(f"  Пропущен (неподдерживаемый формат): {fname}")
            else:
                print(f"  Файл не найден: {fname}")
    else:
        files = collect_files(input_dir)

    if dry_run:
        print(f"Найдено файлов для обработки: {len(files)}")
        for fpath, ext in files:
            print(f"  {ext}: {os.path.basename(fpath)}")
        return {"total": len(files), "success": 0, "errors": 0}

    stats = {"total": len(files), "success": 0, "errors": 0, "files": []}

    # Инициализируем OCR-клиент
    ocr_config = load_ocr_config()
    ocr_client = OCRCient(ocr_config)

    print(f"OCR: {ocr_config['model']} @ {ocr_config['url']}")
    print(f"Папка ввода: {input_dir}")
    print(f"Папка вывода: {output_dir}")
    print(f"Всего файлов: {len(files)}")
    print()

    # Создаём временную папку
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for fpath, ext in files:
        fname = os.path.basename(fpath)
        print(f"Обработка: {fname}")

        try:
            md_content = process_file(fpath, ext, ocr_client)

            if md_content:
                out_path = save_markdown(md_content, output_dir, fname)
                stats["success"] += 1
                stats["files"].append({
                    "source": fname,
                    "output": out_path,
                    "size": len(md_content),
                })
            else:
                print(f"  Внимание: пустой результат")
                stats["errors"] += 1

        except Exception as e:
            print(f"  ОШИБКА: {e}")
            stats["errors"] += 1

        print()

    return stats


def main():
    """Точка входа из командной строки."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Универсальный парсер DOCX/PDF/XLSX → Markdown с OCR"
    )

    parser.add_argument(
        "--input", "-i",
        default=INPUT_DIR,
        help="Входная папка с документами",
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_DIR,
        help="Выходная папка для Markdown",
    )
    parser.add_argument(
        "--files", "-f",
        nargs="*",
        default=None,
        help="Конкретные файлы для обработки (имена относительно input)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Только показать список файлов без обработки",
    )
    parser.add_argument(
        "--type", "-t",
        choices=["docx", "pdf", "xlsx"],
        default=None,
        help="Обработать только файлы указанного типа",
    )

    args = parser.parse_args()

    # Фильтр по типу
    files_filter = args.files
    if args.type and not files_filter:
        input_dir = args.input
        all_files = os.listdir(input_dir)
        files_filter = [
            f for f in all_files
            if f.lower().endswith(f".{args.type}")
        ]

    stats = run(
        input_dir=args.input,
        output_dir=args.output,
        files_filter=files_filter,
        dry_run=args.dry_run,
    )

    # Итог
    print("=" * 60)
    print(f"Готово. Успешно: {stats['success']}, Ошибок: {stats['errors']}, Всего: {stats['total']}")

    return stats


if __name__ == "__main__":
    main()
