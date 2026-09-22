#!/usr/bin/env python3
"""Построение бенчмарк-датасета: var/benchmark/dataset.jsonl.

Части:
  1. XLSX (режим по умолчанию): 80 вопросов из Voprosy_dlya_avtotesta.xlsx,
     лист «Тестирование» (№ / Категория / Вопрос / Ответ / Наименование НПА).
     source_doc сопоставляется с файлами MARKDOWN_DIR (нормализация имён),
     ground_truth_chunk_ids — чанки документа с максимальным word-overlap
     с экспертным ответом (все чанки с overlap >= 0.5 x best, минимум 1).
  2. RAGAS (--ragas --n N --seed S): синтетические вопросы через
     ragas TestsetGenerator (LLM = vLLM Qwen, вопросы на русском),
     документы = чанки из chunks.pkl.

Запуск:
  PYTHONPATH=src venv/bin/python src/scripts/build_dataset.py            # xlsx
  PYTHONPATH=src venv/bin/python src/scripts/build_dataset.py --ragas --n 100 --seed 42
  PYTHONPATH=src venv/bin/python src/scripts/build_dataset.py --ragas --deepseek --n 100 --seed 42
  PYTHONPATH=src venv/bin/python src/scripts/build_dataset.py --deepseek-list-models   # модели по токену DEEPSEEK_API_KEY
  (--max-chunks N — ограничить число чанков для ragas-генерации; только для smoke-тестов)
"""

import argparse
import contextlib
import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpyxl import load_workbook

from common.chunk_store import ChunkDatabase
from common.config import (
    BENCHMARK_DIR,
    CHUNKS_PATH,
    MARKDOWN_DIR,
    SUMMARIZER_LLM_BASE_URL,
    SUMMARIZER_LLM_MODEL,
)
from common.logging_utils import setup_logging
from indexing.bm25_indexer import tokenize

logger = logging.getLogger(__name__)

XLSX_PATH = Path(__file__).resolve().parents[2] / "Voprosy_dlya_avtotesta.xlsx"
XLSX_SHEET = "Тестирование"

# ── DeepSeek API (OpenAI-совместимый, Bearer-токен) ──
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL_DEFAULT = "deepseek-flash"
# Thinking включён по умолчанию (reasoning в reasoning_content, content — чистый
# JSON). Конкурентность: 2500 (flash) / 500 (pro) — кап воркеров не нужен.
# Внимание: в thinking-режиме temperature игнорируется (без ошибки).


def deepseek_list_models() -> int:
    """GET /models по токену DEEPSEEK_API_KEY — список доступных моделей."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ОШИБКА: не задан DEEPSEEK_API_KEY (export DEEPSEEK_API_KEY=...)", file=sys.stderr)
        return 1
    import httpx

    try:
        resp = httpx.get(
            f"{DEEPSEEK_BASE_URL.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"ОШИБКА: {e.response.status_code} {e.response.text[:300]}", file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"ОШИБКА сети: {e}", file=sys.stderr)
        return 1
    models = [m.get("id", "?") for m in resp.json().get("data", [])]
    print(f"Доступные модели ({DEEPSEEK_BASE_URL}):")
    for m in sorted(models):
        print(f"  {m}")
    if not models:
        print("  (пусто)")
    return 0

# Ручные алиасы для source_doc из xlsx, не matchящихся нормализацией (пусто = всё matchится)
MANUAL_ALIASES: dict[str, str] = {}


def normalize_name(name: str) -> str:
    """Нормализация имени файла для сопоставления: без расширения и суффикса (1), пробелы/подчёркивания схлопнуты."""
    s = re.sub(r"\.(pdf|docx|xlsx|md)$", "", name.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*\(\d+\)$", "", s)
    s = re.sub(r"[\s_]+", " ", s)
    return s.lower().strip()


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def build_md_index() -> dict[str, str]:
    """norm_name -> имя .md файла в MARKDOWN_DIR."""
    md_dir = Path(MARKDOWN_DIR)
    index: dict[str, str] = {}
    for f in sorted(md_dir.glob("*.md")):
        index[normalize_name(f.name)] = f.name
    return index


def match_source_doc(source_doc: str | None, md_index: dict[str, str]) -> str | None:
    if not source_doc:
        return None
    name = str(source_doc).strip()
    if name in MANUAL_ALIASES:
        name = MANUAL_ALIASES[name]
    hit = md_index.get(normalize_name(name))
    if hit:
        return hit
    # prefix-совпадение (единственное)
    n = normalize_name(name)
    prefix = n[:25]
    cands = [v for k, v in md_index.items() if k.startswith(prefix) or n.startswith(k[:25])]
    if len(cands) == 1:
        return cands[0]
    return None


def _corpus_idf(db: ChunkDatabase) -> dict[str, float]:
    """idf-веса токенов по всему корпусу: частая лексика (проект/отчёт/значение)
    получает низкий вес, различительные токены — высокий."""
    import math
    from collections import Counter

    df: Counter = Counter()
    for c in db.chunks:
        for tok in set(tokenize(c.text)):
            df[tok] += 1
    n = len(db.chunks)
    return {tok: math.log(n / (1 + d)) for tok, d in df.items()}


def _gt_tokens(text: str) -> set[str]:
    """Токены для gt-матчинга с нормализацией ё→е («отчётным» == «отчетным»)."""
    return set(tokenize(text.replace("ё", "е").replace("Ё", "Е")))


def ground_truth_chunk_ids(
    db: ChunkDatabase,
    md_file: str,
    reference_answer: str,
    idf: dict[str, float] | None = None,
) -> list[str]:
    """Чанки документа md_file с максимальным word-overlap с ответом.

    Токенизация как в indexing.bm25_indexer.tokenize (плюс ё→е); overlap
    взвешен idf (без весов общая лексика документа вытесняет различительные
    токены). Отбор: все чанки с overlap >= max(0.6 * best, best - 3),
    не более TOP_K.
    """
    TOP_K = 3
    ref_tokens = _gt_tokens(reference_answer) if reference_answer else set()
    doc_chunks = [c for c in db.chunks if Path(c.doc_path).name == md_file]
    if not doc_chunks or not ref_tokens:
        return [c.id for c in doc_chunks[:1]] if doc_chunks else []

    scored = []
    for chunk in doc_chunks:
        common = _gt_tokens(chunk.text) & ref_tokens
        overlap = sum(idf.get(t, 0.0) for t in common) if idf is not None else len(common)
        scored.append((overlap, chunk.id))
    best = max(s for s, _ in scored)
    if best == 0:
        return [scored[0][1]]
    threshold = max(0.6 * best, best - 3)
    selected = [(s, cid) for s, cid in scored if s >= threshold]
    if len(selected) > TOP_K:
        selected = sorted(selected, key=lambda x: -x[0])[:TOP_K]
    return [cid for _, cid in selected]


def build_xlsx_part(db: ChunkDatabase) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    unmatched: list[str] = []
    md_index = build_md_index()
    idf = _corpus_idf(db)
    logger.info("Corpus idf computed for ground-truth scoring")

    wb = load_workbook(XLSX_PATH, read_only=True, data_only=True)
    ws = wb[XLSX_SHEET]
    for row in ws.iter_rows(min_row=2, values_only=True):
        num, category, question, answer, source_doc = (list(row) + [None] * 5)[:5]
        if question is None or not str(question).strip():
            continue
        question = str(question).strip()
        answer = str(answer).strip() if answer is not None else ""

        md_file = match_source_doc(source_doc, md_index)
        if md_file is None and source_doc:
            unmatched.append(str(source_doc).strip())

        records.append({
            "question_id": f"xlsx_{num}",
            "origin": "xlsx",
            "question": question,
            "reference_answer": answer,
            "source_doc": md_file,
            "ground_truth_chunk_ids": ground_truth_chunk_ids(db, md_file, answer, idf) if md_file else [],
            "question_type": f"category:{category}" if category else "category:unknown",
        })
    wb.close()
    return records, unmatched


RU_INSTRUCTION_SUFFIX = (
    "\n\nIMPORTANT: Write the generated question (user_input) and the reference answer "
    "strictly in Russian (русский язык). Use the official style of Russian normative "
    "legal documents (НПА) in the question and answer. Do not use any other language."
)


def build_ragas_llm(
    base_url: str | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    api_key: str = "dummy",
    use_deepseek: bool = False,
):
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from common.config import SUMMARIZER_LLM_MAX_TOKENS, create_llm

    if use_deepseek:
        base_url = base_url or DEEPSEEK_BASE_URL
        model = model or DEEPSEEK_MODEL_DEFAULT
    else:
        base_url = base_url or SUMMARIZER_LLM_BASE_URL
        model = model or SUMMARIZER_LLM_MODEL
    max_tokens = max_tokens or min(SUMMARIZER_LLM_MAX_TOKENS, 16384)

    if use_deepseek:
        from langchain_openai import ChatOpenAI

        # DeepSeek API OpenAI-совместимый; thinking включён по умолчанию —
        # reasoning в reasoning_content, content чистым JSON, extra_body не нужен.
        # reasoning_effort=low: transforms-задачи простые, глубокий reasoning
        # только сжигает токены (default high ~в 2 раза дороже на вызов).
        # max_retries повышен: 429/сбой сети отбиваются бэкоффом openai-клиента
        chat = ChatOpenAI(
            base_url=base_url.rstrip("/") + "/",
            api_key=api_key,
            model=model,
            temperature=0.3,
            max_tokens=max_tokens,
            max_retries=6,
            timeout=300,
            reasoning_effort="low",
        )
    else:
        # локальный vLLM: thinking ON — reasoning в reasoning_content,
        # content остаётся чистым JSON (эмпирически JSON-комплаенс лучше)
        chat = create_llm(
            base_url=base_url,
            model=model,
            temperature=0.3,
            max_tokens=max_tokens,
            enable_thinking=True,
        )
    return LangchainLLMWrapper(
        chat,
        # и vLLM, и DeepSeek возвращают finish_reason="length", который дефолтный
        # парсер ragas не признаёт
        is_finished_parser=lambda r: True,
        run_config=RunConfig(timeout=600, max_retries=3, max_wait=10),
    )


def _install_tolerant_run_async_tasks():
    """Гасит исключения отдельных ragas-задач (transforms-фаза).

    run_async_tasks поднимает ПЕРВОЕ исключение и роняет весь многочасовой прогон
    (raise_exceptions=False покрывает только фазу синтеза). Единственный плохой
    LLM-ответ (не-JSON) не должен убивать генерацию — узел просто останется
    без summary/theme.
    """
    import ragas.async_utils as async_utils
    import ragas.testset.transforms.engine as engine

    if getattr(async_utils.run_async_tasks, "_ragas_tolerant", False):
        return
    orig = async_utils.run_async_tasks

    def tolerant(tasks, *args, **kwargs):
        async def guard(coro):
            try:
                return await coro
            except Exception as e:  # noqa: BLE001
                logger.warning(f"ragas task failed (skipped): {type(e).__name__}: {str(e)[:200]}")
                return None
        return orig([guard(c) for c in tasks], *args, **kwargs)

    tolerant._ragas_tolerant = True
    async_utils.run_async_tasks = tolerant
    engine.run_async_tasks = tolerant


@contextlib.contextmanager
def _tolerant_executor_nan():
    """Обход бага ragas: Executor с raise_exceptions=False возвращает np.nan
    для упавшей задачи, а generate() потом падает на нём
    (TypeError: 'float' object is not iterable для сценариев;
    ValidationError TestsetSample для синтеза). На время генерации
    заменяем nan на [] — пустые результаты отфильтровываются ниже.
    Патч локальный: метрики (шаг E) не затрагиваются.
    """
    from ragas.executor import Executor

    orig = Executor.wrap_callable_with_index

    def tolerant(self, callable, counter):
        wrapped = orig(self, callable, counter)

        async def wrapped_async(*args, **kwargs):
            idx, result = await wrapped(*args, **kwargs)
            if isinstance(result, float) and math.isnan(result):
                return idx, []
            return idx, result

        return wrapped_async

    Executor.wrap_callable_with_index = tolerant
    try:
        yield
    finally:
        Executor.wrap_callable_with_index = orig


def _install_tolerant_testset_sample():
    """Обход бага ragas: упавшие synthesis-задачи (raise_exceptions=False)
    возвращают nan, а generate() в конце заворачивает их в TestsetSample
    и падает с ValidationError ПОСЛЕ всей генерации, теряя результат.
    Подмена TestsetSample толерантным подклассом (eval_sample: Any),
    nan-сэмплы отфильтровываются при извлечении записей.
    """
    import typing as t

    import ragas.testset.synthesizers.generate as gen_module
    from ragas.testset.synthesizers.testset_schema import TestsetSample

    if getattr(gen_module.TestsetSample, "_ragas_tolerant", False):
        return

    class TolerantTestsetSample(TestsetSample):
        eval_sample: t.Any

    TolerantTestsetSample._ragas_tolerant = True
    gen_module.TestsetSample = TolerantTestsetSample


def build_ragas_part(
    db,
    n: int,
    seed: int,
    max_chunks: int | None,
    workers: int = 16,
    llm_base_url: str | None = None,
    llm_max_tokens: int | None = None,
    llm_model: str | None = None,
    llm_api_key: str = "dummy",
    use_deepseek: bool = False,
    embedding_device: str = "cuda",
) -> list[dict]:
    from langchain_core.documents import Document as LCDocument
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import default_query_distribution
    from common.jina_embeddings import JinaRagasEmbeddings

    chunks = db.chunks
    if max_chunks is not None and max_chunks < len(chunks):
        rng = random.Random(seed)
        chunks = rng.sample(chunks, max_chunks)
        logger.info(f"RAGAS generation restricted to {len(chunks)} chunks (smoke mode)")

    docs = [
        LCDocument(page_content=c.text, metadata={"chunk_id": c.id})
        for c in chunks
    ]

    llm = build_ragas_llm(
        base_url=llm_base_url,
        max_tokens=llm_max_tokens,
        model=llm_model,
        api_key=llm_api_key,
        use_deepseek=use_deepseek,
    )
    embeddings = JinaRagasEmbeddings(device=embedding_device)
    _install_tolerant_run_async_tasks()
    _install_tolerant_testset_sample()
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    # Локализация промптов генерации на русский (ключевой промпт вопрос+ответ)
    distribution = default_query_distribution(llm)
    for syn, _prob in distribution:
        prompts = syn.get_prompts()
        for name, prompt in prompts.items():
            if name == "query_answer_generation_prompt":
                prompt.instruction = (prompt.instruction or "") + RU_INSTRUCTION_SUFFIX
        syn.set_prompts(**prompts)

    logger.info(f"Generating {n} synthetic questions with ragas (this is the longest stage)...")
    with _tolerant_executor_nan():
        testset = generator.generate_with_chunks(
            docs,
            testset_size=n,
            query_distribution=distribution,
            run_config=RunConfig(timeout=600, max_retries=3, max_wait=10, max_workers=workers),
            raise_exceptions=False,
        )

    # соответствие текстов чанков -> chunk_id (точное, fallback — первые 200 символов);
    # multi-hop контексты имеют префикс "<N-hop>\n\n" — срезаем его перед сравнением
    hop_prefix = re.compile(r"^<\d+-hop>\s*")

    def _strip_hop(ctx: str) -> str:
        return hop_prefix.sub("", ctx).strip()

    by_text = {c.text: c.id for c in chunks}
    by_prefix = {c.text[:200]: c.id for c in chunks}

    from ragas.dataset_schema import SingleTurnSample

    records: list[dict] = []
    matched = 0
    dropped_placeholders = 0
    degenerate = 0
    failed = 0
    # шаблонные заглушки из таблиц НПА: вопрос про них — мусор
    placeholder_re = re.compile(r"[«\"\"]\s*(Наименование|[Xx](?: [Xx]){2,})\s*[»\"\"]")
    # мета-вопросы про структуру таблицы изменений («как соотносятся колонки
    # "Действующая редакция" и "новая редакция"») — про устройство документа,
    # а не про содержание НПА
    table_meta_re = re.compile(r"действующ\w+ редакци\w+", re.IGNORECASE)
    for i, sample in enumerate(testset.samples):
        es = sample.eval_sample
        if not isinstance(es, SingleTurnSample):  # nan-заглушка упавшей задачи
            failed += 1
            continue
        question = (es.user_input or "").strip()
        if not question:
            continue
        if placeholder_re.search(question):
            dropped_placeholders += 1
            logger.warning(f"dropped placeholder question: {question[:100]}")
            continue
        if table_meta_re.search(question) and re.search(r"нов\w+ редакци\w+", question, re.IGNORECASE):
            dropped_placeholders += 1
            logger.warning(f"dropped table-meta question: {question[:100]}")
            continue
        ref_contexts = list(es.reference_contexts or [])
        gt_ids = []
        for ctx in ref_contexts:
            text = _strip_hop(ctx)
            cid = by_text.get(text) or by_prefix.get(text[:200])
            if cid and cid not in gt_ids:
                gt_ids.append(cid)
        matched += 1 if gt_ids else 0
        question_type = f"synth:{sample.synthesizer_name}"
        # вырожденный multi-hop: фактически один источник → это single-hop
        if "multi_hop" in question_type and len(gt_ids) < 2:
            question_type = "synth:single_hop_degenerate"
            degenerate += 1
        records.append({
            "question_id": f"ragas_{i}",
            "origin": "ragas",
            "question": question,
            "reference_answer": (es.reference or "").strip(),
            "source_doc": None,
            "ground_truth_chunk_ids": gt_ids,
            "question_type": question_type,
        })
    total = len(records)
    pct = 100.0 * matched / total if total else 0.0
    logger.info(
        f"RAGAS: {total} questions (failed tasks: {failed}, dropped placeholders: {dropped_placeholders}, "
        f"degenerate multi-hop relabeled: {degenerate}), "
        f"contexts matched to chunk_id: {matched}/{total} ({pct:.1f}%)"
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ragas", action="store_true", help="добавить ragas-часть (медленно)")
    parser.add_argument("--n", type=int, default=100, help="число ragas-вопросов")
    parser.add_argument("--seed", type=int, default=42, help="seed для перемешивания/сэмплирования")
    parser.add_argument("--max-chunks", type=int, default=None, help="ограничить чанки для ragas (smoke)")
    parser.add_argument("--workers", type=int, default=16, help="параллелизм LLM-вызовов ragas")
    parser.add_argument("--llm-base-url", default=None, help="endpoint vLLM (по умолчанию SUMMARIZER_LLM_BASE_URL из .env)")
    parser.add_argument("--max-tokens", type=int, default=None, help="max_tokens LLM для ragas (по умолчанию min(SUMMARIZER_LLM_MAX_TOKENS, 16384))")
    parser.add_argument("--deepseek", action="store_true", help="генерировать через DeepSeek API (токен DEEPSEEK_API_KEY)")
    parser.add_argument("--deepseek-model", default=DEEPSEEK_MODEL_DEFAULT, help=f"модель DeepSeek (по умолчанию {DEEPSEEK_MODEL_DEFAULT}; список: --deepseek-list-models)")
    parser.add_argument("--deepseek-list-models", action="store_true", help="вывести доступные модели по токену DEEPSEEK_API_KEY и выйти")
    parser.add_argument("--embedding-device", default="cuda", choices=["cuda", "cpu"], help="устройство jina-эмбеддингов для ragas (по умолчанию cuda)")
    parser.add_argument("--out", type=Path, default=None, help="путь выходного jsonl")
    args = parser.parse_args()

    if args.deepseek_list_models:
        sys.exit(deepseek_list_models())

    ds_kwargs = {}
    if args.deepseek:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("ОШИБКА: --deepseek требует DEEPSEEK_API_KEY (export DEEPSEEK_API_KEY=...)", file=sys.stderr)
            sys.exit(1)
        ds_kwargs = dict(
            llm_base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
            llm_model=args.deepseek_model,
            llm_api_key=api_key,
            use_deepseek=True,
        )
        logger.info(f"RAGAS generation via DeepSeek: model={args.deepseek_model}, endpoint={ds_kwargs['llm_base_url']}")

    setup_logging("build_dataset")

    out_path = args.out or (BENCHMARK_DIR / "dataset.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading chunks from {CHUNKS_PATH}")
    db = ChunkDatabase.load(CHUNKS_PATH)
    logger.info(f"Loaded {len(db.chunks)} chunks")

    records, unmatched = build_xlsx_part(db)
    n_xlsx = len(records)
    n_gt = sum(1 for r in records if r["ground_truth_chunk_ids"])
    logger.info(f"XLSX part: {n_xlsx} questions, with ground truth: {n_gt}, unmatched source_doc: {len(unmatched)}")
    for d in unmatched:
        logger.warning(f"  unmatched source_doc: {d}")

    if args.ragas:
        llm_kwargs: dict = {}
        if args.llm_base_url:
            llm_kwargs["llm_base_url"] = args.llm_base_url
        if args.max_tokens:
            llm_kwargs["llm_max_tokens"] = args.max_tokens
        llm_kwargs.update(ds_kwargs)  # --deepseek перекрывает базовые флаги
        records += build_ragas_part(
            db, args.n, args.seed, args.max_chunks, args.workers,
            embedding_device=args.embedding_device, **llm_kwargs
        )

    # дедупликация вопросов + перемешивание с seed
    seen: set[str] = set()
    unique_records = []
    for r in records:
        key = normalize_question(r["question"])
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(r)
    dropped = len(records) - len(unique_records)
    rng = random.Random(args.seed)
    rng.shuffle(unique_records)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in unique_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"Dataset saved: {out_path}")
    print(f"Total: {len(unique_records)} (dropped duplicates: {dropped})")
    print(f"By origin: {dict(Counter(r['origin'] for r in unique_records))}")
    print(f"By type:   {dict(Counter(r['question_type'] for r in unique_records))}")
    print(f"XLSX with ground truth: {n_gt}/{n_xlsx}")
    if unmatched:
        print(f"UNMATCHED source_doc ({len(unmatched)}):")
        for d in unmatched:
            print(f"  - {d}")


if __name__ == "__main__":
    main()
