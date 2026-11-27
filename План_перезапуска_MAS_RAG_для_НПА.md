# План полного перезапуска MAS-RAG для русскоязычных НПА

**Техническое задание для кодингового агента**
**Protocol ID:** `npa_mas_component_v1`
**Базовый проверенный snapshot:** `1d6e10d75ced83368997bba1df4f203395a33b1e`
**Дата составления:** 2026-11-27

## 1. Назначение и ожидаемый результат

Нужно не продолжать `coordinated_mechanism_v1`, а после контролируемой очистки построить новый экспериментальный стенд для сравнения компонентов централизованно оркестрируемой многоагентной retrieval-системы на русскоязычных нормативно-правовых актах (НПА).

Исследуемая система имеет **фиксированный порядок вычислений**:

```text
запрос
  → [опционально planner]
  → формирование lexical/semantic queries
  → BM25-канал и Dense-канал
  → детерминированное объединение кандидатов
  → Jina или LLM-selector
  → top-10
  → метрики, traces, стоимость и reliability
```

Многоагентность здесь означает специализированные роли, локальные решения и структурированные сообщения внутри общего pipeline. Не заявлять децентрализацию, самоорганизацию, динамическое планирование или автономный выбор порядка шагов.

**Главный научный результат:** не утверждение «первый MAS-RAG для права», а воспроизводимый компонентный анализ того, какие механизмы — специализация retrieval workers, planner, LLM-reranking и worker reports — дают эффект на разных типах запросов НПА и какой ценой.

Кодинговый агент должен передать:

1. очищенный и собираемый проект;
2. новый versioned protocol и prompts;
3. новые реализации двух retrieval workers, planner и двух selectors;
4. полный набор новых runs без использования старых evaluation results;
5. same-pool артефакты для чистых ranking/communication contrasts;
6. таблицы качества, стоимости, надёжности и результатов по типологиям НПА;
7. машинно проверяемый results pack и отчёт без автоматического «решения о публикации».

## 2. Жёсткие ограничения

Эти ограничения действуют на всех этапах:

- **не изменять benchmark, qrels, исходные документы, chunk store и индексы**;
- **не раскрывать `.env`, API keys и другие секреты**;
- **не включать секреты в artifacts/results pack**;
- **не подбирать prompts, seeds, подгруппы или output policies после просмотра результатов**;
- **не скрывать errors/fallback**;
- **не считать отсутствующие token usage нулевыми**;
- benchmark остаётся прежним, несмотря на известные неточности; неточности описываются как limitations, а не исправляются неявно;
- старые runs, snapshots и выводы `coordinated_mechanism_v1` не используются как результаты нового эксперимента;
- не добавлять Jev, Dynamic MAS, GRPO, обучение retriever или второй раунд peer communication в core protocol;
- не пропускать конфигурации по промежуточным результатам: data-dependent gates в `npa_mas_component_v1` запрещены;
- не менять score threshold, chunking, embedding model, BM25/Dense индексы или текст документов ради улучшения метрик;
- не выдавать development benchmark за независимый test set;
- не делать вывод о статистической значимости по повторяющимся исходным строкам вопросов: независимая единица анализа — normalized unique-query group;
- не принимать фактический недобор search hits за ошибку: `actual ≤ target/request`, а saturation подлежит учёту;
- не компенсировать недобор hits повторными запросами, duplicates, дополнительными paraphrases в baseline или перераспределением бюджета между каналами.

## 3. Этап 1 — сначала удалить старый проект эксперимента

### 3.1. Граница удаления

В проверенном snapshot `src/retrieval` содержит 13 файлов. Удаляются только шесть legacy algorithm modules:

```text
src/retrieval/agentic.py
src/retrieval/agentic_service.py
src/retrieval/coordinated.py
src/retrieval/dynamic_mas.py
src/retrieval/multiagent.py
src/retrieval/multiagent_v2.py
```

Одновременно удаляются зависимые A2A wrappers:

```text
src/agents/agentic_retriever.py
src/agents/agentic_retriever_alpha.py
src/agents/dynamic_mas.py
src/agents/multiagent_coordinated.py
src/agents/multiagent_retriever.py
src/agents/multiagent_retriever_v2.py
```

Удаляются старый экспериментальный стенд, его конфигурация и все производные evaluation results:

```text
src/scripts/coordinated_experiment/
src/scripts/run_coordinated_experiment.py
experiments/coordinated_mechanism_v1.yaml
var/benchmark/final/evaluation/
```

Если в рабочей копии существует `tests/coordinated_experiment/`, её также удалить. В проверенном snapshot каталог `tests` не обнаружен, поэтому агент обязан проверить фактическое дерево, а не считать это утверждение универсальным.

### 3.2. Что сохранить

Не удалять общие компоненты retrieval:

```text
src/retrieval/a2a.py
src/retrieval/base.py
src/retrieval/baseline_reranker.py
src/retrieval/contracts.py
src/retrieval/engine.py
src/retrieval/reranker.py
src/retrieval/rrf.py
```

Назначение сохраняемых файлов:

- `a2a.py` — transport, не исследуемый фактор;
- `engine.py` — доступ к уже построенным BM25/FAISS индексам и chunk store;
- `contracts.py` — общие dataclasses/budget primitives, которые можно рефакторить с тестами;
- `reranker.py` — Jina adapter;
- `rrf.py` — deterministic fusion, который нужно усилить явным tie-breaking;
- `base.py`, `baseline_reranker.py` — исходные сильные baselines, которые будут переведены на новый config-driven candidate builder.

Сохранить wrappers:

```text
src/agents/base_retriever.py
src/agents/baseline_reranker.py
src/agents/summarizer.py
```

Сохранить generic scripts, если после миграции они проходят import/smoke checks:

```text
src/scripts/build_indices.py
src/scripts/compare_retrieval_results.py
src/scripts/prepare_annotation_packs.py
src/scripts/run_a2a_benchmark.py
src/scripts/score_retrieval.py
src/scripts/test_e2e.py
```

### 3.3. Защищаемые benchmark-файлы

Как минимум следующие файлы должны остаться byte-identical:

```text
var/benchmark/final/benchmark_v3_review.xlsx
var/benchmark/final/coverage.jsonl
var/benchmark/final/evidence.jsonl
var/benchmark/final/manifest.json
var/benchmark/final/questions.jsonl
var/benchmark/final/qrels.jsonl
var/benchmark/final/unresolved.jsonl
var/benchmark/final/validation.json
```

Также защищаются все пути из `CHUNKS_PATH`, `FAISS_INDEX_PATH`, `BM25_INDEX_PATH`, исходные документы и secrets. Запрещён широкий `rm -rf var/benchmark/final`: удаляется только его подкаталог `evaluation`.

Для контрольного snapshot получены следующие SHA-256; они полезны для сверки, но агент обязан заново вычислить hashes в фактической рабочей копии:

| Файл | SHA-256 в проверенном snapshot |
|---|---|
| `questions.jsonl` | `aff110752180fcbad3323dea034f659fc8f46ce628f714eee7b34e5e0d7c4d23` |
| `qrels.jsonl` | `a162f4be821ee5c7deb26217fd6d867a6f5f07bc1906e834f222044cd4985212` |
| `evidence.jsonl` | `7a9c6704e6e39428c15f2e36e68adfb9efe4b1c7c1bfc1834f44dcaef26f874f` |
| `manifest.json` | `bf39fb64fea3bb8e186f15ca4c814953d277fdf32405fac7acf20d7f8e293a36` |
| `coverage.jsonl` | `2ee6c4673410c9e17dafeadbcd949527884e64f4a129e75cd9410eb63a33036a` |
| `unresolved.jsonl` | `07ff328d358378620cb8d2524c777f0f5fe105a207dc4027fc26a8d52f9960c7` |
| `validation.json` | `fd7b88d8770cffc9a895e3d417e82524ee6531ba92fa740b560c5ed477f38758` |
| `benchmark_v3_review.xlsx` | `141dadb4210cdd5d4685898467fd0440ee9679a854a6ab82b67089ab2d2671b7` |

Если HEAD или hashes отличаются, не выполнять удаление вслепую: зафиксировать diff, повторить inventory и применить тот же allowlist только после подтверждения назначения файлов.

### 3.4. Безопасная последовательность cleanup

Выполнять из корня репозитория. Старые процессы benchmark/retrieval должны быть остановлены до удаления, чтобы они не записали новые файлы в `evaluation`.

```bash
set -euo pipefail

# 1. Убедиться, что команда выполняется в git-корне.
test "$(git rev-parse --show-toplevel)" = "$PWD"
git rev-parse HEAD
git status --short

# При tracked изменениях — остановиться и не удалять файлы.
git diff --quiet
git diff --cached --quiet

# Untracked evaluation разрешён как derived artifact. Любые другие untracked paths
# сначала разобрать: автоматически удалять пользовательские файлы запрещено.
unexpected_untracked="$({ git ls-files --others --exclude-standard || true; } | grep -Ev '^var/benchmark/final/evaluation(/|$)' || true)"
if [ -n "$unexpected_untracked" ]; then
  printf '%s\n' 'Unexpected untracked paths:' "$unexpected_untracked"
  exit 1
fi

# 2. Проверить обязательные защищаемые входы.
protected=(
  var/benchmark/final/benchmark_v3_review.xlsx
  var/benchmark/final/coverage.jsonl
  var/benchmark/final/evidence.jsonl
  var/benchmark/final/manifest.json
  var/benchmark/final/questions.jsonl
  var/benchmark/final/qrels.jsonl
  var/benchmark/final/unresolved.jsonl
  var/benchmark/final/validation.json
)
for path in "${protected[@]}"; do test -f "$path"; done

mkdir -p var/audit/npa_mas_component_v1
sha256sum "${protected[@]}" > var/audit/npa_mas_component_v1/protected_before.sha256
find src/retrieval src/agents src/scripts experiments \
  -maxdepth 3 -type f | sort \
  > var/audit/npa_mas_component_v1/source_inventory_before.txt

# 3. Удалить явно перечисленные legacy algorithm modules и wrappers.
git rm -- \
  src/retrieval/agentic.py \
  src/retrieval/agentic_service.py \
  src/retrieval/coordinated.py \
  src/retrieval/dynamic_mas.py \
  src/retrieval/multiagent.py \
  src/retrieval/multiagent_v2.py \
  src/agents/agentic_retriever.py \
  src/agents/agentic_retriever_alpha.py \
  src/agents/dynamic_mas.py \
  src/agents/multiagent_coordinated.py \
  src/agents/multiagent_retriever.py \
  src/agents/multiagent_retriever_v2.py \
  experiments/coordinated_mechanism_v1.yaml \
  src/scripts/run_coordinated_experiment.py

git rm -r -- src/scripts/coordinated_experiment
if [ -d tests/coordinated_experiment ]; then git rm -r -- tests/coordinated_experiment; fi

# 4. Удалить ТОЛЬКО производные результаты. Не заменять эту строку удалением final/.
test -d var/benchmark/final/evaluation
rm -rf -- var/benchmark/final/evaluation

# 5. Сразу проверить, что protected inputs не изменились.
sha256sum -c var/audit/npa_mas_component_v1/protected_before.sha256
for path in "${protected[@]}"; do test -f "$path"; done
```

Если часть удаляемых файлов не отслеживается git, `git rm` завершится ошибкой. Не применять массовый `rm` вместо разбора: сопоставить каждый путь с allowlist, удалить только подтверждённые legacy-файлы и записать отклонение в cleanup report.

### 3.5. Обязательная миграция ссылок

Одного удаления модулей недостаточно. Обновить:

1. `cli.py` — убрать пункты `agentic`, `agentic_alpha`, `multiagent` и другие удалённые системы;
2. `src/scripts/start_agents.sh` — оставить только реально существующие modules и новый wrapper, когда он появится;
3. `src/scripts/run_a2a_benchmark.py::RETRIEVER_CARDS` — удалить cards шести legacy wrappers;
4. `README.md` — удалить команды запуска старых систем и старого experiment CLI;
5. `src/scripts/test_e2e.py` — убрать/заменить сценарии удалённых retrievers;
6. любые imports и docs, найденные поиском по старым именам.

Проверка после миграции:

```bash
grep -RInE \
  'retrieval\.(agentic|agentic_service|coordinated|dynamic_mas|multiagent|multiagent_v2)|coordinated_mechanism_v1|run_coordinated_experiment' \
  src cli.py README.md experiments || true

PYTHONPATH=src python -m compileall -q src cli.py
# Если pytest/tests присутствуют:
PYTHONPATH=src pytest -q
```

Пустой grep не заменяет import tests. Отдельно проверить запуск `--help` у сохраняемых CLI в корректно настроенном окружении, не печатая значения secrets.

### 3.6. Cleanup acceptance и отдельный commit

Cleanup считается завершённым только когда:

- все перечисленные legacy paths удалены;
- `var/benchmark/final/evaluation` отсутствует;
- protected hashes совпадают;
- не осталось imports/commands несуществующих модулей;
- `compileall` и доступные regression tests проходят;
- BM25, Dense и Jina smoke-тесты способны загрузить прежние индексы/model без их перезаписи;
- создан файл `var/audit/npa_mas_component_v1/cleanup_report.json` с HEAD, списком удалённых путей, hashes и результатами проверок.

После этого сделать отдельный commit, например:

```bash
git add -A
git commit -m "Remove legacy retrievers and coordinated experiment artifacts"
```

Новую архитектуру начинать следующим commit/серией commits. Не смешивать cleanup и новый эксперимент в одном необратимом изменении.

## 4. Научная рамка нового эксперимента

### 4.1. Исследовательские вопросы

- **RQ1 — role specialization:** дают ли специализированные BM25-agent и Dense-agent преимущество над одним generic LLM-rewriter при одинаковых channel budgets и Jina-selector?
- **RQ2 — planner:** какой эффект добавляет planner с распределением правовых аспектов между каналами относительно тех же workers без planner?
- **RQ3 — selector:** что даёт LLM-selector относительно Jina на **идентичном candidate pool**?
- **RQ4 — coordination reports:** что добавляют структурированные worker reports относительно того же LLM-selector без reports на **идентичном candidate pool**?
- **RQ5 — НПА и стоимость:** на каких типах запросов НПА проявляются положительные/отрицательные эффекты, и каковы tokens, calls, latency, fallback и search cost?

Главные confirmatory contrasts заранее фиксируются:

```text
C1 = M1 - S    # специализированные workers против generic single-LLM rewriting
C2 = M2 - M1   # добавление planner
C3 = M3 - M2   # LLM-selector против Jina, same pool
C4 = M4 - M3   # worker reports против no-reports, same pool
```

C1 и C2 — end-to-end component contrasts: candidate pool может измениться. C3 и C4 — механистические same-pool contrasts: любое различие pool делает сравнение невалидным.

### 4.2. Что является и что не является вкладом

Допустимые формулировки после получения результатов:

- компонентная модель оценки fixed-pipeline MAS retrieval;
- разделение candidate generation, ranking и information exchange;
- сравнение специализированных каналов с сильными hybrid/query-expansion baselines;
- эмпирический анализ механизмов на русскоязычных НПА;
- типология запросов и анализ границ применимости;
- стоимость и надёжность как часть архитектурного сравнения.

Запрещённые без дополнительных доказательств формулировки:

- «первая многоагентная система legal retrieval»;
- «динамическая/самоорганизующаяся MAS»;
- «доказана общая эффективность MAS-RAG»;
- «независимый test подтверждает обобщение»;
- «LLM-selector лучше», если same-pool контраст этому не соответствует;
- «planner улучшает ranking напрямую», если изменился прежде всего candidate pool.

### 4.3. Ближайший аналог LegalMALR и обязательное отличие

`LegalMALR` уже объединяет multi-agent query understanding, legal reformulations, iterative dense retrieval и LLM-based reranking для Chinese statute retrieval. Поэтому общая идея «legal MAS + query rewriting + LLM reranker» не может быть заявлена как новая.

Наш проект отличается постановкой и должен проверить эти отличия экспериментально:

| Ось | LegalMALR | `npa_mas_component_v1` |
|---|---|---|
| Домен | китайские statutes | русскоязычные НПА, прежний benchmark |
| Retrieval channels | iterative dense retrieval | отдельные BM25 и Dense workers + deterministic fusion |
| Обучение | unified MAS policy оптимизируется через GRPO | zero-shot/frozen prompts, без policy training |
| Главный вопрос | эффективность интегрированной legal framework | самостоятельный вклад workers, planner, selector и reports |
| Контроль ranking | LLM reranker как компонент framework | обязательные same-pool Jina/LLM и reports/no-reports contrasts |
| Анализ домена | difficult/colloquial/multi-issue legal queries | заранее зафиксированная multi-label типология русскоязычных запросов |
| Стоимость/надёжность | зависит от анализа конкретной статьи | обязательные per-stage traces, tokens/calls/latency/fallback |

Не утверждать, что в LegalMALR отсутствуют ablations или какие-либо таблицы, не проверив полный текст. Ссылка нужна для позиционирования, а не для отрицания чужих результатов.

## 5. Литературная основа архитектурных решений

Ссылки ниже обосновывают конкретные design choices. Они не означают дословного воспроизведения чужой системы и не доказывают эффективность на русском НПА.

| Компонент/метод | Что именно заимствуется на уровне подхода | Первичный источник | Что не следует копировать или заявлять |
|---|---|---|---|
| BM25 | lexical retrieval baseline и channel tool | Robertson, Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond*. 2009. [DOI](https://dl.acm.org/doi/10.1561/1500000019) | BM25 сам по себе не является LLM-агентом |
| Dense semantic rewriting | генерация semantic reformulations для снижения lexical gap | Ma et al. *Query Rewriting in Retrieval-Augmented Large Language Models*. EMNLP 2023. [ACL](https://aclanthology.org/2023.emnlp-main.322/) | Не называть обычные paraphrases методом HyDE |
| HyDE, только conceptual background | пример использования generated text как dense query representation | Gao et al. *Precise Zero-Shot Dense Retrieval without Relevance Labels*. ACL 2023. [ACL](https://aclanthology.org/2023.acl-long.99/) | Если не генерируется hypothetical document embedding, это не реализация HyDE |
| Multi-perspective rewriting | роли/перспективы как источник различных rewrites | *Agent4Ranking: Semantic Robust Ranking via Personalized Query Rewriting Using Multi-agent LLM*. [arXiv](https://arxiv.org/abs/2312.15450), [ACM TOIS DOI](https://dl.acm.org/doi/10.1145/3749099) | Не переносить demographic personas или MMoE в core |
| Role-specialized sequential MAS | декомпозиция на planner и специализированные роли с промежуточными сообщениями | *MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning*. [arXiv](https://arxiv.org/abs/2505.20096) | Не копировать QA/Extractor roles и не заявлять dynamic planning |
| Legal multi-agent query understanding | ближайший архитектурный ориентир для planner + legal rewrites + LLM reranking | *LegalMALR: Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval*. 2026, preprint. [arXiv](https://arxiv.org/abs/2601.17692) | Не переносить GRPO и не заявлять первенство общей legal MAS-идеи |
| RRF | детерминированный hybrid baseline и fusion кандидатов | Cormack, Clarke, Büttcher. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009, pp. 758–759. [DOI](https://dl.acm.org/doi/10.1145/1571941.1572114), [author PDF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) | Не подбирать `alpha/k` по итоговому test quality |
| LLM relevance ranking | listwise/permutation ranking и необходимость контроля исходного порядка | Sun et al. *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents*. EMNLP 2023, pp. 14918–14937. [ACL](https://aclanthology.org/2023.emnlp-main.923/), DOI `10.18653/v1/2023.emnlp-main.923` | Не называть selector «точным RankGPT», если нет его sliding-window algorithm |
| Multi-role reranking и ablations | methodological ориентир для role ablations и cost-aware reranking | Jin et al. *RankFlow: A Multi-Role Collaborative Reranking Workflow Utilizing Large Language Models*. [arXiv](https://arxiv.org/abs/2502.00709), WWW Companion 2025, DOI `10.1145/3701716.3717575` | Не добавлять Answerer/Summarizer только для внешнего сходства |
| Мотивация НПА-типологии | colloquial non-professional queries и lexical gap в statute retrieval | Su et al. *STARD: A Chinese Statute Retrieval Dataset Derived from Real-life Queries by Non-professionals*. Findings of EMNLP 2024, pp. 10658–10671. [ACL](https://aclanthology.org/2024.findings-emnlp.625/), DOI `10.18653/v1/2024.findings-emnlp.625` | Не переносить численные результаты китайского STARD на русский benchmark |

Перед включением библиографии в статью агент должен сохранить metadata источников в `experiments/npa_mas_component_v1_literature.json`: title, authors, venue, year, DOI/URL, accessed_at, архитектурный тезис и ограничение заимствования. Для LegalMALR и MA-RAG явно указать статус preprint на дату обращения.

## 6. Новая программная архитектура

### 6.1. Предлагаемая структура файлов

После cleanup создать новый модуль, не восстанавливая старые классы под другими именами:

```text
src/retrieval/npa_mas/
  __init__.py
  schemas.py          # TypedDict/Pydantic contracts всех сообщений и traces
  budget.py           # requested/actual search accounting и saturation
  candidates.py       # retrieval, within-channel fusion, final RRF, pool hashing
  workers.py          # BM25Worker и DenseWorker
  planner.py          # PlannerAgent
  reports.py          # WorkerReport generation и observation policy
  selectors.py        # JinaSelector и LLMSelector
  pipeline.py         # единственный fixed-pipeline orchestrator
  variants.py         # декларативный registry B0/B1/S/M1–M4

src/agents/npa_pipeline.py       # опциональный тонкий A2A wrapper, не новая логика

src/scripts/npa_mas_experiment/
  __init__.py
  __main__.py         # freeze/pilot/run/score/analyze/stability/pack/validate
  runner.py
  metrics.py
  statistics.py
  analysis.py
  packaging.py
  schema.py

experiments/npa_mas_component_v1.yaml
experiments/annotations/npa_query_types_v1.md
experiments/annotations/npa_query_types_v1.jsonl
prompts/npa_mas_component_v1/
  single_rewriter.txt
  planner.txt
  bm25_worker.txt
  dense_worker.txt
  worker_report.txt
  llm_selector.txt
```

Можно объединить мелкие модули, но нельзя размножать независимые implementations одного pipeline. Все варианты должны собираться из одних и тех же компонентов через config.

Для core runs использовать **in-process runner**. A2A — только integration/transport smoke test. Это исключает влияние сериализации, разных services и сетевых задержек на механистические сравнения.

### 6.2. Общие contracts

Минимальные типизированные сообщения:

```python
QueryPlan = {
    "aspects": [
        {
            "aspect_id": "a1",
            "legal_issue": "...",
            "bm25_instruction": "...",
            "dense_instruction": "..."
        }
    ],
    "planner_notes": "...",       # короткое пояснение, без скрытого CoT
    "fallback_used": False
}

WorkerQueries = {
    "channel": "bm25" | "dense",
    "queries": ["..."],
    "aspects_addressed": ["a1"],
    "fallback_used": False
}

WorkerReport = {
    "channel": "bm25" | "dense",
    "aspects_addressed": ["a1"],
    "generated_queries": ["..."],
    "observed_chunk_ids": ["..."],
    "coverage": "...",            # краткое структурированное резюме
    "gaps": ["..."],
    "supporting_chunk_ids": ["..."],
    "fallback_used": False
}

Candidate = {
    "chunk_id": "...",
    "pool_rank": 1,
    "selector_text": "...",       # одинаковый текст для Jina и LLM
    "text_sha256": "...",
    "doc_path": "...",
    "title": "...",
    "source_ranks": {},
    "rrf_score": 0.0
}

CandidatePool = {
    "query_group": "...",
    "pool_id": "...",
    "pool_sha256": "...",
    "candidates": [Candidate],
    "budget": {},
    "generated_queries": {},
    "provenance": {}
}
```

Все LLM outputs валидируются строгой JSON schema. Не передавать между компонентами произвольные chain-of-thought. Передаются только краткие пояснения, queries, aspects, IDs, coverage/gaps и служебные статусы.

### 6.3. Роли компонентов

#### Generic single rewriter — S

Один LLM call получает исходный запрос и возвращает два списка:

```json
{
  "lexical_queries": ["...", "...", "..."],
  "semantic_queries": ["...", "...", "..."]
}
```

Он не является MAS. Это обязательный сильный query-expansion control с теми же максимальными числами queries, retrieval budgets и Jina-selector, что у M1.

#### BM25Worker

Содержательная роль, а не wrapper `BM25.search(original_query)`:

- выделяет точные названия НПА, номера, даты, статьи, пункты, реквизиты, редкие термины и устойчивые словосочетания;
- сохраняет точные реквизиты без «творческой» замены;
- формирует до `max_generated_queries_per_channel` lexical queries;
- не имеет доступа к qrels/gold IDs;
- возвращает только lexical queries и краткие aspect labels;
- в M4 дополнительно формирует `WorkerReport` по своим observed hits.

#### DenseWorker

- выделяет смысловую потребность, бытовые формулировки, синонимы и правовые последствия;
- формирует до того же числа semantic queries;
- не обязан сохранять поверхностную лексику, но не должен выдумывать номера статей и факты;
- не имеет доступа к qrels/gold IDs;
- в M4 формирует отдельный `WorkerReport`.

#### PlannerAgent

Planner работает только до retrieval workers:

1. выделяет не более `max_aspects` правовых подвопросов;
2. назначает каждому каналу инструкции в рамках его специализации;
3. возвращает фиксированный `QueryPlan`;
4. не выбирает следующий этап, не запускает повторный retrieval и не меняет pipeline;
5. не видит hits, qrels, метрики или будущие reports.

В M1 planner отсутствует, но workers получают тот же исходный запрос и имеют те же query/output limits. В M2 добавляется один planner call; его дополнительная стоимость не скрывается.

#### Worker reports

Reports формируются **после построения M2 pool** из сохранённых worker observations и не могут изменить pool:

- BM25 report использует только BM25 observations;
- Dense report использует только Dense observations;
- каждый report может ссылаться только на уже observed IDs;
- reports не добавляют новые chunks в selector pool;
- M3 получает пустой reports block;
- M4 получает оба reports;
- остальные selector inputs и decoding parameters совпадают.

#### JinaSelector

- получает original query и `Candidate.selector_text`;
- возвращает scores и deterministic ranking;
- не получает planner notes, generated queries или worker reports;
- используется в B1, B1W, S, M1, M2.

#### LLMSelector

- в M3 получает original query и тот же ordered candidate list, что Jina;
- в M4 дополнительно получает два worker reports;
- должен вернуть полную permutation всех `pool_id::chunk_id`, без новых/duplicate/missing IDs;
- не имеет права выполнять retrieval или изменять кандидатов;
- использует тот же `selector_text`, что Jina;
- результат top-10 получается из первых десяти позиций validated permutation.

Если полный listwise prompt не помещается даже после engineering pilot, до freeze разрешено перейти на заранее документированный sliding-window algorithm по RankGPT. Нельзя в одном run часть групп обрабатывать full-list, а часть — другим algorithm без отдельного variant ID.

### 6.4. Candidate generation и fusion

Для каждого normalized unique-query group:

1. representative query выбирается по минимальному `source_row`, как в существующем runner;
2. original query всегда включается в соответствующий channel retrieval;
3. generated queries очищаются от surrounding whitespace и пустых строк;
4. точные duplicates внутри канала удаляются до retrieval, с сохранением первого occurrence;
5. число generated queries не может превышать frozen limit; лишние не используются и логируются как `truncated_queries`;
6. для каждой query сохраняются requested hits, actual hits, ranks и scores;
7. hits дедуплицируются по `chunk_id` только внутри соответствующего ranking/fusion по frozen rule;
8. rankings нескольких queries внутри одного канала объединяются RRF;
9. итоговые Dense/BM25 channel rankings объединяются weighted RRF с `alpha=0.5`;
10. первые `pool_size` кандидатов образуют ordered candidate pool;
11. selector получает pool без повторного изменения порядка;
12. top-10 — единственный финальный scoring output.

Frozen budget allocation для S/M1/M2 в каждом канале:

```text
original query:                 target 30 raw hits
каждый unique generated query:  target 10 raw hits
максимум generated queries:     3
максимум target на канал:       30 + 3 × 10 = 60
```

Порядок — original query, затем generated queries в порядке validated worker output. Если worker вернул меньше трёх unique queries, невостребованные по 10 hits **не перераспределяются** на original или другие queries. Поэтому при fallback к original-only в expanded system сохраняются те же 30 requested hits, что у B0/B1, а не скрытый wide budget. B1W отдельно запрашивает до 60 hits для original query. Эти правила одинаковы для S/M1/M2 и фиксируются до full runs.

Двухступенчатый RRF нужен, чтобы вариант с большим числом успешных rewrites не получал дополнительные «голоса» непосредственно на финальном channel fusion. Для `B0_RRF` двухступенчатый algorithm вырождается в обычный RRF двух исходных rankings.

**Deterministic tie-breaking обязателен** на каждом fusion:

```text
(-score, first_registration_index, chunk_id)
```

Текущий `rrf.py` использует стабильность dictionary/insertion order, но не задаёт полный явный ключ. Исправить это до новых runs и покрыть тестами на одинаковый результат при разном порядке обхода источников. `first_registration_index` фиксируется по frozen source/query order; `chunk_id` — окончательный tie-break.

### 6.5. Pool identity

`pool_sha256` вычисляется по canonical JSON ordered list, включающему все selector-visible поля:

```text
pool_rank, chunk_id, selector_text, doc_path, title
```

Отдельно сохранить `source_ranks` и `rrf_score`, но не допускать ситуации, когда IDs совпадают, а texts/aliases/order различаются.

Для всех групп должны выполняться assertions:

```text
M3.pool_sha256 == M2.pool_sha256
M4.pool_sha256 == M3.pool_sha256
M3.candidate_ids_in_order == M2.candidate_ids_in_order
M4.candidate_ids_in_order == M3.candidate_ids_in_order
M3.selector_texts == M2.selector_texts == M4.selector_texts
```

Reports generation, caching и повторный selector call не имеют права изменять `pool_sha256`. При mismatch группа не допускается к same-pool scoring; это protocol error, а не повод удалить группу из отчёта.

### 6.6. Budget и saturation

Разделить три величины:

- `target_raw_hits` — заранее назначенный cap для канала/variant;
- `requested_raw_hits` — сумма фактических `top_k`, зарезервированных для search calls;
- `actual_raw_hits` — суммарное число возвращённых raw hits до deduplication.

Инварианты:

```text
0 <= requested_raw_hits <= target_raw_hits
0 <= actual_raw_hits <= requested_raw_hits
unspent_raw_hits = requested_raw_hits - actual_raw_hits
```

`channel_status`:

- `exact_match` — `actual == requested`;
- `saturated` — `0 < actual < requested`;
- `empty` — поиск выполнен корректно, но `actual == 0`;
- `error` — исключение/index/transport failure.

Saturation не является fallback и не требует повторного search. Не требовать равенства actual hits между системами. Пример `target=11, actual=1` валиден, если search выполнен без ошибки.

Текущий `SearchBudget.record()` учитывает actual hits и возвращает неиспользованное через `refund()`. Для нового протокола этого недостаточно: ledger должен отдельно сохранять reservations/requested и actual. Не использовать refund как скрытое перераспределение между каналами или дополнительные queries.

### 6.7. Fallback и output policies

Каждый LLM stage имеет статусы:

```text
success
success_after_transient_retry
success_after_json_repair
fallback
failed
```

Reliability events включают:

```text
timeout
rate_limit
provider_error
malformed_json
schema_violation
empty_output
invalid_candidate_id
duplicate_candidate_id
missing_candidate_id
context_length_exceeded
search_error
```

Engineering defaults до pilot:

- timeout одного LLM call — 180 секунд;
- не более двух retries для transient/network/rate-limit errors;
- не более одного JSON/schema repair call;
- exponential backoff с frozen seed;
- нулевая температура, если provider поддерживает;
- retry не считается успешным независимым экспериментом, все calls учитываются в cost.

Pre-registered practical fallback:

| Сбой | Practical complete-system behavior | Strict sensitivity |
|---|---|---|
| generic rewriter | original query в обоих каналах | failed group, empty final ranking |
| planner | workers запускаются без planner assignments | failed group, empty final ranking |
| BM25 worker | original query только в BM25-канале | failed group, empty final ranking |
| Dense worker | original query только в Dense-канале | failed group, empty final ranking |
| worker report | соответствующий report заменяется явным empty/failed report | failed group, empty final ranking |
| LLM-selector | deterministic RRF pool order | failed group, empty final ranking |
| Jina/index failure | результат не подделывать; group `failed` | failed group |

В practical output fallback всегда помечается `fallback_used=true`, указываются `fallback_stage`, `fallback_reason`, все attempts и исходный error class. В strict analysis неуспешная группа остаётся в denominator с нулевыми ranking metrics; исключать её запрещено.

`success_after_json_repair` не считается fallback, но включается отдельно в reliability rate. Частично валидная permutation не дополняется молча: сначала repair, затем declared fallback/failure.

### 6.8. Token usage и стоимость

Для каждого LLM call сохранять:

```json
{
  "stage": "planner",
  "model": "...",
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "total_tokens": 0,
  "token_usage_status": "reported",
  "cached": false,
  "latency_ms": 0,
  "attempt": 1
}
```

Правила:

- если provider не вернул usage — значения `null`, `token_usage_status="missing"`, а не `0`;
- для non-LLM BM25/Dense/RRF usage — `not_applicable`;
- для Jina отдельно учитывать local compute/latency/model/device, а не приписывать ей LLM API tokens;
- monetary cost вычисляется только при наличии явно записанного pricing profile и даты цены; иначе `null`;
- разделять actual experimental spend и reconstructed full-system cost при reuse общего M2 pool;
- stage latency, measured end-to-end latency и reconstructed latency не смешивать в одной колонке.

## 7. Матрица систем

### 7.1. Core matrix — выполнить полностью

| ID | Query generation / agents | Planner | Candidate pool | Selector | Reports | Назначение |
|---|---|---|---|---|---|---|
| `B0_BM25` | original query | нет | BM25 original | исходный BM25 ranking | нет | single-channel baseline |
| `B0_DENSE` | original query | нет | Dense original | исходный Dense ranking | нет | single-channel baseline |
| `B0_RRF` | original query | нет | BM25 + Dense original | RRF | нет | hybrid baseline без reranker |
| `B1` | original query | нет | BM25 + Dense original, baseline budget | Jina | нет | сильный non-MAS baseline |
| `B1W` | original query | нет | BM25 + Dense original, expanded requested budget | Jina | нет | wide-budget control с saturation logging |
| `S` | один generic LLM call → lexical + semantic queries | нет | свой expanded pool | Jina | нет | сильный single-LLM expansion control |
| `M1` | BM25Worker + DenseWorker | нет | свой expanded pool | Jina | нет | эффект role-specialized workers |
| `M2` | те же workers | есть | свой expanded pool | Jina | не подаются selector | эффект planner; общий frozen pool для M3/M4 |
| `M3` | ровно кандидаты M2 | уже отражён в pool | **тот же M2 pool** | LLM | нет | same-pool замена Jina на LLM |
| `M4` | ровно кандидаты M2 | уже отражён в pool | **тот же M2 pool** | LLM | BM25 + Dense reports | same-pool эффект worker reports |

Все десять configurations обязательны. Не исключать B1W, S или любой M-вариант из-за промежуточных metrics.

### 7.2. Два контура интерпретации

**Practical end-to-end контур:**

```text
B0/B1/B1W, S, M1, M2, M3, M4 как полные системы
```

Здесь candidate pools могут различаться. Сравнение отвечает на вопрос практической ценности, но не изолирует ranking mechanism. Для M3/M4 reconstructed full-system path включает M2 candidate generation.

**Mechanistic same-pool контур:**

```text
M2 → frozen candidate pools
   ├─ Jina, no reports             = M2
   ├─ LLM, no reports              = M3
   └─ LLM, BM25+Dense reports      = M4
```

Только этот контур поддерживает выводы:

- `M3-M2`: эффект замены relevance selector при неизменных кандидатах;
- `M4-M3`: эффект дополнительной worker coordination information при неизменных кандидатах.

### 7.3. Что не входит в core

Второй раунд reassessment, peer-to-peer exchange, dynamic stopping, обучение planner/rewriters, GRPO, HyDE variant и дополнительные Answerer/Summarizer roles не добавлять. Если позднее понадобится communication round-2, оформить новый protocol ID после завершения текущего core, не изменяя текущие runs.

## 8. НПА-типология запросов

### 8.1. Разметка до просмотра full quality results

Создать новые derived annotations, не редактируя исходный benchmark:

```text
experiments/annotations/npa_query_types_v1.md
experiments/annotations/npa_query_types_v1.jsonl
```

Одна строка на `query_group`, всего ожидается 556 groups. Основные labels — multi-label boolean:

| Label | Критерий | Примеры признаков |
|---|---|---|
| `exact_reference` | есть точный указатель на НПА/структуру | название, номер, дата, статья, пункт |
| `lay_paraphrase` | бытовое описание без точной юридической терминологии | «что делать, если…», разговорные синонимы |
| `multi_issue` | в запросе несколько различимых правовых вопросов | несколько условий/требований/субъектов |
| `applicability_conditions` | важны условия применимости нормы | субъект, срок, исключение, территория, порог |
| `cross_reference_multi_document` | требуется связь норм/документов | отсылки, несколько НПА, соотношение положений |
| `other_unclear` | нельзя уверенно отнести или формулировка неясна | неполный/неоднозначный запрос |

Labels не mutually exclusive. `other_unclear` не выставлять автоматически всем запросам без первых пяти labels: это отдельная оценка неопределённости.

Каждая строка:

```json
{
  "query_group": "...",
  "representative_question_id": "...",
  "labels": {
    "exact_reference": false,
    "lay_paraphrase": true,
    "multi_issue": true,
    "applicability_conditions": true,
    "cross_reference_multi_document": false,
    "other_unclear": false
  },
  "annotation_confidence": "high",
  "annotator": "human_a",
  "notes": ""
}
```

В annotations не записывать retrieved IDs, system outputs или будущие metrics.

### 8.2. Контроль качества разметки

- сначала составить codebook с положительными/отрицательными примерами;
- LLM может подготовить draft labels, но final labels должны пройти human review;
- не менее 20% groups или не менее 100 groups, что больше при доступности, разметить независимо повторно;
- при двух независимых annotators сообщить per-label Cohen's kappa и разрешить disagreements;
- при повторной разметке тем же человеком называть это audit agreement, а не межэкспертной надёжностью;
- freeze annotations и их SHA-256 до full scoring;
- не перестраивать labels после просмотра subgroup effects.

Известные неточности benchmark отдельно описать в `limitations.md`. Не исправлять qrels или questions под желаемый результат.

## 9. Конфигурация и engineering pilot

### 9.1. Начальный config template

Скопировать в `experiments/npa_mas_component_v1.yaml`, затем заполнить только engineering parameters до freeze:

```yaml
protocol_version: npa_mas_component_v1
seed: 20261127

benchmark:
  root: var/benchmark/final
  questions: questions.jsonl
  qrels: qrels.jsonl
  evidence: evidence.jsonl
  expected_rows: 630
  expected_groups: 556
  expected_answerable_rows: 617
  expected_answerable_groups: 544
  expected_rows_without_qrels: 13
  duplicate_policy: normalized_group_representative_min_source_row
  group_qrels_policy: union_max_relevance

annotations:
  codebook: experiments/annotations/npa_query_types_v1.md
  labels: experiments/annotations/npa_query_types_v1.jsonl

retrieval:
  channels: [bm25, dense]
  score_threshold_source: frozen_environment
  baseline_target_raw_hits_per_channel: 30
  expanded_target_raw_hits_per_channel: 60
  expanded_original_query_target_raw_hits: 30
  expanded_each_generated_query_target_raw_hits: 10
  redistribute_unused_query_budget: false
  wide_target_raw_hits_per_channel: 60
  max_generated_queries_per_channel: 3
  include_original_query: true
  max_queries_per_channel_including_original: 4
  max_generated_query_chars: 300
  pool_size: 30
  output_chunks: 10
  rrf_k: 60
  final_alpha_dense: 0.5
  tie_break: [score_desc, first_registration_index, chunk_id]
  allow_cross_channel_compensation: false
  allow_saturation_backfill: false

selector:
  text_policy: identical_prefix_for_jina_and_llm
  text_max_chars: 2000
  llm_output: full_permutation
  jina_model_source: RERANKER_MODEL
  jina_device_source: RERANKER_DEVICE

planner:
  max_aspects: 4

reports:
  max_observed_chunks_per_worker: 10
  text_max_chars: 2000
  max_gap_items: 4

llm:
  model_env: NPA_LLM_MODEL
  temperature: 0
  timeout_s: 180
  max_transient_retries: 2
  json_repair_attempts: 1
  max_output_tokens:
    single_rewriter: 1200
    planner: 800
    bm25_worker: 800
    dense_worker: 800
    worker_report: 1200
    llm_selector: 2000
  concurrency: 2

output_policy:
  primary: complete_with_declared_fallback
  sensitivity: strict_failure_as_empty

run:
  variants: [B0_BM25, B0_DENSE, B0_RRF, B1, B1W, S, M1, M2, M3, M4]
  adaptive_gates: false
  reuse_old_evaluation: false

statistics:
  primary_metric: recall@10
  primary_unit: answerable_unique_query_group
  bootstrap_iterations: 10000
  randomization_iterations: 10000
  confidence_level: 0.95
  multiplicity: holm
  confirmatory_contrasts: [M1-S, M2-M1, M3-M2, M4-M3]

stability:
  groups: 50
  additional_repeats: 2
  selection: deterministic_hash_before_quality_results

pilot:
  groups: 24
  scoring_enabled: false
```

`60/30/2000/180` — стартовые engineering values, а не результаты оптимизации. Они специально увеличивают search/request limits относительно прежнего стенда, но остаются конечными и воспроизводимыми.

### 9.2. Единая LLM configuration

Для S, planner, BM25/Dense workers, reports и LLM-selector использовать один и тот же frozen `NPA_LLM_MODEL` и decoding profile в core. Иначе различие компонентов смешивается с различием моделей.

Model name, provider, revision (если доступен), decoding parameters и context limit записать в manifest. API keys, `.env`, Authorization headers и полные URLs с credentials не записывать.

### 9.3. Цель и границы pilot

Pilot выполняется после unit/integration tests и annotations freeze, но до full core scoring:

- выбрать 24 groups детерминированным hash от `query_group + seed`;
- не использовать qrels и не вычислять ranking quality;
- проверить context lengths, JSON validity, latency, timeout/rate-limit rates, search saturation, requested/actual hits, disk size и примерную стоимость;
- при необходимости изменить только engineering limits: timeouts, concurrency, retries, output tokens, candidate text chars, pool size и search caps;
- prompts можно исправлять только для schema compliance/роли, не для улучшения ответов на конкретных benchmark questions;
- все pilot runs пометить `engineering_only=true` и не включать в quality tables.

После pilot зафиксировать config/prompts/models/limits. Дальнейшие изменения после просмотра full quality запрещены. Если обнаружен общий implementation bug, остановить matrix, исправить, создать новый run/protocol revision и повторить все зависимые runs с нуля.

## 10. Порядок реализации и полных прогонов

### 10.1. Implementation sequence

Строго соблюдать последовательность:

1. завершить cleanup и отдельный cleanup commit;
2. добавить schemas, budget ledger, deterministic RRF и candidate builder;
3. перенести B0/B1 на новый candidate builder без изменения индексов;
4. реализовать generic single rewriter `S`;
5. реализовать BM25Worker и DenseWorker, собрать `M1`;
6. реализовать PlannerAgent, собрать `M2`;
7. реализовать JinaSelector через общий pool interface;
8. реализовать LLMSelector `M3` без reports;
9. реализовать WorkerReport и `M4` на том же M2 pool;
10. реализовать scoring, paired statistics, cost/reliability/subgroup analysis;
11. выполнить tests и no-scoring engineering pilot;
12. заморозить protocol;
13. выполнить всю core matrix и только затем открыть сводные quality results;
14. выполнить stability и упаковать результаты.

Это соответствует исследовательской лестнице: сначала два retrieval workers с reranker, затем planner, затем замена reranker LLM-агентом, затем reports. Не начинать с M4 и не объявлять его единственным baseline.

### 10.2. Pre-run input audit

До любого платного full run:

- проверить protected hashes;
- проверить, что chunk store и индексы доступны read-only;
- вычислить их hashes/манифесты без модификации;
- проверить количество questions/qrels/groups;
- проверить 630 rows, 556 normalized groups, 617 answerable rows, 544 answerable groups и 13 rows без qrels;
- если фактические counts отличаются, остановиться: нельзя молча адаптировать protocol под другой dataset;
- проверить, что все ranked IDs принадлежат chunk store;
- проверить 556 annotation rows и отсутствие unknown query groups;
- записать model/prompt/config/source hashes.

Повторяющиеся вопросы выполняются один раз на normalized group, затем ranking проецируется на исходные rows. Для qrels внутри group использовать union с `max(relevance)` по каждому `chunk_id`.

### 10.3. Core run без adaptive gates

Одна логическая команда должна построить все зависимости:

```bash
export PYTHONPATH=src

python -m scripts.npa_mas_experiment audit-inputs \
  --config experiments/npa_mas_component_v1.yaml

python -m scripts.npa_mas_experiment pilot \
  --config experiments/npa_mas_component_v1.yaml \
  --run-id pilot_001 \
  --no-scoring

# После инженерной проверки и окончательного freeze:
python -m scripts.npa_mas_experiment freeze \
  --config experiments/npa_mas_component_v1.yaml \
  --run-id core_001

python -m scripts.npa_mas_experiment run-core \
  --config experiments/npa_mas_component_v1.yaml \
  --run-id core_001

python -m scripts.npa_mas_experiment validate \
  --config experiments/npa_mas_component_v1.yaml \
  --run-id core_001 \
  --require-complete

# Не выполнять до успешного validate всей core matrix:
python -m scripts.npa_mas_experiment score \
  --config experiments/npa_mas_component_v1.yaml \
  --run-id core_001

python -m scripts.npa_mas_experiment analyze \
  --config experiments/npa_mas_component_v1.yaml \
  --run-id core_001

python -m scripts.npa_mas_experiment stability \
  --config experiments/npa_mas_component_v1.yaml \
  --run-id stability_001

python -m scripts.npa_mas_experiment pack \
  --config experiments/npa_mas_component_v1.yaml \
  --run-id core_001
```

`run-core` должен внутри выполнить:

1. B0/B1/B1W для всех groups;
2. S, M1 и M2 candidate generation для всех groups;
3. Jina outputs S/M1/M2;
4. сохранить M2 pools и worker observations;
5. M3 LLM-selector на frozen M2 pools;
6. сформировать M4 reports из M2 observations;
7. M4 LLM-selector на тех же pools;
8. проверить pool hashes, coverage и schemas;
9. сформировать 630-row `run.jsonl` для каждого variant.

Нельзя после этапа 2/3/5 открывать aggregate quality и затем менять prompts, budgets, variants или output policy. Технические progress logs разрешены, но они не должны содержать qrels-based quality.

### 10.4. Resume и invalidation

Resume допускается только при совпадении:

```text
protocol_version
config_sha256
prompt_bundle_sha256
benchmark protected hashes
chunk/index manifest hashes
model/decoding manifest
source_code_sha256
variant + query_group
```

Иначе создать новый run ID. Не смешивать group files из разных configs в одном run.

При изменении:

- candidate builder/workers/planner — invalidate S/M1/M2 и все зависимые M3/M4 по затронутым компонентам; для core consistency предпочтительно полный новый run;
- M2 pool — обязательно invalidate M2/M3/M4;
- reports только — invalidate M4, но не переписывать M2/M3;
- selector только — invalidate соответствующий selector variant;
- scoring code — пересчитать scores всех variants из неизменных raw rankings;
- prompts/model/decoding после full quality — начать новый protocol revision, а не заменять неудобный результат.

## 11. Метрики и механистический анализ

### 11.1. Основная оценка качества

Primary:

```text
Recall@10, macro по 544 answerable unique-query groups
```

Secondary:

```text
Recall@5
Hit Rate@5 / Hit Rate@10
nDCG@5 / nDCG@10, gain = 2^relevance - 1
MRR@5 / MRR@10
```

Дополнительно публиковать row-macro по 617 answerable rows только для сопоставимости с прежними таблицами. Primary unit остаётся unique group, чтобы duplicates не увеличивали вес вопроса.

13 rows без qrels не исключаются из runs: для них отчитываться по completeness, latency, calls, tokens и fallback. Они не входят в denominators quality metrics.

Empty/short ranking не дополнять посторонними IDs. Метрики считаются по фактически возвращённым уникальным валидным IDs, максимум 10.

### 11.2. Candidate generation analysis

До selector для каждого pool считать:

- `pool_recall@30` — доля union qrels, найденных в pool;
- число relevant IDs и unique IDs в pool;
- Recall@30 отдельно для BM25/Dense channel rankings;
- пересечение и уникальный вклад каналов;
- новые relevant IDs относительно original-query B0/B1 pool;
- число generated/unique/truncated queries;
- raw hits requested/actual, duplicates removed, saturation;
- pool size до/после cap;
- provenance каждого relevant ID: channel, query, first raw rank, fusion rank.

Эти результаты отвечают, **откуда появился возможный прирост**: lexical exact matching, semantic expansion, planner assignments или просто больший search budget.

### 11.3. Same-pool ranking analysis

Для M2/M3/M4:

- Recall@5/10, Hit Rate, nDCG и MRR;
- per-group paired deltas;
- перемещение каждого relevant ID: `old_rank → new_rank`;
- число relevant IDs, вошедших в top-10 и выпавших из top-10;
- top-10 overlap и rank correlation;
- pool recall как контрольная константа — он обязан совпадать;
- reports references: какие `supporting_chunk_ids` совпадают с поднявшимися/опустившимися IDs.

Trace может объяснить возможный механизм, но не доказывает причинность отдельного движения документа. Не писать «planner нашёл документ», если документ уже был в original pool; проверять provenance.

### 11.4. Типология НПА

Для каждого pre-registered label сообщить:

- число answerable groups с label и без label;
- primary/secondary metrics каждой системы;
- paired effect C1–C4 внутри label;
- interaction: различие эффекта `label=true` против `label=false`;
- 95% CI и число groups;
- пересечение labels и multi-label nature.

При `n < 30` давать только descriptive estimates без сильных выводов. Не создавать новые подгруппы по длине запроса, fallback или «удачным» случаям после просмотра результатов. Такие дополнительные срезы допустимы только как явно exploratory.

### 11.5. Стоимость и reliability

Для каждого variant:

- LLM calls total и per group, включая retries/repair;
- prompt/completion/total tokens: mean, median, p95, sum;
- доля groups с reported/missing/partial usage;
- requested/actual raw hits и unique candidates;
- Jina calls и selector latency;
- per-stage latency и measured group latency;
- reconstructed full-system latency/cost для M3/M4;
- timeout, malformed JSON, repair, fallback и failure rates по stages;
- empty/short output rates;
- monetary cost только при documented pricing profile.

Quality-per-cost графики допустимы, но missing usage нельзя заменять нулём или silently удалять. При высокой доле missing usage стоимостный вывод ограничить calls/latency/search counts.

## 12. Статистический протокол

### 12.1. Confirmatory analysis

Primary output policy — `complete_with_declared_fallback`, поскольку она сохраняет работоспособный system output для всех groups. `strict_failure_as_empty` — заранее обязательная sensitivity analysis, а не способ выбрать более красивый результат позднее.

Для четырёх confirmatory contrasts на Recall@10:

1. вычислить per-group paired difference;
2. mean difference и 95% CI через paired bootstrap, 10 000 iterations;
3. two-sided sign-flip randomization test, 10 000 iterations;
4. seed `20261127`;
5. Holm correction внутри одного семейства из четырёх contrasts;
6. wins/losses/ties и median difference как descriptive statistics.

Общий seed управляет генерацией индексов, но bootstrap/randomization streams должны быть разделены и детерминированы. Одинаковые resampling indices использовать для paired систем внутри contrast.

Таблица обязана содержать:

```text
contrast
n_groups
mean_left
mean_right
mean_delta
ci_low
ci_high
p_randomization
p_holm
wins
losses
ties
fallback_groups_left/right
```

### 12.2. Secondary и exploratory analysis

- B1/B1W/B0 и качество по другим метрикам — secondary;
- `M4-B1`, `M4-B1W`, `M4-S` — practical system comparisons, не чистый эффект одного компонента;
- subgroup и quality-cost analyses — exploratory, если отдельно не зафиксировано иное;
- для secondary families можно применять Holm внутри заранее объявленного семейства, но нельзя переносить confirmatory статус по итогам p-values;
- matched groups внутри duplicate query group не считаются независимыми повторениями;
- нулевой/отрицательный эффект не удалять и не заменять другой метрикой как новой primary.

### 12.3. B1W и saturation

B1W получает тот же expanded per-channel requested cap, но только original query. Он не должен получить synthetic paraphrases или backfill.

Если original query насыщается выше score threshold и возвращает меньше requested hits:

- принять фактический результат;
- записать `saturated` и `unspent_raw_hits`;
- не требовать равенства actual raw hits с S/M1/M2;
- не переносить остаток в другой канал;
- интерпретировать B1W как requested-budget control, а не доказательство равного фактически выполненного compute.

S является главным более сильным control для query expansion, поскольку имеет те же generation/query caps и expanded candidate budget, что M1.

## 13. Stability protocol

До просмотра core quality выбрать 50 answerable groups детерминированным hash от `query_group + seed`. Выбор не основывать на wins/losses.

Выполнить два additional repeats с финальным config:

1. **candidate stability:** S, M1, M2 — повторить generation, retrieval и Jina;
2. **selector stability:** M3 — повторить selector на исходном frozen M2 pool;
3. **reports+selector stability:** M4 — повторить reports и selector на исходном frozen M2 pool.

Сохранять отдельно:

- exact query-set match;
- generated query similarity;
- pool ID Jaccard и ordered overlap;
- `pool_sha256` exact match rate;
- top-10 Jaccard;
- Kendall tau на общих IDs;
- per-group metric range;
- fallback/repair variability;
- tokens/calls/latency variability.

Repeats одной и той же group не увеличивают независимый `n` в primary statistics. Для same-pool stability не смешивать pool одного repeat с reports/selector другого repeat.

Если заявляется cold end-to-end latency M3/M4, дополнительно выполнить cache-disabled performance run на тех же 50 groups. Иначе публиковать только measured stage latency и явно названный reconstructed full-system latency.

## 14. Тесты

### 14.1. Unit tests без внешних API

Обязательные тесты:

1. **normalization/grouping:** 630/556/617/544/13 audit на read-only fixture или фактическом benchmark;
2. **qrels union:** max relevance внутри group, без дублирования веса;
3. **metrics:** hand-calculated Recall/Hit Rate/nDCG/MRR @5/@10;
4. **RRF:** score, limit, deterministic tie `(score desc, first registration, chunk_id)`;
5. **source-order invariance:** эквивалентный input с другим dictionary iteration не меняет ranking после frozen registration;
6. **budget:** `actual ≤ requested ≤ target`;
7. **saturation:** `target=11, requested=11, actual=1` валиден и имеет `saturated`;
8. **no compensation:** saturation не вызывает новые queries, duplicates или cross-channel transfer;
9. **deduplication:** duplicates не увеличивают raw unique pool и не дают дополнительный финальный RRF vote;
10. **pool hash:** изменение order/text/title/doc_path меняет hash; metadata вне selector-visible contract не меняет;
11. **same-pool guard:** M2/M3/M4 mismatch вызывает protocol error;
12. **selector schema:** extra/missing/duplicate/out-of-pool ID отклоняется;
13. **repair/fallback:** malformed JSON → один repair → declared fallback, все события в trace;
14. **partial permutation:** не принимается молча как полный результат;
15. **reports:** не могут добавить новый candidate ID или изменить pool;
16. **planner:** не может выполнить search или изменить порядок этапов;
17. **token usage:** missing остаётся `null`, non-LLM — `not_applicable`;
18. **resume:** config/prompt/source hash mismatch запрещает reuse;
19. **projection:** один group ranking корректно проецируется на исходные duplicate rows;
20. **secrets:** artifact redaction scanner не записывает значения чувствительных env variables.

### 14.2. Integration tests

С fake LLM и fake engine прогнать все variants на небольших fixtures:

- B0/B1 не выполняют LLM calls;
- S выполняет один generic rewriter call;
- M1 выполняет два worker calls и не вызывает planner;
- M2 добавляет один planner call;
- M3 не запрашивает reports;
- M4 запрашивает два reports, но pool остаётся неизменным;
- Jina и LLM получают идентичный `selector_text`;
- top-10 содержит не более десяти уникальных валидных IDs;
- все fallback events доступны в serialized group artifact.

Затем сделать минимальный real smoke на 2–3 groups после настройки окружения. Smoke не использовать для выбора prompts по qrels.

## 15. Структура artifacts

Новый evaluation root создаётся только после cleanup:

```text
var/benchmark/final/evaluation/npa_mas_component_v1/
  pilot_001/
    manifest.json
    reliability.json
    groups/
  core_001/
    freeze/
      protocol.yaml
      config_sha256.txt
      prompt_bundle/
      prompt_bundle_sha256.txt
      source_code_hashes.json
      benchmark_hashes_before.json
      chunk_index_manifest.json
      model_manifest.json
      annotations_codebook.md
      annotations.jsonl
      annotations_sha256.txt
    variants/
      B0_BM25/
      B0_DENSE/
      B0_RRF/
      B1/
      B1W/
      S/
      M1/
      M2/
      M3/
      M4/
    pools/
      M2/
    scores/
      <variant>/
    analysis/
      core_metrics.csv
      candidate_generation.csv
      confirmatory_contrasts.csv
      same_pool_ranking.csv
      subgroup_effects.csv
      cost.csv
      reliability.csv
      trace_examples.md
      limitations.md
      report.md
    run_manifest.json
    validation.json
  stability_001/
  results_pack.zip
```

Каждый variant должен содержать:

```text
groups/group_0000.json ... group_0555.json
run.jsonl                 # 630 исходных rows
detailed.jsonl            # 556 unique groups
manifest.json
errors.jsonl              # даже если пустой
```

Каждый group artifact включает:

- representative query и все source question IDs;
- generated queries/plan/reports, если применимо;
- raw retrieval counts и channel provenance;
- candidate pool reference/hash;
- final ranked IDs;
- per-stage timings/calls/token usage;
- attempts, errors, repairs, fallback;
- config/prompt/model/source hashes;
- cache status.

Использовать canonical JSON и atomic write (`tmp` + rename). Не перезаписывать успешный group artifact другим config.

### 15.1. Results pack validation

`pack` и независимый `validate` должны проверить:

- 10/10 variants;
- 556 group files и 630 run rows на variant;
- 544 answerable groups в quality denominators;
- отсутствие unknown/duplicate final IDs;
- M2=M3=M4 pool hashes;
- budget invariants;
- полноту score/contrast/cost/reliability tables;
- hashes prompts/config/annotations/source;
- protected hashes after = before;
- отсутствие secrets и непредусмотренных env dumps;
- отсутствие старых snapshots/runs внутри нового pack.

Token usage aggregates должны включать coverage. Отчёт не должен содержать значения `0` там, где usage фактически missing.

## 16. Итоговый отчёт без data-dependent переориентации

`analysis/report.md` должен содержать как минимум:

1. **T1:** качество всех десяти systems, unique-query macro;
2. **T2:** candidate pool quality и уникальный вклад BM25/Dense;
3. **T3:** C1–C4 с CI, randomization p и Holm;
4. **T4:** same-pool M2/M3/M4 и перемещения relevant IDs;
5. **T5:** эффекты по шести НПА-labels с `n` и interactions;
6. **T6:** calls/tokens/latency/search cost/missing usage;
7. **T7:** retries/repairs/fallback/saturation/failures;
8. **T8:** stability;
9. 3–5 заранее отобранных trace examples разных исходов: planner нашёл новый relevant ID, semantic channel дополнил BM25, LLM улучшил/ухудшил ranking, reports не дали эффекта;
10. limitations: development benchmark, известные неточности qrels/questions, отсутствие независимого test, возможная contamination LLM, preprint status части литературы, multi-label subgroup overlap и missing usage.

Trace examples можно выбирать по результатам только для иллюстрации уже посчитанных эффектов; нельзя на их основе создавать новые confirmatory выводы или удалять противоречащие случаи.

Правила интерпретации:

| Наблюдение | Что можно сказать | Чего нельзя сказать |
|---|---|---|
| `M1 > S` | специализированные workers дали эффект в данном expanded-budget pipeline | любые MAS всегда лучше single LLM |
| `M2 > M1` | planner package улучшил candidate+ranking pipeline | planner напрямую улучшил relevance scoring |
| `M3 > M2` same pool | LLM-selector улучшил ranking данных кандидатов | LLM улучшил candidate recall |
| `M4 > M3` same pool | reports добавили полезную coordination information | доказана общая польза peer communication |
| эффект только в expanded pool, но не same pool | механизм связан прежде всего с candidate generation | прирост получен за счёт reranking |
| эффект только при practical fallback | полная система с declared recovery оказалась лучше | чистый LLM component всегда успешнее |
| нулевой/отрицательный эффект | компонент не оправдал стоимость/сложность в данном домене и бюджете | результат нужно скрыть или заменить протокол |

Все основные ветки выполняются независимо от промежуточного знака эффекта. Переориентация статьи допускается позднее при научном обсуждении, но код/протокол не подстраиваются под желаемый вывод.

## 17. Definition of Done

### Cleanup

- [ ] Шесть legacy retrieval modules и шесть wrappers удалены.
- [ ] `src/scripts/coordinated_experiment`, старый CLI и `coordinated_mechanism_v1.yaml` удалены.
- [ ] Весь `var/benchmark/final/evaluation` удалён перед новым проектом.
- [ ] `cli.py`, `start_agents.sh`, `run_a2a_benchmark.py`, README и e2e references мигрированы.
- [ ] Protected benchmark/data/index hashes не изменились.
- [ ] Cleanup commit существует отдельно от новой архитектуры.

### Реализация

- [ ] Есть один config-driven fixed pipeline и registry всех десяти variants.
- [ ] BM25Worker и DenseWorker формируют разные channel-specific queries, а не просто вызывают search с original query.
- [ ] S, M1, M2, M3, M4 реализованы без imports старого Coordinated.
- [ ] Planner не управляет динамическим порядком pipeline.
- [ ] Jina/LLM selectors используют общий candidate contract и identical text.
- [ ] RRF tie-breaking, budget/saturation и fallback покрыты tests.
- [ ] Все prompts/config/models/annotations заморожены до full quality.

### Прогоны

- [ ] Engineering pilot выполнен без qrels-based scoring.
- [ ] Все десять core variants выполнены для 556 groups и спроецированы на 630 rows.
- [ ] 544 answerable groups включены в primary quality analysis; 13 rows без qrels учтены в cost/completeness.
- [ ] Ни одна ветка не пропущена из-за промежуточных metrics.
- [ ] M2/M3/M4 same-pool assertions пройдены для каждой группы.
- [ ] Saturation не вызывал backfill/compensation; requested/actual hits сохранены.
- [ ] Все retries/repairs/fallback/failures видны в artifacts.
- [ ] Stability repeats и 556-group annotations завершены.

### Анализ и упаковка

- [ ] Primary Recall@10 unique-query macro посчитан по frozen qrels policy.
- [ ] C1–C4 содержат paired bootstrap CI, sign-flip p и Holm.
- [ ] Secondary/practical/subgroup результаты не выдаются за чистые механистические доказательства.
- [ ] Есть candidate, same-pool, НПА-типология, cost, reliability и stability tables.
- [ ] Missing token usage не заменён нулём.
- [ ] Protected hashes после всех runs совпадают с исходными.
- [ ] Results pack не содержит secrets и старых evaluation snapshots.
- [ ] Независимый `validate` завершился успешно.
- [ ] Отчёт содержит отрицательные/нулевые результаты и limitations без автоматической переориентации протокола.

Только после выполнения этих пунктов результаты можно передавать на научную интерпретацию и обсуждение публикационного вклада.
