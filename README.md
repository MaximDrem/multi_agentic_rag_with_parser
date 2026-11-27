# Document Parser + Multi-Agent RAG (A2A v1.0)

Система парсинга нормативно-правовых актов (DOCX/PDF/XLSX → Markdown) с последующей индексацией и retrieval-augmented generation на протоколе Agent-to-Agent (A2A) v1.0.

Вся конфигурация — в `.env`. Ни один URL, порт или путь не захардкожен в коде.

## Архитектура

Оркестратор — центральный агент на LangGraph ReAct. LLM **сам решает** всю цепочку: каких агентов вызвать, какие данные им передать (`data_from`), отправить результат через `forward_agent_artifacts` (артефакты агента) или `send_text` (собственный текст), продолжать поиск или завершить.

```
НПА для НИР/ (документы)
    ↓ Парсинг (опция 1)
var/markdown_output/ (сырые .md)
    ↓ Очистка
var/markdown_output_clean/ (очищенные .md)
    ↓ Индексация (опция 2)
var/data/ (FAISS + BM25 + chunks.pkl)
    ↓ RAG (опции 3-4)
Пользователь
    ↕ A2A (JSON-RPC + streaming TaskArtifactUpdateEvent)
Orchestrator ←── LangGraph ReAct
    │               forward_agent_artifacts(agent_name) — переслать артефакты агента
    │               send_text(message) — отправить свой текст
    │               data_from=[...] — явная передача данных между агентами
    ↕ A2A
Selected Retriever (base | baseline_reranker)
    ↕ A2A
Summarizer
```

Агенты регистрируют `supportedInterfaces[].protocolVersion: "1.0"`. Клиент и сервер передают заголовок `A2A-Version: 1.0`. Части Part используют unified oneof (`text | raw | url | data`), строятся через SDK `a2a.helpers.proto_helpers`. Статусы задач: `TASK_STATE_SUBMITTED → TASK_STATE_WORKING → TASK_STATE_COMPLETED/FAILED/REJECTED/CANCELED`.

Оркестратор стримит артефакты через `TaskArtifactUpdateEvent` во время обработки — клиент видит ответы по мере готовности, не дожидаясь конца цепочки. Передача данных между агентами — через `data_from` (generic, не привязан к конкретному типу данных).

## Структура проекта

```
.
├── cli.py                          # Интерактивный CLI: парсинг, индексация, агенты, запросы
├── README.md
├── requirements.txt
├── .env                            # ВСЕ параметры (RAG + Parser + OCR), без defaults в коде
├── src/
│   ├── main.py                     # entrypoint оркестратора (python -m main)
│   ├── parser/                     # Парсер DOCX/PDF/XLSX → Markdown
│   │   ├── config.py               # Конфиг OCR (только из .env)
│   │   ├── parser.py               # Оркестратор парсинга
│   │   ├── ocr_client.py           # OCR через vLLM
│   │   ├── docx_handler.py         # DOCX → Markdown (python-docx)
│   │   ├── pdf_handler.py          # PDF → Markdown (pymupdf + OCR)
│   │   ├── xlsx_handler.py         # XLSX → Markdown (openpyxl)
│   │   ├── omml_to_latex.py        # OMML-формулы → LaTeX
│   │   ├── markdown_cleaner.py     # Очистка inline
│   │   └── clean_markdown.py       # Пост-обработка: var/markdown_output → var/markdown_output_clean
│   ├── common/                     # RAG: общая инфраструктура
│   │   ├── config.py               # Читает .env (os.environ, без defaults)
│   │   ├── a2a_server.py           # A2AServer + A2AExecutor (v1.0)
│   │   ├── a2a_client.py           # A2AClientManager (v1.0)
│   │   ├── models.py               # Chunk
│   │   ├── chunk_store.py          # ChunkDatabase
│   │   ├── logging_utils.py        # Онлайн-логи → var/logs/
│   │   └── benchmark_logger.py     # Бенчмарк-логи → var/logs_benchmark/
│   ├── indexing/                   # RAG: индексация
│   │   ├── chunker.py              # Markdown → chunks
│   │   ├── faiss_indexer.py        # FAISS IndexFlatIP + Jina v5
│   │   └── bm25_indexer.py         # BM25Okapi
│   ├── agents/                     # RAG: тонкие A2A-entrypoints
│   │   ├── base_retriever.py       # Dense + BM25 + RRF
│   │   ├── baseline_reranker.py    # RRF + Jina reranking
│   │   └── summarizer.py           # LLM-генерация ответа
│   ├── orchestrator/               # RAG: оркестрация
│   │   ├── orchestrator.py         # ReAct-оркестратор, динамически строит цепочку
│   │   └── discovery.py            # AgentDiscoverer + AgentRegistry
│   └── scripts/                    # Утилиты
│       ├── build_indices.py
│       ├── start_agents.sh
│       └── test_e2e.py
├── НПА для НИР/                     # Вход: исходные документы (tracked)
└── var/                             # Всё генерируемое (gitignored)
    ├── markdown_output/             # Сырые .md после парсинга
    ├── markdown_output_clean/       # Очищенные .md для RAG
    ├── temp_images/                 # Временные изображения OCR
    ├── data/                        # Индексы: chunks.pkl, faiss_index/, bm25_index.pkl
    ├── logs/                        # Онлайн-логи процессов
    └── logs_benchmark/              # Бенчмарк-логи (experiment_*/)
```

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Конфигурация (.env)

Все параметры задаются исключительно в `.env`. `common/config.py` не содержит значений по умолчанию — пропущенная переменная вызывает `KeyError` с именем ключа.

### Оркестратор

| Переменная | Назначение |
|-----------|-----------|
| `ORCHESTRATOR_PORT` | Порт сервера |
| `ORCHESTRATOR_URL` | URL сервера (например `http://localhost:8101`) |
| `ORCHESTRATOR_LLM_BASE_URL` | vLLM/OpenAI-compatible endpoint |
| `ORCHESTRATOR_LLM_MODEL` | Название модели |
| `ORCHESTRATOR_LLM_TEMPERATURE` | Температура генерации |
| `ORCHESTRATOR_LLM_MAX_TOKENS` | Максимум выходных токенов |
| `ORCHESTRATOR_ENABLE_THINKING` | `true`/`false` — включать reasoning (vLLM thinking) |

### Retrieval и Summarizer

| Переменная | Назначение |
|-----------|-----------|
| `RETRIEVER_PORT` / `RETRIEVER_URL` | Общий endpoint выбранного retriever |
| `SUMMARIZER_PORT` / `SUMMARIZER_URL` | Endpoint summarizer |
| `SUMMARIZER_LLM_BASE_URL` | vLLM/OpenAI-compatible endpoint |
| `SUMMARIZER_LLM_MODEL` | Название модели |
| `SUMMARIZER_LLM_TEMPERATURE` | Температура генерации |
| `SUMMARIZER_LLM_MAX_TOKENS` | Максимум выходных токенов |
| `SUMMARIZER_ENABLE_THINKING` | `true`/`false` |

### Индексы и поиск

| Переменная | Назначение |
|-----------|-----------|
| `CHUNKS_PATH` | Путь к chunks.pkl |
| `FAISS_INDEX_PATH` | Путь к FAISS-индексу |
| `BM25_INDEX_PATH` | Путь к BM25-индексу |
| `EMB_TOP_K` | top-K для FAISS |
| `BM25_TOP_K` | top-K для BM25 |
| `SCORE_THRESHOLD` | Порог релевантности |

### Таймауты и пути

| Переменная | Назначение |
|-----------|-----------|
| `REQUEST_TIMEOUT` | Таймаут A2A-запроса (секунды) |
| `TASK_POLL_INTERVAL` | Интервал опроса задачи |
| `MARKDOWN_DIR` | Директория с очищенными .md для RAG |

### OCR и парсер

| Переменная | Назначение |
|-----------|-----------|
| `OCR_API_URL` | vLLM endpoint для OCR |
| `OCR_MODEL` | Модель для OCR |
| `OCR_TIMEOUT` | Таймаут OCR |
| `OCR_MAX_TOKENS` | Максимум выходных токенов OCR |
| `OCR_TEMPERATURE` | Температура OCR |
| `OCR_TOP_P` | top-p для OCR |
| `OCR_TOP_K` | top-K для OCR |
| `PARSER_INPUT_DIR` | Входная директория с документами |
| `PARSER_OUTPUT_DIR` | Выходная директория сырых .md |
| `PARSER_CLEAN_OUTPUT_DIR` | Выходная директория очищенных .md |
| `PARSER_TEMP_DIR` | Директория временных изображений |

### Discovery оркестратора

| Переменная | Назначение |
|-----------|-----------|
| `SUB_AGENT_URLS` | URL агентов через запятую — оркестратор читает AgentCard каждого и строит тулы динамически |

## Масштабирование: добавление нового агента

Добавление агента не требует правок кода:

1. Реализовать A2A-совместимого агента с корректным AgentCard:
   - `id`, `name`, `description` — идентификация
   - `inputModes`, `outputModes` — что принимает/отдаёт
   - `tags` — `intermediate`, `final`, `shared-corpus` для роутинга оркестратором
   - `examples` — примеры запросов (помогают LLM выбрать агента)
2. Запустить агента на любом хосте/порту
3. Дописать его URL в `.env` → `SUB_AGENT_URLS`
4. Перезапустить оркестратор

Оркестратор при `initialize()` читает AgentCard каждого URL, динамически создаёт LangChain-тулы из скиллов и форматирует промпт для LLM с описаниями всех агентов. При изменении состава агентов (rediscovery) инструменты перестраиваются автоматически без перезапуска.

## CLI (venv/bin/python cli.py)

```
============================================================
  Document Parser + A2A Multi-Agent RAG System
============================================================
  Agents: [stopped]  |  Index: [no index]  |  Parse: [no parse]
============================================================
  1. Parse documents         → вход → markdown_output → markdown_output_clean
  2. Index documents         → чанки + FAISS + BM25
  3. Start agents            → A2A-серверы
  4. Query the system        → вопрос → ответ
  0. Exit
============================================================
```

### 1. Парсинг документов

- DOCX: извлечение текста, таблиц, формул (OMML → LaTeX), встроенных изображений через OCR
- PDF: извлечение текста (pymupdf), детекция формул, таблиц через OCR
- XLSX: конвертация листов в Markdown-таблицы
- Двухэтапная очистка: inline + пост-обработка (склейка абзацев, TOC, оптимизация для RAG)

### 2. Индексация

- Чанкер: разбивка по h1-h3, fallback 512 токенов
- FAISS: IndexFlatIP + sentence-transformers
- BM25: rank-bm25, BM25Okapi

### 3. Запуск агентов

Orchestrator + один выбранный Retriever (`base` или `baseline_reranker`) + Summarizer — независимые FastAPI + uvicorn процессы. Логи в `var/logs/`.

### 4. Запрос

Оркестратор через ReAct-агента строит цепочку, сам принимает все решения:

- **Приветствия и мета-вопросы** — `send_text` с прямым ответом
- **Поисковые запросы** — вызов выбранного retriever, передача чанков через `data_from`, вызов summarizer, `forward_agent_artifacts`
- **Многошаговость** — в одном запросе можно отправить несколько результатов подряд (`send_text` → приветствие, `forward_agent_artifacts` → поиск)
- **Retrieval** — `base` выполняет Dense + BM25 и RRF; `baseline_reranker` дополнительно применяет Jina
- **Контроль качества** — LLM может повторить вызов retriever, если результат неудовлетворительный

Артефакты доставляются через A2A-стриминг (`TaskArtifactUpdateEvent`) — клиент получает каждый ответ сразу, не дожидаясь завершения всей цепочки.

## Использование

```bash
# Интерактивный режим
venv/bin/python cli.py

# Или отдельно (из корня репозитория; PYTHONPATH=src обязателен для -m запусков):
PYTHONPATH=src venv/bin/python -m parser.parser                        # только парсинг
PYTHONPATH=src venv/bin/python -m parser.clean_markdown               # только очистка
PYTHONPATH=src venv/bin/python src/scripts/build_indices.py           # только индексация
bash src/scripts/start_agents.sh base                              # base + summarizer + orchestrator
bash src/scripts/start_agents.sh baseline_reranker                  # Jina reranker + summarizer + orchestrator
PYTHONPATH=src venv/bin/python src/scripts/test_e2e.py                # E2E тест

# Прямой A2A-запрос к оркестратору (JSON-RPC v1.0)
curl -X POST "$ORCHESTRATOR_URL/" \
  -H "Content-Type: application/json" \
  -H "A2A-Version: 1.0" \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"role":"ROLE_USER","parts":[{"text":"поисковый запрос","mediaType":"text/plain"}]}}}'
```


## Эксперимент `npa_mas_component_v1`

Воспроизводимый fixed-pipeline стенд описан в `План_перезапуска_MAS_RAG_для_НПА.md`. Для всех LLM-компонентов используются `NPA_LLM_BASE_URL`, `NPA_LLM_MODEL` и необязательный `NPA_LLM_API_KEY`; секреты не записываются в manifests.

```bash
export PYTHONPATH=src
venv/bin/python -m scripts.npa_mas_experiment annotations-template
# Заполнить human_a/human_b/final/adjudications согласно codebook.
venv/bin/python -m scripts.npa_mas_experiment annotations-validate
venv/bin/python -m scripts.npa_mas_experiment audit-inputs
venv/bin/python -m scripts.npa_mas_experiment pilot --run-id pilot_001 --no-scoring
venv/bin/python -m scripts.npa_mas_experiment freeze --run-id core_001
venv/bin/python -m scripts.npa_mas_experiment run-core --run-id core_001
venv/bin/python -m scripts.npa_mas_experiment validate --run-id core_001 --require-complete
venv/bin/python -m scripts.npa_mas_experiment score --run-id core_001
venv/bin/python -m scripts.npa_mas_experiment analyze --run-id core_001
venv/bin/python -m scripts.npa_mas_experiment stability --run-id stability_001
venv/bin/python -m scripts.npa_mas_experiment pack --run-id core_001
```

Draft или LLM labels не принимаются как final. До pilot/freeze требуются 556 human-reviewed groups, исходные primary/secondary annotations для независимой 112-group контрольной выборки, per-label reliability и документированные adjudications. Pilot не использует qrels и не вычисляет quality; aggregate quality открывается только после завершения всех десяти core variants.

## Технический стек

- Python 3.12+
- `a2a-sdk[fastapi]` — A2A v1.0
- `langgraph` + `langchain-openai` — ReAct агенты
- FastAPI + uvicorn — A2A серверы
- `sentence-transformers` + FAISS — эмбеддинг-ретривер
- `rank-bm25` — ключевой ретривер
- `python-docx`, `pymupdf`, `openpyxl` — парсеры документов
- `lxml` + `Pillow` — OMML/OCR
- vLLM — LLM и OCR
