#!/usr/bin/env python3
"""End-to-end test for the A2A multi-agent RAG system using the A2A SDK client (v1.0)."""

import asyncio
import json
import sys
import time
import uuid

import httpx

from a2a.client import create_client, Client, ClientConfig
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    SendMessageConfiguration,
    StreamResponse,
)

sys.path.insert(0, __file__.rsplit("/src/", 1)[0] + "/src")
from common.config import A2A_VERSION, ORCHESTRATOR_URL, REQUEST_TIMEOUT  # noqa: E402


ORCH_URL = ORCHESTRATOR_URL

QUERIES = [
    {
        "id": "test_1_financial_agreements",
        "text": "Каким нормативно-правовым актом регламентируется процедура подписания финансовых соглашений и существуют ли утвержденные сроки по подписанию данных соглашений?",
        "expected_keywords": ["Постановление", "Правительства", "соглашени"],
    },
    {
        "id": "test_2_project_start",
        "text": "Мы хотим запустить новый проект, с чего нам нужно начать?",
        "expected_keywords": ["шаг", "методическ", "национальн"],
    },
    {
        "id": "test_3_reporting_frequency",
        "text": "Как часто я должен отчитываться по проекту?",
        "expected_keywords": ["месяц", "квартал", "отчет"],
    },
]


def extract_answer(stream_responses: list[StreamResponse]) -> str:
    for resp in stream_responses:
        if resp.HasField("task"):
            for artifact in resp.task.artifacts:
                for part in artifact.parts:
                    if part.HasField("data"):
                        try:
                            from google.protobuf.json_format import MessageToDict
                            d = MessageToDict(part.data)
                            if isinstance(d, dict) and "answer" in d:
                                return d["answer"]
                        except Exception:
                            pass
                    if part.text:
                        return part.text
        elif resp.HasField("artifact_update"):
            artifact = resp.artifact_update.artifact
            for part in artifact.parts:
                if part.HasField("data"):
                    try:
                        from google.protobuf.json_format import MessageToDict
                        d = MessageToDict(part.data)
                        if isinstance(d, dict) and "answer" in d:
                            return d["answer"]
                    except Exception:
                        pass
                if part.text:
                    return part.text
    return ""


def check_keywords(text: str, keywords: list[str]) -> list[str]:
    lower = text.lower()
    missing = [kw for kw in keywords if kw.lower() not in lower]
    return missing


async def run_query(client: Client, query_text: str) -> list[StreamResponse]:
    request = SendMessageRequest(
        message=Message(
            role=Role.ROLE_USER,
            parts=[Part(text=query_text)],
            message_id=f"msg-{uuid.uuid4().hex[:12]}",
        ),
        configuration=SendMessageConfiguration(return_immediately=False),
    )
    responses = []
    async for resp in client.send_message(request):
        responses.append(resp)
    return responses


async def test_all():
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        headers={"A2A-Version": A2A_VERSION},
    )
    client = await create_client(ORCH_URL, client_config=ClientConfig(httpx_client=http_client))
    all_passed = True

    try:
        for q in QUERIES:
            print(f"\n--- {q['id']} ---")
            print(f"Query: {q['text'][:80]}...")
            start = time.time()

            try:
                responses = await run_query(client, q["text"])
            except Exception as e:
                print(f"FAIL: request failed: {e}")
                all_passed = False
                continue

            elapsed = time.time() - start
            answer = extract_answer(responses)

            print(f"Time: {elapsed:.1f}s")

            if not answer:
                print("FAIL: empty answer")
                all_passed = False
                continue

            missing = check_keywords(answer, q["expected_keywords"])
            if missing:
                print(f"WARN: missing keywords: {missing}")

            print("Answer preview:")
            print(answer[:600].replace("\n", " "))
            print("...")
            print("OK")

    finally:
        await client.close()
        await http_client.aclose()

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed.")
        return 0
    else:
        print("Some tests failed.")
        return 1


def main() -> int:
    return asyncio.run(test_all())


if __name__ == "__main__":
    sys.exit(main())
