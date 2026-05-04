"""Tests for KnowledgeServiceHttpAdapter.

Uses ``httpx.MockTransport`` to cover all four public methods plus their
happy-path, error-path, and fallback-mapping branches without standing
up a real Knowledge Service.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from classification_agent.adapters.knowledge_service_http import (
    KnowledgeServiceHttpAdapter,
)
from classification_agent.api.schemas import (
    SubTopic,
    Topic,
    TopicHierarchy,
)


_KS_URL = "http://test-ks:8020"


def _adapter_with_handler(handler):
    """Wrap a mock transport into a real adapter for assertion convenience."""
    adapter = KnowledgeServiceHttpAdapter(base_url=_KS_URL)
    adapter._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=_KS_URL,
    )
    return adapter


# ---------------------------------------------------------------------------
# get_chunks_by_workflow
# ---------------------------------------------------------------------------


class TestGetChunksByWorkflow:
    async def test_returns_mapped_chunks(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = request.read()
            payload = {
                "chunks": [
                    {
                        "chunk_id": "c-1",
                        "workflow_id": "wf-1",
                        "content": "Alpha",
                        "source_type": "direct_text",
                        "metadata": {"source_file": "a.pdf", "chunk_index": 0},
                    },
                    {
                        # 'id' alias instead of chunk_id
                        "id": "c-2",
                        "content": "Beta",
                        # no source_type => default 'direct_text'
                        # no metadata => falls back to the assembled dict
                        "source_file": "b.pdf",
                        "chunk_index": 1,
                    },
                ]
            }
            return httpx.Response(200, json=payload)

        adapter = _adapter_with_handler(handler)
        chunks = await adapter.get_chunks_by_workflow("wf-1", assessor_id="assessor-1")

        assert len(chunks) == 2
        assert chunks[0].chunk_id == "c-1"
        assert chunks[1].chunk_id == "c-2"
        # fallback workflow_id used when the payload omits it
        assert chunks[1].workflow_id == "wf-1"
        # Body includes assessor_id when provided
        assert b"assessor-1" in captured["body"]
        assert captured["url"].endswith("/api/v1/internal/chunks-by-workflow")

    async def test_omits_assessor_id_when_none(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.read()
            return httpx.Response(200, json={"chunks": []})

        adapter = _adapter_with_handler(handler)
        await adapter.get_chunks_by_workflow("wf-no-assessor")

        assert b"assessor_id" not in captured["body"]

    async def test_swallows_http_error_and_returns_empty(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"detail": "boom"})

        adapter = _adapter_with_handler(handler)
        chunks = await adapter.get_chunks_by_workflow("wf-fail")
        assert chunks == []


# ---------------------------------------------------------------------------
# store_topics
# ---------------------------------------------------------------------------


class TestStoreTopics:
    async def test_posts_each_topic(self) -> None:
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            body = json.loads(request.read())
            calls.append(body)
            return httpx.Response(200, json={})

        adapter = _adapter_with_handler(handler)
        topics = TopicHierarchy(
            workflow_id="wf-store",
            topics=[
                Topic(
                    topic_id="t-1",
                    name="OOP",
                    subtopics=[SubTopic(topic_id="s-1", name="Inheritance")],
                ),
                Topic(topic_id="t-2", name="Algorithms", subtopics=[]),
            ],
        )

        await adapter.store_topics("wf-store", topics)

        assert len(calls) == 2
        assert {c["main_topic"] for c in calls} == {"OOP", "Algorithms"}
        assert calls[0]["subtopics"] == ["Inheritance"]

    async def test_suppresses_per_topic_errors(self) -> None:
        """A failing topic is logged and skipped; the remaining still POST."""
        attempt = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempt["n"] += 1
            if attempt["n"] == 1:
                return httpx.Response(500, json={"detail": "db flaky"})
            return httpx.Response(200, json={})

        adapter = _adapter_with_handler(handler)
        topics = TopicHierarchy(
            workflow_id="wf-partial",
            topics=[
                Topic(topic_id="t-1", name="T1", subtopics=[]),
                Topic(topic_id="t-2", name="T2", subtopics=[]),
            ],
        )

        # Should not raise even though the first POST errored.
        await adapter.store_topics("wf-partial", topics)
        assert attempt["n"] == 2


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------


class TestSimilaritySearch:
    async def test_maps_results_and_uses_aliases(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "chunks": [
                    {
                        "chunk_id": "c-1",
                        "content": "alpha",
                        "similarity_score": 0.91,
                        "source_file": "a.pdf",
                    },
                    {
                        # 'id' alias + 'score' alias
                        "id": "c-2",
                        "content": "beta",
                        "score": 0.42,
                        "metadata": {"source_file": "b.pdf"},
                    },
                ]
            })

        adapter = _adapter_with_handler(handler)
        results = await adapter.similarity_search(
            query="trees",
            knowledge_base_target="workflow_materials",
            workflow_id="wf-1",
            top_k=3,
            assessor_id="assessor-2",
        )

        assert len(results) == 2
        assert results[0].chunk_id == "c-1"
        assert results[0].similarity_score == 0.91
        assert results[1].chunk_id == "c-2"
        assert results[1].similarity_score == 0.42
        assert results[1].source_document == "b.pdf"

    async def test_returns_empty_on_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "index offline"})

        adapter = _adapter_with_handler(handler)
        results = await adapter.similarity_search(
            query="x", knowledge_base_target="workflow_materials",
            workflow_id="wf-fail",
        )
        assert results == []


# ---------------------------------------------------------------------------
# search_policies
# ---------------------------------------------------------------------------


class TestSearchPolicies:
    async def test_maps_results_and_filters_by_policy_type(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "results": [
                    {
                        "id": "p-1",
                        "content": "System default policy",
                        "policy_type": "system_default",
                        "source": "admin_seeded",
                        "score": 0.77,
                    },
                    {
                        "chunk_id": "p-2",
                        "content": "Assessor rubric chunk",
                        "policy_type": "assessor_rubric",
                        "source": "assessor_upload",
                        "assessment_id": "a-1",
                        "similarity_score": 0.61,
                    },
                ]
            })

        adapter = _adapter_with_handler(handler)
        chunks = await adapter.search_policies(
            query="rubric", policy_type="assessor_rubric", assessment_id="a-1",
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "p-2"
        assert chunks[0].policy_type == "assessor_rubric"

    async def test_falls_back_to_chunks_key(self) -> None:
        """KS sometimes returns results under 'chunks' rather than 'results'."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "chunks": [
                    {
                        "chunk_id": "p-3",
                        "content": "c",
                        "policy_type": "system_default",
                        "source": "admin_seeded",
                    },
                ]
            })

        adapter = _adapter_with_handler(handler)
        chunks = await adapter.search_policies(query="anything")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "p-3"

    async def test_returns_empty_on_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, json={"detail": "upstream down"})

        adapter = _adapter_with_handler(handler)
        chunks = await adapter.search_policies(query="x")
        assert chunks == []


# ---------------------------------------------------------------------------
# Defaults + close()
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    def test_default_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KS_URL", "http://ks-from-env:9000")
        adapter = KnowledgeServiceHttpAdapter()
        assert adapter._base_url == "http://ks-from-env:9000"

    def test_missing_ks_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # PROD-safety: no source-code localhost default. Missing env must raise.
        monkeypatch.delenv("KS_URL", raising=False)
        with pytest.raises(RuntimeError, match="KS_URL env var is required"):
            KnowledgeServiceHttpAdapter()

    async def test_close_closes_client(self) -> None:
        adapter = KnowledgeServiceHttpAdapter(base_url=_KS_URL)
        # sanity: close() should simply call aclose on the underlying client
        await adapter.close()
        assert adapter._client.is_closed is True
