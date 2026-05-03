"""KnowledgeServiceHttpAdapter — real HTTP client for Knowledge Service.

Calls the Knowledge Service FastAPI internal endpoints via HTTP.
Returns Classification Agent-specific types (ChunkData, SimilarityResult,
PolicyChunk) by mapping from the KS JSON responses.

Environment Variables:
    KS_URL: Base URL for the Knowledge Service (default: http://localhost:8020)
"""
from __future__ import annotations

import os
from typing import Any

import httpx
import structlog

from classification_agent.api.schemas import (
    ChunkData,
    PolicyChunk,
    SimilarityResult,
    TopicHierarchy,
)
from classification_agent.ports.knowledge_service_port import KnowledgeServicePort

logger = structlog.get_logger(__name__)


class KnowledgeServiceHttpAdapter(KnowledgeServicePort):
    """HTTP client adapter for the real Knowledge Service at :8020."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url or os.environ.get(
            "KS_URL", "http://localhost:8020"
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )

    async def get_chunks_by_workflow(
        self,
        workflow_id: str,
        assessor_id: str | None = None,
    ) -> list[ChunkData]:
        body: dict[str, Any] = {"workflow_id": workflow_id}
        if assessor_id:
            body["assessor_id"] = assessor_id

        try:
            resp = await self._client.post(
                "/api/v1/internal/chunks-by-workflow", json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            chunks = [
                ChunkData(
                    chunk_id=c.get("chunk_id") or c.get("id", ""),
                    workflow_id=c.get("workflow_id", workflow_id),
                    content=c["content"],
                    source_type=c.get("source_type", "direct_text"),
                    metadata=c.get("metadata") or {"source_file": c.get("source_file"), "chunk_index": c.get("chunk_index")},
                )
                for c in data.get("chunks", [])
            ]
            logger.info("ks_get_chunks", workflow_id=workflow_id, count=len(chunks))
            return chunks
        except Exception:
            logger.warning("ks_get_chunks_failed", workflow_id=workflow_id, exc_info=True)
            return []

    async def store_topics(self, workflow_id: str, topics: TopicHierarchy) -> None:
        stored = 0
        for topic in topics.topics:
            body = {
                "workflow_id": workflow_id,
                "main_topic": topic.name,
                "subtopics": [s.name if hasattr(s, 'name') else str(s) for s in topic.subtopics],
            }
            try:
                resp = await self._client.post(
                    "/api/v1/internal/store-topics", json=body,
                )
                resp.raise_for_status()
                stored += 1
            except Exception:
                logger.warning("ks_store_topic_failed", workflow_id=workflow_id, topic=topic.name, exc_info=True)
        logger.info("ks_store_topics", workflow_id=workflow_id, stored=stored, total=len(topics.topics))

    async def similarity_search(
        self,
        query: str,
        knowledge_base_target: str,
        workflow_id: str,
        top_k: int = 5,
        assessor_id: str | None = None,
    ) -> list[SimilarityResult]:
        body: dict[str, Any] = {
            "query": query,
            "workflow_id": workflow_id,
            "kb_type": knowledge_base_target,
            "top_k": top_k,
        }
        if assessor_id:
            body["assessor_id"] = assessor_id

        try:
            resp = await self._client.post(
                "/api/v1/internal/similarity-search", json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            results = [
                SimilarityResult(
                    chunk_id=c.get("chunk_id") or c.get("id", ""),
                    content=c["content"],
                    similarity_score=c.get("similarity_score", c.get("score", 0.0)),
                    source_document=c.get("source_file") or c.get("metadata", {}).get("source_file"),
                    metadata=c.get("metadata"),
                )
                for c in data.get("chunks", [])
            ]
            logger.info("ks_similarity_search", workflow_id=workflow_id, results=len(results))
            return results
        except Exception:
            logger.warning("ks_similarity_search_failed", workflow_id=workflow_id, exc_info=True)
            return []

    async def search_policies(
        self,
        query: str,
        policy_type: str | None = None,
        assessment_id: str | None = None,
    ) -> list[PolicyChunk]:
        body: dict[str, Any] = {"query": query, "top_k": 5}
        if assessment_id:
            body["assessment_id"] = assessment_id

        try:
            resp = await self._client.post(
                "/api/v1/internal/search-policies", json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            chunks = [
                PolicyChunk(
                    chunk_id=c.get("id", c.get("chunk_id", "")),
                    content=c["content"],
                    policy_type=c.get("policy_type", "system_default"),
                    source=c.get("source", "admin_seeded"),
                    assessment_id=c.get("assessment_id"),
                    similarity_score=c.get("score", c.get("similarity_score")),
                )
                for c in data.get("results", data.get("chunks", []))
            ]
            if policy_type:
                chunks = [c for c in chunks if c.policy_type == policy_type]
            logger.info("ks_search_policies", results=len(chunks))
            return chunks
        except Exception:
            logger.warning("ks_search_policies_failed", exc_info=True)
            return []

    async def close(self) -> None:
        await self._client.aclose()
