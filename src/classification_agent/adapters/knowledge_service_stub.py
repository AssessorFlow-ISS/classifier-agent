from __future__ import annotations

from classification_agent.api.schemas import (
    ChunkData,
    PolicyChunk,
    SimilarityResult,
    TopicHierarchy,
)
from classification_agent.ports.knowledge_service_port import KnowledgeServicePort


class StubKnowledgeServiceAdapter(KnowledgeServicePort):
    """In-memory stub for Knowledge Service.

    Pre-load chunks per workflow_id for deterministic testing.
    Stored topics are kept in memory for assertion checking.
    Similarity search and policy search return configurable canned responses.
    """

    _POLICY_UNSET = object()

    def __init__(self) -> None:
        self._chunks: dict[str, list[ChunkData]] = {}
        self._stored_topics: dict[str, TopicHierarchy] = {}
        self._similarity_results: dict[str, list[SimilarityResult]] = {}
        self._policy_chunks: list[PolicyChunk] | object = self._POLICY_UNSET
        # Default chunks (22 total) — enough to pass sufficiency threshold (20).
        topics = ["Encapsulation"] * 8 + ["Polymorphism"] * 7 + ["Inheritance"] * 7
        self._default_chunks = [
            ChunkData(
                chunk_id=f"default-chunk-{i:03d}",
                workflow_id="default",
                content=f"Stub chunk {i} covering {t} concepts for E2E testing.",
                source_type="direct_text",
                metadata={"topic_hint": t},
            )
            for i, t in enumerate(topics, start=1)
        ]
        # Default similarity results for ReAct probing
        self._default_similarity_results = [
            SimilarityResult(
                chunk_id=f"sim-chunk-{i:03d}",
                content=f"Semantically similar content for probing query {i}.",
                similarity_score=0.9 - (i * 0.05),
                source_document=f"doc-{i}.pdf",
            )
            for i in range(1, 4)
        ]
        # Default policy chunks for rubric fitness assessment
        self._default_policy_chunks = [
            PolicyChunk(
                chunk_id="policy-001",
                content="Default grading rubric: assess understanding of core concepts.",
                policy_type="system_default",
                source="admin_seeded",
                similarity_score=0.85,
            ),
        ]

    # -----------------------------------------------------------------------
    # Test helpers
    # -----------------------------------------------------------------------

    def add_chunks(self, workflow_id: str, chunks: list[ChunkData]) -> None:
        """Pre-load chunks for a given workflow (test setup)."""
        self._chunks[workflow_id] = chunks

    def get_stored_topics(self, workflow_id: str) -> TopicHierarchy | None:
        """Return topics stored via store_topics (test assertion)."""
        return self._stored_topics.get(workflow_id)

    def set_similarity_results(
        self, knowledge_base_target: str, results: list[SimilarityResult],
    ) -> None:
        """Pre-set similarity search results for a knowledge base target (test setup)."""
        self._similarity_results[knowledge_base_target] = results

    def set_policy_chunks(self, chunks: list[PolicyChunk]) -> None:
        """Pre-set policy chunks for SearchPolicies (test setup)."""
        self._policy_chunks = chunks

    # -----------------------------------------------------------------------
    # Port implementation
    # -----------------------------------------------------------------------

    async def get_chunks_by_workflow(
        self,
        workflow_id: str,
        assessor_id: str | None = None,
    ) -> list[ChunkData]:
        return self._chunks.get(workflow_id, self._default_chunks)

    async def store_topics(self, workflow_id: str, topics: TopicHierarchy) -> None:
        self._stored_topics[workflow_id] = topics

    async def similarity_search(
        self,
        query: str,
        knowledge_base_target: str,
        workflow_id: str,
        top_k: int = 5,
        assessor_id: str | None = None,
    ) -> list[SimilarityResult]:
        results = self._similarity_results.get(
            knowledge_base_target, self._default_similarity_results,
        )
        return results[:top_k]

    async def search_policies(
        self,
        query: str,
        policy_type: str | None = None,
        assessment_id: str | None = None,
    ) -> list[PolicyChunk]:
        if self._policy_chunks is self._POLICY_UNSET:
            chunks = self._default_policy_chunks
        else:
            chunks = self._policy_chunks  # type: ignore[assignment]
        if policy_type is not None:
            return [c for c in chunks if c.policy_type == policy_type]
        return chunks
