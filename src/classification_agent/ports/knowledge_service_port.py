from __future__ import annotations

from abc import ABC, abstractmethod

from classification_agent.api.schemas import (
    ChunkData,
    PolicyChunk,
    SimilarityResult,
    TopicHierarchy,
)


class KnowledgeServicePort(ABC):
    """Port for reading chunks, searching, and storing topics via Knowledge Service.

    Classification Agent reads pre-stored chunks (stored by Validator Agent
    in Phase 3) and uses SimilaritySearch for ReAct sufficiency probing and
    SearchPolicies for rubric fitness assessment (CR-CLS-001).
    """

    @abstractmethod
    async def get_chunks_by_workflow(
        self,
        workflow_id: str,
        assessor_id: str | None = None,
    ) -> list[ChunkData]:
        """Retrieve document chunks for a workflow (gRPC GetChunksByWorkflow 3.2.3).

        Returns all chunks stored by the Validator Agent during Phase 3.
        When assessor_id is provided, returns cumulative chunks across all
        of the assessor's workflows (CR-RAG-001).
        """

    @abstractmethod
    async def store_topics(self, workflow_id: str, topics: TopicHierarchy) -> None:
        """Store extracted topic hierarchy for a workflow (gRPC StoreTopics 3.2.4)."""

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        knowledge_base_target: str,  # "document" | "policy" | "enriched"
        workflow_id: str,
        top_k: int = 5,
        assessor_id: str | None = None,
    ) -> list[SimilarityResult]:
        """Search for semantically similar chunks in a knowledge base (gRPC SimilaritySearch 3.2.2).

        Used by the Classification Agent's ReAct sufficiency probing loop to
        formulate depth queries per borderline topic at specific difficulty levels.
        When assessor_id is provided, search spans all of the assessor's
        workflows for cumulative knowledge (CR-RAG-001).
        """

    @abstractmethod
    async def search_policies(
        self,
        query: str,
        policy_type: str | None = None,  # "system_default" | "assessor_rubric" | None (both)
        assessment_id: str | None = None,
    ) -> list[PolicyChunk]:
        """Search policy knowledge base for rubrics and policies (gRPC SearchPolicies 3.3.2).

        Used for rubric fitness assessment: checks assessor rubric first, then
        falls back to system defaults. Returns matching policy chunks ranked
        by semantic similarity.
        """

