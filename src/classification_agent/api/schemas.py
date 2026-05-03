from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClassificationType(str, Enum):
    """Classification pipeline variant."""

    SUFFICIENCY_AND_TOPICS = "sufficiency_and_topics"
    SUFFICIENCY_ONLY = "sufficiency_only"
    TOPICS_ONLY = "topics_only"


class SourceType(str, Enum):
    """How the chunk content was obtained."""

    DIRECT_TEXT = "direct_text"
    OCR_EXTRACTED = "ocr_extracted"


class DifficultyLevel(str, Enum):
    """Assessment difficulty setting."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class WebResearchMode(str, Enum):
    """Web research autonomy setting for the Classification Agent."""

    MANUAL = "manual"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Knowledge Service data
# ---------------------------------------------------------------------------

class ChunkData(BaseModel):
    """A single document chunk retrieved from Knowledge Service."""

    chunk_id: str
    workflow_id: str
    content: str
    source_type: SourceType = SourceType.DIRECT_TEXT
    metadata: dict | None = None


class SimilarityResult(BaseModel):
    """A single result from semantic similarity search in a knowledge base."""

    chunk_id: str
    content: str
    similarity_score: float
    source_document: str | None = None
    metadata: dict | None = None


class PolicyChunk(BaseModel):
    """A policy chunk from the Policy Knowledge Base."""

    chunk_id: str
    content: str
    policy_type: str  # "system_default" | "assessor_rubric"
    source: str  # "admin_seeded" | "assessor_upload"
    assessment_id: str | None = None
    similarity_score: float | None = None


# ---------------------------------------------------------------------------
# Assessment configuration
# ---------------------------------------------------------------------------

class AssessmentConfig(BaseModel):
    """Assessment parameters from Assessment Submission Service."""

    assessment_id: str
    assessment_title: str = "Untitled Assessment"
    structured_question_count: int = Field(ge=0, default=10)
    non_structured_question_count: int = Field(ge=0, default=5)
    difficulty_level: DifficultyLevel = DifficultyLevel.MEDIUM
    web_research_mode: WebResearchMode = WebResearchMode.MANUAL


# ---------------------------------------------------------------------------
# Gap analysis and rubric fitness
# ---------------------------------------------------------------------------

class GapAnalysisEntry(BaseModel):
    """A single gap identified during material sufficiency analysis."""

    topic: str
    current_depth: str  # "surface" | "moderate" | "deep"
    required_depth: str  # "surface" | "moderate" | "deep"
    gap_description: str
    fillable_by_web: bool = False
    confidence: float = 0.0


class RubricFitnessResult(BaseModel):
    """Result of rubric fitness assessment via SearchPolicies."""

    is_aligned: bool
    rubric_source: str  # "assessor_upload" | "system_default" | "none"
    alignment_score: float = 0.0
    gap_description: str | None = None
    recommendation: str | None = None  # "use_as_is" | "supplement" | "synthesize_new"


# ---------------------------------------------------------------------------
# Sufficiency result
# ---------------------------------------------------------------------------

class SufficiencyResult(BaseModel):
    """Output of the sufficiency checker."""

    sufficient: bool
    reason: str
    gap_analysis: list[GapAnalysisEntry] = Field(default_factory=list)
    chunk_count: int = 0
    threshold: int = 0
    confidence: float = 0.0  # Average of LLM gap scores, or heuristic when no LLM call


# ---------------------------------------------------------------------------
# Topic hierarchy
# ---------------------------------------------------------------------------

class SubTopic(BaseModel):
    """A leaf-level subtopic."""

    topic_id: str
    name: str
    subtopics: list[SubTopic] = Field(default_factory=list)


class Topic(BaseModel):
    """A top-level topic with nested subtopics."""

    topic_id: str
    name: str
    subtopics: list[SubTopic] = Field(default_factory=list)


class TopicHierarchy(BaseModel):
    """Complete extracted topic hierarchy for a workflow."""

    workflow_id: str
    topics: list[Topic] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0,
        description="LLM self-rated confidence in this hierarchy")


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------

class ClassificationRequest(BaseModel):
    """Request body for POST /invoke.

    ``chunks`` is an optional test-mode override. When provided, the
    ClassificationService will skip the Knowledge Service
    ``get_chunks_by_workflow`` call and use the supplied chunk dicts
    directly. This exists so Depth-2 adversarial drivers (DeepTeam,
    Promptfoo, Guardrail regression) can supply poisoned chunks without
    having to pre-seed the Knowledge Service for a synthetic workflow
    id. Each dict must match the ``ChunkData`` shape:
    ``{"chunk_id": str, "workflow_id": str, "content": str,
       "source_type": "direct_text"|"ocr_extracted",
       "metadata": dict | None}``.
    In production this field is always ``None``; the Orchestrator never
    sets it.
    """

    workflow_id: str
    assessment_id: str
    assessor_id: str | None = None
    classification_type: ClassificationType = ClassificationType.SUFFICIENCY_AND_TOPICS
    chunks: list[dict] | None = None


class ClassificationResponse(BaseModel):
    """Response body for POST /invoke."""

    workflow_id: str
    sufficient: bool
    reason: str
    gap_analysis: list[GapAnalysisEntry] = Field(default_factory=list)
    topics: TopicHierarchy | None = None
    rubric_fitness: RubricFitnessResult | None = None
    rubric_source: str | None = None
    autonomy_exercised: bool = False
    search_queries: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response body for GET /health and GET /ready."""

    status: str = "ok"
    service: str = "classification-agent"
