"""Pydantic response models for structured LLM output.

Each LLM call in the Classification Agent has a corresponding Pydantic model
that defines the expected response structure. These models serve dual purposes:

1. Generate JSON schemas via ``model_json_schema()`` + ``clean_for_gemini()``
   for Model Broker's structured output enforcement (Gemini/OpenAI).
2. Validate and parse LLM responses via ``model_validate()`` for type-safe
   access in domain code (replacing manual ``dict.get()`` calls).

Usage::

    from af_shared.utils.schema_compat import clean_for_gemini
    schema = clean_for_gemini(ReactSufficiencyCheckResponse.model_json_schema())
    # Pass schema to Model Broker as response_schema
    # Parse response: ReactSufficiencyCheckResponse.model_validate(llm_result)
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared gap item (used by ReAct unified probe)
# ---------------------------------------------------------------------------

class SufficiencyGapItem(BaseModel):
    """A single gap identified in material sufficiency analysis."""

    topic: str
    current_depth: str  # "none" | "surface" | "moderate" | "deep"
    required_depth: str  # "surface" | "moderate" | "deep"
    gap_description: str
    fillable_by_web: bool = False
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Unified ReAct probe response (classification.react_sufficiency)
# ---------------------------------------------------------------------------

class ReactSufficiencyCheckResponse(BaseModel):
    """LLM response for unified ReAct sufficiency + rubric fitness probe.

    A single structured payload covering both material sufficiency
    (TASK 1) and rubric fitness (TASK 2) in one tool-calling session.
    """

    sufficient: bool
    reason: str = ""
    gap_analysis: list[SufficiencyGapItem] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    autonomy_exercised: bool = False
    # Rubric fitness verdict: "ALIGNED" | "MISALIGNED" | "NO_RUBRIC"
    rubric_fitness: str = "NO_RUBRIC"
    rubric_reasoning: str = ""
    rubric_source: str = "none"


# ---------------------------------------------------------------------------
# Topic Extraction (classification.topic_extraction)
# ---------------------------------------------------------------------------

class TopicSubtopicItem(BaseModel):
    """A subtopic within a parent topic."""

    topic_id: str = ""
    name: str


class TopicItem(BaseModel):
    """A top-level topic with optional subtopics."""

    topic_id: str = ""
    name: str
    subtopics: list[TopicSubtopicItem] = Field(default_factory=list)


class TopicExtractionResponse(BaseModel):
    """LLM response for hierarchical topic extraction from document chunks."""

    topics: list[TopicItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Rubric Fitness (kept for backward compat with rubric_fitness.py class file)
# ---------------------------------------------------------------------------

class RubricFitnessResponse(BaseModel):
    """LLM response for rubric-material alignment assessment."""

    is_aligned: bool
    alignment_score: float
    gap_description: str | None = None
    recommendation: str  # "use_as_is" | "supplement" | "synthesize_new"
