"""Vendored shim — only the classes the classifier-agent imports.

See vendor/af_shared/__init__.py for context.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ModelResponse(BaseModel):
    content: str
    model_used: str
    model_tier: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    latency_ms: float


class DecisionLogEntry(BaseModel):
    workflow_id: str
    agent_name: str
    decision_type: str
    assessor_id: str | None = None
    input_summary: dict[str, Any] | None = None
    output_summary: dict[str, Any] | None = None
    reasoning_steps: list[dict[str, Any]] | None = None
    confidence_score: float | None = None
    prompt_version: str | None = None
    model_id: str | None = None
    grounding_sources: list[str] | None = None


class TokenUsageEntry(BaseModel):
    workflow_id: str
    agent_name: str
    model_id: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    assessor_id: str | None = None


class EvaluationAuditEntry(BaseModel):
    workflow_id: str
    participant_id: str
    question_id: str
    evaluation_method: str
    score: float
    max_score: float
    reasoning: str | None = None
    assessor_id: str | None = None
