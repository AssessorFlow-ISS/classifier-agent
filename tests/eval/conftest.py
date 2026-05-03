"""Shared fixtures for DeepEval quality evaluation tests.

Provides the golden dataset loader, classification pipeline runner
(using stub adapters), and helper functions for converting golden
test cases into classification requests and configuring stub adapters.
"""
from __future__ import annotations

import json
import json as _json
import os as _os
from pathlib import Path
from pathlib import Path as _Path
from typing import Any

from classification_agent.adapters.assessment_config_stub import StubAssessmentConfigAdapter
from classification_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from classification_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
    ClassificationRequest,
    ClassificationType,
    DifficultyLevel,
    PolicyChunk,
    WebResearchMode,
)
from classification_agent.domain.services import ClassificationService
from classification_agent.domain.topic_extractor import TopicExtractor
from classification_agent.tools.registry import build_react_prober_factory

# Registry for live model broker instances (for token/cost stats)
_live_model_brokers: list = []

GOLDEN_DIR = Path(__file__).parent / "golden"
FIXTURES_DIR = GOLDEN_DIR / "fixtures"


# ---------------------------------------------------------------------------
# Golden dataset loader
# ---------------------------------------------------------------------------


def load_golden(filename: str) -> list[dict[str, Any]]:
    """Load a golden dataset JSON file."""
    with open(GOLDEN_DIR / filename) as f:
        return json.load(f)


def load_fixture_chunks(filename: str) -> list[ChunkData]:
    """Load chunk fixtures from the fixtures directory."""
    with open(FIXTURES_DIR / filename) as f:
        raw = json.load(f)
    return [ChunkData(**item) for item in raw]


# ---------------------------------------------------------------------------
# Case preparation helpers
# ---------------------------------------------------------------------------


def _difficulty_from_str(val: str) -> DifficultyLevel:
    return DifficultyLevel(val)


def _web_mode_from_str(val: str) -> WebResearchMode:
    return WebResearchMode(val)


def _classification_type_from_str(val: str) -> ClassificationType:
    return ClassificationType(val)


def prepare_config(case: dict[str, Any]) -> AssessmentConfig:
    """Build an AssessmentConfig from a golden test case."""
    cfg = case["input"]["config"]
    return AssessmentConfig(
        assessment_id=case["input"]["assessment_id"],
        assessment_title=cfg["assessment_title"],
        structured_question_count=cfg["structured_question_count"],
        non_structured_question_count=cfg["non_structured_question_count"],
        difficulty_level=_difficulty_from_str(cfg["difficulty_level"]),
        web_research_mode=_web_mode_from_str(cfg.get("web_research_mode", "manual")),
    )


def prepare_chunks(case: dict[str, Any]) -> list[ChunkData]:
    """Load chunks from fixture file or inline definition."""
    if "chunks_fixture" in case["input"]:
        return load_fixture_chunks(case["input"]["chunks_fixture"])
    if "chunks" in case["input"]:
        return [ChunkData(**c) for c in case["input"]["chunks"]]
    return []


def prepare_request(case: dict[str, Any]) -> ClassificationRequest:
    """Build a ClassificationRequest from a golden test case."""
    cls_type = case["input"].get("classification_type", "sufficiency_and_topics")
    return ClassificationRequest(
        workflow_id=case["input"]["workflow_id"],
        assessment_id=case["input"]["assessment_id"],
        classification_type=_classification_type_from_str(cls_type),
    )


def prepare_rubric_policy(case: dict[str, Any]) -> list[PolicyChunk] | None:
    """Build policy chunks from rubric_setup, or None if no rubric."""
    rubric_setup = case["input"].get("rubric_setup")
    if rubric_setup is None:
        return []  # empty list signals "no rubric found"
    return [
        PolicyChunk(
            chunk_id=f"rubric-{case['test_id']}",
            content=rubric_setup["content"],
            policy_type=rubric_setup["policy_type"],
            source=rubric_setup["source"],
            assessment_id=case["input"]["assessment_id"],
            similarity_score=0.85,
        ),
    ]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


async def run_classification_pipeline(case: dict[str, Any]) -> dict[str, Any]:
    """Execute the full classification pipeline for a golden test case.

    Uses stub adapters with configurable responses to produce deterministic
    outputs without real LLM calls.

    Returns the ClassificationResponse serialized as a dict.
    """
    # Set up adapters — use real Model Broker when EVAL_LIVE_MODE=true
    import os
    live = os.environ.get("EVAL_LIVE_MODE", "").lower() in ("true", "1", "yes")

    ks = StubKnowledgeServiceAdapter()
    ac = StubAssessmentConfigAdapter()
    da = StubDecisionAuditAdapter()
    ep = StubEventPublisherAdapter()

    if live:
        from classification_agent.adapters.model_broker_http import ModelBrokerHttpAdapter
        mb = ModelBrokerHttpAdapter()
        _live_model_brokers.append(mb)
    else:
        mb = StubModelBrokerAdapter()

    # Load and register chunks
    chunks = prepare_chunks(case)
    workflow_id = case["input"]["workflow_id"]
    assessment_id = case["input"]["assessment_id"]

    ks.add_chunks(workflow_id, chunks)

    # Register assessment config
    config = prepare_config(case)
    ac.set_config(assessment_id, config)

    # Configure rubric policies
    rubric_policies = prepare_rubric_policy(case)
    if rubric_policies is not None:
        ks.set_policy_chunks(rubric_policies)

    # Configure model broker responses based on expected outcome
    expected = case["expected_output"]

    # Configure stub model broker responses (skipped in live mode)
    expected_rubric = expected.get("rubric_fitness")
    is_sufficient = expected.get("sufficient", True)
    is_aligned = (expected_rubric or {}).get("is_aligned", True) if expected_rubric else True

    if not live:
        # Unified ReAct probe response covers both sufficiency and rubric fitness
        if is_sufficient:
            rubric_fitness_val = "ALIGNED" if (expected_rubric and is_aligned) else "NO_RUBRIC"
            rubric_source_val = (expected_rubric or {}).get("rubric_source", "none") if expected_rubric else "none"
            mb.set_tool_call_responses("classification.react_sufficiency", [
                {
                    "tool_calls": [],
                    "content": {
                        "sufficient": True,
                        "reason": "Material covers all required topics with adequate depth",
                        "gap_analysis": [],
                        "search_queries": [],
                        "autonomy_exercised": False,
                        "rubric_fitness": rubric_fitness_val,
                        "rubric_reasoning": "",
                        "rubric_source": rubric_source_val,
                    },
                }
            ])
            mb.set_response("classification.topic_extraction", {
                "topics": [
                    {
                        "topic_id": "t-001",
                        "name": "Object-Oriented Programming",
                        "subtopics": [
                            {"topic_id": "t-001-1", "name": "Encapsulation"},
                            {"topic_id": "t-001-2", "name": "Polymorphism"},
                            {"topic_id": "t-001-3", "name": "Inheritance"},
                        ],
                    },
                    {
                        "topic_id": "t-002",
                        "name": "Data Structures",
                        "subtopics": [
                            {"topic_id": "t-002-1", "name": "Arrays and Lists"},
                            {"topic_id": "t-002-2", "name": "Hash Maps"},
                        ],
                    },
                    {
                        "topic_id": "t-003",
                        "name": "Software Design",
                        "subtopics": [
                            {"topic_id": "t-003-1", "name": "Design Patterns"},
                            {"topic_id": "t-003-2", "name": "SOLID Principles"},
                        ],
                    },
                ],
            })
        else:
            # Insufficient path: probe returns sufficient=False with gap analysis
            web_mode = config.web_research_mode
            autonomy = web_mode == WebResearchMode.AUTO
            mb.set_tool_call_responses("classification.react_sufficiency", [
                {
                    "tool_calls": [],
                    "content": {
                        "sufficient": False,
                        "reason": "Insufficient material: too few chunks to cover required question count",
                        "gap_analysis": [
                            {
                                "topic": "Object-Oriented Programming",
                                "current_depth": "surface",
                                "required_depth": "moderate",
                                "gap_description": "Need more content on OOP principles and design patterns",
                                "fillable_by_web": True,
                                "confidence": 0.8,
                            },
                            {
                                "topic": "Data Structures and Algorithms",
                                "current_depth": "surface",
                                "required_depth": "deep",
                                "gap_description": "No content on advanced data structures or algorithm analysis",
                                "fillable_by_web": True,
                                "confidence": 0.75,
                            },
                        ],
                        "search_queries": [
                            "OOP design patterns tutorial",
                            "data structures algorithms comprehensive guide",
                        ] if autonomy else [],
                        "autonomy_exercised": autonomy,
                        "rubric_fitness": "NO_RUBRIC",
                        "rubric_reasoning": "",
                        "rubric_source": "none",
                    },
                }
            ])

    # Build domain components — unified probe handles sufficiency + rubric
    topic_extractor = TopicExtractor(model_broker=mb)
    react_prober_factory = build_react_prober_factory(
        model_broker=mb,
        knowledge_service=ks,
    )

    # Build service
    service = ClassificationService(
        knowledge_service=ks,
        assessment_config=ac,
        topic_extractor=topic_extractor,
        decision_audit=da,
        event_publisher=ep,
        react_prober_factory=react_prober_factory,
    )

    # Execute
    request = prepare_request(case)
    response = await service.classify(request)

    # Serialize response
    result = response.model_dump()

    # Add derived fields for evaluation convenience
    result["has_topics"] = response.topics is not None and len(response.topics.topics) > 0
    result["has_gap_analysis"] = len(response.gap_analysis) > 0
    result["has_search_queries"] = len(response.search_queries) > 0
    result["topic_count"] = len(response.topics.topics) if response.topics else 0

    return result


# Per-test result tracking for O+ dashboard accordion
_test_details: list[dict] = []


def pytest_runtest_logreport(report):
    """Capture per-test results for detailed stats output."""
    if report.when == "call":
        _test_details.append({
            "name": report.nodeid.split("::")[-1],
            "module": report.nodeid.split("::")[0].split("/")[-1].replace(".py", ""),
            "passed": report.passed,
            "duration_s": round(report.duration, 3),
            "message": str(report.longrepr).split("\n")[-1] if report.failed else None,
        })


def pytest_sessionfinish(session, exitstatus):
    """Print LLM Router (CI) usage summary when running in live mode."""
    live = _os.environ.get("EVAL_LIVE_MODE", "").lower() in ("true", "1", "yes")
    if not live or not _live_model_brokers:
        return

    reqs = sum(a.request_count for a in _live_model_brokers)
    prompt_tokens = sum(a.total_prompt_tokens for a in _live_model_brokers)
    completion_tokens = sum(a.total_completion_tokens for a in _live_model_brokers)
    total_tokens = sum(a.total_tokens for a in _live_model_brokers)
    cost = sum(a.total_cost_usd for a in _live_model_brokers)
    tests = session.testscollected
    failed = getattr(session, "testsfailed", 0)

    print("\n")
    print("=" * 70)
    print("  DeepEval Quality Gate — LLM Router (CI) Usage Summary")
    print("=" * 70)
    print(f"  Tests:              {tests}")
    print(f"  LLM Router calls:   {reqs} ({reqs / tests:.1f} per test)" if tests else "")
    print(f"  Prompt Tokens:      {prompt_tokens:,}")
    print(f"  Completion Tokens:  {completion_tokens:,}")
    print(f"  Total Tokens:       {total_tokens:,}")
    print(f"  Cost (USD):         ${cost:.6f}")
    print(f"  Route:              {_os.environ.get('MODEL_BROKER_URL', 'http://localhost:8010')}")
    print("=" * 70)

    stats_path = _Path(__file__).parent / "last_run_stats.json"
    stats_path.write_text(_json.dumps({
        "tests": tests,
        "passed": tests - failed,
        "failed": failed,
        "request_count": reqs,
        "requests_per_test": round(reqs / tests, 1) if tests else 0,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost, 6),
        "route": _os.environ.get("MODEL_BROKER_URL", "http://localhost:8010"),
        "mode": "live",
        "details": _test_details,
    }, indent=2))
