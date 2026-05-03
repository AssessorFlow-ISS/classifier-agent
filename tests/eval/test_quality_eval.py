"""DeepEval quality evaluation for Classification Agent (#4).

Two modes:
  SKIP (default, $0 cost):
    .venv/bin/python -m pytest tests/eval/test_quality_eval.py -v
    Skips all tests (no API key or EVAL_LIVE_MODE not set).
    For deterministic CI tests, use test_golden_validation.py instead.

  LIVE MODE (real LLM-as-Judge, ~$0.30 per run):
    EVAL_LIVE_MODE=true OPENAI_API_KEY=sk-... .venv/bin/python -m pytest tests/eval/test_quality_eval.py -v

CI gate: blocks merge if any metric drops >5 points from baseline.json.
Switch: EVAL_LIVE_MODE=true + OPENAI_API_KEY to enable (costs money).
Default: skipped (zero cost).

For Walfa: Set EVAL_LIVE_MODE=true in the CI pipeline ONLY for milestone releases.
"""
from __future__ import annotations

import json
import os

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from tests.eval.conftest import load_golden, run_classification_pipeline

# Skip all tests unless BOTH EVAL_LIVE_MODE and OPENAI_API_KEY are set
_live = os.environ.get("EVAL_LIVE_MODE", "").lower() in ("true", "1", "yes")
_has_key = bool(os.environ.get("OPENAI_API_KEY"))
pytestmark = pytest.mark.skipif(
    not (_live and _has_key),
    reason="EVAL_LIVE_MODE not set or OPENAI_API_KEY missing; skipping LLM-as-Judge tests (set both to enable, costs ~$0.30)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarize_input(case: dict) -> str:
    """Create a concise input summary for the LLM evaluator."""
    cfg = case["input"]["config"]
    chunks_source = case["input"].get("chunks_fixture", "inline")
    rubric = case["input"].get("rubric_setup")
    rubric_desc = (
        f"rubric_type={rubric['policy_type']}, source={rubric['source']}"
        if rubric
        else "no rubric"
    )
    return (
        f"Assessment: {cfg['assessment_title']}\n"
        f"MCQ: {cfg['structured_question_count']}, "
        f"OE: {cfg['non_structured_question_count']}, "
        f"Difficulty: {cfg['difficulty_level']}\n"
        f"Web research mode: {cfg.get('web_research_mode', 'manual')}\n"
        f"Chunks source: {chunks_source}\n"
        f"Rubric: {rubric_desc}"
    )


def _get_metrics():
    """Import metrics lazily to avoid LLM client init at collection time."""
    from tests.eval.metrics import (
        get_autonomy_decision,
        get_classification_correctness,
        get_rubric_alignment,
        get_topic_extraction,
    )
    return {
        "correctness": get_classification_correctness(),
        "rubric": get_rubric_alignment(),
        "topics": get_topic_extraction(),
        "autonomy": get_autonomy_decision(),
    }


# ---------------------------------------------------------------------------
# 1. Sufficient + Aligned rubric tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    load_golden("sufficient_aligned.json"),
    ids=lambda c: c["test_id"],
)
@pytest.mark.asyncio
async def test_sufficient_aligned_correctness(case: dict) -> None:
    """Verify classification correctness for sufficient material with aligned rubric."""
    metrics = _get_metrics()
    actual = await run_classification_pipeline(case)
    test_case = LLMTestCase(
        input=_summarize_input(case),
        actual_output=json.dumps(actual, default=str),
        expected_output=json.dumps(case["expected_output"], default=str),
    )
    assert_test(test_case, [metrics["correctness"]])


@pytest.mark.parametrize(
    "case",
    [c for c in load_golden("sufficient_aligned.json") if c["expected_output"].get("rubric_fitness")],
    ids=lambda c: c["test_id"],
)
@pytest.mark.asyncio
async def test_sufficient_aligned_rubric(case: dict) -> None:
    """Verify rubric alignment assessment for sufficient material with aligned rubric."""
    metrics = _get_metrics()
    actual = await run_classification_pipeline(case)
    test_case = LLMTestCase(
        input=_summarize_input(case),
        actual_output=json.dumps(actual, default=str),
        expected_output=json.dumps(case["expected_output"], default=str),
    )
    assert_test(test_case, [metrics["rubric"]])


@pytest.mark.parametrize(
    "case",
    [c for c in load_golden("sufficient_aligned.json") if c["expected_output"].get("has_topics")],
    ids=lambda c: c["test_id"],
)
@pytest.mark.asyncio
async def test_sufficient_aligned_topics(case: dict) -> None:
    """Verify topic extraction quality for sufficient material."""
    metrics = _get_metrics()
    actual = await run_classification_pipeline(case)
    test_case = LLMTestCase(
        input=_summarize_input(case),
        actual_output=json.dumps(actual, default=str),
        expected_output=json.dumps(case["expected_output"], default=str),
    )
    assert_test(test_case, [metrics["topics"]])


# ---------------------------------------------------------------------------
# 2. Sufficient + Misaligned rubric tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    load_golden("sufficient_misaligned.json"),
    ids=lambda c: c["test_id"],
)
@pytest.mark.asyncio
async def test_sufficient_misaligned_correctness(case: dict) -> None:
    """Verify classification correctness for sufficient material with misaligned rubric."""
    metrics = _get_metrics()
    actual = await run_classification_pipeline(case)
    test_case = LLMTestCase(
        input=_summarize_input(case),
        actual_output=json.dumps(actual, default=str),
        expected_output=json.dumps(case["expected_output"], default=str),
    )
    assert_test(test_case, [metrics["correctness"]])


@pytest.mark.parametrize(
    "case",
    load_golden("sufficient_misaligned.json"),
    ids=lambda c: c["test_id"],
)
@pytest.mark.asyncio
async def test_sufficient_misaligned_rubric(case: dict) -> None:
    """Verify rubric alignment correctly identifies misalignment."""
    metrics = _get_metrics()
    actual = await run_classification_pipeline(case)
    test_case = LLMTestCase(
        input=_summarize_input(case),
        actual_output=json.dumps(actual, default=str),
        expected_output=json.dumps(case["expected_output"], default=str),
    )
    assert_test(test_case, [metrics["rubric"]])


# ---------------------------------------------------------------------------
# 3. Insufficient + Auto web research tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    load_golden("insufficient_auto.json"),
    ids=lambda c: c["test_id"],
)
@pytest.mark.asyncio
async def test_insufficient_auto_correctness(case: dict) -> None:
    """Verify classification correctness for insufficient material with auto web research."""
    metrics = _get_metrics()
    actual = await run_classification_pipeline(case)
    test_case = LLMTestCase(
        input=_summarize_input(case),
        actual_output=json.dumps(actual, default=str),
        expected_output=json.dumps(case["expected_output"], default=str),
    )
    assert_test(test_case, [metrics["correctness"]])


@pytest.mark.parametrize(
    "case",
    load_golden("insufficient_auto.json"),
    ids=lambda c: c["test_id"],
)
@pytest.mark.asyncio
async def test_insufficient_auto_autonomy(case: dict) -> None:
    """Verify autonomy decision for insufficient material with auto web research."""
    metrics = _get_metrics()
    actual = await run_classification_pipeline(case)
    test_case = LLMTestCase(
        input=_summarize_input(case),
        actual_output=json.dumps(actual, default=str),
        expected_output=json.dumps(case["expected_output"], default=str),
    )
    assert_test(test_case, [metrics["autonomy"]])


# ---------------------------------------------------------------------------
# 4. Insufficient + Manual/Disabled web research tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    load_golden("insufficient_manual.json"),
    ids=lambda c: c["test_id"],
)
@pytest.mark.asyncio
async def test_insufficient_manual_correctness(case: dict) -> None:
    """Verify classification correctness for insufficient material without auto web research."""
    metrics = _get_metrics()
    actual = await run_classification_pipeline(case)
    test_case = LLMTestCase(
        input=_summarize_input(case),
        actual_output=json.dumps(actual, default=str),
        expected_output=json.dumps(case["expected_output"], default=str),
    )
    assert_test(test_case, [metrics["correctness"]])


@pytest.mark.parametrize(
    "case",
    load_golden("insufficient_manual.json"),
    ids=lambda c: c["test_id"],
)
@pytest.mark.asyncio
async def test_insufficient_manual_autonomy(case: dict) -> None:
    """Verify autonomy is NOT exercised for manual/disabled web research modes."""
    metrics = _get_metrics()
    actual = await run_classification_pipeline(case)
    test_case = LLMTestCase(
        input=_summarize_input(case),
        actual_output=json.dumps(actual, default=str),
        expected_output=json.dumps(case["expected_output"], default=str),
    )
    assert_test(test_case, [metrics["autonomy"]])


# ---------------------------------------------------------------------------
# 5. Edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    load_golden("edge_cases.json"),
    ids=lambda c: c["test_id"],
)
@pytest.mark.asyncio
async def test_edge_case_correctness(case: dict) -> None:
    """Verify classification correctness for edge cases."""
    metrics = _get_metrics()
    actual = await run_classification_pipeline(case)
    test_case = LLMTestCase(
        input=_summarize_input(case),
        actual_output=json.dumps(actual, default=str),
        expected_output=json.dumps(case["expected_output"], default=str),
    )
    assert_test(test_case, [metrics["correctness"]])
