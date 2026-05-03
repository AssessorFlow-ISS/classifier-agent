"""Deterministic golden dataset validation tests (no LLM required).

These tests validate that the classification pipeline produces correct
outputs for each golden test case using stub adapters. Unlike the DeepEval
tests (test_quality_eval.py), these do NOT require an LLM for evaluation
and can run in CI without API keys.

Run: .venv/bin/python -m pytest tests/eval/test_golden_validation.py -v
"""
from __future__ import annotations

import pytest

from tests.eval.conftest import load_golden, run_classification_pipeline


# ---------------------------------------------------------------------------
# 1. Sufficient + Aligned
# ---------------------------------------------------------------------------


class TestSufficientAligned:
    """Golden dataset: sufficient material with aligned assessor rubric."""

    @pytest.mark.parametrize(
        "case",
        load_golden("sufficient_aligned.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_sufficiency_verdict(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected = case["expected_output"]
        assert actual["sufficient"] is expected["sufficient"], (
            f"{case['test_id']}: expected sufficient={expected['sufficient']}, "
            f"got {actual['sufficient']}"
        )

    @pytest.mark.parametrize(
        "case",
        load_golden("sufficient_aligned.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_topic_count_in_range(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected = case["expected_output"]
        if expected.get("has_topics"):
            topic_range = expected.get("topic_count_range", [1, 99])
            assert topic_range[0] <= actual["topic_count"] <= topic_range[1], (
                f"{case['test_id']}: topic_count {actual['topic_count']} "
                f"not in range {topic_range}"
            )

    @pytest.mark.parametrize(
        "case",
        load_golden("sufficient_aligned.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_no_autonomy_exercised(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        assert actual["autonomy_exercised"] is False

    @pytest.mark.parametrize(
        "case",
        load_golden("sufficient_aligned.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_no_gap_analysis(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected = case["expected_output"]
        if not expected.get("has_gap_analysis", False):
            assert actual["gap_analysis"] == [], (
                f"{case['test_id']}: expected empty gap_analysis"
            )

    @pytest.mark.parametrize(
        "case",
        [c for c in load_golden("sufficient_aligned.json") if c["expected_output"].get("rubric_fitness")],
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_rubric_aligned(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected_rubric = case["expected_output"]["rubric_fitness"]
        assert actual["rubric_fitness"] is not None
        assert actual["rubric_fitness"]["is_aligned"] is expected_rubric["is_aligned"]
        assert actual["rubric_fitness"]["rubric_source"] == expected_rubric["rubric_source"]


# ---------------------------------------------------------------------------
# 2. Sufficient + Misaligned
# ---------------------------------------------------------------------------


class TestSufficientMisaligned:
    """Golden dataset: sufficient material with misaligned rubric."""

    @pytest.mark.parametrize(
        "case",
        load_golden("sufficient_misaligned.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_sufficiency_verdict(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        assert actual["sufficient"] is True

    @pytest.mark.parametrize(
        "case",
        load_golden("sufficient_misaligned.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_rubric_misaligned(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected_rubric = case["expected_output"]["rubric_fitness"]
        assert actual["rubric_fitness"] is not None
        assert actual["rubric_fitness"]["is_aligned"] is expected_rubric["is_aligned"]

    @pytest.mark.parametrize(
        "case",
        load_golden("sufficient_misaligned.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_rubric_source(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected_rubric = case["expected_output"]["rubric_fitness"]
        assert actual["rubric_fitness"]["rubric_source"] == expected_rubric["rubric_source"]


# ---------------------------------------------------------------------------
# 3. Insufficient + Auto
# ---------------------------------------------------------------------------


class TestInsufficientAuto:
    """Golden dataset: insufficient material with auto web research."""

    @pytest.mark.parametrize(
        "case",
        load_golden("insufficient_auto.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_insufficient_verdict(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        assert actual["sufficient"] is False

    @pytest.mark.parametrize(
        "case",
        load_golden("insufficient_auto.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_gap_analysis_populated(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected = case["expected_output"]
        min_gaps = expected.get("gap_analysis_min_count", 1)
        assert len(actual["gap_analysis"]) >= min_gaps, (
            f"{case['test_id']}: expected at least {min_gaps} gap entries, "
            f"got {len(actual['gap_analysis'])}"
        )

    @pytest.mark.parametrize(
        "case",
        load_golden("insufficient_auto.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_no_topics_extracted(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        assert actual["has_topics"] is False


# ---------------------------------------------------------------------------
# 4. Insufficient + Manual/Disabled
# ---------------------------------------------------------------------------


class TestInsufficientManual:
    """Golden dataset: insufficient material with manual/disabled web research."""

    @pytest.mark.parametrize(
        "case",
        load_golden("insufficient_manual.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_insufficient_verdict(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        assert actual["sufficient"] is False

    @pytest.mark.parametrize(
        "case",
        load_golden("insufficient_manual.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_no_autonomy(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        assert actual["autonomy_exercised"] is False

    @pytest.mark.parametrize(
        "case",
        load_golden("insufficient_manual.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_no_search_queries(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        assert actual["search_queries"] == []

    @pytest.mark.parametrize(
        "case",
        load_golden("insufficient_manual.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_gap_analysis_populated(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected = case["expected_output"]
        min_gaps = expected.get("gap_analysis_min_count", 1)
        assert len(actual["gap_analysis"]) >= min_gaps


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Golden dataset: edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "case",
        load_golden("edge_cases.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_sufficiency_verdict(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected = case["expected_output"]
        assert actual["sufficient"] is expected["sufficient"], (
            f"{case['test_id']}: expected sufficient={expected['sufficient']}, "
            f"got {actual['sufficient']}"
        )

    @pytest.mark.parametrize(
        "case",
        load_golden("edge_cases.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_topics_presence(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected = case["expected_output"]
        assert actual["has_topics"] is expected.get("has_topics", False), (
            f"{case['test_id']}: expected has_topics={expected.get('has_topics')}, "
            f"got {actual['has_topics']}"
        )

    @pytest.mark.parametrize(
        "case",
        [c for c in load_golden("edge_cases.json") if c["expected_output"].get("topic_count_range")],
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_topic_count_in_range(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        expected = case["expected_output"]
        topic_range = expected["topic_count_range"]
        assert topic_range[0] <= actual["topic_count"] <= topic_range[1], (
            f"{case['test_id']}: topic_count {actual['topic_count']} "
            f"not in range {topic_range}"
        )

    @pytest.mark.parametrize(
        "case",
        load_golden("edge_cases.json"),
        ids=lambda c: c["test_id"],
    )
    @pytest.mark.asyncio
    async def test_no_autonomy(self, case: dict) -> None:
        actual = await run_classification_pipeline(case)
        assert actual["autonomy_exercised"] is False
