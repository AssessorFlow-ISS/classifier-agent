"""Per-agent GEval smoke quality evaluation — Classification Agent.

Runs 1 golden case per category (5 total) through all 4 GEval quality
metrics via a real Model Broker. Captures per-metric scores and writes
them to baseline-classification.json for the O+ dashboard.

Usage:
  cd ~/Desktop/5008-workspace/classification-agent
  EVAL_LIVE_MODE=true \
  MODEL_BROKER_URL=http://localhost:8010 \
  .venv/bin/python tests/eval/smoke_quality_eval.py

Pre-requisites:
  - Model Broker running at :8010
  - OPENAI_API_KEY or GOOGLE_AI_API_KEY set (for GEval judge LLM)
  - Docker stack up (PostgreSQL, Redis, Langfuse)

Cost: ~$0.02 to $0.05 USD (5 pipeline calls + 20 judge calls)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# noqa: E402 — imports below intentionally placed after sys.path manipulation
from tests.eval.conftest import load_golden, run_classification_pipeline, prepare_chunks  # noqa: E402
from tests.eval.metrics import (  # noqa: E402
    get_autonomy_decision,
    get_classification_correctness,
    get_faithfulness,
    get_rubric_alignment,
    get_topic_extraction,
)

from deepeval.test_case import LLMTestCase  # noqa: E402

# ── Configuration ──

SMOKE_CASES = {
    "sufficient_aligned": "cls-gold-001",
    "sufficient_misaligned": "cls-gold-010",
    "insufficient_auto": "cls-gold-020",
    "insufficient_manual": "cls-gold-030",
    "edge_cases": "cls-gold-040",
}

METRICS = {
    "classification_correctness": get_classification_correctness,
    "rubric_alignment": get_rubric_alignment,
    "topic_extraction": get_topic_extraction,
    "autonomy_decision": get_autonomy_decision,
    "faithfulness": get_faithfulness,
}

# Which metrics apply to which categories
# faithfulness included for all categories that read from RAG (chunks as retrieval_context)
METRIC_APPLICABILITY = {
    "sufficient_aligned": ["classification_correctness", "rubric_alignment", "topic_extraction", "faithfulness"],
    "sufficient_misaligned": ["classification_correctness", "rubric_alignment", "topic_extraction", "faithfulness"],
    "insufficient_auto": ["classification_correctness", "autonomy_decision", "faithfulness"],
    "insufficient_manual": ["classification_correctness", "autonomy_decision", "faithfulness"],
    "edge_cases": ["classification_correctness", "autonomy_decision"],  # edge: empty chunks, no retrieval_context
}

BASELINE_DIR = os.environ.get(
    "BASELINE_DIR",
    str(Path.home() / "Desktop/5008-workspace/UI_e2eTesting"),
)


def _summarize_input(case: dict) -> str:
    """Create a concise input summary for the LLM evaluator."""
    cfg = case["input"]["config"]
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
        f"Rubric: {rubric_desc}"
    )


async def run_smoke():
    """Execute the smoke quality evaluation."""
    print("=" * 70)
    print("  Classification Agent — Smoke Quality Evaluation")
    print("  5 golden cases x 4 GEval metrics (per-agent baseline)")
    print("=" * 70)
    print()

    # Verify live mode
    live = os.environ.get("EVAL_LIVE_MODE", "").lower() in ("true", "1", "yes")
    if not live:
        print("ERROR: Set EVAL_LIVE_MODE=true to enable real LLM calls.")
        print("       Without this, the pipeline uses stub adapters (deterministic).")
        sys.exit(1)

    broker_url = os.environ.get("MODEL_BROKER_URL", "http://localhost:8010")
    print(f"  Model Broker:  {broker_url}")
    print(f"  Baseline dir:  {BASELINE_DIR}")
    print()

    # Collect per-metric scores
    metric_scores: dict[str, list[float]] = {k: [] for k in METRICS}
    case_results: list[dict] = []
    total_start = time.time()

    for category, case_id in SMOKE_CASES.items():
        # Load the target case
        cases = load_golden(f"{category}.json")
        case = next((c for c in cases if c["test_id"] == case_id), None)
        if not case:
            print(f"  SKIP: {case_id} not found in {category}.json")
            continue

        print(f"  ── {case_id} ({category}) ──")
        case_start = time.time()

        # Run pipeline (real LLM via Model Broker)
        try:
            actual = await run_classification_pipeline(case)
        except Exception as e:
            print(f"    PIPELINE ERROR: {e}")
            case_results.append({"case_id": case_id, "category": category, "error": str(e)})
            continue

        pipeline_time = time.time() - case_start
        print(f"    Pipeline: {pipeline_time:.1f}s — sufficient={actual.get('sufficient')}")

        # Load fixture chunks as retrieval_context for Faithfulness metric
        retrieval_ctx: list[str] = []
        try:
            chunks = prepare_chunks(case)
            retrieval_ctx = [c.content for c in chunks if c.content]
        except Exception:
            pass  # edge cases with no chunks

        # Build LLM test case for GEval judge (with RAG grounding context)
        test_case = LLMTestCase(
            input=_summarize_input(case),
            actual_output=json.dumps(actual, default=str),
            expected_output=json.dumps(case["expected_output"], default=str),
            retrieval_context=retrieval_ctx if retrieval_ctx else None,
        )

        # Evaluate applicable metrics
        applicable = METRIC_APPLICABILITY.get(category, list(METRICS.keys()))
        case_scores: dict[str, float | None] = {}

        for metric_key in applicable:
            metric_factory = METRICS[metric_key]
            metric = metric_factory()
            try:
                metric.measure(test_case)
                score = metric.score
                if score is not None:
                    metric_scores[metric_key].append(score)
                    case_scores[metric_key] = score
                    status = "PASS" if score >= metric.threshold else "WARN"
                    print(f"    {metric_key}: {score:.4f} ({status}, threshold={metric.threshold})")
                else:
                    case_scores[metric_key] = None
                    print(f"    {metric_key}: null (metric returned no score)")
            except Exception as e:
                case_scores[metric_key] = None
                print(f"    {metric_key}: ERROR — {e}")

        case_results.append({
            "case_id": case_id,
            "category": category,
            "sufficient": actual.get("sufficient"),
            "scores": case_scores,
            "pipeline_time_s": round(pipeline_time, 2),
        })
        print()

    total_time = time.time() - total_start

    # ── Compute averages ──
    print("=" * 70)
    print("  Per-Metric Averages (baseline candidates)")
    print("=" * 70)

    baseline: dict[str, float] = {}
    for metric_key, scores in metric_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            baseline[metric_key] = round(avg, 4)
            print(f"  {metric_key:30s}  {avg:.4f}  (n={len(scores)})")
        else:
            print(f"  {metric_key:30s}  --      (no data)")

    print()
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Cases run:  {len(case_results)}")
    print()

    # ── Write baseline ──
    baseline_path = Path(BASELINE_DIR) / "baseline-classification.json"

    # Preserve existing scores (e.g., answer_relevancy from golden-eval.py)
    existing: dict = {}
    if baseline_path.exists():
        try:
            existing = json.loads(baseline_path.read_text())
        except Exception:
            pass

    # Merge: new scores overwrite, existing scores preserved
    merged = {**existing, **baseline}
    baseline_path.write_text(json.dumps(merged, indent=2))
    print(f"  Baseline written: {baseline_path}")
    print(f"  Keys: {list(merged.keys())}")
    print()

    # ── Write detailed results ──
    results_path = Path(BASELINE_DIR) / "smoke-classification-results.json"
    results_path.write_text(json.dumps({
        "agent": "classification-agent",
        "type": "smoke_quality_eval",
        "cases": case_results,
        "averages": baseline,
        "total_time_s": round(total_time, 1),
        "model_broker_url": broker_url,
    }, indent=2, default=str))
    print(f"  Detailed results: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_smoke())
