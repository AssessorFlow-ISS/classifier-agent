"""Adversarial test session fixtures.

Sums LLM token/cost stats from all ModelBrokerHttpAdapter instances
created during the session and prints a summary + writes JSON for
the O+ dashboard to consume.

Enhanced: per-category breakdown mapped to Starter Kit 5 risk categories.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ── Map test class names to Starter Kit risk categories ──
_CLASS_TO_RISK = {
    "TestPromptInjectionResistance": "adversarial_prompts",
    "TestJailbreakResistance": "inappropriate_content",
    "TestPiiLeakageResistance": "data_leakage",
    "TestHallucinationResistance": "hallucination",
    "TestRagPoisoningResistance": "adversarial_prompts",
    "TestGoalHijackResistance": "adversarial_prompts",
    "TestToolAbuseResistance": "adversarial_prompts",
    "TestBiasResistance": "bias",
    "TestCombinedAttackResistance": "adversarial_prompts",
    "TestOcrSourceTypeAdversarial": "adversarial_prompts",
}

# Per-test results collected during session
_per_test_results: list[dict] = []

# Per-test LLM response capture (tests register via adv_response fixture)
_current_response: dict = {}


@pytest.fixture(autouse=True)
def _adv_response_reset():
    """Reset response capture before each test."""
    _current_response.clear()
    yield


@pytest.fixture
def adv_response():
    """Fixture for tests to register the LLM response for O+ drill-down.

    Usage in test:
        async def test_example(self, payload, adv_response):
            result = await checker.check(...)
            adv_response(payload=payload, response=result.reason)
            assert result.sufficient is False
    """
    def _capture(*, payload: str = "", response: str = ""):
        _current_response["payload"] = payload
        _current_response["response"] = response
    return _capture


def pytest_runtest_logreport(report):
    """Collect per-test pass/fail with category info and LLM response."""
    if report.when != "call":
        return
    # Extract class name from nodeid: tests/adversarial/test_adversarial.py::TestClassName::test_name[param]
    parts = report.nodeid.split("::")
    class_name = parts[1] if len(parts) >= 3 else "Unknown"
    risk = _CLASS_TO_RISK.get(class_name, "other")

    # Extract payload from parametrized test ID: test_name[payload_text...]
    payload_text = ""
    if "[" in report.nodeid:
        payload_text = report.nodeid.split("[", 1)[1].rstrip("]")[:200]

    entry: dict = {
        "nodeid": report.nodeid,
        "class": class_name,
        "test": parts[2].split("[")[0] if len(parts) >= 3 else "",
        "risk": risk,
        "passed": report.passed,
        "failed": report.failed,
        "payload": _current_response.get("payload") or payload_text,
    }
    # Attach captured LLM response if available
    if _current_response.get("response"):
        entry["response"] = _current_response["response"][:500]
    # On failure, capture assertion message as fallback
    if report.failed and report.longreprtext:
        entry["failure_detail"] = report.longreprtext[:300]

    # Per-test model attribution (read from live adapter if available)
    if os.environ.get("EVAL_LIVE_MODE", "").lower() in ("true", "1", "yes"):
        try:
            from tests.adversarial.test_adversarial import _live_adapters
            if _live_adapters:
                adapter = _live_adapters[-1]
                entry["model_used"] = getattr(adapter, "last_model_used", "")
                entry["model_tier"] = getattr(adapter, "last_model_tier", "")
        except ImportError:
            pass

    _per_test_results.append(entry)


def pytest_sessionfinish(session, exitstatus):
    """Print cumulative LLM usage summary and write JSON for O+ dashboard."""
    live = os.environ.get("EVAL_LIVE_MODE", "").lower() in ("true", "1", "yes")

    # Always write per-category breakdown (even in stub mode)
    categories: dict[str, dict] = defaultdict(lambda: {"tests": 0, "passed": 0, "failed": 0})
    for r in _per_test_results:
        cat = categories[r["risk"]]
        cat["tests"] += 1
        if r["passed"]:
            cat["passed"] += 1
        if r["failed"]:
            cat["failed"] += 1

    tests = session.testscollected
    failed = getattr(session, "testsfailed", 0)

    stats: dict = {
        "tests": tests,
        "passed": tests - failed,
        "failed": failed,
        "mode": "live" if live else "stub",
        "date": datetime.now(timezone.utc).isoformat(),
        "categories": dict(categories),
    }

    if live:
        # Import the adapter registry from the test module
        try:
            from tests.adversarial.test_adversarial import _live_adapters
        except ImportError:
            _live_adapters = []

        reqs = sum(a.request_count for a in _live_adapters)
        prompt_tokens = sum(a.total_prompt_tokens for a in _live_adapters)
        completion_tokens = sum(a.total_completion_tokens for a in _live_adapters)
        total_tokens = sum(a.total_tokens for a in _live_adapters)
        cost = sum(a.total_cost_usd for a in _live_adapters)

        # Aggregate per-model breakdown across all adapters
        merged_models: dict[str, dict] = {}
        for adapter in _live_adapters:
            for model_id, pm in getattr(adapter, "_per_model", {}).items():
                if model_id not in merged_models:
                    merged_models[model_id] = {
                        "request_count": 0, "prompt_tokens": 0, "completion_tokens": 0,
                        "total_tokens": 0, "cost_usd": 0.0, "tests_passed": 0, "tests_failed": 0,
                    }
                for k in ("request_count", "prompt_tokens", "completion_tokens", "total_tokens"):
                    merged_models[model_id][k] += pm.get(k, 0)
                merged_models[model_id]["cost_usd"] += pm.get("cost_usd", 0.0)

        # Count per-model pass/fail from per-test results
        for r in _per_test_results:
            mid = r.get("model_used", "")
            if mid and mid in merged_models:
                if r.get("passed"):
                    merged_models[mid]["tests_passed"] += 1
                if r.get("failed"):
                    merged_models[mid]["tests_failed"] += 1

        stats.update({
            "request_count": reqs,
            "requests_per_test": round(reqs / tests, 1) if tests else 0,
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 6),
            "route": os.environ.get("MODEL_BROKER_URL", "http://localhost:8010"),
            "models": {mid: {k: round(v, 6) if isinstance(v, float) else v for k, v in pm.items()} for mid, pm in merged_models.items()},
        })

        print("\n")
        print("=" * 70)
        print("  LLM Router (CI) Usage Summary")
        print("=" * 70)
        print(f"  Tests:              {tests}")
        print(f"  LLM Requests:       {reqs} ({reqs / tests:.1f} per test)" if tests else "")
        print(f"  Prompt Tokens:      {prompt_tokens:,}")
        print(f"  Completion Tokens:  {completion_tokens:,}")
        print(f"  Total Tokens:       {total_tokens:,}")
        print(f"  Cost (USD):         ${cost:.4f}")
        print(f"  Route:              {os.environ.get('MODEL_BROKER_URL', 'http://localhost:8010')}")
        if merged_models:
            print(f"  Models:             {', '.join(merged_models.keys())}")
        print("=" * 70)
    else:
        stats.update({
            "request_count": 0,
            "requests_per_test": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0,
            "route": "stub",
        })

    # Print per-category summary
    print("\n  Per-category breakdown:")
    for risk, counts in sorted(categories.items()):
        status = "PASS" if counts["failed"] == 0 else "FAIL"
        print(f"    {risk}: {counts['passed']}/{counts['tests']} passed [{status}]")

    # Write stats to JSON for O+ dashboard consumption
    stats_path = Path(__file__).parent / "last_run_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\n  Stats written to {stats_path}")

    # Write per-test details for O+ drill-down (payload, pass/fail, response)
    if _per_test_results:
        responses_path = Path(__file__).parent / "last_run_responses.json"
        responses_path.write_text(json.dumps(_per_test_results, indent=2))
        print(f"  Per-test details written to {responses_path} ({len(_per_test_results)} entries)")
