"""Golden re-baseline runner for the classifier-agent.

Drives a smoke-level golden replay (one representative case per scenario),
collects all 9 drift-runner metrics, and OVERWRITES the canonical baseline
at gs://${GCS_BUCKET}/golden/baselines/baseline-classifier-agent.json.

Triggered by .github/workflows/golden-rebaseline.yml (weekly, Mon 14:00 SGT).

NO defaults: GCS_BUCKET, SCENARIO, MODEL_BROKER_URL, KS_URL must all be set
by the caller. Each invoked drift_runner.py also requires its own env. Hard
fail on any missing — there is no silent fallback in this script or its
sub-process invocations. All scoring goes through real LLM judges via the
Model Broker port-forward established by golden-rebaseline.yml.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

GCS_BUCKET = os.environ["GCS_BUCKET"]  # required — no default
AGENT_NAME = "classifier-agent"
BASELINE_KEY = f"golden/baselines/baseline-{AGENT_NAME}.json"


def _run_drift(drift_kind: str, scratch_dir: Path) -> dict:
    """Invoke scripts/drift_runner.py for one kind, return its current.json."""
    artifact_dir = scratch_dir / drift_kind
    subprocess.run(
        [
            sys.executable, "scripts/drift_runner.py",
            "--drift-kind", drift_kind,
            "--artifact-dir", str(artifact_dir),
        ],
        check=True,
    )
    return json.loads((artifact_dir / "current.json").read_text())


def main() -> int:
    # SCENARIO is required — CI sets via inputs.scenario || 'sufficient'.
    # No silent default in script: if CI forgets to pass it, fail loud.
    try:
        scenario = os.environ["SCENARIO"]
    except KeyError as exc:
        raise SystemExit(
            "SCENARIO env var is required (set by golden-rebaseline.yml). "
            "No script-level default — CI must pass it explicitly."
        ) from exc
    print(f"[golden_rebaseline] scenario={scenario}", file=sys.stderr)

    scratch = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "golden-rebaseline"
    scratch.mkdir(parents=True, exist_ok=True)

    drift_kinds = [
        "faithfulness", "answer-relevancy", "contextual-precision",
        "contextual-recall", "score-consistency", "llm-base",
        "bias", "canary-leak", "retrieval-poisoning",
    ]
    aggregated_scores: dict[str, float] = {}
    for kind in drift_kinds:
        result = _run_drift(kind, scratch)
        aggregated_scores.update(result["scores"])

    new_baseline = {
        "agent": AGENT_NAME,
        "scenario": scenario,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scores": aggregated_scores,
        "source": "golden-rebaseline.yml",
        "git_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "run_url": (
            f"{os.environ.get('GITHUB_SERVER_URL', '')}/"
            f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/"
            f"{os.environ.get('GITHUB_RUN_ID', '')}"
        ),
    }
    baseline_path = scratch / "new-baseline.json"
    baseline_path.write_text(json.dumps(new_baseline, indent=2))

    gcs_uri = f"gs://{GCS_BUCKET}/{BASELINE_KEY}"
    subprocess.run(["gsutil", "cp", str(baseline_path), gcs_uri], check=True)
    print(f"[golden_rebaseline] uploaded new baseline to {gcs_uri}", file=sys.stderr)

    summary_path = Path(os.environ.get("GITHUB_STEP_SUMMARY", "/tmp/_summary.md"))
    rows = [
        f"| {kind} | {round(score, 4)} |"
        for kind, score in sorted(aggregated_scores.items())
    ]
    body = "\n".join([
        f"## Golden — Rebaseline / {AGENT_NAME} ({scenario})",
        "",
        f"**New baseline written to** `{gcs_uri}`  ",
        f"**Captured:** {new_baseline['captured_at']}  ",
        f"**Scenario:** {scenario}  ",
        "",
        "| Drift kind | Score |",
        "|---|---|",
        *rows,
        "",
        "Subsequent drift workflows will compare against this baseline.",
    ])
    with summary_path.open("a") as f:
        f.write(body + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
