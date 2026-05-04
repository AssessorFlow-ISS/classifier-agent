# Classifier Agent — Test Run Tracking (real LLM call confirmation)

> Stock-take ledger for whether each scheduled CI/drift run actually completed real LLM calls (i.e. NOT a hollow pass via `|| true`, placeholder fallback, regex stand-in, or unexpanded env-var). Update after each run completes; one row per scenario per run.
>
> Methodology: see [TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md).
> Operational state: see [HONEST_GOLDEN_TESTING.md](HONEST_GOLDEN_TESTING.md).

## How to mark a row "Real LLM ✅"

A run counts as a real LLM call if and only if ALL of the following are true:

1. The job exited 0 (no `|| true` masking)
2. The artifact (`deepeval-report/`, `llm-adversarial-report/`) contains real `score`, `reason`, `model_used` fields — NOT `{"skipped": true}` or absent
3. The script log shows `[info] model_broker_tool_request task_key=…` lines proving HTTP calls landed at the broker
4. (Promptfoo) `promptfoo.expanded.yaml` was generated and passes the `${VAR}`-not-left-in check

If any of those is missing, mark `Hollow ❌` and link the run for forensic review.

Verify command:

```bash
# Pull the latest run for a workflow and grep its log for real-LLM signals
id=$(gh run list -R AssessorFlow-ISS/classifier-agent --workflow=ci.yml -L 1 --json databaseId --jq '.[0].databaseId')
gh run view "$id" -R AssessorFlow-ISS/classifier-agent --log \
  | grep -iE "model_broker_tool_request|model_used|judge_model|deepeval|deepteam|promptfoo"
```

## Tool 1 — DeepEval (CI: 4 quality + Drift: 5 = 9 scenarios)

### CI — pre-merge (every push to main)

| # | Scenario | Workflow | Last run | Run ID | Real LLM | Notes |
|---|---|---|---|---|---|---|
| 1 | Faithfulness | `ci.yml` → DeepEval step | (pending verification of `6475188` push) | — | ⏳ | First post-fix push |
| 2 | Answer relevancy | `ci.yml` → DeepEval step | — | — | ⏳ | |
| 3 | Contextual precision | `ci.yml` → DeepEval step | — | — | ⏳ | |
| 4 | Contextual recall | `ci.yml` → DeepEval step | — | — | ⏳ | |

### Drift — weekly Mon SGT (one workflow per scenario)

| # | Scenario | Workflow | Last scheduled fire (UTC) | Last run | Run ID | Real LLM | Notes |
|---|---|---|---|---|---|---|---|
| 5 | `score-consistency` | `drift-score-consistency.yml` | Mon 02:00 UTC | — | — | ⏳ | Post-fix not yet fired |
| 6 | `llm-base` | `drift-llm-base.yml` | Mon 04:00 UTC | — | — | ⏳ | |
| 7 | `bias` | `drift-bias.yml` | Sun 23:00 UTC | — | — | ⏳ | |
| 8 | `canary-leak` | `drift-canary-leak.yml` | Mon 03:00 UTC | — | — | ⏳ | UUID canary per `_fresh_canary()` |
| 9 | `retrieval-poisoning` | `drift-retrieval-poisoning.yml` | Mon 05:00 UTC | — | — | ⏳ | Real poisoned chunk in fixture |

Plus drift coverage of the 4 quality metrics (separate workflows, same `quality.json` fixture):

| # | Scenario | Workflow | Last scheduled fire (UTC) | Last run | Run ID | Real LLM | Notes |
|---|---|---|---|---|---|---|---|
| 1d | Faithfulness drift | `drift-faithfulness.yml` | Sun 21:00 UTC | run 25300735952 | 25300735952 | ❌ Hollow (HTTP 500 KS pre-fix) | Pre-fix; expected pass after `6475188` |
| 2d | Answer relevancy drift | `drift-answer-relevancy.yml` | Sun 22:00 UTC | — | — | ⏳ | |
| 3d | Contextual precision drift | `drift-contextual-precision.yml` | Mon 00:00 UTC | — | — | ⏳ | |
| 4d | Contextual recall drift | `drift-contextual-recall.yml` | Mon 01:00 UTC | — | — | ⏳ | |

## Tool 2 — DeepTeam (8 vulnerability categories, 1 PromptInjection attack each)

Runs in `ci.yml` → "LLM — DeepTeam + Promptfoo" job → "DeepTeam — 8 vulnerability categories against REAL Model Broker" step. Every push to main.

| # | Vulnerability | Type literal | Last run | Run ID | Real LLM | Notes |
|---|---|---|---|---|---|---|
| 1 | Bias | `gender` | (pending verification of `6475188` push) | — | ⏳ | |
| 2 | Toxicity | `insults` | — | — | ⏳ | |
| 3 | PIILeakage | `direct_disclosure` | — | — | ⏳ | Pre-fix crashed at `"direct disclosure"` ValueError |
| 4 | PromptLeakage | `instructions` | — | — | ⏳ | |
| 5 | ExcessiveAgency | `functionality` | — | — | ⏳ | |
| 6 | Misinformation | `factual_errors` | — | — | ⏳ | NEW in `6475188` |
| 7 | IllegalActivity | `cybercrime` | — | — | ⏳ | NEW in `6475188` |
| 8 | IntellectualProperty | `copyright_violations` | — | — | ⏳ | NEW in `6475188` |

Pre-fix run **25301087185**: 0/8 real LLM calls (script crashed at vuln #3, masked by `|| true` + placeholder fallback).

## Tool 3 — Promptfoo (3 OWASP categories, non-overlapping with DeepTeam)

Runs in `ci.yml` → "LLM — DeepTeam + Promptfoo" job → "Promptfoo — 3 OWASP categories…" step. Every push to main.

| # | Scenario | OWASP | Last run | Run ID | Real LLM | Notes |
|---|---|---|---|---|---|---|
| 1 | XSS payload in classifier output | LLM05 (Improper Output Handling) | (pending verification of `6475188` push) | — | ⏳ | |
| 2 | Cross-context retrieval leak probe | LLM08 (Vector/Embedding Weaknesses) | — | — | ⏳ | |
| 3 | Unbounded recursive expansion | LLM10 (Unbounded Consumption) | — | — | ⏳ | |

Pre-fix run **25301087185**: 0/3 real LLM calls (`${MODEL_BROKER_URL}` never expanded → URL TypeError → silent "3/3 pass").

## Tool 4 — DeepEval golden-rebaseline (composite drift refresh)

Runs `golden-rebaseline.yml` weekly Mon 14:00 SGT (06:00 UTC). Re-fires all 9 drift kinds and overwrites the canonical baseline at `gs://thet-integration-af-assessorflow-materials/golden/baselines/baseline-classifier-agent.json`.

| Scheduled date | Run ID | All 9 drifts real LLM? | New baseline written? |
|---|---|---|---|
| (pending) | — | ⏳ | — |

## Tool 5 — E2E Golden Pipeline (cross-agent, real validator-write + KS)

Runs `e2e-golden-pipeline.yml` weekly Mon 14:00 SGT (06:00 UTC), workflow_dispatch with `{insufficient, sufficient}`. Drives the full 13-stage workflow through every agent in af-platform.

| Scheduled date | Run ID | Scenario | Real LLM (per agent) | Notes |
|---|---|---|---|---|
| (pending) | — | — | ⏳ | First fire pending — secrets set 2026-05-04 04:34 UTC |

## Roll-up — last 7 days (target = all ✅)

| Tool | Scenarios | Real LLM | Hollow | Pending |
|---|---|---|---|---|
| DeepEval (CI) | 4 | 0 | 0 | 4 |
| DeepEval (drift) | 5 | 0 | 0 | 5 |
| DeepEval (drift coverage of CI metrics) | 4 | 0 | 1 | 3 |
| DeepTeam | 8 | 0 | 8 | 0 (rerun pending) |
| Promptfoo | 3 | 0 | 3 | 0 (rerun pending) |
| Golden rebaseline | 1 | 0 | 0 | 1 |
| E2E golden pipeline | 1 | 0 | 0 | 1 |
| **TOTAL** | **26** | **0** | **12** | **14** |

## Update protocol

After each new CI/drift/e2e run completes:

1. Run the verify command at the top of this doc
2. Locate the row(s) for that run
3. Update `Last run`, `Run ID`, `Real LLM` (✅/❌/⏳)
4. If ❌, add a one-line `Notes` describing the failure mode (e.g. "HTTP 500 from KS", "envsubst left `${VAR}`", "ValueError on type literal")
5. Update the roll-up table at the bottom
6. Commit with subject `docs(testing): update CLASSIFIER_TEST_RUNS.md (run <id>)`

If a row stays ⏳ for more than 8 days past its scheduled fire, that's a CI scheduler bug — open an issue.

## Forensic links — runs found to be hollow (don't lose these)

- **25291601980** — drift constant-collision (Baseline=Current=0.85). Pre-fix `score_map`. Fixed by `drift_runner` v3.
- **25300735952** — drift HTTP 500 KS error. Real workflow_id mismatch with synthetic chunks. Fixed by switching drift to stub KS + real Model Broker.
- **25301087185** — DeepTeam ValueError + Promptfoo URL-parse error, both masked by `|| true` and placeholder fallback. Fixed by `6475188`.
