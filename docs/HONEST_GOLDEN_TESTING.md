# Honest Golden Testing + Refactor — classifier-agent

> Persisted plan-of-record so future operators/agents can pick up without re-deriving the trade-offs. Last updated 2026-05-04.

## Context

This repo (`AssessorFlow-ISS/classifier-agent`) was created as a CI/CD-tier sister of the production `assessorflow/classification-agent`. Two correctness audits drove a substantial overhaul:

1. **Constant-collision audit (run #25291601980)** — drift workflows printed `Baseline = Current = 0.85`, `Delta = 0` for every run. Root cause: `scripts/drift_runner.py` returned the same hard-coded `score_map` dict for both the GCS-stored baseline AND the current-run scores. Drift detection was mathematically a no-op.
2. **No-localhost / no-cheating audit** — PROD source carried 9 localhost defaults (`config.py`, `submission_client.py`, `model_broker_http.py`, `knowledge_service_http.py`, `progress_emitter.py`) and a `dev_password` fallback. A misconfigured deploy would silently route to a non-existent local socket. Adversarial test runners (DeepTeam, Promptfoo) used regex pattern-match judges instead of real LLM judges.

Outcome: drift_runner rewritten with real DeepEval/DeepTeam LLM-judge calls; localhost defaults stripped repo-wide; new `e2e-golden-pipeline.yml` workflow added for true cross-agent validation.

## Three-tier testing architecture

| Tier | Workflow(s) | What it tests | Cost | Cadence |
|---|---|---|---|---|
| **0 — Pre-merge CI** | `ci.yml` (14-card pipeline ending in `Hand-off to Cloud Deploy`) | Lint, SAST, SCA, secrets, GHA-audit, unit, integration, DeepEval (LLM-judge quality), DeepTeam + Promptfoo (LLM-judge adversarial), DAST (ZAP), Container build + Trivy, GAR push, hand-off probe. Real LLM via Model Broker port-forward. Stub KS (no cluster KS roundtrip). | ~$0.10 / merge | every push to `main` |
| **1 — Per-agent drift** (smoke) | 9× `drift-*.yml` + `golden-rebaseline.yml` | Single-agent: classifier topic-extractor invoked with synthetic chunks (override). DeepEval/DeepTeam LLM-judge scores response. Real Model Broker, **stub KS**. Catches LLM-judge regression, prompt drift, classifier behavior drift. **Does NOT exercise** validator-write, real KS read/write, L-10 guardrails, Pub/Sub orchestration. | ~$0.05 / drift | weekly Mon SGT, hourly stagger 05:00–13:00; rebaseline at 14:00 |
| **2 — True end-to-end** | `e2e-golden-pipeline.yml` | Full 13-stage assessment driven via API server → real validator-agent writes chunks to KS → real classifier reads them via KS gRPC → Q&A Gen → Evaluator → Reporting. Real Pub/Sub, real L-10 guardrails, real Langfuse trace publish. Per-agent scores read from Langfuse. **This is the only path that exercises validator-write + real RAG retrieval + L-10.** | ~$0.10–0.30 / run | weekly Mon 14:00 SGT (06:00 UTC), workflow_dispatch with `{insufficient, sufficient}` |

## Per-tool real-LLM call audit (no cheating)

| Tool | Where wired | Judge type | Localhost in source? |
|---|---|---|---|
| **DeepEval** (FaithfulnessMetric / AnswerRelevancyMetric / ContextualPrecisionMetric / ContextualRecallMetric / BiasMetric) | `scripts/drift_runner.py` via `ModelBrokerJudge(DeepEvalBaseLLM)` | Real LLM through Model Broker port-forward at `${MODEL_BROKER_URL}` | No — `os.environ["MODEL_BROKER_URL"]` required, raises if missing |
| **DeepTeam** (Bias / Toxicity / PIILeakage / PromptLeakage / ExcessiveAgency / PromptInjection) | `scripts/deepteam_smoke.py` via `from deepteam import red_team` + `simulator_model=judge`, `evaluation_model=judge` | Real LLM through Model Broker (separate task_keys for telemetry separation) | No — same env requirement |
| **Promptfoo** (LLM01 prompt-injection, LLM02 PII disclosure, LLM06 excessive agency) | `promptfoo.yaml` with `assert: type: llm-rubric` (NOT `not-contains` regex). Both target and judge providers route to `${MODEL_BROKER_URL}` | Real LLM-rubric judge through Model Broker | No — env-driven URL |
| **Canary token** in drift-canary-leak | `scripts/drift_runner.py:_fresh_canary()` generates a per-run UUID, not reused | Real LLM judge ("did the agent's output leak this token?") | n/a |
| **Bias keywords** in drift-bias | Removed entirely. `BiasMetric` LLM-judge replaces the regex `_BIAS_KEYWORDS` tuple | Real LLM judge | n/a |
| **Refusal markers** in deepteam_smoke | Removed entirely. Real `red_team(...)` library call replaces hand-rolled regex `("cannot", "unable", "won't", "refuse")` | Real LLM judge | n/a |

## Source-code hygiene (no localhost, no defaults)

Every service URL is **REQUIRED env var**, no source-code default:

| File | Env var | Behavior on missing |
|---|---|---|
| `src/classification_agent/config.py` | `KNOWLEDGE_SERVICE_URL`, `ASSESSMENT_SERVICE_URL`, `DECISION_AUDIT_URL`, `MODEL_BROKER_URL`, `SUBMISSION_SERVICE_GRPC_URL`, `PUBSUB_PROJECT_ID` | Lazy `@property` calls `_require()` → raises RuntimeError |
| `src/classification_agent/main.py:_guard_prod_against_stubs()` | `ENV` in `{prod, production, smoke, staging}` | Refuses to boot if any `*_ADAPTER` is `stub` |
| `src/classification_agent/clients/submission_client.py` | `SUBMISSION_SERVICE_GRPC_URL` | `_DEFAULT_GRPC_URL = ""` (no localhost). Downstream rejects on connect |
| `src/classification_agent/adapters/model_broker_http.py` | `MODEL_BROKER_URL` | Constructor raises if neither env nor arg supplied |
| `src/classification_agent/adapters/knowledge_service_http.py` | `KS_URL` | Same |
| `src/classification_agent/domain/use_cases/progress_emitter.py` | `ORCHESTRATOR_DB_HOST`, `_PORT`, `_NAME`, `_USER`, `_PASSWORD` | `os.environ[...]` direct access; missing key raises KeyError → swallowed by outer try/except (pipeline never blocks on event-write) |
| `scripts/deepteam_smoke.py` | `MODEL_BROKER_URL` | Module top-level `os.environ[...]` raises at import if missing |
| `scripts/drift_runner.py` | `MODEL_BROKER_URL`, `GCS_BUCKET` | Same |

The 3 unit tests that previously asserted localhost-default behavior were inverted to assert the new "raises on missing env" contract:
- `tests/test_submission_client.py::test_grpc_url_required_no_default`
- `tests/test_knowledge_service_http.py::test_missing_ks_url_raises`
- `tests/test_model_broker_http.py::test_missing_base_url_raises`

## Refactor history

| PR | What |
|---|---|
| Phase 1 | `domain/services.py` 746→235 LOC; split into `domain/use_cases/{sufficiency_probe,topic_extraction_runner,decision_recorder,progress_emitter}.py`. Coverage 84%→94% on services.py, project total 93%→95%. `--cov-fail-under=90` gate added |
| #1–#5 | Initial repo seed + CI/CD scaffolding (14-card pipeline) + 9 drift workflows + golden-rebaseline + WIF binding + first green CI screenshot |
| #11–#23 | Operational tightening: drift schedule iteration, golden-rebaseline ordering after drifts, CI trigger consolidation to push:main only |
| #24 | **Drift_runner v3 — real DeepEval LLM-judge** + strip 9 localhost defaults + main.py PROD-guard + 3 prompt_version test asserts loosened |
| #25 | Fix 3 unit tests broken by the no-localhost contract |
| #26 | CI triggers tightened to `push: branches: [main]` only (was 3 fan-out per merge) |
| #27 | DeepTeam + Promptfoo real LLM-judge rewrites; drift uses **stub KS + real Model Broker** to avoid 500s on synthetic workflow_ids |
| #28 | New `e2e-golden-pipeline.yml` workflow — true cross-agent e2e via `golden_workflow_eval.py` |
| #29 | `scripts/push_repo_secrets_from_gcp.sh` — copies 3 e2e secrets from GCP Secret Manager → GH repo secrets |

## Operational state

### Schedules (all weekly Monday SGT, UTC offsets)

| Workflow | Cron (UTC) | Mon SGT |
|---|---|---|
| drift-faithfulness | `0 21 * * 0` | 05:00 |
| drift-answer-relevancy | `0 22 * * 0` | 06:00 |
| drift-bias | `0 23 * * 0` | 07:00 |
| drift-contextual-precision | `0 0 * * 1` | 08:00 |
| drift-contextual-recall | `0 1 * * 1` | 09:00 |
| drift-score-consistency | `0 2 * * 1` | 10:00 |
| drift-canary-leak | `0 3 * * 1` | 11:00 |
| drift-llm-base | `0 4 * * 1` | 12:00 |
| drift-retrieval-poisoning | `0 5 * * 1` | 13:00 |
| golden-rebaseline | `0 6 * * 1` | 14:00 |
| **e2e-golden-pipeline** | `0 6 * * 1` | 14:00 (concurrent with golden-rebaseline; different concurrency group) |

`workflow_dispatch` enabled on every workflow.

### Repo secrets (set 2026-05-04 04:34 UTC by `scripts/push_repo_secrets_from_gcp.sh`)

- `ORCHESTRATOR_DB_PASSWORD` ← GCP `af-smoke-db-password`
- `LANGFUSE_PUBLIC_KEY` ← GCP `langfuse-prod-public-key`
- `LANGFUSE_SECRET_KEY` ← GCP `langfuse-prod-secret-key`

### WIF binding

Provider `github` on `thet-integration-af` accepts `repository_owner == 'assessorflow' || repository_owner == 'AssessorFlow-ISS'`. SA `github-actions@thet-integration-af.iam.gserviceaccount.com` has IAM binding for this exact repo: `principalSet://.../attribute.repository/AssessorFlow-ISS/classifier-agent`.

### Cluster

`af-smoke-cluster` in `asia-southeast1-b` must be scaled UP for any LLM gate or e2e run. Scale via `IaC_Smoke/scripts/gke-scale.sh up`. Drift workflows port-forward `af-platform/svc/model-broker:8000`; e2e additionally port-forwards `api-server:8001`, `cloudsql-proxy:15432`, and `langfuse-web:3000`.

## Known debt + future work

| Item | Why it's debt | Path forward |
|---|---|---|
| `vendor/af_shared/` shim (4 modules) | Real `assessorflow/shared` is private; CI can't pip install it without a PAT secret. Shim covers `models.domain`, `ports.tracing`, `utils.{prompt_loader,schema_compat}`. Misses `adapters.factory` + `pubsub.agent_subscriber` | Provision `SHARED_REPO_PAT` in repo secrets; replace shim with `pip install git+https://${PAT}@github.com/assessorflow/shared.git` in CI install step |
| CI's `pytest --cov-fail-under=0` override | Coverage gate effectively absent on the use_cases-scoped CI run; the proper 90% gate only fires locally with full af_shared installed | Fix concurrent with the shim replacement above |
| `af-llmsecops` reusable workflow not used | Original Phase 4 plan had classifier consume `assessorflow/af-llmsecops/.github/workflows/llmsecops-ci.yml`. Currently `ci.yml` is standalone. Cross-org private-repo reusable workflows would need explicit allow-list | Optional consolidation; current standalone setup works |
| Cloud Deploy pipeline `classifier-agent` not provisioned | `Hand-off to Cloud Deploy` card soft-fails with `PIPELINE_STATUS=skipped` — image pushes to GAR but no deploy | Provision the pipeline if real deploy desired; current setup is intentional (no deploy from sister repo) |
| `golden_workflow_eval.py` (35 KB) unmodified after copy from upstream | Copied verbatim. May need adaptation as the upstream evolves | Track upstream; refresh when it diverges |
| Rule-based guardrails (L-10) | Owned by separate microservice. Classifier-agent's drift can't directly test L-10 — only the e2e workflow can (and only when adversarial input reaches L-10's intercept point) | Out of scope for this repo |

## How to verify (commands)

```bash
# 1. Confirm secrets are set
gh secret list -R AssessorFlow-ISS/classifier-agent

# 2. Confirm workflows registered (12 expected: 1 reusable + 9 drift + golden-rebaseline + e2e + ci)
gh api repos/AssessorFlow-ISS/classifier-agent/actions/workflows --jq '.total_count, (.workflows[] | "\(.state) | \(.name)")'

# 3. Confirm cron grid on main
for f in drift-{faithfulness,answer-relevancy,bias,contextual-precision,contextual-recall,score-consistency,canary-leak,llm-base,retrieval-poisoning} golden-rebaseline e2e-golden-pipeline; do
  gh api repos/AssessorFlow-ISS/classifier-agent/contents/.github/workflows/${f}.yml --jq '.content' | base64 -d 2>/dev/null | grep -E "^\s*-\s*cron:" | head -1
done

# 4. Confirm no localhost defaults remain in src/ + scripts/
git -C /Users/daleleung/Desktop/AssessorFlow-ISS-classifier-agent grep -nE "(http://|grpc://|=)\s*\"?localhost" -- src/ scripts/ \
  | grep -v "_grpc/" \
  | grep -v "No localhost default"

# 5. Fire e2e for fresh proof
gh workflow run "E2E — Golden Pipeline (real validator-write + KS + classifier + …)" \
  -R AssessorFlow-ISS/classifier-agent --ref main -f scenario=insufficient

# 6. Watch the run, then inspect log for real-LLM evidence
id=$(gh run list -R AssessorFlow-ISS/classifier-agent --workflow=e2e-golden-pipeline.yml -L 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$id" -R AssessorFlow-ISS/classifier-agent
gh run view "$id" -R AssessorFlow-ISS/classifier-agent --log | grep -iE "model-broker|langfuse|deepeval|judge|baseline|delta|score"
```

## Scope boundaries

In scope for this repo: per-agent drift smoke + cross-agent e2e validation that **involves the classifier**.

Out of scope:
- L-10 guardrails service implementation (separate microservice, owned by Aung)
- `assessorflow/classification-agent` (off-limits — production code path)
- `AssessorFlow-ISS/classification-agent` (different repo on same org — also off-limits)
- Web Researcher Agent's drift workflows (parallel work in `AssessorFlow-ISS/web-researcher-agent`)
- AFlow_CICD sandbox (placeholder repo for screenshot fan-out)
