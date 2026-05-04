# Classifier Agent (#4)

Stateless FastAPI microservice for material sufficiency assessment and topic extraction.

> **Naming note** — this repo (`AssessorFlow-ISS/classifier-agent`) is the home of the CICD for the Classification Agent with golden automation + drift CI pipeline. The sister repo `AssessorFlow-ISS/classification-agent` is archived and off-limits for SIT Env only.

## Provenance

| Source | Role |
|---|---|
| `assessorflow/classification-agent@feat/AF-classifier-refactor` | Upstream — Phase 1 use-case refactor lands here |
| `locoroco-git/AFlow_CICD/sandboxes/classifier-agent/` | Sandbox mirror — exercises the 9-card LLMOps PR DAG with **placeholder** drift jobs |
| `AssessorFlow-ISS/classifier-agent` (this repo) | Production CI tier — REAL golden + drift workflows wired to `thet-integration-af` GCP via WIF |

## What It Does (COntent Fitness Gating)

1. **Sufficiency Check** — reads chunks stored by Validator Agent, runs unified ReAct probe (sufficiency + rubric fitness) via `SimilaritySearch` and `SearchPolicies` tool calls.
2. **Topic Extraction** — synthesises hierarchical topics from chunks via Model Broker (CHEAP tier), persists to Knowledge Service.

It never have to call `ProcessMaterial` - chunks are already stored by the Validator Agent in in upstream processing.

## Quick Start

```bash
# Install dependencies (Python 3.12+)
pip install -e ".[dev]"

# Run tests + coverage gate (>=90% enforced)
pytest

# Start the server
uvicorn classification_agent.main:app --reload
```

## API

- `POST /invoke` — main classification endpoint
- `GET /health` — liveness probe
- `GET /ready` — readiness probe

## Architecture

Port-first hexagonal architecture (ADR-42). All external dependencies are behind abstract port interfaces. Internal layout:

```
src/classification_agent/
├── main.py                              FastAPI lifespan + DI
├── api/                                 routes + Pydantic schemas
├── ports/                               5 ABCs (knowledge_service, model_broker,
│                                        event_publisher, assessment_config,
│                                        decision_audit)
├── adapters/                            stub + real (HTTP/gRPC/Pub/Sub) per port
├── domain/
│   ├── services.py                      thin orchestrator (235 LOC)
│   ├── use_cases/                       split from the legacy 746-LOC services.py
│   │   ├── sufficiency_probe.py         unified ReAct probe + rubric helpers
│   │   ├── topic_extraction_runner.py   LLM topic synthesis + KS storage +
│   │   │                                guardrail-blocked terminal path
│   │   ├── decision_recorder.py         success-path decision log + completion
│   │   └── progress_emitter.py          workflow_events sub-card writer
│   ├── sufficiency.py                   ReAct prober internals
│   ├── topic_extractor.py               LLM topic extraction internals
│   ├── rubric_fitness.py                rubric helpers
│   └── response_models.py               pydantic models
├── tools/                               LLM tool definitions (SimilaritySearch,
│                                        SearchPolicies)
└── clients/                             gRPC clients
```

### Ports

| Port | Purpose |
|------|---------|
| `KnowledgeServicePort` | Read chunks, store topics (gRPC 3.2.x) |
| `AssessmentConfigPort` | Read assessment config (gRPC 2.2.1) |
| `ModelBrokerPort` | LLM inference (L-09) |
| `DecisionAuditPort` | Audit logging (gRPC Section 4) |
| `EventPublisherPort` | Pub/Sub events |

### Pub/Sub Topics

- Subscribes to: `assessorflow.classification.trigger` (Topic #4)
- Publishes to: `assessorflow.classification.complete` (Topic #5) or `assessorflow.classification.insufficient` (Topic #6)

## CI Tiers

| Tier | Where | What | Cost |
|---|---|---|---|
| Sandbox PR DAG (placeholder) | `locoroco-git/AFlow_CICD/.github/workflows/sandbox-classifier-agent-ci.yml` | 9-card lint/SAST/SCA/secret/unit/integration/build/quality/adversarial fan-out — placeholder LLM cards | ~$0 |
| **Golden Pipeline + Support drift detection scheduled Jobs** | `AssessorFlow-ISS/classifier-agent/.github/workflows/` | 9 drift workflows + 1 golden-rebaseline against `af-golden` namespace via WIF; PROD LLM/KS/Pub/Sub/Langfuse | ~$0.30 per cycle |

## License

Inherits the AssessorFlow project license.
