"""Use-case modules extracted from the legacy ``services.py`` god-object.

Each module owns one responsibility from the Phase 4 classification pipeline:

- ``progress_emitter`` — workflow_events sub-card writer (cross-cutting infra)
- ``sufficiency_probe`` — unified ReAct material + rubric probing
- ``topic_extraction_runner`` — LLM topic synthesis + KS storage + guardrail-terminal handling
- ``decision_recorder`` — success-path decision log + completion event composition

``ClassificationService`` in ``domain.services`` is now a thin orchestrator that
wires these use cases together via constructor injection.
"""
