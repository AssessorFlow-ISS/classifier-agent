"""Minimal HTTP surface for the ZAP baseline scan.

ZAP baseline tests **HTTP-layer security** (security headers, methods,
cookies, error-code disclosure). It does NOT test classification logic
or LLM behavior — those are exercised by the DeepEval/DeepTeam/Promptfoo
jobs against the real af-platform Model Broker.

This file mirrors only the route SHAPES of the real classifier
(`classification_agent.main:app`) — same paths, same request/response
schemas — so ZAP's spider has surfaces to crawl and the security checks
fire on the same body validation the real app uses. We do not boot the
real app here because it imports `af_shared.adapters.factory` which is
not in the vendored shim (debt: PAT install of assessorflow/shared).

The /classify endpoint returns a deterministic, schema-conformant response
without invoking the classification pipeline. ZAP does not score this
content — it scores HTTP-layer behavior. Replacing this file with a real
boot of classification_agent.main:app is the documented path forward.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="classifier-agent (HTTP surface for ZAP baseline)",
    description="Mirrors classification_agent route shapes; HTTP-layer scan only.",
    version="0.1.0",
)


class ClassifyRequest(BaseModel):
    workflow_id: str
    assessment_id: str
    classification_type: str = "sufficiency_and_topics"


class ClassifyResponse(BaseModel):
    workflow_id: str
    sufficient: bool
    reason: str


@app.get("/")
async def root():
    return {"agent": "classifier-agent", "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    return {"status": "ready"}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    # Schema-conformant response. ZAP scores HTTP-layer security, not the
    # contents of this body. LLM behavior is tested by DeepEval/DeepTeam/
    # Promptfoo against the real af-platform Model Broker.
    return ClassifyResponse(
        workflow_id=req.workflow_id,
        sufficient=False,
        reason="schema_conformant_response",
    )
