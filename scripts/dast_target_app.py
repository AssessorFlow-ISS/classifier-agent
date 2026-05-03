"""Minimal FastAPI target for ZAP baseline scans.

The full classification_agent.main:app pulls in af_shared.adapters.factory
which is not in the vendored shim and would force the shim to grow
unbounded. For DAST we only need an HTTP surface that mirrors the routes
ZAP will probe — /, /health, /ready, /invoke — so the spider has paths
to crawl and the security checks can fire.

This module only exists for the CI DAST card. Production traffic always
goes through classification_agent.main:app.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="classifier-agent (DAST target)",
    description="Minimal HTTP surface for ZAP baseline.",
    version="0.1.0",
)


class InvokeRequest(BaseModel):
    workflow_id: str
    assessment_id: str
    classification_type: str = "sufficiency_and_topics"


class InvokeResponse(BaseModel):
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


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest):
    return InvokeResponse(
        workflow_id=req.workflow_id,
        sufficient=True,
        reason="DAST stub — no real classification performed",
    )
