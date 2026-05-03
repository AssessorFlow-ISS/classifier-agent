from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from classification_agent.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    HealthResponse,
)
from classification_agent.domain.services import ClassificationService

router = APIRouter()


def _get_service(request: Request) -> ClassificationService:
    """Retrieve the ClassificationService from app state."""
    return request.app.state.classification_service


@router.post("/invoke", response_model=ClassificationResponse)
async def invoke(
    body: ClassificationRequest,
    service: ClassificationService = Depends(_get_service),
) -> ClassificationResponse:
    """Main classification endpoint.

    Runs the sufficiency check and topic extraction pipeline.
    """
    return await service.classify(body)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", service="classification-agent")


@router.get("/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    """Readiness probe."""
    return HealthResponse(status="ok", service="classification-agent")
