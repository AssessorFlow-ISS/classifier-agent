"""Tests for SubmissionClient — gRPC client helper.

Covers retry policy, channel lifecycle, and happy-path get_assessment_config
without requiring a real Submission Service. We mock the gRPC stub so only
the client's orchestration logic is exercised.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from classification_agent._grpc import submission_pb2
from classification_agent.clients.submission_client import SubmissionClient


def _make_client_with_mock_stub(stub: MagicMock) -> SubmissionClient:
    """Build a client with a preconfigured mock stub attached."""
    client = SubmissionClient(grpc_url="localhost:9999")
    client._stub = stub
    client._channel = MagicMock()
    return client


def _aio_rpc_error(code: grpc.StatusCode, detail: str = "boom") -> grpc.aio.AioRpcError:
    err = grpc.aio.AioRpcError(
        code=code,
        initial_metadata=grpc.aio.Metadata(),
        trailing_metadata=grpc.aio.Metadata(),
        details=detail,
    )
    return err


class TestSubmissionClientLifecycle:
    def test_default_grpc_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUBMISSION_SERVICE_GRPC_URL", "submission:5001")
        client = SubmissionClient()
        assert client._grpc_url == "submission:5001"

    def test_grpc_url_required_no_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # PROD-safety: no source-code localhost default. Missing env -> "" url
        # which downstream rejects on connect. Verify the empty-default contract.
        monkeypatch.delenv("SUBMISSION_SERVICE_GRPC_URL", raising=False)
        client = SubmissionClient()
        assert client._grpc_url == ""

    def test_explicit_grpc_url_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUBMISSION_SERVICE_GRPC_URL", "ignored:1")
        client = SubmissionClient(grpc_url="explicit:2222")
        assert client._grpc_url == "explicit:2222"

    def test_ensure_stub_creates_channel_once(self) -> None:
        client = SubmissionClient(grpc_url="localhost:9999")
        with patch(
            "classification_agent.clients.submission_client.grpc.aio.insecure_channel",
            return_value=MagicMock(),
        ) as mock_channel:
            stub_a = client._ensure_stub()
            stub_b = client._ensure_stub()
        assert stub_a is stub_b
        assert mock_channel.call_count == 1

    async def test_close_is_idempotent(self) -> None:
        client = SubmissionClient()
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        client._channel = mock_channel
        client._stub = MagicMock()

        await client.close()
        assert client._channel is None
        assert client._stub is None

        # Second call is safe
        await client.close()


class TestGetAssessmentConfig:
    async def test_happy_path_returns_response(self) -> None:
        stub = MagicMock()
        fake_response = submission_pb2.GetAssessmentConfigResponse(
            config=submission_pb2.AssessmentConfig(
                assessment_id="a-1",
                assessment_title="Test",
                structured_question_count=10,
                non_structured_question_count=3,
                difficulty_level="medium",
                web_research_mode="manual",
            )
        )
        stub.GetAssessmentConfig = AsyncMock(return_value=fake_response)
        client = _make_client_with_mock_stub(stub)

        response = await client.get_assessment_config(assessment_id="a-1")

        assert response.config.assessment_id == "a-1"
        stub.GetAssessmentConfig.assert_awaited_once()

    async def test_retries_on_unavailable_and_succeeds(self) -> None:
        stub = MagicMock()
        fake_response = submission_pb2.GetAssessmentConfigResponse()
        stub.GetAssessmentConfig = AsyncMock(
            side_effect=[
                _aio_rpc_error(grpc.StatusCode.UNAVAILABLE),
                fake_response,
            ]
        )
        client = _make_client_with_mock_stub(stub)

        with patch(
            "classification_agent.clients.submission_client.asyncio.sleep",
            new=AsyncMock(),
        ):
            response = await client.get_assessment_config(
                assessment_id="a-2", timeout_seconds=1.0,
            )
        assert response is fake_response
        assert stub.GetAssessmentConfig.await_count == 2

    async def test_non_retriable_error_propagates(self) -> None:
        stub = MagicMock()
        stub.GetAssessmentConfig = AsyncMock(
            side_effect=_aio_rpc_error(grpc.StatusCode.NOT_FOUND, "missing")
        )
        client = _make_client_with_mock_stub(stub)

        with pytest.raises(grpc.aio.AioRpcError):
            await client.get_assessment_config(assessment_id="a-404")
        assert stub.GetAssessmentConfig.await_count == 1

    async def test_retries_exhausted_raises(self) -> None:
        stub = MagicMock()
        stub.GetAssessmentConfig = AsyncMock(
            side_effect=_aio_rpc_error(grpc.StatusCode.UNAVAILABLE)
        )
        client = _make_client_with_mock_stub(stub)

        with patch(
            "classification_agent.clients.submission_client.asyncio.sleep",
            new=AsyncMock(),
        ):
            with pytest.raises(grpc.aio.AioRpcError):
                await client.get_assessment_config(assessment_id="a-dead")
        # Max 3 attempts per _RETRY_MAX_ATTEMPTS
        assert stub.GetAssessmentConfig.await_count == 3
