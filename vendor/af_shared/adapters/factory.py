"""Factory functions for cross-cutting adapters (decision audit + tracing).

Vendored shim of upstream ``af_shared.adapters.factory`` — provides the
two callables that ``classification_agent.main`` imports unconditionally.

Selection contract:

| Env var                  | Values            | Returned adapter                        |
|--------------------------|-------------------|-----------------------------------------|
| DECISION_AUDIT_ADAPTER   | unset / stub      | StubDecisionAuditAdapter (in-memory)    |
|                          | grpc / real       | RuntimeError — real adapter not vendored|
| TRACING_ADAPTER          | unset / stub      | StubTracingAdapter (in-memory)          |
|                          | langfuse / real   | RuntimeError — real adapter not vendored|

Real-tier resolution is intentionally a hard error in the shim: the
vendored shim is for CI smoke + local dev only. Production deploys must
replace the shim with the real ``pip install`` of ``assessorflow/shared``
once ``SHARED_REPO_PAT`` is provisioned.
"""

from __future__ import annotations

import os
from typing import Any


_REAL_DECISION_AUDIT_VALUES = {"grpc", "real"}
_REAL_TRACING_VALUES = {"langfuse", "real"}


def _shim_real_unavailable(env_name: str, value: str) -> None:
    raise RuntimeError(
        f"{env_name}={value!r} requires the real assessorflow/shared package, "
        "which is not in the vendored shim. Provision SHARED_REPO_PAT and pip "
        "install git+https://github.com/assessorflow/shared.git in CI to enable."
    )


def get_decision_audit() -> Any:
    """Return a DecisionAuditPort-compatible adapter.

    Defaults to ``StubDecisionAuditAdapter`` (in-memory list of entries).
    """
    value = os.environ.get("DECISION_AUDIT_ADAPTER", "stub").lower()
    if value in _REAL_DECISION_AUDIT_VALUES:
        _shim_real_unavailable("DECISION_AUDIT_ADAPTER", value)
    from af_shared.adapters.stubs.decision_audit_stub import StubDecisionAuditAdapter
    return StubDecisionAuditAdapter()


def get_tracing() -> Any:
    """Return a TracingPort-compatible adapter.

    Defaults to ``StubTracingAdapter`` (in-memory lists of decisions /
    LLM calls / tool calls — used by tests/test_tracing_wiring.py).
    """
    value = os.environ.get("TRACING_ADAPTER", "stub").lower()
    if value in _REAL_TRACING_VALUES:
        _shim_real_unavailable("TRACING_ADAPTER", value)
    from af_shared.adapters.stubs.tracing_stub import StubTracingAdapter
    return StubTracingAdapter()
