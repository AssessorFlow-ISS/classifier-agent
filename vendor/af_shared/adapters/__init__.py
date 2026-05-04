"""Adapter factories shipped with the af_shared shim.

`factory.get_decision_audit()` and `factory.get_tracing()` are the entry
points used by `classification_agent.main:_build_service()`. Both default
to in-memory stubs in dev/test and resolve to real implementations only
when the corresponding env var is set to a real-tier value.
"""
