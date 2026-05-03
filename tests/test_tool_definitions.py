"""Tests for AF-137: Tool definitions for SimilaritySearch and SearchPolicies.

Each tool wraps a KnowledgeServicePort method as an LLM-callable function
with OpenAI function-calling compatible schema. The LLM sees these tools
and decides when to invoke them during ReAct reasoning.
"""
from __future__ import annotations


from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.api.schemas import SimilarityResult, PolicyChunk


class TestSimilaritySearchToolDefinition:
    """Tests for SimilaritySearch tool schema (AF-137 AC-1)."""

    def test_tool_has_openai_function_schema(self) -> None:
        """AC-1: SimilaritySearch tool exposes OpenAI function-calling schema."""
        from classification_agent.tools.similarity_search_tool import SimilaritySearchTool

        ks = StubKnowledgeServiceAdapter()
        tool = SimilaritySearchTool(knowledge_service=ks, workflow_id="wf-test")
        schema = tool.to_openai_function()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "similarity_search"
        assert "parameters" in schema["function"]
        params = schema["function"]["parameters"]
        assert "query" in params["properties"]
        assert "knowledge_base_target" in params["properties"]
        assert "top_k" in params["properties"]
        assert params["required"] == ["query", "knowledge_base_target"]

    def test_tool_has_descriptive_docstring(self) -> None:
        """AC-1: Tool has natural language description for LLM."""
        from classification_agent.tools.similarity_search_tool import SimilaritySearchTool

        ks = StubKnowledgeServiceAdapter()
        tool = SimilaritySearchTool(knowledge_service=ks, workflow_id="wf-test")
        schema = tool.to_openai_function()

        desc = schema["function"]["description"]
        assert len(desc) > 20
        assert "semantic" in desc.lower() or "similar" in desc.lower()

    async def test_tool_invocation_calls_port_method(self) -> None:
        """AC-2: Invoking the tool calls KnowledgeServicePort.similarity_search."""
        from classification_agent.tools.similarity_search_tool import SimilaritySearchTool

        ks = StubKnowledgeServiceAdapter()
        ks.set_similarity_results("document", [
            SimilarityResult(
                chunk_id="sim-001",
                content="Deep OOP concepts",
                similarity_score=0.92,
                source_document="oop.pdf",
            ),
        ])
        tool = SimilaritySearchTool(knowledge_service=ks, workflow_id="wf-test")

        result = await tool.execute(
            query="advanced object-oriented design patterns",
            knowledge_base_target="document",
            top_k=5,
        )

        assert len(result) > 0
        assert "sim-001" in result[0]["chunk_id"]
        assert result[0]["similarity_score"] == 0.92

    async def test_tool_returns_serializable_dicts(self) -> None:
        """AC-2: Tool returns JSON-serializable dicts (not Pydantic models)."""
        from classification_agent.tools.similarity_search_tool import SimilaritySearchTool

        ks = StubKnowledgeServiceAdapter()
        tool = SimilaritySearchTool(knowledge_service=ks, workflow_id="wf-test")

        result = await tool.execute(
            query="test query",
            knowledge_base_target="document",
        )

        # Each result should be a plain dict
        for item in result:
            assert isinstance(item, dict)
            assert "chunk_id" in item
            assert "content" in item
            assert "similarity_score" in item

    async def test_tool_respects_top_k(self) -> None:
        """AC-2: Tool passes top_k to the port method."""
        from classification_agent.tools.similarity_search_tool import SimilaritySearchTool

        ks = StubKnowledgeServiceAdapter()
        # Default stub returns 3 results
        tool = SimilaritySearchTool(knowledge_service=ks, workflow_id="wf-test")

        result = await tool.execute(
            query="test query",
            knowledge_base_target="document",
            top_k=1,
        )

        assert len(result) <= 1


class TestSearchPoliciesToolDefinition:
    """Tests for SearchPolicies tool schema (AF-137 AC-3)."""

    def test_tool_has_openai_function_schema(self) -> None:
        """AC-3: SearchPolicies tool exposes OpenAI function-calling schema."""
        from classification_agent.tools.search_policies_tool import SearchPoliciesTool

        ks = StubKnowledgeServiceAdapter()
        tool = SearchPoliciesTool(knowledge_service=ks, assessment_id="assess-001")
        schema = tool.to_openai_function()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_policies"
        assert "parameters" in schema["function"]
        params = schema["function"]["parameters"]
        assert "query" in params["properties"]
        assert "policy_type" in params["properties"]

    def test_tool_has_descriptive_docstring(self) -> None:
        """AC-3: Tool has natural language description for LLM."""
        from classification_agent.tools.search_policies_tool import SearchPoliciesTool

        ks = StubKnowledgeServiceAdapter()
        tool = SearchPoliciesTool(knowledge_service=ks, assessment_id="assess-001")
        schema = tool.to_openai_function()

        desc = schema["function"]["description"]
        assert len(desc) > 20
        assert "polic" in desc.lower() or "rubric" in desc.lower()

    async def test_tool_invocation_calls_port_method(self) -> None:
        """AC-4: Invoking the tool calls KnowledgeServicePort.search_policies."""
        from classification_agent.tools.search_policies_tool import SearchPoliciesTool

        ks = StubKnowledgeServiceAdapter()
        ks.set_policy_chunks([
            PolicyChunk(
                chunk_id="pol-001",
                content="Assessor rubric: evaluate critical thinking",
                policy_type="assessor_rubric",
                source="assessor_upload",
                assessment_id="assess-001",
                similarity_score=0.90,
            ),
        ])
        tool = SearchPoliciesTool(knowledge_service=ks, assessment_id="assess-001")

        result = await tool.execute(
            query="grading criteria",
            policy_type="assessor_rubric",
        )

        assert len(result) > 0
        assert result[0]["chunk_id"] == "pol-001"
        assert result[0]["policy_type"] == "assessor_rubric"

    async def test_tool_returns_serializable_dicts(self) -> None:
        """AC-4: Tool returns JSON-serializable dicts."""
        from classification_agent.tools.search_policies_tool import SearchPoliciesTool

        ks = StubKnowledgeServiceAdapter()
        tool = SearchPoliciesTool(knowledge_service=ks, assessment_id="assess-001")

        result = await tool.execute(query="rubric")

        for item in result:
            assert isinstance(item, dict)
            assert "chunk_id" in item
            assert "content" in item
            assert "policy_type" in item

    async def test_tool_filters_by_policy_type(self) -> None:
        """AC-4: Tool passes policy_type filter to port method."""
        from classification_agent.tools.search_policies_tool import SearchPoliciesTool

        ks = StubKnowledgeServiceAdapter()
        ks.set_policy_chunks([
            PolicyChunk(
                chunk_id="pol-sys",
                content="System default grading",
                policy_type="system_default",
                source="admin_seeded",
            ),
            PolicyChunk(
                chunk_id="pol-asr",
                content="Assessor rubric",
                policy_type="assessor_rubric",
                source="assessor_upload",
                assessment_id="assess-001",
            ),
        ])
        tool = SearchPoliciesTool(knowledge_service=ks, assessment_id="assess-001")

        result = await tool.execute(
            query="rubric",
            policy_type="assessor_rubric",
        )

        assert all(r["policy_type"] == "assessor_rubric" for r in result)


class TestToolRegistry:
    """Tests for tool registry (AF-137 AC-5)."""

    def test_registry_collects_all_tools(self) -> None:
        """AC-5: Registry returns all tool definitions."""
        from classification_agent.tools.registry import build_tool_registry

        ks = StubKnowledgeServiceAdapter()
        registry = build_tool_registry(
            knowledge_service=ks,
            workflow_id="wf-test",
            assessment_id="assess-001",
        )

        assert len(registry.tools) == 2
        names = [t.name for t in registry.tools]
        assert "similarity_search" in names
        assert "search_policies" in names

    def test_registry_returns_openai_function_schemas(self) -> None:
        """AC-5: Registry exports all tool schemas in OpenAI format."""
        from classification_agent.tools.registry import build_tool_registry

        ks = StubKnowledgeServiceAdapter()
        registry = build_tool_registry(
            knowledge_service=ks,
            workflow_id="wf-test",
            assessment_id="assess-001",
        )

        schemas = registry.to_openai_functions()
        assert len(schemas) == 2
        assert all(s["type"] == "function" for s in schemas)

    async def test_registry_dispatches_tool_call(self) -> None:
        """AC-5: Registry can dispatch a tool call by name."""
        from classification_agent.tools.registry import build_tool_registry

        ks = StubKnowledgeServiceAdapter()
        registry = build_tool_registry(
            knowledge_service=ks,
            workflow_id="wf-test",
            assessment_id="assess-001",
        )

        result = await registry.execute(
            tool_name="similarity_search",
            arguments={"query": "test", "knowledge_base_target": "document"},
        )

        assert isinstance(result, list)
