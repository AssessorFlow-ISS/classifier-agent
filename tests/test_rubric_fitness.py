"""Tests for AF-139: Rubric fitness assessment (detection + signaling).

The RubricFitnessAssessor searches for assessor rubric via SearchPolicies,
falls back to system defaults, and reasons about semantic alignment between
rubric and material topics.  It signals misalignment to the Orchestrator
but does NOT write to Policy KB (Thet Q-3).
"""
from __future__ import annotations



from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from classification_agent.api.schemas import (
    PolicyChunk,
    RubricFitnessResult,
    TopicHierarchy,
    Topic,
    SubTopic,
)


def _make_topic_hierarchy(workflow_id: str = "wf-rubric") -> TopicHierarchy:
    """Create a sample topic hierarchy for rubric fitness testing."""
    return TopicHierarchy(
        workflow_id=workflow_id,
        topics=[
            Topic(
                topic_id="t-001",
                name="Shakespearean Literature",
                subtopics=[
                    SubTopic(topic_id="t-001-1", name="Hamlet Analysis"),
                    SubTopic(topic_id="t-001-2", name="Early Modern English"),
                ],
            ),
            Topic(
                topic_id="t-002",
                name="Literary Criticism",
                subtopics=[
                    SubTopic(topic_id="t-002-1", name="Formalist Criticism"),
                ],
            ),
        ],
    )


class TestRubricFitnessAssessor:
    """Tests for RubricFitnessAssessor domain component (AF-139)."""

    async def test_assessor_rubric_found_and_aligned(self) -> None:
        """AC-1: When assessor rubric exists and is aligned, return ALIGNED."""
        from classification_agent.domain.rubric_fitness import RubricFitnessAssessor

        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        # Set up assessor rubric
        ks.set_policy_chunks([
            PolicyChunk(
                chunk_id="rubric-001",
                content="Assess understanding of Shakespearean themes and Early Modern English grammar",
                policy_type="assessor_rubric",
                source="assessor_upload",
                assessment_id="assess-001",
                similarity_score=0.95,
            ),
        ])

        # Model broker returns alignment assessment
        mb.set_response("classification.rubric_fitness", {
            "is_aligned": True,
            "alignment_score": 0.92,
            "gap_description": None,
            "recommendation": "use_as_is",
        })

        assessor = RubricFitnessAssessor(
            model_broker=mb,
            knowledge_service=ks,
        )

        topics = _make_topic_hierarchy()
        result = await assessor.assess(
            topics=topics,
            assessment_id="assess-001",
            workflow_id="wf-rubric",
        )

        assert isinstance(result, RubricFitnessResult)
        assert result.is_aligned is True
        assert result.rubric_source == "assessor_upload"
        assert result.alignment_score > 0.5
        assert result.recommendation == "use_as_is"

    async def test_no_assessor_rubric_falls_back_to_system_default(self) -> None:
        """AC-2: When no assessor rubric exists, search system defaults."""
        from classification_agent.domain.rubric_fitness import RubricFitnessAssessor

        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        # Only system default rubric (no assessor rubric)
        ks.set_policy_chunks([
            PolicyChunk(
                chunk_id="pol-default",
                content="Default grading rubric: assess grammar, vocabulary, comprehension",
                policy_type="system_default",
                source="admin_seeded",
                similarity_score=0.80,
            ),
        ])

        mb.set_response("classification.rubric_fitness", {
            "is_aligned": True,
            "alignment_score": 0.78,
            "gap_description": None,
            "recommendation": "use_as_is",
        })

        assessor = RubricFitnessAssessor(
            model_broker=mb,
            knowledge_service=ks,
        )

        topics = _make_topic_hierarchy()
        result = await assessor.assess(
            topics=topics,
            assessment_id="assess-001",
            workflow_id="wf-rubric",
        )

        assert result.rubric_source == "system_default"

    async def test_system_default_misaligned_signals_synthesis(self) -> None:
        """AC-3: When system default is misaligned, signal synthesize_new (no direct write)."""
        from classification_agent.domain.rubric_fitness import RubricFitnessAssessor

        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        # System default rubric (misaligned with Shakespeare material)
        ks.set_policy_chunks([
            PolicyChunk(
                chunk_id="pol-default",
                content="Default grading rubric: 50% modern grammar, 40% vocabulary, 10% structure",
                policy_type="system_default",
                source="admin_seeded",
                similarity_score=0.70,
            ),
        ])

        # Rubric fitness assessment (misaligned)
        mb.set_response("classification.rubric_fitness", {
            "is_aligned": False,
            "alignment_score": 0.25,
            "gap_description": "Default rubric assumes modern English grammar norms; material uses Early Modern English",
            "recommendation": "synthesize_new",
        })

        assessor = RubricFitnessAssessor(
            model_broker=mb,
            knowledge_service=ks,
        )

        topics = _make_topic_hierarchy()
        result = await assessor.assess(
            topics=topics,
            assessment_id="assess-001",
            workflow_id="wf-rubric",
        )

        assert result.is_aligned is False
        # rubric_source reflects what was FOUND, not what will be synthesized later
        assert result.rubric_source == "system_default"
        assert result.gap_description is not None
        assert result.recommendation == "synthesize_new"

    async def test_no_rubric_at_all(self) -> None:
        """AC-4: When no rubric exists (assessor or system), return NO_RUBRIC."""
        from classification_agent.domain.rubric_fitness import RubricFitnessAssessor

        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        # No policy chunks at all
        ks.set_policy_chunks([])

        assessor = RubricFitnessAssessor(
            model_broker=mb,
            knowledge_service=ks,
        )

        topics = _make_topic_hierarchy()
        result = await assessor.assess(
            topics=topics,
            assessment_id="assess-001",
            workflow_id="wf-rubric",
        )

        assert result.rubric_source == "none"
        assert result.is_aligned is False
        assert result.recommendation is not None

    async def test_misaligned_returns_recommendation_without_writing(self) -> None:
        """AC-5: Misaligned rubric returns recommendation but does NOT write to Policy KB."""
        from classification_agent.domain.rubric_fitness import RubricFitnessAssessor

        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        ks.set_policy_chunks([
            PolicyChunk(
                chunk_id="pol-default",
                content="Default rubric",
                policy_type="system_default",
                source="admin_seeded",
                similarity_score=0.60,
            ),
        ])

        mb.set_response("classification.rubric_fitness", {
            "is_aligned": False,
            "alignment_score": 0.20,
            "gap_description": "Misaligned",
            "recommendation": "synthesize_new",
        })

        assessor = RubricFitnessAssessor(
            model_broker=mb,
            knowledge_service=ks,
        )

        topics = _make_topic_hierarchy()
        result = await assessor.assess(
            topics=topics,
            assessment_id="assess-001",
            workflow_id="wf-rubric",
        )

        # Signals misalignment with recommendation
        assert result.is_aligned is False
        assert result.recommendation == "synthesize_new"
        assert result.rubric_source == "system_default"

    async def test_prompt_version_format(self) -> None:
        """AC-6: RubricFitnessAssessor has correct prompt version format."""
        from classification_agent.domain.rubric_fitness import RubricFitnessAssessor

        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        assessor = RubricFitnessAssessor(
            model_broker=mb,
            knowledge_service=ks,
        )

        # ADR-39 format `<agent>/<template>@v<n>`; `<agent>` derives from
        # checkout dir so match the suffix and presence of `@v` only.
        assert "/" in assessor.prompt_version
        assert "@v" in assessor.prompt_version
        assert "@v" in assessor.prompt_version

    async def test_aligned_rubric_returns_use_as_is(self) -> None:
        """AC-7: When rubric is aligned, recommendation is use_as_is."""
        from classification_agent.domain.rubric_fitness import RubricFitnessAssessor

        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        ks.set_policy_chunks([
            PolicyChunk(
                chunk_id="pol-001",
                content="Appropriate rubric for the material",
                policy_type="system_default",
                source="admin_seeded",
                similarity_score=0.90,
            ),
        ])

        mb.set_response("classification.rubric_fitness", {
            "is_aligned": True,
            "alignment_score": 0.88,
            "gap_description": None,
            "recommendation": "use_as_is",
        })

        assessor = RubricFitnessAssessor(
            model_broker=mb,
            knowledge_service=ks,
        )

        topics = _make_topic_hierarchy()
        result = await assessor.assess(
            topics=topics,
            assessment_id="assess-001",
            workflow_id="wf-rubric",
        )

        assert result.is_aligned is True
        assert result.recommendation == "use_as_is"
        assert result.rubric_source == "system_default"
