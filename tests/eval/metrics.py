"""Custom DeepEval GEval metrics for Classification Agent quality evaluation.

Each metric evaluates a specific dimension of classification quality using
LLM-as-Judge (GEval). Thresholds are calibrated for stub-mode testing
where the Model Broker returns canned responses.

Metrics are created via factory functions (not module-level) to avoid
initializing LLM clients at import time. This allows the test file to
be collected by pytest even without an API key set.

Metrics:
    - classification_correctness: sufficiency verdict accuracy
    - rubric_alignment: rubric fitness assessment quality
    - topic_extraction: topic hierarchy quality
    - autonomy_decision: web research autonomy correctness
"""
from __future__ import annotations

import os
from functools import lru_cache

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# Judge model override: set DEEPEVAL_JUDGE_MODEL to use a non-OpenAI model
# e.g. DEEPEVAL_JUDGE_MODEL=gemini/gemini-2.5-pro (uses GEMINI_API_KEY)
_JUDGE_MODEL = os.environ.get("DEEPEVAL_JUDGE_MODEL")


# ---------------------------------------------------------------------------
# Metric definitions (criteria + params only, no LLM client init)
# ---------------------------------------------------------------------------

_FAITHFULNESS_CONFIG = {
    "name": "Classification Grounding Fidelity",
    "criteria": (
        "Evaluate whether the classification output is genuinely grounded in the "
        "retrieval_context (source chunks), not just superficially matching keywords. "
        "Be strict — a perfect score requires ALL of the following:\n"
        "1. Every topic name in the output must correspond to a substantive knowledge "
        "area actually covered in the retrieval_context chunks. If a topic appears in "
        "the output but no chunk discusses it in depth, that is a hallucinated topic. "
        "Penalize heavily.\n"
        "2. The sufficiency verdict (sufficient/insufficient) must be justified by the "
        "volume and depth of the retrieval_context. If there are only 3 short chunks "
        "but the output says 'sufficient for 15 questions at hard difficulty', that is "
        "unfaithful to the evidence.\n"
        "3. Gap analysis entries (if present) must reference actual gaps observable in "
        "the retrieval_context, not generic domain gaps. A gap like 'needs more on "
        "algorithms' is only valid if the chunks genuinely lack algorithm content.\n"
        "4. The rubric fitness assessment (if present) must be consistent with both "
        "the rubric content in the input AND the material covered in the chunks. "
        "Claiming alignment when the rubric domain differs from the chunk content "
        "is unfaithful.\n"
        "5. If the output contains quantitative claims (e.g., 'covers 5 major topics'), "
        "they must be verifiable from the retrieval_context."
    ),
    "evaluation_params": [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    "threshold": 0.7,
}

_METRIC_CONFIGS = {
    "classification_correctness": {
        "name": "Classification Correctness",
        "criteria": (
            "Evaluate whether the classification output correctly identifies "
            "material sufficiency. Check that:\n"
            "1. The 'sufficient' field matches the expected value\n"
            "2. When sufficient, topics are extracted and gap_analysis is empty\n"
            "3. When insufficient, gap_analysis is populated with specific gaps\n"
            "4. The autonomy_exercised flag matches the web_research_mode setting\n"
            "5. Search queries are present only when autonomy is exercised"
        ),
        "evaluation_params": [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        "threshold": 0.7,
    },
    "rubric_alignment": {
        "name": "Rubric Alignment Quality",
        "criteria": (
            "Evaluate whether the rubric fitness assessment is correct:\n"
            "1. When the rubric covers the same domain as the material, "
            "is_aligned should be true\n"
            "2. When the rubric covers a different domain, is_aligned should "
            "be false and recommendation should be synthesize_new\n"
            "3. When no rubric exists, rubric_source should be 'none'\n"
            "4. The rubric_source correctly reflects what was found: "
            "assessor_upload, system_default, or none"
        ),
        "evaluation_params": [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        "threshold": 0.7,
    },
    "topic_extraction": {
        "name": "Topic Extraction Quality",
        "criteria": (
            "Evaluate the quality and completeness of the extracted topic hierarchy. "
            "Be strict — a perfect score requires ALL of the following:\n"
            "1. Topics MUST cover the breadth of the retrieval_context chunks, not "
            "just a subset. If 22 chunks span OOP, data structures, and algorithms, "
            "all three domains must appear as topics. Penalize heavily if major "
            "knowledge areas present in the chunks are missing from the topics.\n"
            "2. The number of topics must fall within the expected range in expected_output. "
            "Too few topics (under-segmentation) or too many (over-segmentation) should "
            "reduce the score.\n"
            "3. Each topic must have at least 2 subtopics that represent specific, "
            "assessable concepts (not vague restatements of the parent).\n"
            "4. Subtopic depth should be proportional to the difficulty_level in the input. "
            "A 'hard' difficulty assessment needs deeper subtopics (e.g., 'Template Method Pattern') "
            "than an 'easy' one (e.g., 'What is inheritance?').\n"
            "5. No duplicate, overlapping, or near-synonym topics exist.\n"
            "6. Topic names must be precise domain terms, not generic labels like "
            "'General Concepts' or 'Miscellaneous'."
        ),
        "evaluation_params": [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        "threshold": 0.7,
    },
    "autonomy_decision": {
        "name": "Autonomy Decision Correctness",
        "criteria": (
            "Evaluate whether the web research autonomy decision is correct:\n"
            "1. When web_research_mode='auto' and material is insufficient, "
            "autonomy_exercised should be true with search_queries populated\n"
            "2. When web_research_mode='manual' or 'disabled', "
            "autonomy_exercised must be false\n"
            "3. Search queries (when present) should be relevant to the "
            "identified content gaps\n"
            "4. Gap analysis entries should indicate which gaps are fillable "
            "by web research"
        ),
        "evaluation_params": [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        "threshold": 0.7,
    },
}


# ---------------------------------------------------------------------------
# Factory functions (deferred LLM client initialization)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _get_metric(key: str) -> GEval:
    """Create a GEval metric instance on first use (deferred init)."""
    config = dict(_METRIC_CONFIGS[key])
    if _JUDGE_MODEL:
        config["model"] = _JUDGE_MODEL
    return GEval(**config)


def get_classification_correctness() -> GEval:
    return _get_metric("classification_correctness")


def get_rubric_alignment() -> GEval:
    return _get_metric("rubric_alignment")


def get_topic_extraction() -> GEval:
    return _get_metric("topic_extraction")


def get_autonomy_decision() -> GEval:
    return _get_metric("autonomy_decision")


@lru_cache(maxsize=1)
def get_faithfulness() -> GEval:
    """Classification-specific grounding fidelity metric.

    Replaces the generic DeepEval FaithfulnessMetric with a GEval metric
    that tests topic-chunk traceability and verdict-evidence consistency,
    not just keyword overlap.
    """
    config = dict(_FAITHFULNESS_CONFIG)
    if _JUDGE_MODEL:
        config["model"] = _JUDGE_MODEL
    return GEval(**config)
