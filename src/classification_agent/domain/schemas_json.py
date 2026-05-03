"""JSON Schemas for structured LLM output.

Generated from Pydantic response models via ``model_json_schema()`` and
cleaned for Gemini compatibility via ``clean_for_gemini()``.

Passed to the Model Broker via response_schema so Gemini/OpenAI enforce
the output structure at the model level (response_mime_type / json_schema).
"""
from af_shared.utils.schema_compat import clean_for_gemini

from classification_agent.domain.response_models import (
    ReactSufficiencyCheckResponse,
    RubricFitnessResponse,
    TopicExtractionResponse,
)

REACT_SUFFICIENCY_SCHEMA = clean_for_gemini(ReactSufficiencyCheckResponse.model_json_schema())

TOPIC_EXTRACTION_SCHEMA = clean_for_gemini(TopicExtractionResponse.model_json_schema())

# Kept for backward compat with rubric_fitness.py (class file not yet archived)
RUBRIC_FITNESS_SCHEMA = clean_for_gemini(RubricFitnessResponse.model_json_schema())
