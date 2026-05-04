# Prompts directory: versioning convention

This directory holds the YAML system prompts the Classification Agent loads at
startup via `vendor/af_shared/utils/prompt_loader.load_prompt`.

## Single canonical file per prompt; Git history is the version archive

Each task has exactly **one** YAML file (e.g. `react_sufficiency.yaml`); the
production loader reads it via a hardcoded path in
`src/classification_agent/domain/{sufficiency,rubric_fitness,topic_extractor}.py`.
The version inside each file (`version:` field) is the deployed version; older
versions are recoverable from Git history via
`git show <SHA>:prompts/<file>.yaml` and via the `changelog:` field that each
prompt maintains as a running narrative.

This mirrors the Validator Agent's prompt-versioning pattern documented at
`prompt_regression_n_data_versioning_ci.md` (single-file-per-prompt, version
field bumps inline, Git is source of truth).

## When to bump a prompt's version

| Trigger | Action |
|---|---|
| New persona, new instructions, new examples, new constraints | Bump `version`; append a changelog entry; commit. |
| Model-tier swap, temperature or sampling-parameter change | Bump `version`; changelog entry. |
| Schema change in the output contract | Bump `version`; coordinate with the consuming Pydantic model in `domain/response_models.py` in the same PR. |
| Typo fix, comment-only change, formatting tidy | Do **not** bump `version`; the deployed behaviour does not change. |

## Status field convention

Each YAML's frontmatter may declare its status:

| `status:` value (or omitted) | Meaning |
|---|---|
| (omitted) | Canonical production prompt actively loaded by the agent. |
| `legacy` | Canonical filename retained for backward compatibility, but the production code path no longer loads it (see `superseded_by:` field). |

`sufficiency_check.yaml` is the current legacy example: it predates the unified
ReAct probe in `react_sufficiency.yaml` v5+ and is preserved as a regression
baseline only.

## Current state (2026-05-04)

| File | Version | Status | Notes |
|---|---|---|---|
| `react_sufficiency.yaml` | v6 | active | CREATE-framework adoption; centerpiece. |
| `rubric_fitness.yaml` | v3 | active | CREATE-framework adoption. |
| `topic_extraction.yaml` | v2 | active | CREATE-framework adoption. |
| `rubric_synthesis.yaml` | v3 | active | CREATE-framework adoption. |
| `sufficiency_check.yaml` | v2 | legacy | Superseded by `react_sufficiency.yaml v5+` in the production code path; kept as regression baseline. |

The CREATE prompt-engineering framework (Birss: Character / Request / Examples
/ Adjustments / Type of output / Extras) is the explicit prompt scaffold for
all four active prompts. The most consequential change introduced by this
adoption was the EXAMPLES section that was previously absent across all five
prompts; see Section 2.6 of the Classification Agent individual report for the
full coverage audit and design-intent rationale.

## Promotion safety

When a prompt's `version` is bumped, the deployment is gated by:

1. Per-PR DeepEval smoke (12 cases, 4 metrics) on the dev branch.
2. Per-PR Promptfoo OWASP smoke.
3. Full prompt-regression gate on merge to `main` (planned; modelled on the
   Validator Agent's `prompt_regression_n_data_versioning_ci.md` pattern: pull
   golden corpus from GCS, run real-pipeline scoring with DeepEval GEval,
   absolute thresholds + 10% relative tolerance against the latest baseline).

Until the full regression gate is wired up, prompt bumps are gated only by the
per-PR smoke; this is tracked as **debt** under Section B.6.7 of the agent's
individual report.
