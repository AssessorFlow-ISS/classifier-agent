# Prompts directory: versioning convention

This directory holds the YAML system prompts the Classification Agent loads at
startup via `vendor/af_shared/utils/prompt_loader.load_prompt`.

## Filename convention

| Filename pattern | Meaning | Loader behaviour |
|---|---|---|
| `<task>.yaml` | **Canonical / production** prompt for that task. The version inside the file (`version:` field) is the deployed version. | Loaded by hardcoded path in `src/classification_agent/domain/*.py` (e.g. `prompts/react_sufficiency.yaml`). |
| `<task>_v<N>.yaml` | **Design-intent** prompt awaiting promotion. Carries the proposed next version of the task's prompt; sibling to the canonical file. | Not loaded by production. Visible to reviewers, regression tests, and CI as the candidate for the next deploy. |
| `_versions/<task>_v<N>.yaml` | **Archived superseded** prompt. Frozen historical record of a version that has been promoted out of production. | Not loaded by production. Available for regression baselines and rollback. |

## Promotion flow (when a `_v<N>.yaml` ships to production)

1. Run the prompt-regression CI gate (`.github/workflows/prompt-regression.yml`)
   against the candidate. The gate reads the `_v<N>.yaml` file directly.
2. If the gate passes (absolute thresholds + 10% relative tolerance against the
   latest baseline on Confident AI), the promotion PR does the swap:
   - Move the current canonical to `_versions/<task>_v<oldN>.yaml`.
   - Rename `<task>_v<newN>.yaml` to `<task>.yaml` (the new canonical).
   - Update the `agent_decision_log.prompt_version` write path so audit rows
     show the new version from the next workflow run onward.
3. The promotion PR runs the full per-merge gate again. Merge to `main` is
   what flips production to the new prompt.

## Why two-file (canonical + `_v<N>`) instead of in-place version bumps?

The Validator Agent's `prompt_regression_n_data_versioning_ci.md` describes
single-file-per-prompt versioning where Git history is the source of truth.
That works when the deploy cadence is "every merge to main is a deploy".
The Classification Agent uses a slower promotion cadence (CREATE-framework
upgrades, eval-baseline shifts) where the design-intent file needs to live
in the repo for review and regression *without* being loaded by production
until a deliberate promotion PR is opened. Keeping both files visible in
`prompts/` makes the diff explicit and lets reviewers see exactly what is
being promoted.

## Status field convention

Each YAML's frontmatter declares its status explicitly:

| `status:` value | Meaning |
|---|---|
| (omitted) | Canonical production prompt. |
| `design-intent` | Awaiting promotion via the regression gate. |
| `archived` | Lives under `_versions/`; preserved for regression baselines and rollback. |
| `legacy` | Canonical filename retained for backward compatibility, but the production code path no longer loads it (see `superseded_by:` field). |

## Current state (2026-05-04)

| Canonical file | Current production version | Design-intent candidate |
|---|---|---|
| `react_sufficiency.yaml` | v5 | `react_sufficiency_v6.yaml` (CREATE-framework adoption) |
| `rubric_fitness.yaml` | v2 | `rubric_fitness_v3.yaml` (CREATE-framework adoption) |
| `topic_extraction.yaml` | v1 | `topic_extraction_v2.yaml` (CREATE-framework adoption) |
| `rubric_synthesis.yaml` | v2 | `rubric_synthesis_v3.yaml` (CREATE-framework adoption) |
| `sufficiency_check.yaml` | v2 (legacy, superseded by `react_sufficiency.yaml v5+`) | None planned. Kept as regression baseline only. |

The four `_v<N>` files in this PR introduce the CREATE prompt-engineering
framework (Birss: Character / Request / Examples / Adjustments / Type of
output / Extras) as the explicit prompt scaffold. The most consequential
change is the EXAMPLES section that was previously absent across all five
prompts; its absence was the dominant gap in the prior CREATE coverage
audit (see `[draft v1] Dale Leung - Classification Agent.md` Section 2.6).
