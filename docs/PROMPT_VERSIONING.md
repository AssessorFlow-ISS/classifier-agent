# Prompt Versioning — classifier-agent

> Operating procedure for the per-prompt directory + `MANIFEST.yaml`
> layout used by all prompts in this repo. Last updated 2026-05-05.

This document is the source of truth for how prompt versions are stored,
how the runtime resolves "the active version", how to bump a version, how
to roll one back, and how to audit "what prompt produced run X".

---

## Layout

```
prompts/
  <template>/
    v<N>.yaml          # one file per version; current is the highest N
    v<N-1>.yaml        # prior version (browseable; not loaded at runtime)
    ...
    MANIFEST.yaml      # 'current: vN' pointer + history list
```

**Why directories instead of flat files or sibling `_vN.yaml` files?**

1. **Flat-file in-place bumps lose history.** The convention before this
   refactor was to bump the `version:` frontmatter field on a single file
   per template; the prior version's body was overwritten. History was
   only recoverable via `git checkout <SHA>:prompts/<template>.yaml`.
   For audit ("what was the active prompt during run X?"), that requires
   knowing the SHA — which is fine in theory, but not surfaced anywhere in
   the trace data unless we attach it explicitly.

2. **Sibling `_vN.yaml` files in the same flat directory are fragile.**
   An earlier attempt (`a985b37` on `feat/prompts-create-framework`) used
   `react_sufficiency_v6.yaml` next to `react_sufficiency.yaml`. The
   problem: Python import paths and tooling expect a single canonical
   filename, and a future contributor can accidentally edit the canonical
   one when they meant to bump. Commit `d98cc18` reverted that approach
   for exactly this reason.

3. **Subdirectory per template is structurally distinct.** There is no
   `prompts/react_sufficiency.yaml` — only `prompts/react_sufficiency/`.
   You cannot "accidentally edit the canonical file in place" because
   there is no canonical flat file. Every edit is a discrete `vN.yaml`
   addition + `MANIFEST.yaml` update.

The trade-off: the directory grows over time. That is the intended
behaviour — older versions remain visible for inspection and comparison
without a `git checkout`. See `prompts/react_sufficiency/` for the
worked example: v3.yaml, v4.yaml, v5.yaml, v6.yaml are all browseable.

---

## MANIFEST.yaml schema

Every template directory contains exactly one `MANIFEST.yaml`:

```yaml
template: <template-name>          # must match the directory name
current: v<N>                      # active version; the loader resolves to this
history:                           # newest first
  - version: v<N>
    file: v<N>.yaml
    body_preserved: true           # true if the v<N>.yaml in this dir holds the
                                   # actual body that was active at this version
    reconstructed: false           # true if the file was reconstructed after the
                                   # fact (body_preserved must then be false)
    summary: |
      One- to two-sentence prose explaining what changed in this version.
    created: "YYYY-MM-DD"          # optional — frontmatter `created` is canonical
```

`body_preserved: false` is permitted and honest. It means the version is
recorded in the history list for provenance but its body was never
committed as a standalone file (e.g. an in-place bump from before this
convention existed). The corresponding `vN.yaml` file in the directory
will then be a frontmatter-only placeholder with a `preservation_note:`
field. This is the case for `react_sufficiency/v3.yaml` and `v4.yaml`.

Future versions starting from any first-bump-after-this-doc-lands MUST
ship with `body_preserved: true`. Reconstruction-flagged entries are not
permitted for new bumps; either preserve or do not record.

---

## Loader behaviour

`vendor/af_shared/utils/prompt_loader.py` exposes:

| Function | Accepts | Returns |
|---|---|---|
| `load_prompt(path)` | file or directory | `(frontmatter, body)` |
| `get_prompt_version(path)` | file or directory | `<agent>/<template>@v<N>` tag |
| `list_prompt_versions(template_dir)` | directory only | `["v3", "v4", ...]` sorted |

When `path` is a directory, the loader reads `MANIFEST.yaml`, takes the
`current` field, and resolves to that `vN.yaml`. When `path` is a file,
the file is loaded directly (legacy flat-file behaviour preserved for
backwards compatibility — but the repo no longer uses flat files).

The path constants in `domain/*.py` point at the directory:

```python
# src/classification_agent/domain/sufficiency.py
_REACT_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "react_sufficiency"
)
```

---

## Runtime tag and the broker

Every LLM call carries a `prompt_version` tag in the format
`<agent>/<template>@v<N>` per ADR-39. The tag is:

1. Resolved at agent construction time via
   `get_prompt_version(_PROMPT_PATH)` and stored on the prober/extractor
   instance.
2. Passed verbatim to `ModelBrokerHttpAdapter.invoke(...)` /
   `invoke_with_tools(...)` as the `prompt_version=` keyword argument.
3. Forwarded verbatim into the broker request body, which is what
   Langfuse reads and indexes.

The broker adapter raises `ValueError` if `prompt_version` is missing —
there is no silent fallback to `@v1`. (The pre-refactor code hardcoded
`f"classification/{task_key.split('.')[-1]}@v1"`, which mislabelled every
v5/v6 trace as v1.)

**Audit query — "what prompt was active during run X?":**

1. Open the Langfuse trace for run X.
2. Find the `prompt_version` tag, e.g. `classification/react_sufficiency@v6`.
3. Open `prompts/react_sufficiency/v6.yaml` in the SHA that was deployed
   when run X happened. (The deployment SHA is captured in pod labels +
   GHA build artifacts.)

---

## How to bump a version

Mechanical procedure for promoting a prompt change to a new version:

```bash
TEMPLATE=react_sufficiency        # or topic_extraction, etc.
NEW=v7                            # next version
PRIOR=v6                          # current

# 1. Copy the prior version as the starting point
cp prompts/$TEMPLATE/$PRIOR.yaml prompts/$TEMPLATE/$NEW.yaml

# 2. Edit the new file:
#    - bump the `version:` frontmatter field to "7"
#    - update `updated:` to today's date
#    - prepend a one-paragraph entry to the `changelog:` block
#    - make whatever prompt changes drove the bump
$EDITOR prompts/$TEMPLATE/$NEW.yaml

# 3. Update MANIFEST.yaml
#    - flip `current:` to v7
#    - prepend a new history entry with body_preserved: true
$EDITOR prompts/$TEMPLATE/MANIFEST.yaml

# 4. Run unit tests locally
PYTHONPATH=$(pwd)/vendor uv run pytest tests/ -k "not eval and not adversarial" --no-cov

# 5. Commit on a feature branch (NEVER on main directly per
#    feedback_service_repo_main_readonly), open PR
git switch -c feat/prompt-bump-$TEMPLATE-$NEW
git add prompts/$TEMPLATE/
git commit -m "feat(prompts): bump $TEMPLATE to $NEW"
git push origin HEAD
gh pr create
```

After the PR merges to `main`, the path-filtered prompt-regression
workflow (`.github/workflows/prompt-regression.yml`) fires automatically
on the merge commit. See **Regression gate** below.

---

## Regression gate

`.github/workflows/prompt-regression.yml` is a **post-merge** path-filtered
workflow that exercises the full DeepEval drift suite + Promptfoo OWASP
smoke against the bumped prompt.

| Aspect | Value |
|---|---|
| Trigger | `push: branches: [main], paths: ['prompts/**']` |
| Runs | DeepEval drift quality (4 metrics) + Promptfoo OWASP smoke + DeepTeam smoke |
| Gate type | Post-merge — produces a red signal on `main` if scores regress, but does NOT block the PR merge |
| Failure handling | Revert the prompt commit + reset MANIFEST `current` to the prior version; cut a follow-up PR with a fix |

**Why post-merge, not pre-merge?**

The full DeepEval suite costs LLM tokens and ~10–15 min wall. Pre-merge
gating every PR (including non-prompt PRs) would be wasteful — the path
filter already ensures it only runs when prompts change, so a pre-merge
variant would only differ in *when* it fires. Post-merge red on `main` is
actionable: a single revert PR rolls back both the prompt bump and the
MANIFEST `current` pointer in one step.

This is honest. The gate is not a pre-merge guard. The trade-off is
documented as residual debt under §B.6.7 of the individual report; a
future upgrade to pre-merge is in scope when the cost/wall-time profile
becomes acceptable (e.g. when the smoke set is pruned to a 12-case
fast-cycle that runs in 90 seconds).

---

## How to roll back a version

```bash
TEMPLATE=react_sufficiency
PRIOR=v5         # the version we want to roll back TO
BAD=v6           # the version we want to retire

# 1. Edit MANIFEST.yaml: set current: $PRIOR
$EDITOR prompts/$TEMPLATE/MANIFEST.yaml

# 2. (Optional) leave $BAD.yaml in the directory — history is preserved
#    even when the version is rolled back. The MANIFEST history list
#    documents that $BAD was active for the period between its bump
#    and this rollback.

# 3. Commit + PR
git switch -c fix/prompt-rollback-$TEMPLATE-$BAD-to-$PRIOR
git add prompts/$TEMPLATE/MANIFEST.yaml
git commit -m "fix(prompts): roll $TEMPLATE back from $BAD to $PRIOR"
gh pr create
```

The regression gate fires on the rollback merge as well — if the prior
version still scores green against the current drift suite, the rollback
is confirmed safe.

---

## Reconstruction policy

When migrating to this convention, two `react_sufficiency` versions (v3,
v4) had no preserved body — they were created via in-place bumps before
per-version archival existed. They are recorded in `MANIFEST.yaml`
history with `body_preserved: false, reconstructed: true` and shipped as
frontmatter-only placeholder files.

**No fabrication.** The placeholder bodies contain only a
`preservation_note:` field explaining the gap and pointing at the closest
recoverable reference (typically the next-newest version which built on
the missing one). Future operators reading the v3.yaml or v4.yaml files
in `prompts/react_sufficiency/` will see the gap explicitly rather than
encountering invented content.

---

## Cross-references

- `vendor/af_shared/utils/prompt_loader.py` — loader implementation
- `tests/test_prompt_loader.py` — unit tests for both layouts
- `src/classification_agent/adapters/model_broker_http.py` — broker adapter that
  forwards `prompt_version` (no `@v1` hardcode)
- `[draft v1] Dale Leung — Classification Agent.md` §2.6.5 / §2.6.7 —
  the individual report's centerpiece walkthrough and versioning-convention
  narrative both reference this document
- `docs/HONEST_GOLDEN_TESTING.md` — debt table (the "no main-branch
  regression gate" item is closed by `.github/workflows/prompt-regression.yml`)
