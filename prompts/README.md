# prompts/

This directory holds the prompt templates the Classification Agent loads
at runtime. Each template is a subdirectory containing per-version YAML
files plus a `MANIFEST.yaml` pointer.

```
prompts/
  react_sufficiency/    # unified ReAct sufficiency + rubric fitness probe
  rubric_fitness/       # standalone rubric fitness assessor
  rubric_synthesis/     # rubric synthesis (no runtime caller yet)
  sufficiency_check/    # standalone sufficiency check (no runtime caller yet)
  topic_extraction/     # topic + subtopic extraction
```

The runtime loads `MANIFEST.yaml` to resolve `current: vN` and reads
`vN.yaml` for the active prompt body. Older `vN.yaml` files remain in
each directory as browseable artefacts.

For the operating procedure (how to bump a version, the regression gate
that fires on merge, rollback, audit queries, the reconstruction policy
for `react_sufficiency` v3/v4), see **[`docs/PROMPT_VERSIONING.md`](../docs/PROMPT_VERSIONING.md)**.
