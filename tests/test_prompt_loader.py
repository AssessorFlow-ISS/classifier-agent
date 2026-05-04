"""Tests for vendor/af_shared/utils/prompt_loader.py — directory + MANIFEST layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from af_shared.utils.prompt_loader import (
    get_prompt_version,
    list_prompt_versions,
    load_prompt,
)

_FRONTMATTER_V3 = """---
version: "3"
model_tier: CHEAP
description: "v3 reconstructed"
created: "2026-04-05"
---
body v3
"""
_FRONTMATTER_V6 = """---
version: "6"
model_tier: CHEAP
description: "v6 active"
created: "2026-05-04"
---
body v6
"""


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_legacy_flat_file_layout_still_loads(tmp_path):
    flat = tmp_path / "prompts" / "topic_extraction.yaml"
    _write(flat, _FRONTMATTER_V6)

    fm, body = load_prompt(flat)

    assert fm["version"] == "6"
    assert body == "body v6"


def test_directory_layout_resolves_via_manifest(tmp_path):
    template_dir = tmp_path / "prompts" / "react_sufficiency"
    _write(template_dir / "v3.yaml", _FRONTMATTER_V3)
    _write(template_dir / "v6.yaml", _FRONTMATTER_V6)
    _write(
        template_dir / "MANIFEST.yaml",
        "template: react_sufficiency\ncurrent: v6\nhistory: []\n",
    )

    fm, body = load_prompt(template_dir)

    assert fm["version"] == "6"
    assert body == "body v6"


def test_get_prompt_version_directory_layout(tmp_path):
    template_dir = tmp_path / "prompts" / "react_sufficiency"
    _write(template_dir / "v6.yaml", _FRONTMATTER_V6)
    _write(
        template_dir / "MANIFEST.yaml",
        "template: react_sufficiency\ncurrent: v6\nhistory: []\n",
    )

    tag = get_prompt_version(template_dir)

    assert tag.endswith("/react_sufficiency@v6")


def test_get_prompt_version_legacy_flat(tmp_path):
    flat = tmp_path / "prompts" / "topic_extraction.yaml"
    _write(flat, _FRONTMATTER_V6)

    tag = get_prompt_version(flat)

    assert tag.endswith("/topic_extraction@v6")


def test_missing_manifest_raises(tmp_path):
    template_dir = tmp_path / "prompts" / "react_sufficiency"
    _write(template_dir / "v6.yaml", _FRONTMATTER_V6)

    with pytest.raises(FileNotFoundError, match="MANIFEST.yaml"):
        load_prompt(template_dir)


def test_manifest_current_pointing_at_missing_file_raises(tmp_path):
    template_dir = tmp_path / "prompts" / "react_sufficiency"
    _write(template_dir / "v6.yaml", _FRONTMATTER_V6)
    _write(
        template_dir / "MANIFEST.yaml",
        "template: react_sufficiency\ncurrent: v9\nhistory: []\n",
    )

    with pytest.raises(FileNotFoundError, match="no matching file"):
        load_prompt(template_dir)


def test_list_prompt_versions_natural_order(tmp_path):
    template_dir = tmp_path / "prompts" / "react_sufficiency"
    for v in ("v3", "v4", "v5", "v6"):
        body = _FRONTMATTER_V6.replace('version: "6"', f'version: "{v[1:]}"')
        _write(template_dir / f"{v}.yaml", body)
    _write(
        template_dir / "MANIFEST.yaml",
        "template: react_sufficiency\ncurrent: v6\nhistory: []\n",
    )

    versions = list_prompt_versions(template_dir)

    assert versions == ["v3", "v4", "v5", "v6"]


def test_missing_version_field_raises(tmp_path):
    flat = tmp_path / "prompts" / "broken.yaml"
    _write(flat, "---\nmodel_tier: CHEAP\n---\nbody\n")

    with pytest.raises(ValueError, match="missing 'version'"):
        get_prompt_version(flat)
