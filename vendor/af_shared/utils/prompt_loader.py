"""Vendored shim — prompt loader. See vendor/af_shared/__init__.py.

Two layouts are supported:

1. Legacy flat-file layout: ``prompts/<template>.yaml`` (single file per
   template). The version is the ``version`` frontmatter field on that file.

2. Directory + MANIFEST layout: ``prompts/<template>/<vN>.yaml`` with a
   ``MANIFEST.yaml`` sibling whose ``current`` field names the active version
   file. Older versions remain in the directory as browseable artefacts.

The loader accepts a ``Path`` pointing at either a file (legacy) or a
directory (new). The directory variant resolves through MANIFEST.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_MANIFEST_FILENAME = "MANIFEST.yaml"


def _resolve_template_path(path: Path) -> Path:
    """Return the actual prompt YAML file given a template path.

    If ``path`` is a directory, read its MANIFEST.yaml ``current`` field and
    return the resolved YAML. If ``path`` is a file, return it unchanged.
    """
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Prompt path not found: {path}")

    manifest_path = path / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Prompt directory missing {_MANIFEST_FILENAME}: {path}"
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    current = manifest.get("current")
    if not current:
        raise ValueError(
            f"MANIFEST.yaml missing 'current' field: {manifest_path}"
        )
    # 'current' may be 'v6' or 'v6.yaml'
    candidate_names = (current, f"{current}.yaml") if not current.endswith(".yaml") else (current,)
    for name in candidate_names:
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"MANIFEST.yaml 'current'={current!r} but no matching file in {path}"
    )


def load_prompt(prompt_path: Path) -> tuple[dict[str, Any], str]:
    """Load a prompt's frontmatter + body. Accepts a file or a directory."""
    resolved = _resolve_template_path(prompt_path)
    raw = resolved.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"Prompt file must start with '---': {resolved}")
    parts = raw.split("---", maxsplit=2)
    if len(parts) < 3:
        raise ValueError(f"Invalid frontmatter: {resolved}")
    frontmatter: dict[str, Any] = yaml.safe_load(parts[1].strip()) or {}
    return frontmatter, parts[2].strip()


def get_prompt_version(prompt_path: Path) -> str:
    """Return the agent/template@vN tag for a prompt path.

    For directory layout, the template name comes from the directory and the
    version from the MANIFEST-resolved YAML's frontmatter. For legacy flat
    files, the template name comes from the file stem.
    """
    resolved = _resolve_template_path(prompt_path)
    frontmatter, _ = load_prompt(resolved)
    version = frontmatter.get("version")
    if version is None:
        raise ValueError(f"Prompt file missing 'version': {resolved}")

    if prompt_path.is_dir():
        template_name = prompt_path.name
        prompts_dir = prompt_path.parent
    else:
        template_name = resolved.stem
        prompts_dir = resolved.parent
    if prompts_dir.name == "prompts" and prompts_dir.parent.name:
        agent_name = prompts_dir.parent.name
    else:
        agent_name = prompts_dir.name
    return f"{agent_name}/{template_name}@v{version}"


def list_prompt_versions(template_dir: Path) -> list[str]:
    """List all version files (``vN.yaml``) under a template directory.

    Returns version strings (e.g. ``["v3", "v4", "v5", "v6"]``) sorted by
    natural numeric order. Excludes MANIFEST.yaml and any non-vN files.
    """
    if not template_dir.is_dir():
        raise NotADirectoryError(f"Not a template directory: {template_dir}")
    versions: list[tuple[int, str]] = []
    for child in template_dir.iterdir():
        if child.name == _MANIFEST_FILENAME:
            continue
        if child.suffix != ".yaml":
            continue
        stem = child.stem
        if not stem.startswith("v"):
            continue
        try:
            num = int(stem[1:])
        except ValueError:
            continue
        versions.append((num, stem))
    versions.sort(key=lambda t: t[0])
    return [stem for _, stem in versions]
