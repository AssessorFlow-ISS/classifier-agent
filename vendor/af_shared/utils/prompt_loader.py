"""Vendored shim — prompt loader. See vendor/af_shared/__init__.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_prompt(prompt_path: Path) -> tuple[dict[str, Any], str]:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    raw = prompt_path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"Prompt file must start with '---': {prompt_path}")
    parts = raw.split("---", maxsplit=2)
    if len(parts) < 3:
        raise ValueError(f"Invalid frontmatter: {prompt_path}")
    frontmatter: dict[str, Any] = yaml.safe_load(parts[1].strip()) or {}
    return frontmatter, parts[2].strip()


def get_prompt_version(prompt_path: Path) -> str:
    frontmatter, _ = load_prompt(prompt_path)
    version = frontmatter.get("version")
    if version is None:
        raise ValueError(f"Prompt file missing 'version': {prompt_path}")
    template_name = prompt_path.stem
    prompts_dir = prompt_path.parent
    if prompts_dir.name == "prompts" and prompts_dir.parent.name:
        agent_name = prompts_dir.parent.name
    else:
        agent_name = prompts_dir.name
    return f"{agent_name}/{template_name}@v{version}"
