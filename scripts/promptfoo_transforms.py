"""Promptfoo Python transform helpers — kept minimal."""

from __future__ import annotations


def noop(output, _context=None):
    """Identity transform — promptfoo expects this signature."""
    return output
