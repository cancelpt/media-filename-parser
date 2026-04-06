"""Public rule-parser facade for external project integration."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .rules import parse_filename as _parse_filename


def _drop_internal_flags(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal parser-only fields from result payloads."""
    parsed = result.get("parsed")
    if isinstance(parsed, dict):
        parsed.pop("_inherited_title", None)
    return result


class RuleParser:
    """Reusable rule-based parser facade for downstream projects."""

    def parse(self, path_or_name: str) -> Dict[str, Any]:
        """Parse one path/filename with the rules engine."""
        text = (path_or_name or "").strip()
        if not text:
            raise ValueError("Input text is empty.")
        return _drop_internal_flags(_parse_filename(text))

    def parse_many(self, values: Iterable[str]) -> List[Dict[str, Any]]:
        """Parse multiple values, skipping empty items."""
        results: List[Dict[str, Any]] = []
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            results.append(self.parse(text))
        return results


def parse_filename(path_or_name: str) -> Dict[str, Any]:
    """Parse one filename/path with rule parser only."""
    return RuleParser().parse(path_or_name)


def parse_batch(values: Iterable[str]) -> List[Dict[str, Any]]:
    """Parse multiple filenames/paths with rule parser only."""
    return RuleParser().parse_many(values)
