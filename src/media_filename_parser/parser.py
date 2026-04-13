"""Public rule-parser facade for external project integration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ParsedMediaName:  # pylint: disable=too-many-instance-attributes
    """Typed parse payload for stable downstream integrations."""

    raw_path: str = ""
    filename: str = ""
    title: str = ""
    zh_title: str = ""
    year: str = ""
    season_episode: str = ""
    confidence: float = 0.0
    parsed: Dict[str, Any] = field(default_factory=dict)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_filename_typed(path_or_name: str) -> ParsedMediaName:
    """Parse one value and normalize into a typed integration-friendly result."""
    raw_result = parse_filename(path_or_name)
    if not isinstance(raw_result, dict):
        raise TypeError("parse_filename() returned a non-dict result")

    parsed_raw = raw_result.get("parsed", {})
    parsed_dict = parsed_raw if isinstance(parsed_raw, dict) else {}

    confidence_raw = raw_result.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0

    return ParsedMediaName(
        raw_path=_as_text(raw_result.get("raw_path")) or _as_text(path_or_name),
        filename=_as_text(raw_result.get("filename")),
        title=_as_text(parsed_dict.get("title")),
        zh_title=_as_text(parsed_dict.get("zh_title")),
        year=_as_text(parsed_dict.get("year")),
        season_episode=_as_text(parsed_dict.get("season_episode")),
        confidence=confidence,
        parsed=parsed_dict,
    )


def _extract_query_season_token(season_episode: str, season_only: bool) -> str:
    normalized = _as_text(season_episode).upper()
    if not normalized:
        return ""

    if not season_only:
        return normalized

    season_match = re.search(r"S(\d{1,2})", normalized)
    if season_match is None:
        return ""
    return f"S{int(season_match.group(1)):02d}"


def build_query_name(
    path_or_name: str,
    prefer_zh: bool = True,
    season_only: bool = True,
) -> str:
    """
    Build a normalized query name for downstream media search.

    Fallbacks to the input text if parsing fails or title extraction is empty.
    """
    raw_text = _as_text(path_or_name)
    if not raw_text:
        return raw_text

    try:
        typed = parse_filename_typed(raw_text)
    except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover
        return raw_text

    if prefer_zh:
        title = typed.zh_title or typed.title
    else:
        title = typed.title or typed.zh_title
    title = _as_text(title)
    if not title:
        return raw_text

    output_parts = [title]
    if typed.year:
        output_parts.append(typed.year)

    season_token = _extract_query_season_token(typed.season_episode, season_only)
    if season_token:
        output_parts.append(season_token)

    return ".".join(output_parts) if output_parts else raw_text
