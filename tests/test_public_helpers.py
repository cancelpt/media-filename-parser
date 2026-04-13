"""Tests for typed parser facade and query-name helper."""

import pytest

from media_filename_parser.parser import (
    ParsedMediaName,
    build_query_name,
    parse_filename_typed,
)


def test_parse_filename_typed_exposes_stable_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        "media_filename_parser.parser.parse_filename",
        lambda _text: {
            "raw_path": "input-name",
            "filename": "input-name.mkv",
            "parsed": {
                "title": "Black Mirror",
                "zh_title": "黑镜",
                "year": "2011",
                "season_episode": "S02E03",
            },
            "confidence": 0.91,
        },
    )

    result = parse_filename_typed("input-name")
    assert isinstance(result, ParsedMediaName)
    assert result.title == "Black Mirror"
    assert result.zh_title == "黑镜"
    assert result.year == "2011"
    assert result.season_episode == "S02E03"
    assert result.confidence == pytest.approx(0.91)


def test_build_query_name_prefers_zh_title_and_season_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "media_filename_parser.parser.parse_filename",
        lambda _text: {
            "raw_path": "raw",
            "filename": "raw",
            "parsed": {
                "title": "Black Mirror",
                "zh_title": "黑镜",
                "year": "2011",
                "season_episode": "S02E03",
            },
            "confidence": 0.95,
        },
    )

    assert build_query_name("raw") == "黑镜.2011.S02"


def test_build_query_name_can_keep_episode_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(
        "media_filename_parser.parser.parse_filename",
        lambda _text: {
            "raw_path": "raw",
            "filename": "raw",
            "parsed": {
                "title": "Black Mirror",
                "zh_title": "黑镜",
                "year": "2011",
                "season_episode": "S02E03",
            },
            "confidence": 0.95,
        },
    )

    assert (
        build_query_name("raw", prefer_zh=False, season_only=False)
        == "Black Mirror.2011.S02E03"
    )


def test_build_query_name_falls_back_to_raw_text_when_parse_fails(monkeypatch) -> None:
    def _raise(_text: str):
        raise RuntimeError("parse failed")

    monkeypatch.setattr("media_filename_parser.parser.parse_filename", _raise)
    assert build_query_name("Some.File.2020.mkv") == "Some.File.2020.mkv"
