"""Smoke tests for rule-based parsing package surface."""

from media_filename_parser import RuleParser, parse_filename


def test_parse_filename_extracts_episode_not_season_year() -> None:
    """`SEASON2024` should not override concrete episode extraction."""
    text = (
        "[CancelWEB][CONAN][SEASON2024][1109-1147][1080P][HEVC_AAC][CHS_CHT_JP]/"
        "[CancelWEB][CONAN][1134][WEBRIP][1080P][HEVC_AAC][CHS_CHT_JP](2A296F98).mkv"
    )
    result = parse_filename(text)
    assert result["parsed"]["season_episode"] == "1134"


def test_rule_parser_batch_parses_multiple_rows() -> None:
    engine = RuleParser()
    rows = engine.parse_many(
        [
            "[CancelWEB][CONAN][1134][WEBRIP][1080P].mkv",
            "Stranger.S02E14.1080p.BluRay.x265.10bit-CancelHD.mkv",
        ]
    )
    assert len(rows) == 2
    assert rows[0]["parsed"]["group"] == "CancelWEB"
    assert rows[1]["parsed"]["season_episode"] == "S02E14"


def test_parse_filename_mixed_cjk_title_keeps_numeric_english_prefix() -> None:
    text = "[首尔之春].12.12.The.Day.2023.HKG.BluRay.1080p.x264.DDP5.1-CancelHD.mkv"
    result = parse_filename(text)
    assert result["parsed"]["title"] == "12 12 The Day"
    assert result["parsed"]["zh_title"] == "首尔之春"
