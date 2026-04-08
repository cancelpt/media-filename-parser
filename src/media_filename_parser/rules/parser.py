"""Core rule-based parsing pipeline for media filenames and paths."""

import re

from .confidence import calculate_confidence
from .constants import (
    P_ACODEC,
    P_FPS,
    P_MISC,
    P_RES,
    P_SEP,
    P_SOURCE,
    P_VCODEC,
    P_VHDR,
    P_YEAR,
)
from .extraction import (
    extract_season_episode,
    extract_sports_event_title,
    extract_with_pattern,
    looks_like_technical_disc_title,
    resolve_metadata_parent_dir,
)


TECH_META_PATTERNS = (P_RES, P_SOURCE, P_VCODEC, P_ACODEC, P_VHDR, P_FPS, P_YEAR)

RE_PROMO_BLOCK_FULLWIDTH = re.compile(
    r"\u3010[^\u3011]*?(?:www|bbs|\.com|\.net|\.cc|\.tv|\.xyz|vip|\u8bba\u575b)[^\u3011]*?\u3011",
    re.I,
)
RE_PROMO_BLOCK_ASCII = re.compile(
    r"\[[^\]]*?(?:www|bbs|\.com|\.net|\.cc|\.tv|\.xyz|vip|\u8bba\u575b)[^\]]*?\]",
    re.I,
)
RE_KNOWN_MEDIA_EXT = re.compile(
    r"(?i)\.(mkv|mp4|avi|ts|m2ts|iso|vob|flv|wmv|mov|mpg|mpeg|rmvb|webm|m4v)$"
)

RE_NUM_1_TO_4 = re.compile(r"\d{1,4}")
RE_NUM_1_TO_3 = re.compile(r"\d{1,3}")
RE_NUM_LONG = re.compile(r"\d{5,}")
RE_HEXISH_ID = re.compile(r"[A-Fa-f0-9]{7,}")
RE_YEAR_IN_CN = re.compile(r"\d{2,4}年")
RE_YEAR_IN_BRACKET = re.compile(r"(?<!\d)(19\d{2}|20\d{2}|\d{2})年")
RE_EPISODE_TOKEN = re.compile(r"(?i)(S\d{1,2}E?\d{0,3}|E\d{1,3}|EP\d{1,3}|D\d{1,3})")
RE_DISC_INDEX_TOKEN = re.compile(r"D\d{1,3}")
RE_SEASON_TAG = re.compile(r"(?i)^SEASON\s*\d{1,4}$")
RE_SEASON_YEAR = re.compile(r"(?i)^SEASON\s*[-_.]?\s*(19\d{2}|20[0-3]\d)$")
RE_SEASON_YEAR_SEARCH = re.compile(
    r"(?i)(?<![a-zA-Z0-9])SEASON\s*[-_.]?\s*(19\d{2}|20[0-3]\d)(?![a-zA-Z0-9])"
)
RE_EPISODE_RANGE = re.compile(r"^\d{2,4}\s*-\s*\d{2,4}$")

RE_META_WORDS = re.compile(r"(?i)(END|中字|双语|字幕|字幕组|日剧|中日|合集|完结)")
RE_GROUP_ZH_HINT = re.compile(r"(字幕组|字幕社|压制组|汉化组|工作室|剧社)")
RE_SUB_LANG_TAG = re.compile(r"(?i)\b(?:CHS|CHT|ENG|EN|JP|JPN|SUB|SUBBED|CN|ZH)\b")
RE_SUB_LANG_TAG_STRICT = re.compile(
    r"(?i)^(?:CHS|CHT|ENG|EN|JP|JPN|SUB|SUBBED|CN|ZH)(?:[_\-\s](?:CHS|CHT|ENG|EN|JP|JPN|SUB|CN|ZH))*$"
)

RE_HAS_CJK = re.compile(r"[\u4e00-\u9fa5]")
RE_HAS_JA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
RE_HAS_LATIN = re.compile(r"[A-Za-z]")
RE_MULTI_SPACE = re.compile(r"\s+")
RE_NON_LATIN = re.compile(r"[^A-Za-z]+")
RE_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
RE_HDR_BIT_DEPTH = re.compile(r"(?i)^(?:8|10|12)bit$")
RE_FPS_NORMALIZE = re.compile(r"(?i)^(\d{2,3}(?:\.\d{1,3})?)\s*fps$")
RE_HDR_ONLY = re.compile(r"(?i)^HDR$")
RE_SDR_ONLY = re.compile(r"(?i)^SDR$")
RE_HDR_SPECIFIC = re.compile(r"(?i)(?:HDR10\+?|HDR[\s\._-]?Vivid|DV|DoVi|HLG)")
RE_TITLE_NOISE_TOKEN = re.compile(
    r"(?i)^(?:\d{2,3}fps|hdr(?:10\+)?|sdr|dovi|dv|vs|10bit|12bit|8bit)$"
)
RE_WORD_OR_CJK = re.compile(r"[\w\u4e00-\u9fa5]")
RE_UPPER_2_PLUS = re.compile(r"[A-Z]{2,}")
RE_PAREN_BLOCK = re.compile(r"[（(]([^()（）]{1,120})[)）]")
RE_CONTAINER_TAG = re.compile(
    r"(?i)\b(?:mp4|mkv|avi|m2ts|ts|flv|wmv|mov|webm|rmvb|iso|vob)\b"
)
RE_TECH_LIKE_TOKEN = re.compile(
    r"(?i)^(?:\d{2,4}p|\d{2,4}x\d{2,4}|\d+(?:\.\d+)?fps|\d+bit)$"
)

RE_HEAD_BRACKET = re.compile(r"^\[(.*?)\]")
RE_ALL_BRACKETS = re.compile(r"\[(.*?)\]")
RE_BRACKET_BLOCK = re.compile(r"\[.*?\]")
RE_ASCII_BRACKET_BLOCK = re.compile(r"\[[a-zA-Z0-9\s\.\-\_\+\!\&]+?\]")
RE_OUTSIDE_COMPACT_SEP = re.compile(r"[\s\(\)\[\]\{\}\.\-_:：]+")
RE_CCTV_CLEAN_PUNCT = re.compile(r"[\[\]\(\)\._]+")

RE_CHANNEL_TAG = re.compile(r"(?i)\b(?:CCTV\d*|ATV|TVB)\b")
RE_CCTV_EXACT = re.compile(r"(?i)CCTV\d*")
RE_CCTV_PREFIX = re.compile(r"(?i)^\s*CCTV\d*\s*[-_\. ]*")

RE_GROUP_TAIL = re.compile(r"[-\uFFE1]\s*([A-Za-z0-9][A-Za-z0-9_@.\-]{1,31})\s*$")
RE_RELEASE_GROUP = re.compile(r"[A-Za-z0-9_@\uFFE1]{2,24}")
RE_GROUP_ASCII_TAG = re.compile(r"[A-Za-z0-9_\-@&\+\.\s]{1,48}")
RE_PARENT_GROUP_SUFFIX = re.compile(r"-([a-zA-Z0-9_\.@]+)$")
RE_SCENE_GROUP_END = re.compile(
    r"[-\uFFE1]([a-zA-Z0-9_@\uFFE1]+)(?:[.\-_](?:sample|trailer|preview|clip))?$",
    re.I,
)
RE_SCENE_GROUP_DOT = re.compile(
    r"\.([a-zA-Z0-9_@\uFFE1]+)(?:[.\-_](?:sample|trailer|preview|clip))?$",
    re.I,
)

RE_TITLE_SEP = re.compile(r"[\.\_\-\(\)]")
RE_SUBTITLE_EN = re.compile(
    r"(?i)\b(?:ChineseSubbed|ChineseSub|EngSubbed|EngSub|CHS|CHT|CNSub|ZHSub|Subbed)\b"
)
RE_SUBTITLE_ZH = re.compile(
    r"(?:\u4e2d\u5b57|\u5b57\u5e55|\u53cc\u8bed|\u4e2d\u82f1\u53cc\u8bed|\u4e2d\u65e5\u53cc\u8bed|\u7b80\u4e2d|\u7e41\u4e2d|\u7b80\u4f53|\u7e41\u4f53)"
)
RE_STUDIO_ABBR = re.compile(
    r"(?:(?<=\s)|^)(?:西影|上影|北影|长影|峨影|珠影|潇影|中影)(?=\s|$)"
)


def _has_technical_metadata_token(value):
    return any(pat.search(value) for pat in TECH_META_PATTERNS)


def _looks_like_subtitle_lang_tag(value):
    return bool(RE_SUB_LANG_TAG.search(value) or RE_SUB_LANG_TAG_STRICT.search(value))


def _dedupe_tokens_case_insensitive(tokens):
    seen = set()
    out = []
    for token in tokens:
        t = (token or "").strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _normalize_video_hdr_tokens(value):
    if not value:
        return value
    tokens = _dedupe_tokens_case_insensitive(value.split())
    if not tokens:
        return None
    bit_depth = [t.lower() for t in tokens if RE_HDR_BIT_DEPTH.fullmatch(t)]
    hdr_labels = [t for t in tokens if not RE_HDR_BIT_DEPTH.fullmatch(t)]
    has_specific_hdr = any(
        RE_HDR_SPECIFIC.search(t) for t in hdr_labels if not RE_HDR_ONLY.fullmatch(t)
    )
    if has_specific_hdr:
        hdr_labels = [t for t in hdr_labels if not RE_HDR_ONLY.fullmatch(t)]
    has_non_sdr_hdr = any(not RE_SDR_ONLY.fullmatch(t) for t in hdr_labels)
    if has_non_sdr_hdr:
        hdr_labels = [t for t in hdr_labels if not RE_SDR_ONLY.fullmatch(t)]
    ordered = hdr_labels + bit_depth
    return " ".join(ordered) if ordered else None


def _normalize_frame_rate(value):
    if not value:
        return value
    m = RE_FPS_NORMALIZE.fullmatch(value.strip())
    if not m:
        return value
    return f"{m.group(1)}fps"


def _strip_leading_title_noise(text):
    if not text:
        return text
    tokens = text.split()
    idx = 0
    while idx < len(tokens):
        raw = tokens[idx]
        token = raw.strip("[](){}<>")
        if RE_TITLE_NOISE_TOKEN.fullmatch(token):
            idx += 1
            continue
        if token.isdigit() and idx + 1 < len(tokens):
            nxt = tokens[idx + 1].strip("[](){}<>")
            if RE_TITLE_NOISE_TOKEN.fullmatch(nxt):
                idx += 1
                continue
        break
    return " ".join(tokens[idx:]) if idx < len(tokens) else ""


def _strip_parenthetical_title_noise(text):
    if not text:
        return text

    meta_keywords = {
        "BAHA",
        "BILIBILI",
        "BILI",
        "WEB",
        "WEBRIP",
        "WEBDL",
        "BLURAY",
        "BDRIP",
        "REMUX",
        "AVC",
        "HEVC",
        "X264",
        "X265",
        "AAC",
        "FLAC",
        "DTS",
        "TRUEHD",
        "ATMOS",
        "MP4",
        "MKV",
        "AVI",
        "TS",
        "M2TS",
        "ISO",
        "VOB",
        "WEBM",
        "RMVB",
        "NF",
        "NETFLIX",
        "AMZN",
        "ATVP",
        "DSNP",
        "HMAX",
    }

    def _replace(match):
        inner = (match.group(1) or "").strip()
        if not inner:
            return " "
        if RE_HAS_CJK.search(inner):
            return match.group(0)
        if (
            _has_technical_metadata_token(inner)
            or RE_CONTAINER_TAG.search(inner)
            or _looks_like_subtitle_lang_tag(inner)
        ):
            return " "

        tokens = [tok for tok in re.split(r"[\s._-]+", inner) if tok]
        if not tokens:
            return " "
        meta_hits = 0
        for tok in tokens:
            up = tok.upper()
            if up in meta_keywords or RE_TECH_LIKE_TOKEN.fullmatch(tok):
                meta_hits += 1
        if meta_hits >= max(1, len(tokens) - 1):
            return " "
        return match.group(0)

    return RE_PAREN_BLOCK.sub(_replace, text)


def _extract_season_year_tag(text):
    if not text:
        return None

    for bracket_text in RE_ALL_BRACKETS.findall(text):
        match = RE_SEASON_YEAR.fullmatch((bracket_text or "").strip())
        if match:
            return match.group(1)

    match = RE_SEASON_YEAR_SEARCH.search(text)
    if match:
        return match.group(1)
    return None


def parse_filename(filepath):
    """Parse one media path/filename into structured fields with confidence."""
    filepath = (filepath or "").lstrip("\ufeff")

    # Pre-clean promotional wrappers before doing anything to prevent URL/promo pollution.
    filepath = RE_PROMO_BLOCK_FULLWIDTH.sub("", filepath)
    filepath = RE_PROMO_BLOCK_ASCII.sub("", filepath)

    # Normalize full-width punctuation to improve mixed-encoding filename handling.
    filepath_normalized = (
        filepath.replace("\u3010", "[")
        .replace("\u3011", "]")
        .replace("\uff08", "(")
        .replace("\uff09", ")")
    )

    # Extract final file name
    original_parts = filepath.replace("\\", "/").strip().split("/")
    filename = original_parts[-1]
    parts = filepath_normalized.replace("\\", "/").strip().split("/")
    parent_dir, is_disc_layout = resolve_metadata_parent_dir(parts)
    basename = parts[-1]
    # Only strip known media extensions; keep full text for dot-heavy names
    # without real file extensions (e.g. "...Atmos-CancelHD").
    known_ext = RE_KNOWN_MEDIA_EXT.search(basename)
    if known_ext:
        name_no_ext = basename[: known_ext.start()]
    else:
        name_no_ext = basename

    parsed = {
        "title": None,
        "zh_title": "",
        "year": None,
        "season_episode": None,
        "resolution": None,
        "frame_rate": None,
        "source": None,
        "video_codec": None,
        "video_hdr": None,
        "audio_codec": None,
        "group": None,
    }

    working_name = name_no_ext.strip()
    sports_title_override = extract_sports_event_title(name_no_ext)

    def looks_like_group_tag(text):
        t = (text or "").strip()
        if not t:
            return False
        # Technical tags in leading brackets are usually not release groups.
        if (
            _has_technical_metadata_token(t)
            or RE_NUM_1_TO_4.fullmatch(t)
            or RE_META_WORDS.search(t)
        ):
            return False
        if RE_HAS_CJK.search(t):
            return RE_GROUP_ZH_HINT.search(t) is not None
        if RE_HAS_JA.search(t):
            return len(t) <= 20
        tokens = RE_MULTI_SPACE.split(t)
        if len(tokens) > 3:
            return False
        return bool(RE_GROUP_ASCII_TAG.fullmatch(t))

    anime_zh_candidate = None
    anime_en_candidate = None

    # Check for anime style: starts with bracket AND doesn't heavily use dot separators (PT/Scene style)
    dot_count = working_name.count(".")
    if working_name.startswith("[") and dot_count < 3:
        group_match = RE_HEAD_BRACKET.search(working_name)
        if group_match:
            first_bracket = group_match.group(1).strip()
            if looks_like_group_tag(first_bracket):
                parsed["group"] = first_bracket
                working_name = working_name[group_match.end() :]

        # Grab all bracket parts for metadata
        brackets = RE_ALL_BRACKETS.findall(working_name)
        bracket_title_candidates = []
        for b in brackets:
            bracket_text = b.strip()
            if not parsed["resolution"]:
                res, _ = extract_with_pattern(P_RES, bracket_text)
                parsed["resolution"] = res
            if not parsed["source"]:
                src, _ = extract_with_pattern(P_SOURCE, bracket_text)
                parsed["source"] = src
            if not parsed["video_codec"]:
                vc, _ = extract_with_pattern(P_VCODEC, bracket_text)
                parsed["video_codec"] = (
                    vc if "vc" in locals() else None
                )  # safe fallback

            if not parsed["year"]:
                y_match = RE_YEAR_IN_BRACKET.search(bracket_text)
                if y_match:
                    y = y_match.group(1)
                    if len(y) == 2:
                        y_num = int(y)
                        y = f"20{y}" if y_num <= 30 else f"19{y}"
                    parsed["year"] = y
                else:
                    season_year = _extract_season_year_tag(bracket_text)
                    if season_year:
                        parsed["year"] = season_year

            # Keep potential title segments from bracket-heavy anime naming style.
            is_numeric_only = RE_NUM_1_TO_3.fullmatch(bracket_text) is not None
            is_episode_range = RE_EPISODE_RANGE.fullmatch(bracket_text) is not None
            is_season_tag = RE_SEASON_TAG.fullmatch(bracket_text) is not None
            is_hexish_id = RE_HEXISH_ID.fullmatch(bracket_text) is not None
            looks_subtitle_tag = _looks_like_subtitle_lang_tag(bracket_text)
            looks_meta = (
                _has_technical_metadata_token(bracket_text)
                or is_numeric_only
                or is_episode_range
                or is_season_tag
                or is_hexish_id
                or looks_subtitle_tag
                or RE_META_WORDS.search(bracket_text)
                or RE_YEAR_IN_CN.search(bracket_text)
            )
            if not looks_meta and len(bracket_text) >= 2:
                bracket_title_candidates.append(bracket_text)
                if RE_HAS_CJK.search(bracket_text):
                    if anime_zh_candidate is None or len(bracket_text) > len(
                        anime_zh_candidate
                    ):
                        anime_zh_candidate = bracket_text
                elif RE_HAS_LATIN.search(bracket_text):
                    if anime_en_candidate is None or len(bracket_text) > len(
                        anime_en_candidate
                    ):
                        anime_en_candidate = bracket_text
            # Bracket content is removed later; no extra action is needed here.

        # Reparse the full working_name to get all standard attributes out
        # In anime format, attributes are usually within []
        # Let's extract any known attrs from the whole string, including bracket content
        parsed["resolution"], working_name = extract_with_pattern(P_RES, working_name)
        source2, working_name = extract_with_pattern(P_SOURCE, working_name)
        if not parsed["source"] and source2:
            parsed["source"] = source2
        parsed["video_codec"], working_name = extract_with_pattern(
            P_VCODEC, working_name
        )
        parsed["video_hdr"], working_name = extract_with_pattern(
            P_VHDR, working_name, is_multi=True
        )
        parsed["video_hdr"] = _normalize_video_hdr_tokens(parsed["video_hdr"])
        parsed["frame_rate"], working_name = extract_with_pattern(P_FPS, working_name)
        parsed["frame_rate"] = _normalize_frame_rate(parsed["frame_rate"])
        parsed["audio_codec"], working_name = extract_with_pattern(
            P_ACODEC, working_name, is_multi=True
        )
        parsed["year"], working_name = extract_with_pattern(P_YEAR, working_name)
        _, working_name = extract_with_pattern(P_MISC, working_name)

        se, working_name = extract_season_episode(working_name, is_anime=True)
        if se:
            parsed["season_episode"] = se

        # Some anime-style names append release group outside brackets.
        if not parsed["group"]:
            group_tail = RE_GROUP_TAIL.search(working_name)
            if group_tail:
                tail_val = group_tail.group(1)
                if not _has_technical_metadata_token(tail_val):
                    parsed["group"] = tail_val
                    working_name = working_name[: group_tail.start()]

        # Prefer text outside brackets; if empty, fall back to the best non-metadata bracket candidate.
        outside_text = RE_BRACKET_BLOCK.sub(" ", working_name).strip()
        outside_compact = RE_OUTSIDE_COMPACT_SEP.sub("", outside_text)
        outside_is_id_only = bool(outside_compact) and (
            RE_NUM_LONG.fullmatch(outside_compact) is not None
            or RE_HEXISH_ID.fullmatch(outside_compact) is not None
        )
        if outside_text and not outside_is_id_only:
            working_name = outside_text
        elif bracket_title_candidates:
            working_name = max(bracket_title_candidates, key=len)
        else:
            working_name = outside_text

    else:
        # Scene / PT style

        # 1. Determine strict title end BEFORE any property extractions remove text
        matches = list(P_SEP.finditer(working_name))
        title_end = len(working_name)
        for m in matches:
            if m.start() > 3:
                title_end = m.start()
                break

        strict_title = working_name[:title_end]

        # 2. Look for group at the end (usually after '-' or '￡')
        # Some scene groups use '￡' (fullwidth pound U+FFE1), e.g. ￡Cancel@CancelHD
        def is_metadata_like_group_token(val):
            return bool(
                _has_technical_metadata_token(val) or RE_EPISODE_TOKEN.fullmatch(val)
            )

        def looks_like_release_group_token(val):
            if not RE_RELEASE_GROUP.fullmatch(val):
                return False
            if is_metadata_like_group_token(val):
                return False
            upper_count = sum(1 for c in val if "A" <= c <= "Z")
            lower_count = sum(1 for c in val if "a" <= c <= "z")
            digit_count = sum(1 for c in val if c.isdigit())

            # Typical release-group style: mostly uppercase / mixed camel tags / includes digits.
            if upper_count >= 2 and upper_count >= lower_count:
                return True
            if upper_count >= 2 and digit_count > 0:
                return True
            if RE_UPPER_2_PLUS.search(val):
                return True
            return False

        group_match = RE_SCENE_GROUP_END.search(working_name)
        if not group_match:
            dot_match = RE_SCENE_GROUP_DOT.search(working_name)
            if dot_match and looks_like_release_group_token(dot_match.group(1)):
                group_match = dot_match
        if group_match:
            val = group_match.group(1)
            if not is_metadata_like_group_token(val):
                parsed["group"] = val
                working_name = working_name[: group_match.start()]
                if group_match.start() < len(strict_title):
                    strict_title = strict_title[: group_match.start()]

        # 3. Extract standard attributes from metadata suffix to avoid title-token pollution.
        if strict_title and working_name.startswith(strict_title):
            meta_working = working_name[len(strict_title) :]
        else:
            meta_working = working_name
        if not meta_working.strip():
            meta_working = working_name

        parsed["resolution"], meta_working = extract_with_pattern(P_RES, meta_working)
        parsed["source"], meta_working = extract_with_pattern(P_SOURCE, meta_working)
        parsed["video_codec"], meta_working = extract_with_pattern(
            P_VCODEC, meta_working
        )
        parsed["video_hdr"], meta_working = extract_with_pattern(
            P_VHDR, meta_working, is_multi=True
        )
        parsed["video_hdr"] = _normalize_video_hdr_tokens(parsed["video_hdr"])
        parsed["frame_rate"], meta_working = extract_with_pattern(P_FPS, meta_working)
        parsed["frame_rate"] = _normalize_frame_rate(parsed["frame_rate"])
        parsed["audio_codec"], meta_working = extract_with_pattern(
            P_ACODEC, meta_working, is_multi=True
        )
        parsed["year"], meta_working = extract_with_pattern(P_YEAR, meta_working)
        _, meta_working = extract_with_pattern(P_MISC, meta_working)

        # 4. Extract Season/Episode
        se, meta_working = extract_season_episode(meta_working, is_anime=False)
        if se:
            parsed["season_episode"] = se

        # Assign cleanly cut string and clean its season tags
        _, cleaned_strict = extract_season_episode(strict_title, is_anime=False)
        if RE_WORD_OR_CJK.search(cleaned_strict):
            working_name = cleaned_strict
        else:
            working_name = strict_title

    def extract_titles(text):
        def clean_zh_title_noise(zh_text):
            if not zh_text:
                return zh_text
            tokens = zh_text.split()
            if len(tokens) <= 1:
                return zh_text
            noise_tokens = {
                "西影",
                "上影",
                "北影",
                "长影",
                "峨影",
                "珠影",
                "潇影",
                "中影",
            }
            kept = [tok for tok in tokens if tok not in noise_tokens]
            return " ".join(kept) if kept else zh_text

        text = _strip_parenthetical_title_noise(text)
        text = RE_TITLE_SEP.sub(" ", text)
        text = RE_ASCII_BRACKET_BLOCK.sub(" ", text)
        # Channel/broadcaster tags should not be part of the actual title.
        text = RE_CHANNEL_TAG.sub(" ", text)
        # Subtitle/language markers are metadata, not title content.
        text = RE_SUBTITLE_EN.sub(" ", text)
        text = RE_SUBTITLE_ZH.sub(" ", text)
        # Studio abbreviations are often source metadata instead of title text.
        text = RE_STUDIO_ABBR.sub(" ", text)
        text = (
            text.replace("[", " ")
            .replace("]", " ")
            .replace("\uff08", " ")
            .replace("\uff09", " ")
            .replace("\u3010", " ")
            .replace("\u3011", " ")
        )
        text = RE_MULTI_SPACE.sub(" ", text).strip()
        text = _strip_leading_title_noise(text)
        text = RE_MULTI_SPACE.sub(" ", text).strip()
        zh_chars = RE_HAS_CJK.findall(text)
        if not zh_chars:
            return text if text else None, None

        words = text.split()
        en_blocks = []
        current_en_block = []
        pending_non_latin = []

        def _prefix_tokens(tokens):
            # Keep short numeric/Roman prefixes immediately before an
            # English block (e.g. "12 12 The Day").
            result = []
            for tok in tokens:
                cleaned = tok.strip("[](){}<>")
                if re.fullmatch(r"(?i)(?:\d{1,4}|[IVXLCDM]{1,8})", cleaned):
                    result.append(tok)
            return result

        for w in words:
            if RE_HAS_CJK.search(w):
                if current_en_block:
                    en_blocks.append(" ".join(current_en_block))
                    current_en_block = []
                pending_non_latin = []
                continue

            if RE_HAS_LATIN.search(w):
                if not current_en_block and pending_non_latin:
                    current_en_block.extend(_prefix_tokens(pending_non_latin))
                current_en_block.append(w)
                pending_non_latin = []
                continue

            if current_en_block:
                current_en_block.append(w)
                continue

            pending_non_latin.append(w)

        if current_en_block:
            en_blocks.append(" ".join(current_en_block))

        en = max(en_blocks, key=len) if en_blocks else None
        if en:
            alpha_len = len(RE_NON_LATIN.sub("", en))
            if alpha_len <= 1:
                en = None
        if en:
            en_connector = RE_NON_LATIN.sub("", en).lower()
            if en_connector in {"v", "vs", "versus", "at"}:
                en = None
        if en:
            en_norm = RE_NON_ALNUM.sub("", en).upper()
            if en_norm in {"ATV", "TVB"} or RE_CCTV_EXACT.fullmatch(en_norm):
                en = None
        if en:
            zh = text.replace(en, "").strip()
            zh = RE_MULTI_SPACE.sub(" ", zh)
            zh = clean_zh_title_noise(zh)
            return en, zh

        text = clean_zh_title_noise(text)
        return None, text

    t1, z1 = extract_titles(working_name)
    if sports_title_override:
        parsed["title"] = sports_title_override
        parsed["zh_title"] = z1
    else:
        parsed["title"] = t1
        parsed["zh_title"] = z1

    if not parsed["title"] and anime_en_candidate:
        parsed["title"] = anime_en_candidate
    if not parsed["zh_title"] and anime_zh_candidate:
        parsed["zh_title"] = anime_zh_candidate

    # Check if we should fall back to directory metadata.
    # For DVD/Blu-ray structures, this uses the disc root (dir before VIDEO_TS/BDMV/...).
    if parent_dir:
        parent_working = parent_dir

        # Determine strict boundary for parent
        p_matches = list(P_SEP.finditer(parent_working))
        p_title_end = len(parent_working)
        for m in p_matches:
            if m.start() > 3:
                p_title_end = m.start()
                break

        parent_strict = parent_working[:p_title_end]

        # Look for Group
        p_group_match = RE_PARENT_GROUP_SUFFIX.search(parent_working)
        if p_group_match:
            val = p_group_match.group(1)
            if not _has_technical_metadata_token(val):
                if not parsed["group"]:
                    parsed["group"] = val
                parent_working = parent_working[: p_group_match.start()]

        p_res, parent_working = extract_with_pattern(P_RES, parent_working)
        if not parsed["resolution"] and p_res:
            parsed["resolution"] = p_res

        p_source, parent_working = extract_with_pattern(P_SOURCE, parent_working)
        if not parsed["source"] and p_source:
            parsed["source"] = p_source

        p_vcodec, parent_working = extract_with_pattern(P_VCODEC, parent_working)
        if not parsed["video_codec"] and p_vcodec:
            parsed["video_codec"] = p_vcodec

        p_vhdr, parent_working = extract_with_pattern(
            P_VHDR, parent_working, is_multi=True
        )
        if not parsed["video_hdr"] and p_vhdr:
            parsed["video_hdr"] = _normalize_video_hdr_tokens(p_vhdr)

        p_fps, parent_working = extract_with_pattern(P_FPS, parent_working)
        if not parsed["frame_rate"] and p_fps:
            parsed["frame_rate"] = _normalize_frame_rate(p_fps)

        p_acodec, parent_working = extract_with_pattern(
            P_ACODEC, parent_working, is_multi=True
        )
        if not parsed["audio_codec"] and p_acodec:
            parsed["audio_codec"] = p_acodec

        p_year, parent_working = extract_with_pattern(P_YEAR, parent_working)
        if not parsed["year"] and p_year:
            parsed["year"] = p_year
        if not parsed["year"]:
            season_year = _extract_season_year_tag(parent_dir)
            if season_year:
                parsed["year"] = season_year

        _, parent_working = extract_with_pattern(P_MISC, parent_working)

        p_se, parent_working = extract_season_episode(parent_working, is_anime=False)
        if not parsed["season_episode"] and p_se:
            parsed["season_episode"] = p_se
        elif parsed["season_episode"] and p_se:
            # Combine logic
            p1 = parsed["season_episode"].upper().replace(" ", "")
            p2 = p_se.upper().replace(" ", "")
            if "S" in p2 and "E" not in p2:
                if "S" not in p1 and "E" in p1:
                    parsed["season_episode"] = p2 + p1
                elif p1.isdigit():
                    parsed["season_episode"] = f"{p2}E{p1.zfill(2)}"
                elif RE_DISC_INDEX_TOKEN.fullmatch(p1):
                    parsed["season_episode"] = p2 + p1

        # Clean parent_working and get titles
        _, parent_strict = extract_season_episode(parent_strict, is_anime=False)
        parent_working = RE_ASCII_BRACKET_BLOCK.sub(" ", parent_working)
        t2, z2 = extract_titles(parent_strict)

        is_poor_title = (
            (not parsed["title"] and not parsed["zh_title"])
            or (parsed["title"] and parsed["title"].isdigit())
            or (parsed["zh_title"] and parsed["zh_title"].isdigit())
            or (is_disc_layout and looks_like_technical_disc_title(parsed["title"]))
        )

        if is_poor_title:
            if t2:
                parsed["title"] = t2
                parsed["_inherited_title"] = True
            if z2:
                parsed["zh_title"] = z2
        else:
            if not parsed["title"] and t2:
                parsed["title"] = t2
                parsed["_inherited_title"] = True
            if not parsed["zh_title"] and z2:
                parsed["zh_title"] = z2

    # CCTV-style file names often carry program names after channel prefix.
    if (
        not parsed["title"]
        and not parsed["zh_title"]
        and isinstance(parsed.get("source"), str)
        and RE_CCTV_EXACT.fullmatch(parsed["source"])
    ):
        cctv_text = RE_CCTV_PREFIX.sub(" ", name_no_ext)
        for pat in (P_RES, P_FPS, P_SOURCE, P_VCODEC, P_VHDR, P_ACODEC, P_YEAR, P_MISC):
            cctv_text = pat.sub(" ", cctv_text)
        _, cctv_text = extract_season_episode(cctv_text, is_anime=False)
        cctv_text = RE_CCTV_CLEAN_PUNCT.sub(" ", cctv_text)
        cctv_text = RE_MULTI_SPACE.sub(" ", cctv_text).strip(" -")
        t3, z3 = extract_titles(cctv_text)
        if t3:
            parsed["title"] = t3
        if z3:
            parsed["zh_title"] = z3

    # Final normalization
    for k, v in parsed.items():
        if isinstance(v, str):
            if k in {"title", "zh_title"}:
                v = v.strip(" -._:：;；,，。!！?？、·")
            else:
                v = v.strip(" -.")
            parsed[k] = v if v else None

    # Calculate confidence
    confidence = calculate_confidence(parsed, filename, raw_path=filepath)

    return {
        "raw_path": filepath,
        "filename": filename,
        "parsed": parsed,
        "confidence": confidence,
    }
