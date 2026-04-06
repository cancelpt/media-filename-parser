"""Extraction helpers used by the rule-based media parser."""

import re

from .constants import (
    DISC_LAYOUT_DIRS,
    P_ACODEC,
    P_EP_BRACKET,
    P_MISC,
    P_RES,
    P_SE,
    P_SOURCE,
    P_VCODEC,
    P_VHDR,
)


COMMON_RESOLUTION_EP_GUARDS = {360, 480, 540, 576, 720, 900, 1080, 1440, 2160, 4320}


def _parse_cjk_number(token):
    """Parse common Chinese numeral forms used in season/episode markers."""
    if not token:
        return None
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)

    digits = {
        "\u96f6": 0,
        "\u3007": 0,
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u4e24": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
    }

    # e.g. 十 / 十二 / 二十 / 二十五
    if "\u5341" in token:
        left, _, right = token.partition("\u5341")
        if left == "":
            tens = 1
        else:
            if left not in digits:
                return None
            tens = digits[left]
        if right == "":
            ones = 0
        else:
            if len(right) != 1 or right not in digits:
                return None
            ones = digits[right]
        return tens * 10 + ones

    # e.g. 二 / 五 / 两
    if len(token) == 1 and token in digits:
        return digits[token]

    return None


def extract_with_pattern(pattern, text, is_multi=False):
    if is_multi:
        matches = list(pattern.finditer(text))
        results = []
        for m in matches:
            val = m.group(1) if pattern.groups else m.group(0)
            if val:
                results.append(val)
        if results:
            text = pattern.sub(" ", text)
            # Remove duplicates while preserving order
            unique_results = list(dict.fromkeys(results))
            return " ".join(unique_results), text
        return None, text
    match = pattern.search(text)
    if match:
        val = match.group(1) if pattern.groups else match.group(0)
        text = text[: match.start()] + " " + text[match.end() :]
        return val, text
    return None, text


def _is_plausible_bracket_episode(parsed_ep, is_anime=False):
    if parsed_ep <= 0:
        return False
    if parsed_ep <= 200:
        return True
    if not is_anime:
        return False

    # Long-running anime can exceed 200 episodes, but guard against common metadata numbers.
    if parsed_ep > 5000:
        return False
    if parsed_ep in COMMON_RESOLUTION_EP_GUARDS:
        return False
    if 1900 <= parsed_ep <= 2099:  # likely a year marker
        return False
    return True


def extract_season_episode(text, is_anime=False):
    s = None
    e = None
    episode_marker = None  # Distinguish EPxx / Exx / Dxx.

    # 1. DVD disc marker with season, e.g. S01D27.
    for match in reversed(
        list(
            re.finditer(
                r"(?i)(?<![a-zA-Z0-9])S(\d{1,2})D(\d{1,3})(?![a-zA-Z0-9])", text
            )
        )
    ):
        if s is None:
            s = int(match.group(1))
        if e is None:
            e = int(match.group(2))
            episode_marker = "D"
        text = text[: match.start()] + " " + text[match.end() :]

    # 2. Standard P_SE
    for match in reversed(list(P_SE.finditer(text))):
        val = match.group(1).upper()
        if (
            "S" in val
            and "E" in val
            and not any(x in val for x in ["EXTRAS", "BONUS", "SP", "OVA", "OAD"])
        ):
            s_match = re.search(r"S(\d+)", val)
            e_match = re.search(r"E(\d+)", val)
            if s_match:
                s = int(s_match.group(1))
            if e_match:
                e = int(e_match.group(1))
                episode_marker = "E"
        elif "S" in val and not any(
            x in val for x in ["EXTRAS", "BONUS", "SP", "OVA", "OAD"]
        ):
            s_match = re.search(r"S(\d+)", val)
            if s_match:
                s = int(s_match.group(1))
        elif "E" in val and not any(
            x in val for x in ["EXTRAS", "BONUS", "SP", "OVA", "OAD", "COMPLETE"]
        ):
            e_match = re.search(r"E[P]?(?:\s*[-_.]?\s*)(\d+)", val)
            if e_match:
                e = int(e_match.group(1))
                episode_marker = (
                    "EP" if re.search(r"EP(?:\s*[-_.]?\s*)\d+", val) else "E"
                )
        elif "COMPLETE" in val:
            # Keep removing this token from text, but do not use it as season/episode.
            pass
        elif any(x in val for x in ["EXTRAS", "BONUS", "SP", "OVA", "OAD"]):
            original_val = match.group(1).strip()
            up = original_val.upper()

            # Scene/encoder suffix like "@SP" is not an episode marker.
            if up == "SP" and match.start() > 0 and text[match.start() - 1] == "@":
                text = text[: match.start()] + " " + text[match.end() :]
                continue

            # Don't let Extras/Bonus overwrite a real numeric episode
            if e is not None and isinstance(e, int):
                continue

            # Prevent "Extras" (the TV show name) false positive:
            # If it sits at or near the start and standard S/E markers exist in the text
            if up in ["EXTRAS", "BONUS"] and re.search(r"S\d{1,2}E\d{1,3}", text, re.I):
                continue

            # Format nicely
            if "EXTRAS" in up:
                e = up.replace("EXTRAS", "Extras")
            elif "BONUS" in up:
                e = up.replace("BONUS", "Bonus")
            else:
                e = up  # SP, OVA, OAD keep caps
        text = text[: match.start()] + " " + text[match.end() :]

    # 3. Chinese Season
    for match in reversed(
        list(
            re.finditer(
                r"\u7b2c([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24\d]+)\u5b63",
                text,
            )
        )
    ):
        val = match.group(1)
        parsed_s = _parse_cjk_number(val)
        if s is None and parsed_s is not None:
            s = parsed_s
        text = text[: match.start()] + " " + text[match.end() :]

    # 4. English Season (limit to 1-2 digits to avoid "Season2024" false positives).
    for match in reversed(
        list(
            re.finditer(
                r"(?i)(?<![a-zA-Z0-9])Season\s*(\d{1,2})(?!\d)(?![a-zA-Z0-9])", text
            )
        )
    ):
        if s is None:
            s = int(match.group(1))
        text = text[: match.start()] + " " + text[match.end() :]

    # 5. Chinese/Japanese Episode
    for match in reversed(
        list(
            re.finditer(
                r"\u7b2c([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24\d]+)[\u96c6\u8bdd\u8a71]",
                text,
            )
        )
    ):
        parsed_e = _parse_cjk_number(match.group(1))
        if e is None and parsed_e is not None:
            e = parsed_e
        text = text[: match.start()] + " " + text[match.end() :]

    # 6. Hash-style episode marker, e.g. "#4" or "# 12".
    if e is None:
        hash_match = re.search(r"(?<!\d)#\s*(\d{1,3})(?!\d)", text)
        if hash_match:
            parsed_ep = int(hash_match.group(1))
            if 0 < parsed_ep <= 200:
                e = parsed_ep
                episode_marker = "HASH"
                text = text[: hash_match.start()] + " " + text[hash_match.end() :]

    # 7. DVD disc marker without season, e.g. D27.
    if e is None:
        d_match = re.search(r"(?i)(?<![a-zA-Z0-9])D(\d{1,3})(?![a-zA-Z0-9])", text)
        if d_match:
            e = int(d_match.group(1))
            episode_marker = "D"
            text = text[: d_match.start()] + " " + text[d_match.end() :]

    # 8. Bracket Episode (generic): [01], [012], [1134], [v2-ignored by pattern]
    if e is None:
        ep_bracket = P_EP_BRACKET.search(text)
        if ep_bracket:
            parsed_ep = int(ep_bracket.group(1))
            if _is_plausible_bracket_episode(parsed_ep, is_anime=is_anime):
                e = parsed_ep
                text = text[: ep_bracket.start()] + " " + text[ep_bracket.end() :]

    # 9. Anime Loose Episode
    if e is None and is_anime:
        ep_match = re.search(r"\-\s*(\d{2,3})(?:v\d)?(?:[^\d]|$)", text)
        if ep_match:
            e = int(ep_match.group(1))
            text = text[: ep_match.start()] + " " + text[ep_match.end() :]

    # 10. Pure numeric file stems like "05" should map to episode number.
    # This covers common TV naming where the parent folder holds the show metadata.
    if e is None:
        stripped = text.strip()
        if re.fullmatch(r"\d{1,3}", stripped):
            parsed_ep = int(stripped)
            if parsed_ep > 0:
                e = parsed_ep
                text = " "

    res = None
    if s is not None and e is not None and isinstance(e, int):
        if episode_marker == "EP":
            res = f"S{s:02d}EP{e:02d}"
        elif episode_marker == "D":
            res = f"S{s:02d}D{e:02d}"
        else:
            res = f"S{s:02d}E{e:02d}"
    elif s is not None and e is not None and isinstance(e, str):
        if e.lower() == "complete":
            res = f"S{s:02d} Complete"
        else:
            res = f"S{s:02d} {e}"
    elif s is not None:
        res = f"S{s:02d}"
    elif e is not None:
        if isinstance(e, int):
            if episode_marker == "EP":
                res = f"EP{e:02d}"
            elif episode_marker == "D":
                res = f"D{e:02d}"
            elif episode_marker == "HASH":
                res = f"E{e:02d}"
            else:
                res = f"E{e:02d}" if not is_anime else f"{e:02d}"
        else:
            res = e  # Keeps formatted 'Extras-01', 'Complete', 'OVA'

    return res, text


def resolve_metadata_parent_dir(parts):
    if len(parts) <= 1:
        return "", False

    dirs = parts[:-1]
    for idx, segment in enumerate(dirs):
        if segment.strip().lower() in DISC_LAYOUT_DIRS:
            if idx > 0:
                return dirs[idx - 1], True
            return dirs[0], True

    return dirs[-1], False


def looks_like_technical_disc_title(text):
    if not text:
        return False

    normalized = re.sub(r"[\s._-]+", "", text).lower()
    if normalized in {
        "videots",
        "bdmv",
        "stream",
        "playlist",
        "clipinf",
        "certificate",
        "hvdvdts",
        "advobj",
        "movieobject",
        "index",
    }:
        return True

    if re.fullmatch(r"vts\d{3,5}", normalized):
        return True

    if re.fullmatch(r"\d{4,6}", normalized):
        return True

    return False


def extract_sports_event_title(text):
    if not text:
        return None

    has_vs = re.search(r"(?i)\b(?:vs\.?|versus)\b", text)
    has_at = re.search(r"(?i)\b[A-Za-z0-9]{2,}\s+at\s+[A-Za-z0-9]{2,}\b", text)
    if not has_vs and not has_at:
        return None

    if not re.search(
        r"(?i)\b(?:NFL|NBA|MLB|NHL|NCAA|WNBA|EPL|UEFA|FIFA|UFC|WWE|ATP|WTA|F1|Formula\s*1|NASCAR|MotoGP|PGA|LPGA)\b",
        text,
    ):
        return None

    candidate = re.sub(r"^\[[^\]]+\]\s*", " ", text)
    candidate = re.sub(
        r"(?i)\b(19\d{2}|20[0-3]\d)\s*[-_/\.]\s*(0?[1-9]|1[0-2])\s*[-_/\.]\s*(0?[1-9]|[12]\d|3[01])\b",
        " ",
        candidate,
    )
    candidate = re.sub(
        r"(?i)\b(19\d{2}|20[0-3]\d)\s*[-/]\s*(?:19\d{2}|20[0-3]\d|\d{2})\b",
        " ",
        candidate,
    )
    candidate = re.sub(r"(?i)\b(19\d{2}|20[0-3]\d)\b", " ", candidate)
    candidate = re.sub(
        r"(?i)\b(0?[1-9]|1[0-2])\s*[-_/\.]\s*(0?[1-9]|[12]\d|3[01])\b", " ", candidate
    )

    for pat in (P_RES, P_SOURCE, P_VCODEC, P_VHDR, P_ACODEC, P_MISC):
        candidate = pat.sub(" ", candidate)

    candidate = re.sub(r"(?i)\s*[-_@]\s*[A-Za-z0-9_@]{2,20}$", " ", candidate)
    candidate = re.sub(r"@\w+\b", " ", candidate)
    candidate = re.sub(r"\[[^\]]+\]", " ", candidate)
    candidate = re.sub(r"[._\-]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -")

    has_vs_clean = re.search(r"(?i)\b(?:vs\.?|versus)\b", candidate)
    has_at_clean = re.search(
        r"(?i)\b[A-Za-z0-9]{2,}\s+at\s+[A-Za-z0-9]{2,}\b", candidate
    )
    if not has_vs_clean and not has_at_clean:
        return None

    return candidate if len(candidate) >= 8 else None
