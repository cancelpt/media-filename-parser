"""Confidence scoring utilities for rule-based parse results."""

import re

from .constants import GENERIC_SHORT_TITLES


def _compact_text(value):
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", (value or "").lower())


def _season_episode_variants(value):
    v = (value or "").strip().upper().replace(" ", "")
    if not v:
        return []

    variants = {v}
    m_se = re.fullmatch(r"S(\d{1,2})E(\d{1,3})", v)
    if m_se:
        s = int(m_se.group(1))
        e = int(m_se.group(2))
        variants.update(
            {
                f"S{s:02d}E{e:02d}",
                f"S{s}E{e}",
                f"{s:02d}X{e:02d}",
                f"{s}X{e}",
            }
        )
        return list(variants)

    m_sep = re.fullmatch(r"S(\d{1,2})EP(\d{1,3})", v)
    if m_sep:
        s = int(m_sep.group(1))
        e = int(m_sep.group(2))
        variants.update(
            {
                f"S{s:02d}EP{e:02d}",
                f"S{s}EP{e}",
                f"S{s:02d}E{e:02d}",
                f"S{s}E{e}",
            }
        )
        return list(variants)

    m_ep = re.fullmatch(r"EP(\d{1,3})", v)
    if m_ep:
        e = int(m_ep.group(1))
        variants.update({f"EP{e:02d}", f"EP{e}", f"E{e:02d}", f"E{e}"})
        return list(variants)

    m_e = re.fullmatch(r"E(\d{1,3})", v)
    if m_e:
        e = int(m_e.group(1))
        variants.update({f"E{e:02d}", f"E{e}"})
        return list(variants)

    m_d = re.fullmatch(r"S(\d{1,2})D(\d{1,3})", v)
    if m_d:
        s = int(m_d.group(1))
        d = int(m_d.group(2))
        variants.update({f"S{s:02d}D{d:02d}", f"S{s}D{d}"})
        return list(variants)

    m_d_only = re.fullmatch(r"D(\d{1,3})", v)
    if m_d_only:
        d = int(m_d_only.group(1))
        variants.update({f"D{d:02d}", f"D{d}"})
        return list(variants)

    return list(variants)


def _field_variants(field, value):
    text = (value or "").strip()
    if not text:
        return []

    if field == "season_episode":
        return _season_episode_variants(text)

    variants = {text}
    if field == "source":
        variants.add(text.replace("-", ""))
    if field == "frame_rate":
        compact = text.replace(" ", "")
        variants.add(compact)
        variants.add(compact.lower())
        if compact.lower().endswith("fps"):
            variants.add(compact[:-3] + "fps")
            variants.add(compact[:-3] + " fps")
    if field == "video_codec":
        variants.add(text.replace(".", ""))
    if field == "video_hdr":
        parts = [x for x in re.split(r"\s+", text) if x]
        variants.update(parts)
        variants.update(x.replace(".", "") for x in parts)
        if len(parts) >= 2:
            variants.add(" ".join(reversed(parts)))
    if field == "audio_codec":
        parts = [x for x in re.split(r"\s+", text) if x]
        variants.update(parts)
        variants.update(x.replace(".", "") for x in parts)
    if field == "group":
        variants.add(text.replace(" ", ""))

    return list(variants)


def _compute_alignment(parsed_dict, filename, raw_path=None):
    target_raw = f"{raw_path or ''} {filename}".lower()
    target_compact = _compact_text(target_raw)

    considered = 0
    matched = 0
    for field in [
        "title",
        "zh_title",
        "year",
        "season_episode",
        "resolution",
        "frame_rate",
        "source",
        "video_codec",
        "video_hdr",
        "audio_codec",
        "group",
    ]:
        val = parsed_dict.get(field)
        if not isinstance(val, str) or not val.strip():
            continue

        considered += 1
        variants = _field_variants(field, val)
        hit = False
        for var in variants:
            v = var.strip().lower()
            if not v:
                continue
            if v in target_raw:
                hit = True
                break
            vc = _compact_text(v)
            if vc and vc in target_compact:
                hit = True
                break
        if hit:
            matched += 1

    ratio = (matched / considered) if considered else 0.0
    return ratio, matched, considered


def calculate_confidence(parsed_dict, filename, raw_path=None):
    score = 0.0

    title = parsed_dict.get("title", "")
    zh_title = parsed_dict.get("zh_title", "")
    full_title = f"{title or ''} {zh_title or ''}".strip()

    # Check Sample files
    is_sample = "sample" in filename.lower()

    # Check title validity
    has_valid_title = False
    if len(full_title) >= 2:
        # Ensure it's not mostly junk symbols
        symbols = re.sub(
            r"[\w\s\u4e00-\u9fa5\u3040-\u309F\u30A0-\u30FF\.\-]", "", full_title
        )
        if len(symbols) <= len(full_title) / 2:
            has_valid_title = True

    if has_valid_title:
        score += 0.3

    if parsed_dict.get("resolution"):
        score += 0.2

    if parsed_dict.get("season_episode") or parsed_dict.get("year"):
        score += 0.2

    if parsed_dict.get("video_codec") or parsed_dict.get("source"):
        score += 0.2

    if parsed_dict.get("group"):
        score += 0.1

    t = parsed_dict.get("title")
    alignment_ratio, _matched_fields, considered_fields = _compute_alignment(
        parsed_dict, filename, raw_path=raw_path
    )

    # A high parsed-to-filename alignment usually indicates reliable extraction.
    if considered_fields >= 4:
        if alignment_ratio >= 0.85:
            score += 0.1
        elif alignment_ratio >= 0.7:
            score += 0.05
        elif alignment_ratio < 0.4:
            score -= 0.05

    # Extremely generic short titles are often failed parses.
    if t and t.strip().upper() in GENERIC_SHORT_TITLES:
        score -= 0.25

    # Broadcast-channel tags are often noisy metadata for title matching.
    source = parsed_dict.get("source")
    if isinstance(source, str) and re.fullmatch(r"(?i)CCTV\d*", source.strip()):
        score -= 0.1
    if isinstance(t, str) and re.search(r"(?i)\bCCTV\d*\b", t):
        score -= 0.2

    # Penalize common title-pollution markers such as "HDR vs SDR".
    if isinstance(t, str) and re.search(r"(?i)\bhdr\b[\s._-]*vs[\s._-]*\bsdr\b", t):
        score -= 0.2

    # "SDR" mixed with HDR-family tags is usually suspicious noise.
    hdr_value = parsed_dict.get("video_hdr")
    if isinstance(hdr_value, str):
        hdr_tokens = {tok for tok in re.split(r"[\s._-]+", hdr_value.upper()) if tok}
        has_sdr = "SDR" in hdr_tokens
        has_hdr_family = bool(
            {"HDR", "HDR10", "HDR10+", "DV", "DOVI", "HLG"} & hdr_tokens
            or any(tok.startswith("HDR") for tok in hdr_tokens)
        )
        if has_sdr and has_hdr_family:
            score -= 0.15

    # Suspicious inherited titles
    if parsed_dict.get("_inherited_title", False):
        if (
            t
            and parsed_dict.get("zh_title")
            and len(parsed_dict["zh_title"]) >= 4
            and len(t) <= 3
        ):
            score -= 0.5
        elif t and t.isdigit():
            # Numeric inherited titles can still be valid for some standard releases
            # when most parsed fields match the original filename/path.
            if considered_fields >= 4 and alignment_ratio >= 0.8:
                pass
            elif considered_fields >= 4 and alignment_ratio >= 0.6:
                score -= 0.2
            else:
                score -= 0.4

    # Sample files
    lower_path = filename.lower()
    if "sample" in lower_path or "extra" in lower_path:
        score -= 0.5

    # Penalties
    if not has_valid_title or is_sample:
        if score >= 0.3:
            score = 0.29  # Force it below 0.3

    score = min(score, 1.0)

    return round(score, 2)
