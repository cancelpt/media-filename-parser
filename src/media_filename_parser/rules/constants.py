"""Regex constants and shared token fragments for the rule parser."""

import re


def make_pattern(options):
    return re.compile(rf"(?i)(?<![a-zA-Z0-9])({options})(?![a-zA-Z0-9])")


# Reusable token fragments to avoid drift between patterns.
TOKEN_YEAR = r"19\d{2}|20[0-3]\d"
TOKEN_RESOLUTION = (
    r"(?:[1-9]\d{2,3}p|[1-9]\d{2,3}i|[1-9]\d{2,4}x[1-9]\d{2,4}|4k|8k|UHD|mUHD|FHD|HD)"
)
TOKEN_FRAME_RATE = r"\d{2,3}(?:\.\d{1,3})?\s?fps"
TOKEN_SEASON_EP_SEP = r"S\d{1,2}E\d{1,3}|S\d{1,2}|E\d{1,3}|EP\s?\d{1,3}|SP|OVA|OAD"
TOKEN_SOURCE_SEP = r"Blu(?:-?[Rr]ay)?|BDRip|WEB-DL|WEBRip|DVDRip|HDTV"


# Capture Sources (e.g., BluRay, WEB-DL)
P_SOURCE = make_pattern(
    r"UHD|Blu-[Rr]ay|BluRay|BDRip|WEB-DL|WEBRip|HDTV-Rip|HDTVRip|HDTVrip|HDTV|SDTV"
    r"|BD|WEB|TVRip|DVDRip|DVD5|DVD9|DVD|TVING|IQIYI|WeTV|Paramount\+|Disney\+"
    r"|Netflix|Amazon|AppleTV|Hulu|HBO|CCTV\d*|CATCHPLAY\+?|friDay|HamiVideo|Hami|KKTV"
    r"|Peacock|SHO|Max|ATVP|DSNP|PCOK|U-NEXT|CP\+|GagaOOLala"
)
P_RES = make_pattern(TOKEN_RESOLUTION)
P_VCODEC = make_pattern(
    r"x26[4-7]|H\.?26[4-7]"
    r"|AVC|HEVC|VVC"
    r"|AV01|AV1|SVT-?AV1"
    r"|VP9|VP8|VC-?1"
    r"|MPEG-?2|MPEG-?4|XviD|DivX|RV\d{2,3}"
    r"|ProRes|DNxHD|DNxHR"
)
P_ACODEC = make_pattern(
    r"Dolby[\s\._-]?TrueHD(?:[\s\._-]?7\.1)?(?:[\s\._-]?Atmos)?"
    r"|TrueHD(?:[\s\._-]?7\.1)?(?:[\s\._-]?Atmos)?"
    r"|E-?AC-?3(?:[\s\._-]?[0-9]\.[0-9])?(?:[\s\._-]?Atmos)?"
    r"|DDP(?:[\s\._-]?[0-9]\.[0-9])?(?:[\s\._-]?Atmos)?"
    r"|DD\+?(?:[\s\._-]?[0-9]\.[0-9])?"
    r"|DD5\.1|DDP5\.1|DDP2\.0|\d+DDP"
    r"|DTS[-\s]?X|DTS-HD\s?MA|DTS-HD|DTS(?:[\s\._-]?[0-9]\.[0-9])?"
    r"|AAC2\.0|\d+AAC|AAC"
    r"|FLAC|ALAC|OPUS|VORBIS|MP3|WMA(?:PRO|LOSSLESS)?"
    r"|AC-?3|\d+AC3|MLP|LPCM|PCM|WAVPACK|APE"
    r"|AV3A(?:[\s\._-]?\d(?:\.\d){1,3})?"
    r"|\d+[\s\._-]?Audios?"
    r"|2Audio|DUAL|MA(?:\.|\s)?5\.1|MA|Atmos"
)
P_VHDR = make_pattern(
    r"DV[\s\._-]?HDR[\s\._-]?Vivid|DoVi[\s\._-]?HDR[\s\._-]?Vivid|HDR[\s\._-]?Vivid|"
    r"DV[\s\._-]?HDR10\+|DV[\s\._-]?HDR10|DV[\s\._-]?HDR"
    r"|DoVi[\s\._-]?HDR10\+|DoVi[\s\._-]?HDR10|DoVi[\s\._-]?HDR"
    r"|DV|DoVi|HDR10\+|HDR10|HDR|HLG|SDR|10bit|12bit|8bit"
)
P_FPS = make_pattern(TOKEN_FRAME_RATE)
P_YEAR = make_pattern(TOKEN_YEAR)

# Match S01E01, S01, E01, EP01/EP_01, Complete, Extras, OVA
P_SE = make_pattern(
    r"S\d{1,2}E\d{1,3}|S\d{1,2}|E\d{1,3}|EP(?:\s*[-_.]?\s*\d{1,3})|Complete|(?:Extras|Bonus|SP|OVA|OAD)(?:\s*[-_]?\s*\d{1,3})?"
)
# Matches episode markers in brackets or loose forms (e.g., [01], 第x季, 第x集).
P_EP_BRACKET = re.compile(r"\[(?:v\d)?(\d{2,4})(?:v\d)?\]")

# Strict boundary delimiter for Scene/PT formats
P_SEP = make_pattern(
    rf"{TOKEN_YEAR}|{TOKEN_RESOLUTION}|{TOKEN_FRAME_RATE}|{TOKEN_SEASON_EP_SEP}|"
    rf"\u7b2c[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\d]+[\u5b63\u96c6\u8bdd\u8a71]|{TOKEN_SOURCE_SEP}"
)
P_MISC = make_pattern(
    r"MNHD|REMASTERED|PROPER|REPACK|DC|EXTENDED|UNRATED|UNCUT|\d+(?:\.\d+)?\s*(?:MB|GB|KB|mb|gb|kb)"
)

DISC_LAYOUT_DIRS = {
    "video_ts",
    "audio_ts",
    "bdmv",
    "certificate",
    "hvdvd_ts",
    "adv_obj",
    "stream",
    "playlist",
    "clipinf",
    "backup",
    "ssif",
}

GENERIC_SHORT_TITLES = {
    "NFL",
    "NBA",
    "MLB",
    "NHL",
    "WWE",
    "UFC",
    "F1",
}
