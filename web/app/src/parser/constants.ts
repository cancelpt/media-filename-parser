import { SHARED_CONSTANTS } from "./constants.generated.js";

function compilePythonRegex(pyPattern, extraFlags = "") {
  const hasIgnoreCase = pyPattern.includes("(?i)");
  const source = pyPattern.replaceAll("(?i)", "");
  const flagSet = new Set(["u", ...extraFlags.split("")]);
  if (hasIgnoreCase) {
    flagSet.add("i");
  }
  return new RegExp(source, [...flagSet].join(""));
}

export const SHARED = SHARED_CONSTANTS;
export const DISC_LAYOUT_DIRS = new Set(SHARED.lists.DISC_LAYOUT_DIRS);
export const GENERIC_SHORT_TITLES = new Set(SHARED.lists.GENERIC_SHORT_TITLES);

export const P_SOURCE = compilePythonRegex(SHARED.patterns.P_SOURCE);
export const P_RES = compilePythonRegex(SHARED.patterns.P_RES);
export const P_VCODEC = compilePythonRegex(SHARED.patterns.P_VCODEC);
export const P_ACODEC = compilePythonRegex(SHARED.patterns.P_ACODEC);
export const P_VHDR = compilePythonRegex(SHARED.patterns.P_VHDR);
export const P_FPS = compilePythonRegex(SHARED.patterns.P_FPS);
export const P_YEAR = compilePythonRegex(SHARED.patterns.P_YEAR);
export const P_SE = compilePythonRegex(SHARED.patterns.P_SE);
export const P_SEP = compilePythonRegex(SHARED.patterns.P_SEP);
export const P_MISC = compilePythonRegex(SHARED.patterns.P_MISC);
export const P_EP_BRACKET = compilePythonRegex(SHARED.patterns.P_EP_BRACKET);

export const TECH_META_PATTERNS = [P_RES, P_SOURCE, P_VCODEC, P_ACODEC, P_VHDR, P_FPS, P_YEAR];

export const RE_PROMO_BLOCK_FULLWIDTH = /【[^】]*?(?:www|bbs|\.com|\.net|\.cc|\.tv|\.xyz|vip|论坛)[^】]*?】/giu;
export const RE_PROMO_BLOCK_ASCII = /\[[^\]]*?(?:www|bbs|\.com|\.net|\.cc|\.tv|\.xyz|vip|论坛)[^\]]*?\]/giu;
export const RE_KNOWN_MEDIA_EXT = /\.(mkv|mp4|avi|ts|m2ts|iso|vob|flv|wmv|mov|mpg|mpeg|rmvb|webm|m4v)$/i;

export const RE_NUM_1_TO_4 = /^\d{1,4}$/u;
export const RE_NUM_1_TO_3 = /^\d{1,3}$/u;
export const RE_NUM_LONG = /^\d{5,}$/u;
export const RE_YEAR_IN_CN = /\d{2,4}年/u;
export const RE_YEAR_IN_BRACKET = /(?<!\d)(19\d{2}|20\d{2}|\d{2})年?/iu;
export const RE_EPISODE_TOKEN = /^(S\d{1,2}E?\d{0,3}|E\d{1,3}|EP\d{1,3}|D\d{1,3})$/iu;
export const RE_DISC_INDEX_TOKEN = /^D\d{1,3}$/iu;

export const RE_META_WORDS = /(END|中字|双语|字幕|日剧|中日|合集|完结)/iu;
export const RE_GROUP_ZH_HINT = /(字幕组|字幕社|压制组|汉化组|工作室|剧社)/u;
export const RE_SUB_LANG_TAG = /\b(?:CHS|CHT|ENG|EN|JP|JPN|SUB|SUBBED|CN|ZH)\b/iu;
export const RE_SUB_LANG_TAG_STRICT =
  /^(?:CHS|CHT|ENG|EN|JP|JPN|SUB|SUBBED|CN|ZH)(?:[_\-\s](?:CHS|CHT|ENG|EN|JP|JPN|SUB|CN|ZH))*$/iu;

export const RE_HAS_CJK = /[\u4e00-\u9fa5]/u;
export const RE_HAS_JA = /[\u3040-\u30ff\u31f0-\u31ff]/u;
export const RE_HAS_LATIN = /[A-Za-z]/u;
export const RE_MULTI_SPACE = /\s+/gu;
export const RE_NON_LATIN = /[^A-Za-z]+/gu;
export const RE_NON_ALNUM = /[^A-Za-z0-9]+/gu;
export const RE_HDR_BIT_DEPTH = /^(?:8|10|12)bit$/iu;
export const RE_FPS_NORMALIZE = /^(\d{2,3}(?:\.\d{1,3})?)\s*fps$/iu;
export const RE_HDR_ONLY = /^HDR$/iu;
export const RE_SDR_ONLY = /^SDR$/iu;
export const RE_HDR_SPECIFIC = /(?:HDR10\+?|HDR[\s._-]?Vivid|DV|DoVi|HLG)/iu;
export const RE_TITLE_NOISE_TOKEN = /^(?:\d{2,3}fps|hdr(?:10\+)?|sdr|dovi|dv|vs|10bit|12bit|8bit)$/iu;
export const RE_WORD_OR_CJK = /[\w\u4e00-\u9fa5]/u;
export const RE_UPPER_2_PLUS = /[A-Z]{2,}/u;

export const RE_HEAD_BRACKET = /^\[(.*?)\]/u;
export const RE_ALL_BRACKETS_GLOBAL = /\[(.*?)\]/gu;
export const RE_BRACKET_BLOCK_GLOBAL = /\[.*?\]/gu;
export const RE_ASCII_BRACKET_BLOCK = /\[[a-zA-Z0-9\s.\-_+!&]+?\]/gu;
export const RE_OUTSIDE_COMPACT_SEP = /[\s()[\]{}.\-_:，、+]+/gu;
export const RE_CCTV_CLEAN_PUNCT = /[\[\]()._]+/gu;

export const RE_CHANNEL_TAG = /\b(?:CCTV\d*|ATV|TVB)\b/iu;
export const RE_CCTV_EXACT = /^CCTV\d*$/iu;
export const RE_CCTV_PREFIX = /^\s*CCTV\d*\s*[-_. ]*/iu;

export const RE_GROUP_TAIL = /[-￡]\s*([A-Za-z0-9][A-Za-z0-9_@.\-]{1,31})\s*$/u;
export const RE_RELEASE_GROUP = /^[A-Za-z0-9_@￡]{2,24}$/u;
export const RE_PARENT_GROUP_SUFFIX = /-([a-zA-Z0-9_.@]+)$/u;
export const RE_SCENE_GROUP_END = /[-￡]([a-zA-Z0-9_@￡]+)(?:[.\-_](?:sample|trailer|preview|clip))?$/iu;
export const RE_SCENE_GROUP_DOT = /\.([a-zA-Z0-9_@￡]+)(?:[.\-_](?:sample|trailer|preview|clip))?$/iu;

export const RE_TITLE_SEP = /[._\-()]/gu;
export const RE_SUBTITLE_EN =
  /\b(?:ChineseSubbed|ChineseSub|EngSubbed|EngSub|CHS|CHT|CNSub|ZHSub|Subbed)\b/giu;
export const RE_SUBTITLE_ZH = /(?:中字|字幕|双语|中英双语|中日双语|简中|繁中|简体|繁体)/gu;
export const RE_STUDIO_ABBR = /(?:(?<=\s)|^)(?:西影|上影|北影|长影|峨影|珠影|潇影|中影)(?=\s|$)/gu;

export function regexWithGlobal(pattern) {
  const flags = pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`;
  return new RegExp(pattern.source, flags);
}
