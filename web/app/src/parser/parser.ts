import {
  GENERIC_SHORT_TITLES,
  P_ACODEC,
  P_FPS,
  P_MISC,
  P_RES,
  P_SEP,
  P_SOURCE,
  P_VCODEC,
  P_VHDR,
  P_YEAR,
  RE_ALL_BRACKETS_GLOBAL,
  RE_ASCII_BRACKET_BLOCK,
  RE_BRACKET_BLOCK_GLOBAL,
  RE_CCTV_CLEAN_PUNCT,
  RE_CCTV_EXACT,
  RE_CCTV_PREFIX,
  RE_CHANNEL_TAG,
  RE_DISC_INDEX_TOKEN,
  RE_FPS_NORMALIZE,
  RE_GROUP_TAIL,
  RE_GROUP_ZH_HINT,
  RE_HAS_CJK,
  RE_HAS_JA,
  RE_HAS_LATIN,
  RE_HEAD_BRACKET,
  RE_HDR_BIT_DEPTH,
  RE_HDR_ONLY,
  RE_HDR_SPECIFIC,
  RE_KNOWN_MEDIA_EXT,
  RE_META_WORDS,
  RE_MULTI_SPACE,
  RE_NON_ALNUM,
  RE_NON_LATIN,
  RE_NUM_1_TO_3,
  RE_NUM_1_TO_4,
  RE_NUM_LONG,
  RE_OUTSIDE_COMPACT_SEP,
  RE_PARENT_GROUP_SUFFIX,
  RE_PROMO_BLOCK_ASCII,
  RE_PROMO_BLOCK_FULLWIDTH,
  RE_RELEASE_GROUP,
  RE_SCENE_GROUP_DOT,
  RE_SCENE_GROUP_END,
  RE_SDR_ONLY,
  RE_STUDIO_ABBR,
  RE_SUB_LANG_TAG,
  RE_SUB_LANG_TAG_STRICT,
  RE_SUBTITLE_EN,
  RE_SUBTITLE_ZH,
  RE_TITLE_NOISE_TOKEN,
  RE_TITLE_SEP,
  RE_UPPER_2_PLUS,
  RE_WORD_OR_CJK,
  RE_YEAR_IN_BRACKET,
  regexWithGlobal,
  TECH_META_PATTERNS,
} from "./constants";
import {
  extractSeasonEpisode,
  extractSportsEventTitle,
  extractWithPattern,
  looksLikeTechnicalDiscTitle,
  resolveMetadataParentDir,
} from "./extraction";

const RE_HEXISH_ID = /^[A-Fa-f0-9]{7,}$/u;
const RE_SEASON_TAG = /^SEASON\s*\d{1,4}$/iu;
const RE_SEASON_YEAR = /^SEASON\s*[-_.]?\s*(19\d{2}|20[0-3]\d)$/iu;
const RE_SEASON_YEAR_SEARCH = /(?<![a-zA-Z0-9])SEASON\s*[-_.]?\s*(19\d{2}|20[0-3]\d)(?![a-zA-Z0-9])/iu;
const RE_EPISODE_RANGE = /^\d{2,4}\s*-\s*\d{2,4}$/u;
const RE_PAREN_BLOCK = /[（(]([^()（）]{1,120})[)）]/gu;
const RE_CONTAINER_TAG = /\b(?:mp4|mkv|avi|m2ts|ts|flv|wmv|mov|webm|rmvb|iso|vob)\b/iu;
const RE_TECH_LIKE_TOKEN = /^(?:\d{2,4}p|\d{2,4}x\d{2,4}|\d+(?:\.\d+)?fps|\d+bit)$/iu;

function compactText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^0-9a-z\u4e00-\u9fff]+/gu, "");
}

function seasonEpisodeVariants(value) {
  const v = String(value || "")
    .trim()
    .toUpperCase()
    .replace(/\s+/gu, "");
  if (!v) return [];
  const variants = new Set([v]);

  let m = /^S(\d{1,2})E(\d{1,3})$/u.exec(v);
  if (m) {
    const s = Number.parseInt(m[1], 10);
    const e = Number.parseInt(m[2], 10);
    variants.add(`S${String(s).padStart(2, "0")}E${String(e).padStart(2, "0")}`);
    variants.add(`S${s}E${e}`);
    variants.add(`${String(s).padStart(2, "0")}X${String(e).padStart(2, "0")}`);
    variants.add(`${s}X${e}`);
    return [...variants];
  }

  m = /^S(\d{1,2})EP(\d{1,3})$/u.exec(v);
  if (m) {
    const s = Number.parseInt(m[1], 10);
    const e = Number.parseInt(m[2], 10);
    variants.add(`S${String(s).padStart(2, "0")}EP${String(e).padStart(2, "0")}`);
    variants.add(`S${s}EP${e}`);
    variants.add(`S${String(s).padStart(2, "0")}E${String(e).padStart(2, "0")}`);
    variants.add(`S${s}E${e}`);
    return [...variants];
  }

  m = /^EP(\d{1,3})$/u.exec(v);
  if (m) {
    const e = Number.parseInt(m[1], 10);
    variants.add(`EP${String(e).padStart(2, "0")}`);
    variants.add(`EP${e}`);
    variants.add(`E${String(e).padStart(2, "0")}`);
    variants.add(`E${e}`);
    return [...variants];
  }

  m = /^E(\d{1,3})$/u.exec(v);
  if (m) {
    const e = Number.parseInt(m[1], 10);
    variants.add(`E${String(e).padStart(2, "0")}`);
    variants.add(`E${e}`);
    return [...variants];
  }

  m = /^S(\d{1,2})D(\d{1,3})$/u.exec(v);
  if (m) {
    const s = Number.parseInt(m[1], 10);
    const d = Number.parseInt(m[2], 10);
    variants.add(`S${String(s).padStart(2, "0")}D${String(d).padStart(2, "0")}`);
    variants.add(`S${s}D${d}`);
    return [...variants];
  }

  m = /^D(\d{1,3})$/u.exec(v);
  if (m) {
    const d = Number.parseInt(m[1], 10);
    variants.add(`D${String(d).padStart(2, "0")}`);
    variants.add(`D${d}`);
    return [...variants];
  }

  return [...variants];
}

function fieldVariants(field, value) {
  const text = String(value || "").trim();
  if (!text) return [];
  if (field === "season_episode") return seasonEpisodeVariants(text);

  const variants = new Set([text]);
  if (field === "source") variants.add(text.replaceAll("-", ""));
  if (field === "frame_rate") {
    const compact = text.replaceAll(" ", "");
    variants.add(compact);
    variants.add(compact.toLowerCase());
    if (compact.toLowerCase().endsWith("fps")) {
      variants.add(`${compact.slice(0, -3)}fps`);
      variants.add(`${compact.slice(0, -3)} fps`);
    }
  }
  if (field === "video_codec") variants.add(text.replaceAll(".", ""));
  if (field === "video_hdr") {
    const parts = text.split(/\s+/u).filter(Boolean);
    parts.forEach((p) => variants.add(p));
    parts.forEach((p) => variants.add(p.replaceAll(".", "")));
    if (parts.length >= 2) variants.add(parts.slice().reverse().join(" "));
  }
  if (field === "audio_codec") {
    const parts = text.split(/\s+/u).filter(Boolean);
    parts.forEach((p) => variants.add(p));
    parts.forEach((p) => variants.add(p.replaceAll(".", "")));
  }
  if (field === "group") variants.add(text.replaceAll(" ", ""));
  return [...variants];
}

function computeAlignment(parsed, filename, rawPath = "") {
  const targetRaw = `${rawPath} ${filename}`.toLowerCase();
  const targetCompact = compactText(targetRaw);

  const fields = [
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
  ];
  let considered = 0;
  let matched = 0;

  for (const field of fields) {
    const val = parsed[field];
    if (typeof val !== "string" || !val.trim()) continue;
    considered += 1;
    let hit = false;
    for (const variant of fieldVariants(field, val)) {
      const v = variant.trim().toLowerCase();
      if (!v) continue;
      if (targetRaw.includes(v)) {
        hit = true;
        break;
      }
      const vc = compactText(v);
      if (vc && targetCompact.includes(vc)) {
        hit = true;
        break;
      }
    }
    if (hit) matched += 1;
  }
  return { ratio: considered ? matched / considered : 0, matched, considered };
}

function calculateConfidence(parsed, filename, rawPath = "") {
  let score = 0;
  const title = parsed.title || "";
  const zhTitle = parsed.zh_title || "";
  const fullTitle = `${title} ${zhTitle}`.trim();
  const isSample = filename.toLowerCase().includes("sample");

  let hasValidTitle = false;
  if (fullTitle.length >= 2) {
    const symbols = fullTitle.replace(/[\p{L}\p{N}_\s\u4e00-\u9fa5\u3040-\u309F\u30A0-\u30FF.-]/gu, "");
    if (symbols.length <= fullTitle.length / 2) hasValidTitle = true;
  }
  if (hasValidTitle) score += 0.3;
  if (parsed.resolution) score += 0.2;
  if (parsed.season_episode || parsed.year) score += 0.2;
  if (parsed.video_codec || parsed.source) score += 0.2;
  if (parsed.group) score += 0.1;

  const { ratio, considered } = computeAlignment(parsed, filename, rawPath);
  if (considered >= 4) {
    if (ratio >= 0.85) score += 0.1;
    else if (ratio >= 0.7) score += 0.05;
    else if (ratio < 0.4) score -= 0.05;
  }

  if (title && GENERIC_SHORT_TITLES.has(title.trim().toUpperCase())) score -= 0.25;

  if (typeof parsed.source === "string" && RE_CCTV_EXACT.test(parsed.source.trim())) score -= 0.1;
  if (typeof title === "string" && /\bCCTV\d*\b/iu.test(title)) score -= 0.2;
  if (typeof title === "string" && /\bhdr\b[\s._-]*vs[\s._-]*\bsdr\b/iu.test(title)) score -= 0.2;

  const hdrValue = parsed.video_hdr;
  if (typeof hdrValue === "string") {
    const hdrTokens = new Set(hdrValue.toUpperCase().split(/[\s._-]+/u).filter(Boolean));
    const hasSdr = hdrTokens.has("SDR");
    const hasHdrFamily =
      ["HDR", "HDR10", "HDR10+", "DV", "DOVI", "HLG"].some((x) => hdrTokens.has(x)) ||
      [...hdrTokens].some((x) => x.startsWith("HDR"));
    if (hasSdr && hasHdrFamily) score -= 0.15;
  }

  if (parsed._inherited_title) {
    if (title && parsed.zh_title && parsed.zh_title.length >= 4 && title.length <= 3) score -= 0.5;
    else if (title && /^\d+$/u.test(title)) {
      if (considered >= 4 && ratio >= 0.8) {
        // keep
      } else if (considered >= 4 && ratio >= 0.6) {
        score -= 0.2;
      } else {
        score -= 0.4;
      }
    }
  }

  const lowerFilename = filename.toLowerCase();
  if (lowerFilename.includes("sample") || lowerFilename.includes("extra")) score -= 0.5;
  if (!hasValidTitle || isSample) {
    if (score >= 0.3) score = 0.29;
  }
  if (score > 1) score = 1;
  return Math.round(score * 100) / 100;
}

function hasTechnicalMetadataToken(value) {
  return TECH_META_PATTERNS.some((pat) => pat.test(value));
}

function looksLikeSubtitleLangTag(value) {
  return RE_SUB_LANG_TAG.test(value) || RE_SUB_LANG_TAG_STRICT.test(value);
}

function dedupeTokensCaseInsensitive(tokens) {
  const out = [];
  const seen = new Set();
  for (const token of tokens || []) {
    const t = String(token || "").trim();
    if (!t) continue;
    const key = t.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(t);
  }
  return out;
}

function normalizeVideoHdrTokens(value) {
  if (!value) return value;
  const tokens = dedupeTokensCaseInsensitive(value.split(/\s+/u));
  if (tokens.length === 0) return null;
  const bitDepth = tokens.filter((x) => RE_HDR_BIT_DEPTH.test(x)).map((x) => x.toLowerCase());
  let hdrLabels = tokens.filter((x) => !RE_HDR_BIT_DEPTH.test(x));
  const hasSpecificHdr = hdrLabels.some((x) => RE_HDR_SPECIFIC.test(x) && !RE_HDR_ONLY.test(x));
  if (hasSpecificHdr) hdrLabels = hdrLabels.filter((x) => !RE_HDR_ONLY.test(x));
  const hasNonSdrHdr = hdrLabels.some((x) => !RE_SDR_ONLY.test(x));
  if (hasNonSdrHdr) hdrLabels = hdrLabels.filter((x) => !RE_SDR_ONLY.test(x));
  const ordered = [...hdrLabels, ...bitDepth];
  return ordered.length ? ordered.join(" ") : null;
}

function normalizeFrameRate(value) {
  if (!value) return value;
  const m = RE_FPS_NORMALIZE.exec(value.trim());
  if (!m) return value;
  return `${m[1]}fps`;
}

function stripLeadingTitleNoise(text) {
  if (!text) return text;
  const tokens = text.split(/\s+/u);
  let idx = 0;
  while (idx < tokens.length) {
    const raw = tokens[idx];
    const token = raw.replace(/^[\[\](){}<>]+|[\[\](){}<>]+$/gu, "");
    if (RE_TITLE_NOISE_TOKEN.test(token)) {
      idx += 1;
      continue;
    }
    if (/^\d+$/u.test(token) && idx + 1 < tokens.length) {
      const next = tokens[idx + 1].replace(/^[\[\](){}<>]+|[\[\](){}<>]+$/gu, "");
      if (RE_TITLE_NOISE_TOKEN.test(next)) {
        idx += 1;
        continue;
      }
    }
    break;
  }
  return idx < tokens.length ? tokens.slice(idx).join(" ") : "";
}

function stripParentheticalTitleNoise(text) {
  if (!text) return text;
  const metaKeywords = new Set([
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
  ]);

  return text.replace(RE_PAREN_BLOCK, (full, innerRaw) => {
    const inner = String(innerRaw || "").trim();
    if (!inner) return " ";
    if (RE_HAS_CJK.test(inner)) return full;
    if (hasTechnicalMetadataToken(inner) || RE_CONTAINER_TAG.test(inner) || looksLikeSubtitleLangTag(inner)) return " ";

    const tokens = inner.split(/[\s._-]+/u).filter(Boolean);
    if (!tokens.length) return " ";
    let metaHits = 0;
    for (const tok of tokens) {
      const up = tok.toUpperCase();
      if (metaKeywords.has(up) || RE_TECH_LIKE_TOKEN.test(tok)) metaHits += 1;
    }
    if (metaHits >= Math.max(1, tokens.length - 1)) return " ";
    return full;
  });
}

function extractSeasonYearTag(text) {
  if (!text) return null;
  for (const m of text.matchAll(RE_ALL_BRACKETS_GLOBAL)) {
    const bracketText = String(m[1] || "").trim();
    const hit = RE_SEASON_YEAR.exec(bracketText);
    if (hit) return hit[1];
  }
  const hit = RE_SEASON_YEAR_SEARCH.exec(text);
  return hit ? hit[1] : null;
}

function extractTitles(rawText) {
  function cleanZhTitleNoise(zhText) {
    if (!zhText) return zhText;
    const tokens = zhText.split(/\s+/u);
    if (tokens.length <= 1) return zhText;
    const noise = new Set(["西影", "上影", "北影", "长影", "峨影", "珠影", "潇影", "中影"]);
    const kept = tokens.filter((x) => !noise.has(x));
    return kept.length ? kept.join(" ") : zhText;
  }

  let text = String(rawText || "");
  text = stripParentheticalTitleNoise(text);
  text = text.replace(RE_TITLE_SEP, " ");
  text = text.replace(RE_ASCII_BRACKET_BLOCK, " ");
  text = text.replace(RE_CHANNEL_TAG, " ");
  text = text.replace(RE_SUBTITLE_EN, " ");
  text = text.replace(RE_SUBTITLE_ZH, " ");
  text = text.replace(RE_STUDIO_ABBR, " ");
  text = text
    .replaceAll("[", " ")
    .replaceAll("]", " ")
    .replaceAll("（", " ")
    .replaceAll("）", " ")
    .replaceAll("【", " ")
    .replaceAll("】", " ");
  text = text.replace(RE_MULTI_SPACE, " ").trim();
  text = stripLeadingTitleNoise(text);
  text = text.replace(RE_MULTI_SPACE, " ").trim();

  if (!RE_HAS_CJK.test(text)) return [text || null, null];

  const words = text.split(/\s+/u);
  const blocks = [];
  let current = [];
  let pendingNonLatin = [];

  for (const w of words) {
    if (RE_HAS_CJK.test(w)) {
      if (current.length) {
        blocks.push(current.join(" "));
        current = [];
      }
      pendingNonLatin = [];
    } else if (RE_HAS_LATIN.test(w)) {
      if (!current.length && pendingNonLatin.length) {
        // Keep short numeric/Roman prefixes immediately before an English block.
        const prefixTokens = pendingNonLatin.filter((tok) => {
          const cleaned = tok.replace(/^[\[\](){}<>]+|[\[\](){}<>]+$/gu, "");
          return /^(?:\d{1,4}|[IVXLCDM]{1,8})$/iu.test(cleaned);
        });
        current.push(...prefixTokens);
      }
      current.push(w);
      pendingNonLatin = [];
    } else if (current.length) {
      current.push(w);
    } else {
      pendingNonLatin.push(w);
    }
  }
  if (current.length) blocks.push(current.join(" "));

  let en = blocks.length ? blocks.slice().sort((a, b) => b.length - a.length)[0] : null;
  if (en && en.replace(RE_NON_LATIN, "").length <= 1) en = null;
  if (en) {
    const c = en.replace(RE_NON_LATIN, "").toLowerCase();
    if (["v", "vs", "versus", "at"].includes(c)) en = null;
  }
  if (en) {
    const enNorm = en.replace(RE_NON_ALNUM, "").toUpperCase();
    if (["ATV", "TVB"].includes(enNorm) || RE_CCTV_EXACT.test(enNorm)) en = null;
  }
  if (en) {
    let zh = text.replace(en, "").trim();
    zh = zh.replace(RE_MULTI_SPACE, " ");
    zh = cleanZhTitleNoise(zh);
    return [en, zh || null];
  }
  return [null, cleanZhTitleNoise(text) || null];
}

function looksLikeGroupTag(text) {
  const t = String(text || "").trim();
  if (!t) return false;
  if (hasTechnicalMetadataToken(t) || RE_NUM_1_TO_4.test(t) || RE_META_WORDS.test(t)) return false;
  if (RE_HAS_CJK.test(t)) return RE_GROUP_ZH_HINT.test(t);
  if (RE_HAS_JA.test(t)) return t.length <= 20;
  const tokens = t.split(/\s+/u).filter(Boolean);
  if (tokens.length > 3) return false;
  return /^[A-Za-z0-9_\-@&+. ]{1,48}$/u.test(t);
}

function isMetadataLikeGroupToken(val) {
  return hasTechnicalMetadataToken(val) || /^S\d{1,2}E?\d{0,3}|E\d{1,3}|EP\d{1,3}|D\d{1,3}$/iu.test(val);
}

function looksLikeReleaseGroupToken(val) {
  if (!RE_RELEASE_GROUP.test(val)) return false;
  if (isMetadataLikeGroupToken(val)) return false;
  const upper = (val.match(/[A-Z]/g) || []).length;
  const lower = (val.match(/[a-z]/g) || []).length;
  const digit = (val.match(/\d/g) || []).length;
  if (upper >= 2 && upper >= lower) return true;
  if (upper >= 2 && digit > 0) return true;
  return RE_UPPER_2_PLUS.test(val);
}

function parseBasenameFromPath(input) {
  const parts = String(input || "")
    .replaceAll("\\", "/")
    .trim()
    .split("/");
  return parts[parts.length - 1] || "";
}

export function parseFilename(inputPath) {
  const rawPath = String(inputPath || "").replace(/^\uFEFF+/u, "");
  let filepath = rawPath.replace(RE_PROMO_BLOCK_FULLWIDTH, "").replace(RE_PROMO_BLOCK_ASCII, "");
  const filepathNormalized = filepath.replaceAll("【", "[").replaceAll("】", "]").replaceAll("（", "(").replaceAll("）", ")");

  const originalParts = rawPath.replaceAll("\\", "/").trim().split("/");
  const filename = originalParts[originalParts.length - 1] || parseBasenameFromPath(rawPath);
  const parts = filepathNormalized.replaceAll("\\", "/").trim().split("/");
  const [parentDir, isDiscLayout] = resolveMetadataParentDir(parts);

  const basename = parts[parts.length - 1] || "";
  const extMatch = RE_KNOWN_MEDIA_EXT.exec(basename);
  const nameNoExt = extMatch ? basename.slice(0, extMatch.index) : basename;

  const parsed = {
    title: null,
    zh_title: "",
    year: null,
    season_episode: null,
    resolution: null,
    frame_rate: null,
    source: null,
    video_codec: null,
    video_hdr: null,
    audio_codec: null,
    group: null,
  };

  let workingName = nameNoExt.trim();
  const sportsTitleOverride = extractSportsEventTitle(nameNoExt);
  let animeZhCandidate = null;
  let animeEnCandidate = null;

  const dotCount = (workingName.match(/\./g) || []).length;
  if (workingName.startsWith("[") && dotCount < 3) {
    const groupMatch = RE_HEAD_BRACKET.exec(workingName);
    if (groupMatch) {
      const firstBracket = (groupMatch[1] || "").trim();
      if (looksLikeGroupTag(firstBracket)) {
        parsed.group = firstBracket;
        workingName = workingName.slice(groupMatch[0].length);
      }
    }

    const brackets = Array.from(workingName.matchAll(RE_ALL_BRACKETS_GLOBAL), (m) => (m[1] || "").trim());
    const bracketTitleCandidates = [];
    for (const bracketText of brackets) {
      if (!parsed.resolution) {
        const [res] = extractWithPattern(P_RES, bracketText);
        parsed.resolution = res;
      }
      if (!parsed.source) {
        const [src] = extractWithPattern(P_SOURCE, bracketText);
        parsed.source = src;
      }
      if (!parsed.video_codec) {
        const [vc] = extractWithPattern(P_VCODEC, bracketText);
        parsed.video_codec = vc;
      }
      if (!parsed.year) {
        const y = RE_YEAR_IN_BRACKET.exec(bracketText);
        if (y) {
          let yearVal = y[1];
          if (yearVal.length === 2) {
            const yNum = Number.parseInt(yearVal, 10);
            yearVal = yNum <= 30 ? `20${yearVal}` : `19${yearVal}`;
          }
          parsed.year = yearVal;
        } else {
          const seasonYear = extractSeasonYearTag(bracketText);
          if (seasonYear) parsed.year = seasonYear;
        }
      }

      const isNumericOnly = RE_NUM_1_TO_3.test(bracketText);
      const isEpisodeRange = RE_EPISODE_RANGE.test(bracketText);
      const isSeasonTag = RE_SEASON_TAG.test(bracketText);
      const isHexishId = RE_HEXISH_ID.test(bracketText);
      const looksSubtitleTag = looksLikeSubtitleLangTag(bracketText);
      const looksMeta =
        hasTechnicalMetadataToken(bracketText) ||
        isNumericOnly ||
        isEpisodeRange ||
        isSeasonTag ||
        isHexishId ||
        looksSubtitleTag ||
        RE_META_WORDS.test(bracketText) ||
        /\d{2,4}年/u.test(bracketText);

      if (!looksMeta && bracketText.length >= 2) {
        bracketTitleCandidates.push(bracketText);
        if (RE_HAS_CJK.test(bracketText)) {
          if (!animeZhCandidate || bracketText.length > animeZhCandidate.length) animeZhCandidate = bracketText;
        } else if (RE_HAS_LATIN.test(bracketText)) {
          if (!animeEnCandidate || bracketText.length > animeEnCandidate.length) animeEnCandidate = bracketText;
        }
      }
    }

    [parsed.resolution, workingName] = extractWithPattern(P_RES, workingName);
    let source2 = null;
    [source2, workingName] = extractWithPattern(P_SOURCE, workingName);
    if (!parsed.source && source2) parsed.source = source2;
    [parsed.video_codec, workingName] = extractWithPattern(P_VCODEC, workingName);
    [parsed.video_hdr, workingName] = extractWithPattern(P_VHDR, workingName, true);
    parsed.video_hdr = normalizeVideoHdrTokens(parsed.video_hdr);
    [parsed.frame_rate, workingName] = extractWithPattern(P_FPS, workingName);
    parsed.frame_rate = normalizeFrameRate(parsed.frame_rate);
    [parsed.audio_codec, workingName] = extractWithPattern(P_ACODEC, workingName, true);
    [parsed.year, workingName] = extractWithPattern(P_YEAR, workingName);
    [, workingName] = extractWithPattern(P_MISC, workingName);

    const [se, seText] = extractSeasonEpisode(workingName, true);
    workingName = seText;
    if (se) parsed.season_episode = se;

    if (!parsed.group) {
      const groupTail = RE_GROUP_TAIL.exec(workingName);
      if (groupTail) {
        const tailVal = groupTail[1];
        if (!hasTechnicalMetadataToken(tailVal)) {
          parsed.group = tailVal;
          workingName = workingName.slice(0, groupTail.index);
        }
      }
    }

    const outsideText = workingName.replace(RE_BRACKET_BLOCK_GLOBAL, " ").trim();
    const outsideCompact = outsideText.replace(RE_OUTSIDE_COMPACT_SEP, "");
    const outsideIsIdOnly = Boolean(outsideCompact) && (RE_NUM_LONG.test(outsideCompact) || RE_HEXISH_ID.test(outsideCompact));
    if (outsideText && !outsideIsIdOnly) workingName = outsideText;
    else if (bracketTitleCandidates.length) {
      workingName = bracketTitleCandidates.slice().sort((a, b) => b.length - a.length)[0];
    } else {
      workingName = outsideText;
    }
  } else {
    const sepMatches = Array.from(workingName.matchAll(regexWithGlobal(P_SEP)));
    let titleEnd = workingName.length;
    for (const m of sepMatches) {
      if (m.index !== undefined && m.index > 3) {
        titleEnd = m.index;
        break;
      }
    }
    let strictTitle = workingName.slice(0, titleEnd);

    let groupMatch = RE_SCENE_GROUP_END.exec(workingName);
    if (!groupMatch) {
      const dotMatch = RE_SCENE_GROUP_DOT.exec(workingName);
      if (dotMatch && looksLikeReleaseGroupToken(dotMatch[1])) groupMatch = dotMatch;
    }
    if (groupMatch) {
      const val = groupMatch[1];
      if (!isMetadataLikeGroupToken(val)) {
        parsed.group = val;
        workingName = workingName.slice(0, groupMatch.index);
        if (groupMatch.index < strictTitle.length) strictTitle = strictTitle.slice(0, groupMatch.index);
      }
    }

    let metaWorking = strictTitle && workingName.startsWith(strictTitle) ? workingName.slice(strictTitle.length) : workingName;
    if (!metaWorking.trim()) metaWorking = workingName;

    [parsed.resolution, metaWorking] = extractWithPattern(P_RES, metaWorking);
    [parsed.source, metaWorking] = extractWithPattern(P_SOURCE, metaWorking);
    [parsed.video_codec, metaWorking] = extractWithPattern(P_VCODEC, metaWorking);
    [parsed.video_hdr, metaWorking] = extractWithPattern(P_VHDR, metaWorking, true);
    parsed.video_hdr = normalizeVideoHdrTokens(parsed.video_hdr);
    [parsed.frame_rate, metaWorking] = extractWithPattern(P_FPS, metaWorking);
    parsed.frame_rate = normalizeFrameRate(parsed.frame_rate);
    [parsed.audio_codec, metaWorking] = extractWithPattern(P_ACODEC, metaWorking, true);
    [parsed.year, metaWorking] = extractWithPattern(P_YEAR, metaWorking);
    [, metaWorking] = extractWithPattern(P_MISC, metaWorking);

    const [se, seText] = extractSeasonEpisode(metaWorking, false);
    metaWorking = seText;
    if (se) parsed.season_episode = se;

    const [, cleanedStrict] = extractSeasonEpisode(strictTitle, false);
    if (RE_WORD_OR_CJK.test(cleanedStrict)) workingName = cleanedStrict;
    else workingName = strictTitle;
  }

  const [t1, z1] = extractTitles(workingName);
  if (sportsTitleOverride) {
    parsed.title = sportsTitleOverride;
    parsed.zh_title = z1;
  } else {
    parsed.title = t1;
    parsed.zh_title = z1;
  }
  if (!parsed.title && animeEnCandidate) parsed.title = animeEnCandidate;
  if (!parsed.zh_title && animeZhCandidate) parsed.zh_title = animeZhCandidate;

  if (parentDir) {
    let parentWorking = parentDir;
    const pMatches = Array.from(parentWorking.matchAll(regexWithGlobal(P_SEP)));
    let pTitleEnd = parentWorking.length;
    for (const m of pMatches) {
      if (m.index !== undefined && m.index > 3) {
        pTitleEnd = m.index;
        break;
      }
    }
    let parentStrict = parentWorking.slice(0, pTitleEnd);

    const pGroup = RE_PARENT_GROUP_SUFFIX.exec(parentWorking);
    if (pGroup) {
      const val = pGroup[1];
      if (!hasTechnicalMetadataToken(val)) {
        if (!parsed.group) parsed.group = val;
        parentWorking = parentWorking.slice(0, pGroup.index);
      }
    }

    let tmp = null;
    [tmp, parentWorking] = extractWithPattern(P_RES, parentWorking);
    if (!parsed.resolution && tmp) parsed.resolution = tmp;
    [tmp, parentWorking] = extractWithPattern(P_SOURCE, parentWorking);
    if (!parsed.source && tmp) parsed.source = tmp;
    [tmp, parentWorking] = extractWithPattern(P_VCODEC, parentWorking);
    if (!parsed.video_codec && tmp) parsed.video_codec = tmp;
    [tmp, parentWorking] = extractWithPattern(P_VHDR, parentWorking, true);
    if (!parsed.video_hdr && tmp) parsed.video_hdr = normalizeVideoHdrTokens(tmp);
    [tmp, parentWorking] = extractWithPattern(P_FPS, parentWorking);
    if (!parsed.frame_rate && tmp) parsed.frame_rate = normalizeFrameRate(tmp);
    [tmp, parentWorking] = extractWithPattern(P_ACODEC, parentWorking, true);
    if (!parsed.audio_codec && tmp) parsed.audio_codec = tmp;
    [tmp, parentWorking] = extractWithPattern(P_YEAR, parentWorking);
    if (!parsed.year && tmp) parsed.year = tmp;
    if (!parsed.year) {
      const seasonYear = extractSeasonYearTag(parentDir);
      if (seasonYear) parsed.year = seasonYear;
    }
    [, parentWorking] = extractWithPattern(P_MISC, parentWorking);

    let pSe = null;
    [pSe, parentWorking] = extractSeasonEpisode(parentWorking, false);
    if (!parsed.season_episode && pSe) parsed.season_episode = pSe;
    else if (parsed.season_episode && pSe) {
      const p1 = parsed.season_episode.toUpperCase().replace(/\s+/gu, "");
      const p2 = pSe.toUpperCase().replace(/\s+/gu, "");
      if (p2.includes("S") && !p2.includes("E")) {
        if (!p1.includes("S") && p1.includes("E")) parsed.season_episode = `${p2}${p1}`;
        else if (/^\d+$/u.test(p1)) parsed.season_episode = `${p2}E${p1.padStart(2, "0")}`;
        else if (RE_DISC_INDEX_TOKEN.test(p1)) parsed.season_episode = `${p2}${p1}`;
      }
    }

    [, parentStrict] = extractSeasonEpisode(parentStrict, false);
    parentWorking = parentWorking.replace(RE_ASCII_BRACKET_BLOCK, " ");
    const [t2, z2] = extractTitles(parentStrict);

    const isPoorTitle =
      (!parsed.title && !parsed.zh_title) ||
      (parsed.title && /^\d+$/u.test(parsed.title)) ||
      (parsed.zh_title && /^\d+$/u.test(parsed.zh_title)) ||
      (isDiscLayout && looksLikeTechnicalDiscTitle(parsed.title));

    if (isPoorTitle) {
      if (t2) {
        parsed.title = t2;
        parsed._inherited_title = true;
      }
      if (z2) parsed.zh_title = z2;
    } else {
      if (!parsed.title && t2) {
        parsed.title = t2;
        parsed._inherited_title = true;
      }
      if (!parsed.zh_title && z2) parsed.zh_title = z2;
    }
  }

  if (!parsed.title && !parsed.zh_title && typeof parsed.source === "string" && RE_CCTV_EXACT.test(parsed.source)) {
    let cctvText = nameNoExt.replace(RE_CCTV_PREFIX, " ");
    for (const pat of [P_RES, P_FPS, P_SOURCE, P_VCODEC, P_VHDR, P_ACODEC, P_YEAR, P_MISC]) {
      cctvText = cctvText.replace(regexWithGlobal(pat), " ");
    }
    [, cctvText] = extractSeasonEpisode(cctvText, false);
    cctvText = cctvText.replace(RE_CCTV_CLEAN_PUNCT, " ").replace(RE_MULTI_SPACE, " ").trim().replace(/^[\s-]+|[\s-]+$/gu, "");
    const [t3, z3] = extractTitles(cctvText);
    if (t3) parsed.title = t3;
    if (z3) parsed.zh_title = z3;
  }

  for (const [k, v] of Object.entries(parsed)) {
    if (typeof v !== "string") continue;
    const cleaned =
      k === "title" || k === "zh_title"
        ? v.replace(/^[\s\-._:：;；,，。!！?？、·]+|[\s\-._:：;；,，。!！?？、·]+$/gu, "")
        : v.replace(/^[\s\-.]+|[\s\-.]+$/gu, "");
    parsed[k] = cleaned || null;
  }

  const confidence = calculateConfidence(parsed, filename, rawPath);
  return { raw_path: rawPath, filename, parsed, confidence };
}

export function parseFilenameOnly(input) {
  return parseFilename(input).parsed;
}
