import {
  DISC_LAYOUT_DIRS,
  P_ACODEC,
  P_EP_BRACKET,
  P_MISC,
  P_RES,
  P_SE,
  P_SOURCE,
  P_VCODEC,
  P_VHDR,
  regexWithGlobal,
} from "./constants";

function removeSpan(text, start, end) {
  return `${text.slice(0, start)} ${text.slice(end)}`;
}

const COMMON_RESOLUTION_EP_GUARDS = new Set([360, 480, 540, 576, 720, 900, 1080, 1440, 2160, 4320]);

function isPlausibleBracketEpisode(parsedEp, isAnime = false) {
  if (!Number.isInteger(parsedEp) || parsedEp <= 0) return false;
  if (parsedEp <= 200) return true;
  if (!isAnime) return false;
  if (parsedEp > 5000) return false;
  if (COMMON_RESOLUTION_EP_GUARDS.has(parsedEp)) return false;
  if (parsedEp >= 1900 && parsedEp <= 2099) return false;
  return true;
}

function parseCjkNumber(token) {
  if (!token) return null;
  const t = String(token).trim();
  if (!t) return null;
  if (/^\d+$/u.test(t)) return Number.parseInt(t, 10);

  const digits = new Map([
    ["零", 0],
    ["〇", 0],
    ["一", 1],
    ["二", 2],
    ["两", 2],
    ["三", 3],
    ["四", 4],
    ["五", 5],
    ["六", 6],
    ["七", 7],
    ["八", 8],
    ["九", 9],
  ]);

  if (t.includes("十")) {
    const [left, right = ""] = t.split("十", 2);
    let tens = 1;
    if (left !== "") {
      if (!digits.has(left)) return null;
      tens = digits.get(left);
    }
    let ones = 0;
    if (right !== "") {
      if (right.length !== 1 || !digits.has(right)) return null;
      ones = digits.get(right);
    }
    return tens * 10 + ones;
  }

  if (t.length === 1 && digits.has(t)) return digits.get(t);
  return null;
}

export function extractWithPattern(pattern, text, isMulti = false) {
  if (isMulti) {
    const regex = regexWithGlobal(pattern);
    const matches = Array.from(text.matchAll(regex));
    const values = [];
    for (const m of matches) {
      const v = m.length > 1 && m[1] ? m[1] : m[0];
      if (v) values.push(v);
    }
    if (values.length === 0) return [null, text];
    const deduped = [];
    const seen = new Set();
    for (const token of values) {
      const key = token.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      deduped.push(token);
    }
    return [deduped.join(" "), text.replace(regex, " ")];
  }

  const m = pattern.exec(text);
  if (!m || m.index === undefined) return [null, text];
  const val = m.length > 1 && m[1] ? m[1] : m[0];
  return [val, removeSpan(text, m.index, m.index + m[0].length)];
}

export function extractSeasonEpisode(text, isAnime = false) {
  let s = null;
  let e = null;
  let episodeMarker = null;
  let working = text;

  const sDRegex = /(?<![a-zA-Z0-9])S(\d{1,2})D(\d{1,3})(?![a-zA-Z0-9])/giu;
  const sdMatches = Array.from(working.matchAll(sDRegex)).reverse();
  for (const m of sdMatches) {
    if (m.index === undefined) continue;
    if (s === null) s = Number.parseInt(m[1], 10);
    if (e === null) {
      e = Number.parseInt(m[2], 10);
      episodeMarker = "D";
    }
    working = removeSpan(working, m.index, m.index + m[0].length);
  }

  const pSeMatches = Array.from(working.matchAll(regexWithGlobal(P_SE))).reverse();
  for (const m of pSeMatches) {
    if (m.index === undefined) continue;
    const val = (m[1] || m[0] || "").toUpperCase();
    const hasSpecial = /(EXTRAS|BONUS|SP|OVA|OAD)/u.test(val);
    if (val.includes("S") && val.includes("E") && !hasSpecial) {
      const sm = /S(\d+)/u.exec(val);
      const em = /E(\d+)/u.exec(val);
      if (sm) s = Number.parseInt(sm[1], 10);
      if (em) {
        e = Number.parseInt(em[1], 10);
        episodeMarker = "E";
      }
    } else if (val.includes("S") && !hasSpecial) {
      const sm = /S(\d+)/u.exec(val);
      if (sm) s = Number.parseInt(sm[1], 10);
    } else if (val.includes("E") && !/(EXTRAS|BONUS|SP|OVA|OAD|COMPLETE)/u.test(val)) {
      const em = /E[P]?(?:\s*[-_.]?\s*)(\d+)/u.exec(val);
      if (em) {
        e = Number.parseInt(em[1], 10);
        episodeMarker = /EP(?:\s*[-_.]?\s*)\d+/u.test(val) ? "EP" : "E";
      }
    } else if (hasSpecial) {
      const originalVal = (m[1] || "").trim();
      const up = originalVal.toUpperCase();
      if (up === "SP" && m.index > 0 && working[m.index - 1] === "@") {
        working = removeSpan(working, m.index, m.index + m[0].length);
        continue;
      }
      if (Number.isInteger(e)) continue;
      if ((up === "EXTRAS" || up === "BONUS") && /S\d{1,2}E\d{1,3}/iu.test(working)) {
        continue;
      }
      if (up.includes("EXTRAS")) e = up.replace("EXTRAS", "Extras");
      else if (up.includes("BONUS")) e = up.replace("BONUS", "Bonus");
      else e = up;
    }
    working = removeSpan(working, m.index, m.index + m[0].length);
  }

  const seasonCnRegex = /第([一二三四五六七八九十两\d]+)季/gu;
  const seasonCnMatches = Array.from(working.matchAll(seasonCnRegex)).reverse();
  for (const m of seasonCnMatches) {
    if (m.index === undefined) continue;
    const parsed = parseCjkNumber(m[1]);
    if (s === null && parsed !== null) s = parsed;
    working = removeSpan(working, m.index, m.index + m[0].length);
  }

  const seasonEnRegex = /(?<![a-zA-Z0-9])Season\s*(\d{1,2})(?!\d)(?![a-zA-Z0-9])/giu;
  const seasonEnMatches = Array.from(working.matchAll(seasonEnRegex)).reverse();
  for (const m of seasonEnMatches) {
    if (m.index === undefined) continue;
    if (s === null) s = Number.parseInt(m[1], 10);
    working = removeSpan(working, m.index, m.index + m[0].length);
  }

  const epCnRegex = /第([一二三四五六七八九十两\d]+)[集话話]/gu;
  const epCnMatches = Array.from(working.matchAll(epCnRegex)).reverse();
  for (const m of epCnMatches) {
    if (m.index === undefined) continue;
    const parsed = parseCjkNumber(m[1]);
    if (e === null && parsed !== null) e = parsed;
    working = removeSpan(working, m.index, m.index + m[0].length);
  }

  if (e === null) {
    const hash = /(?<!\d)#\s*(\d{1,3})(?!\d)/u.exec(working);
    if (hash && hash.index !== undefined) {
      const parsed = Number.parseInt(hash[1], 10);
      if (parsed > 0 && parsed <= 200) {
        e = parsed;
        episodeMarker = "HASH";
        working = removeSpan(working, hash.index, hash.index + hash[0].length);
      }
    }
  }

  if (e === null) {
    const dOnly = /(?<![a-zA-Z0-9])D(\d{1,3})(?![a-zA-Z0-9])/iu.exec(working);
    if (dOnly && dOnly.index !== undefined) {
      e = Number.parseInt(dOnly[1], 10);
      episodeMarker = "D";
      working = removeSpan(working, dOnly.index, dOnly.index + dOnly[0].length);
    }
  }

  if (e === null) {
    const epBracket = P_EP_BRACKET.exec(working);
    if (epBracket && epBracket.index !== undefined) {
      const parsed = Number.parseInt(epBracket[1], 10);
      if (isPlausibleBracketEpisode(parsed, isAnime)) {
        e = parsed;
        working = removeSpan(working, epBracket.index, epBracket.index + epBracket[0].length);
      }
    }
  }

  if (e === null && isAnime) {
    const loose = /-\s*(\d{2,3})(?:v\d)?(?:[^\d]|$)/iu.exec(working);
    if (loose && loose.index !== undefined) {
      e = Number.parseInt(loose[1], 10);
      working = removeSpan(working, loose.index, loose.index + loose[0].length);
    }
  }

  if (e === null) {
    const stripped = working.trim();
    if (/^\d{1,3}$/u.test(stripped)) {
      const parsed = Number.parseInt(stripped, 10);
      if (parsed > 0) {
        e = parsed;
        working = " ";
      }
    }
  }

  let result = null;
  if (s !== null && Number.isInteger(e)) {
    if (episodeMarker === "EP") result = `S${String(s).padStart(2, "0")}EP${String(e).padStart(2, "0")}`;
    else if (episodeMarker === "D") result = `S${String(s).padStart(2, "0")}D${String(e).padStart(2, "0")}`;
    else result = `S${String(s).padStart(2, "0")}E${String(e).padStart(2, "0")}`;
  } else if (s !== null && typeof e === "string") {
    if (e.toLowerCase() === "complete") result = `S${String(s).padStart(2, "0")} Complete`;
    else result = `S${String(s).padStart(2, "0")} ${e}`;
  } else if (s !== null) {
    result = `S${String(s).padStart(2, "0")}`;
  } else if (e !== null) {
    if (Number.isInteger(e)) {
      if (episodeMarker === "EP") result = `EP${String(e).padStart(2, "0")}`;
      else if (episodeMarker === "D") result = `D${String(e).padStart(2, "0")}`;
      else if (episodeMarker === "HASH") result = `E${String(e).padStart(2, "0")}`;
      else result = isAnime ? String(e).padStart(2, "0") : `E${String(e).padStart(2, "0")}`;
    } else {
      result = e;
    }
  }

  return [result, working];
}

export function resolveMetadataParentDir(parts) {
  if (!Array.isArray(parts) || parts.length <= 1) return ["", false];
  const dirs = parts.slice(0, -1);
  for (let idx = 0; idx < dirs.length; idx += 1) {
    const segment = String(dirs[idx] || "").trim().toLowerCase();
    if (DISC_LAYOUT_DIRS.has(segment)) {
      if (idx > 0) return [dirs[idx - 1], true];
      return [dirs[0], true];
    }
  }
  return [dirs[dirs.length - 1], false];
}

export function looksLikeTechnicalDiscTitle(text) {
  if (!text) return false;
  const normalized = String(text).replace(/[\s._-]+/gu, "").toLowerCase();
  if (
    new Set([
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
    ]).has(normalized)
  ) {
    return true;
  }
  if (/^vts\d{3,5}$/u.test(normalized)) return true;
  if (/^\d{4,6}$/u.test(normalized)) return true;
  return false;
}

export function extractSportsEventTitle(text) {
  if (!text) return null;
  const hasVs = /\b(?:vs\.?|versus)\b/iu.test(text);
  const hasAt = /\b[A-Za-z0-9]{2,}\s+at\s+[A-Za-z0-9]{2,}\b/iu.test(text);
  if (!hasVs && !hasAt) return null;

  const hasSportsDomain =
    /\b(?:NFL|NBA|MLB|NHL|NCAA|WNBA|EPL|UEFA|FIFA|UFC|WWE|ATP|WTA|F1|Formula\s*1|NASCAR|MotoGP|PGA|LPGA)\b/iu.test(
      text
    );
  if (!hasSportsDomain) return null;

  let candidate = text.replace(/^\[[^\]]+\]\s*/u, " ");
  candidate = candidate.replace(
    /\b(19\d{2}|20[0-3]\d)\s*[-_/.]\s*(0?[1-9]|1[0-2])\s*[-_/.]\s*(0?[1-9]|[12]\d|3[01])\b/giu,
    " "
  );
  candidate = candidate.replace(/\b(19\d{2}|20[0-3]\d)\s*[-/]\s*(?:19\d{2}|20[0-3]\d|\d{2})\b/giu, " ");
  candidate = candidate.replace(/\b(19\d{2}|20[0-3]\d)\b/giu, " ");
  candidate = candidate.replace(/\b(0?[1-9]|1[0-2])\s*[-_/.]\s*(0?[1-9]|[12]\d|3[01])\b/giu, " ");

  for (const pat of [P_RES, P_SOURCE, P_VCODEC, P_VHDR, P_ACODEC, P_MISC]) {
    candidate = candidate.replace(regexWithGlobal(pat), " ");
  }

  candidate = candidate.replace(/\s*[-_@]\s*[A-Za-z0-9_@]{2,20}$/giu, " ");
  candidate = candidate.replace(/@\w+\b/gu, " ");
  candidate = candidate.replace(/\[[^\]]+\]/gu, " ");
  candidate = candidate.replace(/[._-]+/gu, " ");
  candidate = candidate.replace(/\s+/gu, " ").trim().replace(/^[\s-]+|[\s-]+$/gu, "");

  const hasVsClean = /\b(?:vs\.?|versus)\b/iu.test(candidate);
  const hasAtClean = /\b[A-Za-z0-9]{2,}\s+at\s+[A-Za-z0-9]{2,}\b/iu.test(candidate);
  if (!hasVsClean && !hasAtClean) return null;

  return candidate.length >= 8 ? candidate : null;
}
