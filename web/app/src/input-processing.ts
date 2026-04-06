export type CandidateSource = "manual" | "torrent";

export interface Candidate {
  source: CandidateSource;
  displayPath: string;
  classifyInput: string;
}

export interface FilteredCandidates {
  valid: Candidate[];
  ignoredCount: number;
  ignoredPreview: string[];
}

const VIDEO_SUFFIXES = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m2ts", ".vob"];
const IGNORED_PREVIEW_CAP = 4;
const IGNORED_NOTICE_PREVIEW_LIMIT = 3;

export function isVideoPath(path: string): boolean {
  const lowerPath = path.toLowerCase();
  return VIDEO_SUFFIXES.some((suffix) => lowerPath.endsWith(suffix));
}

export function collectManualCandidates(raw: string): Candidate[] {
  const seen = new Set<string>();
  const candidates: Candidate[] = [];

  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    const dedupeKey = trimmed.toLowerCase();
    if (seen.has(dedupeKey)) {
      continue;
    }

    seen.add(dedupeKey);
    candidates.push({
      source: "manual",
      displayPath: trimmed,
      classifyInput: trimmed,
    });
  }

  return candidates;
}

export function toTorrentCandidates(fileList: Array<{ path: string; size: number }>): Candidate[] {
  return fileList.map((file) => ({
    source: "torrent",
    displayPath: file.path,
    classifyInput: file.path,
  }));
}

export function filterVideoCandidates(candidates: Candidate[]): FilteredCandidates {
  const valid: Candidate[] = [];
  const ignoredPreview: string[] = [];
  let ignoredCount = 0;

  for (const candidate of candidates) {
    if (isVideoPath(candidate.classifyInput)) {
      valid.push(candidate);
      continue;
    }

    ignoredCount++;
    if (ignoredPreview.length < IGNORED_PREVIEW_CAP) {
      ignoredPreview.push(candidate.displayPath);
    }
  }

  return {
    valid,
    ignoredCount,
    ignoredPreview,
  };
}

export function buildIgnoredNotice(ignoredCount: number, preview: string[]): string {
  if (ignoredCount <= 0) {
    return "";
  }

  const shown = preview.slice(0, IGNORED_NOTICE_PREVIEW_LIMIT).join(", ");
  if (!shown) {
    return `Ignored ${ignoredCount} non-video item(s)`;
  }

  const suffix = ignoredCount > IGNORED_NOTICE_PREVIEW_LIMIT ? "..." : "";
  return `Ignored ${ignoredCount} non-video item(s) (e.g. ${shown}${suffix})`;
}
