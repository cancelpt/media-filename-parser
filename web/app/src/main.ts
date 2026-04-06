import {
  type Candidate,
  buildIgnoredNotice,
  collectManualCandidates,
  filterVideoCandidates,
  toTorrentCandidates,
} from "./input-processing";
import { parseFilename } from "./parser/parser";
import { parseTorrentFile } from "./torrent";

type ParsedRow = ReturnType<typeof parseFilename>;
type ParsedFields = ParsedRow["parsed"];
type ResultItem = {
  candidate: Candidate;
  parsed: ParsedRow;
  filterText: string;
};

const SAMPLE_INPUTS = [
  "[CancelWEB][CONAN][1130][WEBRIP][1080P][HEVC_AAC].mkv",
  "Constantine.2005.2160p.UHD.BluRay.REMUX.DV.HDR.HEVC.TrueHD.7.1.mkv",
  "Life.1984.WEB-DL.1080P.x264.AAC.mp4",
  "README.txt",
] as const;

const inputEl = document.querySelector<HTMLTextAreaElement>("#path-input");
const parseBtn = document.querySelector<HTMLButtonElement>("#parse-btn");
const torrentInputEl = document.querySelector<HTMLInputElement>("#torrent-input");
const dropzoneEl = document.querySelector<HTMLElement>("#torrent-dropzone");
const sourceSummaryEl = document.querySelector<HTMLElement>("#source-summary");
const ignoredNoticeEl = document.querySelector<HTMLElement>("#ignored-notice");
const resultToolbarEl = document.querySelector<HTMLElement>("#result-toolbar");
const resultFilterEl = document.querySelector<HTMLInputElement>("#result-filter");
const pageSizeEl = document.querySelector<HTMLSelectElement>("#page-size");
const pageSummaryEl = document.querySelector<HTMLElement>("#page-summary");
const pageInfoEl = document.querySelector<HTMLElement>("#page-info");
const pagePrevEl = document.querySelector<HTMLButtonElement>("#page-prev");
const pageNextEl = document.querySelector<HTMLButtonElement>("#page-next");
const emptyStateEl = document.querySelector<HTMLElement>("#empty-state");
const resultTableWrapEl = document.querySelector<HTMLElement>("#result-table-wrap");
const resultTableBodyEl = document.querySelector<HTMLTableSectionElement>("#result-table-body");
const sampleWrap = document.querySelector<HTMLElement>("#sample-list");
let dropzoneDragDepth = 0;
let allResults: ResultItem[] = [];
let defaultEmptyMessage = "No results yet.";
let currentPage = 1;

if (
  !inputEl ||
  !parseBtn ||
  !torrentInputEl ||
  !dropzoneEl ||
  !sourceSummaryEl ||
  !ignoredNoticeEl ||
  !resultToolbarEl ||
  !resultFilterEl ||
  !pageSizeEl ||
  !pageSummaryEl ||
  !pageInfoEl ||
  !pagePrevEl ||
  !pageNextEl ||
  !emptyStateEl ||
  !resultTableWrapEl ||
  !resultTableBodyEl ||
  !sampleWrap
) {
  throw new Error("Missing required DOM nodes.");
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "-";
  }

  return String(value);
}

function sourceLabel(source: Candidate["source"]): string {
  return source === "torrent" ? "Torrent" : "Manual";
}

function setIgnoredNotice(ignoredCount: number, ignoredPreview: string[]): void {
  const notice = buildIgnoredNotice(ignoredCount, ignoredPreview);
  ignoredNoticeEl.textContent = notice;
  ignoredNoticeEl.hidden = !notice;
}

function setSourceStatus(message: string): void {
  sourceSummaryEl.textContent = message;
}

function clearResults(message: string): void {
  resultTableBodyEl.innerHTML = "";
  resultTableWrapEl.hidden = true;
  emptyStateEl.textContent = message;
  emptyStateEl.hidden = false;
}

function normalizeKeyword(value: string): string {
  return value.trim().toLowerCase();
}

function getPageSize(): number | "all" {
  return pageSizeEl.value === "all" ? "all" : Number(pageSizeEl.value);
}

function getFilteredResults(): ResultItem[] {
  const keyword = normalizeKeyword(resultFilterEl.value);
  if (!keyword) {
    return allResults;
  }

  return allResults.filter((item) => item.filterText.includes(keyword));
}

function setPaginationState(
  filteredCount: number,
  startIndex: number,
  endIndex: number,
  totalPages: number
): void {
  const hasFilter = normalizeKeyword(resultFilterEl.value).length > 0;

  if (filteredCount === 0) {
    pageSummaryEl.textContent = hasFilter
      ? `过滤后 0 条（总计 ${allResults.length} 条）`
      : "共 0 条";
    pageInfoEl.textContent = "第 0 / 0 页";
    pagePrevEl.disabled = true;
    pageNextEl.disabled = true;
    return;
  }

  if (hasFilter) {
    pageSummaryEl.textContent = `显示 ${startIndex}-${endIndex} 条（过滤后 ${filteredCount} 条 / 总计 ${allResults.length} 条）`;
  } else {
    pageSummaryEl.textContent = `显示 ${startIndex}-${endIndex} 条（共 ${filteredCount} 条）`;
  }

  pageInfoEl.textContent = `第 ${currentPage} / ${totalPages} 页`;
  pagePrevEl.disabled = currentPage <= 1;
  pageNextEl.disabled = currentPage >= totalPages;
}

function renderResultView(): void {
  const filtered = getFilteredResults();

  if (allResults.length === 0) {
    resultToolbarEl.hidden = true;
    setPaginationState(0, 0, 0, 0);
    clearResults(defaultEmptyMessage);
    return;
  }

  resultToolbarEl.hidden = false;

  if (filtered.length === 0) {
    setPaginationState(0, 0, 0, 0);
    clearResults("没有匹配当前文件名过滤条件的结果。");
    return;
  }

  const pageSize = getPageSize();
  const totalPages = pageSize === "all" ? 1 : Math.max(1, Math.ceil(filtered.length / pageSize));
  currentPage = Math.min(Math.max(1, currentPage), totalPages);

  const startOffset = pageSize === "all" ? 0 : (currentPage - 1) * pageSize;
  const endOffset = pageSize === "all" ? filtered.length : Math.min(filtered.length, startOffset + pageSize);
  const pageItems = filtered.slice(startOffset, endOffset);

  resultTableBodyEl.innerHTML = "";
  for (const item of pageItems) {
    appendResultRow(item.candidate, item.parsed);
  }

  emptyStateEl.hidden = true;
  resultTableWrapEl.hidden = false;
  setPaginationState(filtered.length, startOffset + 1, endOffset, totalPages);
}

function makeCell(text: string, className?: string): HTMLTableCellElement {
  const cell = document.createElement("td");
  if (className) {
    cell.className = className;
  }
  cell.textContent = text;
  return cell;
}

function appendResultRow(candidate: Candidate, parsedRow: ParsedRow): void {
  const fields = parsedRow.parsed as ParsedFields;
  const row = document.createElement("tr");

  row.appendChild(makeCell(sourceLabel(candidate.source), "source"));
  row.appendChild(makeCell(candidate.displayPath, "path"));
  row.appendChild(makeCell(displayValue(fields.title), "title"));
  row.appendChild(makeCell(displayValue(fields.zh_title), "title-zh"));
  row.appendChild(makeCell(displayValue(fields.year), "year"));
  row.appendChild(makeCell(displayValue(fields.season_episode), "season"));
  row.appendChild(makeCell(displayValue(fields.resolution), "resolution"));
  row.appendChild(makeCell(displayValue(fields.video_codec), "video-codec"));
  row.appendChild(makeCell(displayValue(fields.audio_codec), "audio-codec"));
  row.appendChild(makeCell(parsedRow.confidence.toFixed(2), "confidence"));

  resultTableBodyEl.appendChild(row);
}

function renderResults(
  candidates: Candidate[],
  summary: string,
  emptyMessage: string,
  ignoredCount: number,
  ignoredPreview: string[]
): void {
  setSourceStatus(summary);
  setIgnoredNotice(ignoredCount, ignoredPreview);

  if (candidates.length === 0) {
    defaultEmptyMessage = emptyMessage;
    allResults = [];
    currentPage = 1;
    renderResultView();
    return;
  }

  defaultEmptyMessage = emptyMessage;
  allResults = candidates.map((candidate) => {
    const parsed = parseFilename(candidate.classifyInput);
    return {
      candidate,
      parsed,
      filterText: `${candidate.displayPath} ${parsed.filename}`.toLowerCase(),
    };
  });
  currentPage = 1;
  renderResultView();
}

function parseManualInput(): void {
  const candidates = collectManualCandidates(inputEl.value);
  const filtered = filterVideoCandidates(candidates);

  renderResults(
    filtered.valid,
    `Manual input: ${filtered.valid.length} video item(s) kept from ${candidates.length}.`,
    "No video item found from manual input.",
    filtered.ignoredCount,
    filtered.ignoredPreview
  );
}

async function parseTorrentEntries(file: File): Promise<void> {
  const torrent = await parseTorrentFile(file);
  const candidates = toTorrentCandidates(torrent.fileList);
  const filtered = filterVideoCandidates(candidates);

  renderResults(
    filtered.valid,
    `Torrent ${torrent.torrentName}: ${filtered.valid.length}/${candidates.length} video item(s).`,
    "No video item found in this torrent.",
    filtered.ignoredCount,
    filtered.ignoredPreview
  );
}

async function handleTorrentFile(file: File | undefined): Promise<void> {
  if (!file) {
    setSourceStatus("No torrent file selected.");
    return;
  }

  if (!/\.torrent$/iu.test(file.name)) {
    setSourceStatus(`Invalid file type: ${file.name}`);
    return;
  }

  setSourceStatus(`Reading torrent: ${file.name}`);

  try {
    await parseTorrentEntries(file);
  } catch (error) {
    setSourceStatus(`Failed to parse torrent: ${file.name}`);
    console.error(error);
  } finally {
    torrentInputEl.value = "";
  }
}

function renderSamples(): void {
  sampleWrap.innerHTML = "";

  for (const sample of SAMPLE_INPUTS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample-chip";
    btn.textContent = sample;
    btn.addEventListener("click", () => {
      inputEl.value = sample;
      parseManualInput();
    });
    sampleWrap.appendChild(btn);
  }
}

parseBtn.addEventListener("click", () => {
  parseManualInput();
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    parseManualInput();
  }
});

resultFilterEl.addEventListener("input", () => {
  currentPage = 1;
  renderResultView();
});

pageSizeEl.addEventListener("change", () => {
  currentPage = 1;
  renderResultView();
});

pagePrevEl.addEventListener("click", () => {
  if (currentPage <= 1) {
    return;
  }
  currentPage -= 1;
  renderResultView();
});

pageNextEl.addEventListener("click", () => {
  const filtered = getFilteredResults();
  const pageSize = getPageSize();
  const totalPages = pageSize === "all" ? 1 : Math.max(1, Math.ceil(filtered.length / pageSize));
  if (currentPage >= totalPages) {
    return;
  }
  currentPage += 1;
  renderResultView();
});

torrentInputEl.addEventListener("change", async () => {
  await handleTorrentFile(torrentInputEl.files?.[0]);
});

dropzoneEl.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzoneEl.classList.add("is-dragover");
});

dropzoneEl.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dropzoneDragDepth += 1;
  dropzoneEl.classList.add("is-dragover");
});

dropzoneEl.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dropzoneDragDepth = Math.max(0, dropzoneDragDepth - 1);
  if (dropzoneDragDepth === 0) {
    dropzoneEl.classList.remove("is-dragover");
  }
});

dropzoneEl.addEventListener("drop", async (event) => {
  event.preventDefault();
  dropzoneDragDepth = 0;
  dropzoneEl.classList.remove("is-dragover");
  const file = Array.from(event.dataTransfer?.files ?? [])[0];
  await handleTorrentFile(file);
});

renderSamples();
inputEl.value = [SAMPLE_INPUTS[0], SAMPLE_INPUTS[2], SAMPLE_INPUTS[3]].join("\n");
parseManualInput();
