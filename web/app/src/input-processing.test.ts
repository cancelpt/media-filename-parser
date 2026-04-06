import { describe, expect, it } from "vitest";
import {
  buildIgnoredNotice,
  collectManualCandidates,
  filterVideoCandidates,
  isVideoPath,
  toTorrentCandidates,
} from "./input-processing";

describe("collectManualCandidates", () => {
  it("splits multiline input and dedupes case-insensitively", () => {
    expect(
      collectManualCandidates("  /media/One.MKV\n\n/media/two.mp4\n/media/one.mkv\n/MEDIA/TWO.MP4  ")
    ).toEqual([
      {
        source: "manual",
        displayPath: "/media/One.MKV",
        classifyInput: "/media/One.MKV",
      },
      {
        source: "manual",
        displayPath: "/media/two.mp4",
        classifyInput: "/media/two.mp4",
      },
    ]);
  });
});

describe("isVideoPath", () => {
  const allowedVideoPaths = [
    "movie.mp4",
    "movie.mkv",
    "movie.avi",
    "movie.mov",
    "movie.wmv",
    "movie.flv",
    "movie.webm",
    "movie.ts",
    "movie.m2ts",
    "movie.vob",
  ];

  it.each(allowedVideoPaths)("accepts allowed video suffix case-insensitively: %s", (path) => {
    expect(isVideoPath(path.toUpperCase())).toBe(true);
  });

  it("rejects non-video paths", () => {
    expect(isVideoPath("notes.txt")).toBe(false);
  });
});

describe("toTorrentCandidates", () => {
  it("maps torrent files to torrent candidates", () => {
    expect(
      toTorrentCandidates([
        { path: "season/episode1.mkv", size: 123 },
        { path: "extras/readme.txt", size: 45 },
      ])
    ).toEqual([
      {
        source: "torrent",
        displayPath: "season/episode1.mkv",
        classifyInput: "season/episode1.mkv",
      },
      {
        source: "torrent",
        displayPath: "extras/readme.txt",
        classifyInput: "extras/readme.txt",
      },
    ]);
  });
});

describe("filterVideoCandidates", () => {
  it("keeps only video candidates", () => {
    expect(
      filterVideoCandidates([
        { source: "manual", displayPath: "movie.mkv", classifyInput: "movie.mkv" },
        { source: "manual", displayPath: "notes.txt", classifyInput: "notes.txt" },
        { source: "torrent", displayPath: "clip.mp4", classifyInput: "clip.mp4" },
      ])
    ).toEqual({
      valid: [
        { source: "manual", displayPath: "movie.mkv", classifyInput: "movie.mkv" },
        { source: "torrent", displayPath: "clip.mp4", classifyInput: "clip.mp4" },
      ],
      ignoredCount: 1,
      ignoredPreview: ["notes.txt"],
    });
  });

  it("caps ignored preview at the first four ignored items", () => {
    expect(
      filterVideoCandidates([
        { source: "manual", displayPath: "a.txt", classifyInput: "a.txt" },
        { source: "manual", displayPath: "b.txt", classifyInput: "b.txt" },
        { source: "manual", displayPath: "c.txt", classifyInput: "c.txt" },
        { source: "manual", displayPath: "d.txt", classifyInput: "d.txt" },
        { source: "manual", displayPath: "e.txt", classifyInput: "e.txt" },
      ])
    ).toEqual({
      valid: [],
      ignoredCount: 5,
      ignoredPreview: ["a.txt", "b.txt", "c.txt", "d.txt"],
    });
  });
});

describe("buildIgnoredNotice", () => {
  it("returns an empty string when there are no ignored items", () => {
    expect(buildIgnoredNotice(0, ["a.txt"])).toBe("");
    expect(buildIgnoredNotice(-1, ["a.txt"])).toBe("");
  });

  it("returns a sane message when there are ignored items but no preview", () => {
    expect(buildIgnoredNotice(2, [])).toBe("Ignored 2 non-video item(s)");
  });

  it("truncates preview to the first three ignored items and appends ellipsis", () => {
    expect(buildIgnoredNotice(5, ["a.txt", "b.txt", "c.txt", "d.txt"])).toBe(
      "Ignored 5 non-video item(s) (e.g. a.txt, b.txt, c.txt...)"
    );
  });
});
