import { describe, expect, it } from "vitest";
import { parseFilename } from "./parser";

describe("parseFilename", () => {
  it("keeps numeric prefixes with english title blocks in mixed cjk names", () => {
    const result = parseFilename("[首尔之春].12.12.The.Day.2023.HKG.BluRay.1080p.x264.DDP5.1-CancelHD.mkv");
    expect(result.parsed.title).toBe("12 12 The Day");
    expect(result.parsed.zh_title).toBe("首尔之春");
  });
});
