import bencode from "bencode";
import { Buffer } from "buffer";

const globalWithBuffer = globalThis as typeof globalThis & { Buffer?: typeof Buffer };

if (!globalWithBuffer.Buffer) {
  globalWithBuffer.Buffer = Buffer;
}

export interface ParsedTorrentFile {
  infoHash: string;
  torrentName: string;
  fileList: Array<{ path: string; size: number }>;
}

type TorrentInfo = {
  name?: unknown;
  length?: unknown;
  files?: Array<{
    length?: unknown;
    path?: unknown[];
  }>;
};

function toUtf8String(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  if (value instanceof Uint8Array || Buffer.isBuffer(value)) {
    return Buffer.from(value).toString("utf8");
  }

  if (value === null || value === undefined) {
    return "";
  }

  return String(value);
}

function toByteSize(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }

  if (typeof value === "bigint") {
    return Number(value);
  }

  return Number(value ?? 0);
}

function toHex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function buildFileList(info: TorrentInfo): Array<{ path: string; size: number }> {
  if (Array.isArray(info.files) && info.files.length > 0) {
    return info.files
      .map((entry) => ({
        path: Array.isArray(entry.path) ? entry.path.map(toUtf8String).filter(Boolean).join("/") : "",
        size: toByteSize(entry.length),
      }))
      .filter((entry) => entry.path);
  }

  const torrentName = toUtf8String(info.name);
  if (!torrentName) {
    return [];
  }

  return [
    {
      path: torrentName,
      size: toByteSize(info.length),
    },
  ];
}

export async function parseTorrentFile(file: File): Promise<ParsedTorrentFile> {
  const source = Buffer.from(await file.arrayBuffer());
  const decoded = bencode.decode(source) as { info?: TorrentInfo };
  const info = decoded?.info;

  if (!info || typeof info !== "object") {
    throw new Error("无效的种子文件");
  }

  const infoBytes = Buffer.from(bencode.encode(info));
  const digest = await crypto.subtle.digest("SHA-1", infoBytes);
  const torrentName = toUtf8String(info.name) || file.name.replace(/\.torrent$/iu, "");

  return {
    infoHash: toHex(digest),
    torrentName,
    fileList: buildFileList(info),
  };
}
