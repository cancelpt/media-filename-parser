#!/usr/bin/env node

import readline from "node:readline";
import { parseFilename } from "./parser";

function printOne(text, pretty) {
  const result = parseFilename(text);
  const serialized = pretty
    ? JSON.stringify(result, null, 2)
    : JSON.stringify(result);
  process.stdout.write(`${serialized}\n`);
}

function main() {
  const args = process.argv.slice(2);
  const pretty = args.includes("--pretty");
  const inputs = args.filter((arg) => arg !== "--pretty");

  if (inputs.length > 0) {
    for (const text of inputs) {
      printOne(text, pretty || inputs.length === 1);
    }
    return;
  }

  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  rl.on("line", (line) => {
    const text = line.trim();
    if (!text) return;
    printOne(text, false);
  });
}

main();
