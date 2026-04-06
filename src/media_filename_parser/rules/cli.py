"""CLI entrypoint for batch parsing with the rules engine."""

import json
import logging
import os
import sys

from .parser import parse_filename

LOGGER = logging.getLogger("media_filename_parser.rules.cli")


def setup_logging() -> None:
    """Configure default CLI logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """Run batch parsing against the default input file paths."""
    setup_logging()
    input_file = "scrape_files_list_merged.txt"
    output_file = "parsed_dataset.json"

    if not os.path.exists(input_file):
        LOGGER.error("%s not found.", input_file)
        sys.exit(1)

    results = []

    LOGGER.info("Parsing %s ...", input_file)
    with open(input_file, "r", encoding="utf-8") as f:
        # Load all lines
        lines = f.readlines()

    for i, line in enumerate(lines):
        filepath = line.strip()
        if not filepath:
            continue

        parsed_data = parse_filename(filepath)

        # Remove the internal flag before saving
        if "_inherited_title" in parsed_data["parsed"]:
            del parsed_data["parsed"]["_inherited_title"]

        results.append(parsed_data)

        if (i + 1) % 2000 == 0:
            LOGGER.info("Processed %d lines ...", i + 1)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    low_conf_results = [r for r in results if r["confidence"] < 1.0]
    low_conf_results.sort(key=lambda x: x["confidence"])

    low_conf_file = "parsed_dataset_low_confidence.json"
    with open(low_conf_file, "w", encoding="utf-8") as f:
        json.dump(low_conf_results, f, ensure_ascii=False, indent=2)

    LOGGER.info("Total processed: %d. Saving to %s ...", len(results), output_file)
    LOGGER.info(
        "Found %d items with confidence < 1.0. Saved to %s ...",
        len(low_conf_results),
        low_conf_file,
    )
    LOGGER.info("Parsing completed.")
