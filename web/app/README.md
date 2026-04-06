# Web Parser Demo

[简体中文](README.zh-CN.md)

This directory contains the browser demo for the media filename parser.

For project setup and package usage, see the root [README](../../README.md).

## Behavior

- Supports manual multi-line input (one candidate per non-empty line)
- Supports `.torrent` drag/upload and file picker
- Only video suffixes are classified
- Non-video items are ignored and summarized in a short notice
- If no valid video entries remain, the result list is cleared and empty state is shown
- Parsed items render as a list

## Run From Repository Root

```bash
python -m tools.export_parser_constants
python -m tools.check_web_parser_parity --input scrape_files_list_merged.txt --limit 200
npm --prefix web/app install
npm --prefix web/app run test
npm --prefix web/app run build-dist
python -m http.server 8080 --directory web/app/dist
```

Then open `http://localhost:8080/`.

## Run Inside `web/app`

```bash
npm install
npm run test
npm run build-dist
python -m http.server 8080 --directory dist
```

## Build Output

`build-dist` creates:

- `dist/index.html`
- `dist/styles.css`
- `dist/app.js`

## Notes

- Constants used by the web parser are generated from Python rules via `tools/export_parser_constants.py`.
- GitHub Pages deployment is handled by `.github/workflows/deploy-pages.yml`.
