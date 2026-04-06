# Web Parser Demo

[English](README.md)

该目录是媒体文件名解析器的浏览器端演示应用。

项目安装与 Python 包说明请见根目录 [README.zh-CN.md](../../README.zh-CN.md)。

## 功能行为

- 支持手动多行输入（每个非空行视为一个候选）
- 支持 `.torrent` 文件拖拽上传与文件选择
- 仅对视频后缀文件执行分类
- 非视频项会被忽略，并通过简短 Notice 提示
- 过滤后若无有效视频项，会清空结果并显示空状态
- 解析结果以列表形式展示

## 在仓库根目录运行

```bash
python -m tools.export_parser_constants
python -m tools.check_web_parser_parity --input scrape_files_list_merged.txt --limit 200
npm --prefix web/app install
npm --prefix web/app run test
npm --prefix web/app run build-dist
python -m http.server 8080 --directory web/app/dist
```

然后访问 `http://localhost:8080/`。

## 在 `web/app` 目录运行

```bash
npm install
npm run test
npm run build-dist
python -m http.server 8080 --directory dist
```

## 构建产物

`build-dist` 会生成：

- `dist/index.html`
- `dist/styles.css`
- `dist/app.js`

## 说明

- Web 端使用的常量由 `tools/export_parser_constants.py` 从 Python 规则导出生成。
- GitHub Pages 发布由 `.github/workflows/deploy-pages.yml` 负责。
