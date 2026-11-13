# PDF

本地 PDF 提取文本时优先 `pdftotext -layout`（Poppler），其次 pypdf；pdfplumber 仅用于表格等结构化场景。若提取结果出现 `(cid:` 占位，说明字体映射缺失，立即改用 pdftotext 或 pypdf，不要渲染图片后再读图。在线 PDF 先使用 browser 提供的 download / pdf_text 工作流；使用工具实际返回的文件信息。

生成 PDF 使用 ReportLab；中文文本选择系统中实际存在的 CJK 字体并嵌入，先检查字体路径和授权。合并、拆分或旋转使用 pypdf，操作后核对页数、顺序和关键内容。扫描页的文本提取为空时，先判断是否需要 OCR 依赖，再报告能力与结果。

最终 PDF 保存到 `output/`，通过 `PdfReader(path)` 重新打开，核对页数和关键字段。版面检查：

```bash
pdftoppm -png -scale-to 1400 output/report.pdf tmp/report
```

核对分页、裁切、中文字体和表格换行。预览放 `tmp/`；仅将用户需要的 PDF 放入 `output/`。提取结果保留页码，引用具体段落、数字或表格时能对应原文。
