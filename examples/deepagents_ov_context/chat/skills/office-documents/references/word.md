# Word

使用 `from docx import Document` 读取或创建 DOCX。修改模板前检查段落、样式、表格、页眉页脚和分节；保留模板格式，注意直接赋值 `paragraph.text` 会重建 runs 并丢失原有局部样式。新文档使用内置标题样式、列表样式、真实表格和明确的页面设置。

保存到 `output/report.docx` 后通过 `Document(path)` 重新打开，检查标题、表格行数、关键文本、单位和来源。页面视觉检查可执行：

```bash
mkdir -p tmp/preview
profile="$(python -c 'from pathlib import Path; print((Path("tmp") / "lo-profile").resolve().as_uri())')"
soffice "-env:UserInstallation=$profile" --headless \
  --convert-to pdf --outdir tmp/preview output/report.docx
pdftoppm -png -scale-to 1400 tmp/preview/report.pdf tmp/preview/page
```

为 LibreOffice 使用本轮唯一配置目录，避免并发运行共用进程。转换后检查输出存在。图像查看使用文件工具返回的图像能力；如当前模型只能读取文本，说明结构与文本已检查、视觉检查待完成。

python-docx 可处理普通段落、表格和样式。复杂修订、批注、嵌入对象或需要精确保真的改动，先评估能力和文件结构，再确定处理范围。
