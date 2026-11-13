# Excel

使用 openpyxl 读取、创建或修改 XLSX。修改前检查所有 sheet、表头、类型、公式、合并单元格、命名范围和样式。保留公式、单位、数字格式和用户模板。金额、百分比与日期使用匹配的显示格式；计算通过脚本完成。

读取公式使用 `load_workbook(path, data_only=False)`；检查已缓存结果使用 `data_only=True`。openpyxl 保存公式文本，公式计算由 Excel / LibreOffice 完成；空缓存作为未知值处理。

需要重算时，先保存到 `tmp/recalc/report.xlsx`，再使用本轮唯一的 LibreOffice 配置目录转换到 `output/`：

```bash
profile="$(python -c 'from pathlib import Path; print((Path("tmp") / "lo-profile").resolve().as_uri())')"
soffice "-env:UserInstallation=$profile" --headless \
  --convert-to xlsx --outdir output tmp/recalc/report.xlsx
```

确认输出文件实际生成，再分别以两种 data_only 模式重新打开。核对关键公式仍存在、关键计算结果与独立计算一致，检查公式错误单元格。统计文件可同时交付来源、口径、行数和总计。

复杂宏、外部连接和不支持的 Excel 特性需要单独评估；常规工作流使用 XLSX。预览、重算副本和辅助文件放在 `tmp/`。
