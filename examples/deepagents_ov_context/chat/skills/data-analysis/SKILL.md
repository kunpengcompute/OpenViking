---
name: data-analysis
description: 对本轮工作区中实际存在的 XLSX、CSV、TSV 文件做结构检查、统计、筛选和关联分析，导出 CSV 或 JSON。生成或修改 Excel 文件时结合 office-documents。
license: MIT
---

# 表格分析

工作流参考 DeerFlow data-analysis，已适配本轮工作区。脚本使用预先安装的 pandas、openpyxl、DuckDB，直接在内存中处理数据。

先确认输入文件实际路径和用户要求的统计口径，再检查列名、类型、行数、空值和样本：

```bash
python skills/data-analysis/scripts/analyze.py --files input.xlsx --action inspect
```

脚本为每个 sheet / 文件返回唯一 SQL 表名。按返回的表名与列名构造查询，列名含空格时用双引号：

```bash
python skills/data-analysis/scripts/analyze.py --files input.xlsx --action query \
  --sql 'SELECT category, SUM(amount) AS total FROM "Sheet1" GROUP BY category' \
  --output-file output/totals.csv
```

数值概览：

```bash
python skills/data-analysis/scripts/analyze.py --files input.csv --action summary
```

核对合计、分母、日期范围、单位与去重口径。保留零值、空值和未知值的区别。跨表关联先检查关联键与一对多关系，核对关联前后行数。输入含公式时，读取的数值取决于 Excel 已保存的缓存；必要时按 office-documents 重新计算后再分析。

交付需要表格时再导出文件；简短问题可以直接根据查询结果回答。中间文件放 `tmp/`，最终 CSV / JSON 放 `output/`。查询脚本输出的是计算证据，分析结论依据用户任务整理。
