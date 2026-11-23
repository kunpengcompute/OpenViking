# 办公案例AI验收

[AI_REVIEW_PROMPT.md](AI_REVIEW_PROMPT.md)定义五案例的证据核验、时间判定、成功率与文档综合得分口径。[policy.json](policy.json)保存统一预算和三轮重复要求。任务正文继续使用 `../tasks/` 的H/Q模板。

## 复核材料

每轮提供 `condition.json`、`result.json`、`trace.json`／SSE、`evidence.json`、下载产物及回执；同时提供Chat数据目录中该Run的 `manifest.json`、`snapshots.jsonl` 和 `blobs/`，以及冻结的任务和原始输入。正式分组清单明确Run ID、组别及重复序号。

AI按提示词核验内容、完成时点和交付要求，输出逐项JSON／CSV账本和分案例报告。复核结果放在本轮 `result/`，与原始证据分别保存。报告先列条件与规则，再给原结果、复核结果、修正原因及证据表。

## 一条命令汇总材料

在Chat容器内或能读取runner结果与Chat数据的环境执行：

```bash
python -m benchmark.office.evaluation.collect \
  --runs /opt/openviking/result/ppt/pair-{1,2,3}/Q-{baseline,memory} \
  --chat-data /data/chat \
  --output /opt/openviking/result/ai-review/ppt
```

`--runs`明确选择本次正式Run的输出目录，可同时选择多个案例和H/Q。`--chat-data`指向包含 `benchmark/`、`inputs/`、`artifacts/` 的Chat数据根目录。输出目录使用新路径，防止覆盖原始材料。

归档材料的目录与当前部署不同，可使用清单逐Run指定原始审计目录：

```json
[
  {
    "directory": "mail/Q-baseline-1",
    "audit_directory": "audit/mail/r-example"
  },
  {
    "directory": "mail/Q-memory-1",
    "audit_directory": "audit/mail/r-another"
  }
]
```

将该JSON保存为 `result/selected-runs.json`，其中相对路径以清单所在目录为起点：

```bash
python -m benchmark.office.evaluation.collect \
  --run-list result/selected-runs.json --output result/ai-review/mail
```

汇总结构如下，整份目录提供给有文件读取能力的AI：

```text
ai-review/
├── README.md               # AI读取顺序与证据定位说明
├── AI_REVIEW_PROMPT.md      # 复核提示词副本
├── policy.json             # 验收预算
├── runs.json               # 正式样本、组别、Run ID及证据目录
├── files.json              # 原来源、包内相对路径、SHA256与大小
├── issues.json             # 缺失、损坏、身份冲突或采集错误
├── manifest.json           # 汇总完整性与待复核状态
├── reference-source/       # 提供的任务模板、数据和案例定义参考
└── runs/001-mail-baseline/
    ├── task.md             # condition中保存的实际任务
    ├── execution/          # 条件、模型与工具轨迹、回执及失败记录
    ├── audit/              # 冻结参考、原始快照及blobs
    │   └── versions/       # 按原文件名展开各哈希版本
    ├── inputs/             # 按上传回执或spec核对哈希的输入
    ├── artifacts/          # 下载／发布产物
    └── prior-reviews/      # 可选的历史评分，独立判断后再对照
```

采集端在manifest中保存日程目标、报工目标、邮件目标正文及PPT／文档spec。汇总脚本核对可用文件哈希，保留原始JSON和字节，同时为截图、正文及PPT生成可直接读取的版本路径。源码参考与实际输入分开放置，实际任务和冻结参考优先。

脚本退出码0表示本次材料检查通过；退出码2表示目录已生成但存在缺项或损坏，先查看 `issues.json`。完整性标记仅表示材料检查结果。失败Run继续保留，质量结果由AI逐项复核。配置凭据、环境文件和SQLite数据库不在收集清单中。

## 执行与统计边界

案例runner发起任务并采集证据，性能汇总脚本统计完整Run耗时和模型usage。Chat仅导出采集结果；任务成功率和分步完成时点通过AI证据复核确认。证据不足的项目保持待核定，按固定分母输出结果范围。

PPT H的业务复核记录格式和执行命令见[案例说明](../cases/ppt_update/README.md)。记忆池完整性、真实召回和模型／输入一致性仍由执行检查保护。
