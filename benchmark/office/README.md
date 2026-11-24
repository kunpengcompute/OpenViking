# 办公案例测试套

提供日程、报工、邮件、PPT 更新和资料生成文档五个固化案例。H 为记忆准备轮，Q 为独立会话中的测试轮；执行通过 Chat HTTP/SSE 接口完成。

| 案例 | 任务与数据 | 执行入口 |
|---|---|---|
| 日程 | `tasks/schedule/{H,Q}.md`；`datasets/schedule/` | `benchmark.office.suites.run --case schedule` |
| 报工 | `tasks/timesheet/{H,Q}.md`；`datasets/timesheet/` | `benchmark.office.suites.run --case timesheet` |
| 邮件 | `tasks/mail/{H,Q}.md`；`datasets/mail/` | `benchmark.office.suites.run --case mail` |
| PPT 更新 | `cases/ppt_update/spec.json`；`tasks/ppt_update/{H,Q}.md`；`datasets/ppt_update/` | `benchmark.office.suites.ppt_update.compare` |
| 资料生成文档 | `cases/reqdoc_generation/spec.json`；`tasks/reqdoc_generation/{H,Q}.md`；`datasets/reqdoc_generation/` | `benchmark.office.suites.reqdoc_generation.run` |

五案例通过Chat采集实际模型输入、工具轨迹、业务状态快照和交付物，使用[evaluation/AI_REVIEW_PROMPT.md](evaluation/AI_REVIEW_PROMPT.md)进行AI证据复核。预算由[evaluation/policy.json](evaluation/policy.json)固定；PPT为120秒，资料生成文档为550秒、六章27项检查点。性能脚本汇总完整耗时和Token，质量结果来自逐项复核账本。

```text
office/
├── admin/       # 数据准备、账号与工作区管理
├── cases/       # 案例实现及公共页面服务
│   ├── portal/             # 公共登录、导航、工作区初始化与存储连接
│   ├── schedule/           # 日程
│   ├── timesheet/          # 报工页面与数据初始化
│   ├── mail/               # 邮件页面数据
│   ├── ppt_update/         # PPT 更新
│   └── reqdoc_generation/  # 资料生成文档
├── datasets/    # 五个案例的固化数据与评分参考
│   ├── schedule/           # 日程规则、背景会议、会议室与占用数据
│   ├── timesheet/          # 报工 H/Q 目标和卡片目录
│   ├── mail/               # 邮件 H/Q 目标和背景邮件
│   ├── ppt_update/         # 原始 PPT
│   └── reqdoc_generation/  # 六份输入资料与评分参考
├── evaluation/  # AI复核提示词、验收预算与复核记录说明
├── suites/      # Chat执行、原始证据收集和性能汇总
│   ├── run.py              # 日程、报工、邮件的统一执行入口
│   ├── common/             # SSE、证据保存、快照统计等公共能力
│   ├── ppt_update/         # PPT对照执行与证据采集
│   └── reqdoc_generation/  # 文档执行与证据采集
├── tasks/       # 每个案例的 H/Q 提示词
└── tests/       # 页面、数据及准备脚本回归
```

部署和执行步骤见[部署指南](../../examples/deepagents_ov_context/docs/DEPLOYMENT.md)与[测试指南](../../examples/deepagents_ov_context/docs/TESTING.md)。原始轨迹、下载文件、日志和报告默认写入仓库根目录的 `result/`；版本演进由 Git 提交记录保存。
