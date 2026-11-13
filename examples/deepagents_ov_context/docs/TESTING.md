# OV Middleware 测试指南

部署环境按 [DEPLOYMENT.md](DEPLOYMENT.md) 准备。单元测试与隔离集成测试使用确定性替身；真实服务和业务验收使用实际 OV、模型与工具，分别记录结果。

正式验证依次执行部署指南中的镜像准备、配置、服务启动与数据集初始化，再执行本文的 UT、ST 和业务案例。部署完成后保存宿主机 IP、源码版本、镜像 ID、有效配置和数据集版本。仅进行部署验收时，以 `container.sh check` 收口；正式测试使用固化的代码、配置和数据集，另行保存测试结果。

## 1. 单元与隔离集成测试

从仓库根目录执行：

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python -m pytest -o addopts='' -q \
  tests/unit/ov_context \
  tests/integration/ov_context \
  tests/client/test_http_request_identity.py
```

容器中使用镜像内的 Python 与依赖：

```bash
bash examples/deepagents_ov_context/scripts/container.sh run \
  env LITELLM_LOCAL_MODEL_COST_MAP=true python -m pytest -o addopts='' -q \
  tests/unit/ov_context tests/integration/ov_context \
  tests/client/test_http_request_identity.py
```

| 目录 | 覆盖范围 |
|---|---|
| `tests/unit/ov_context/` | Reader 排序与预算、消息格式、组装、工具回读、配置、页面及验收脚本的负向控制 |
| `tests/integration/ov_context/` | 真实 LangChain／DeepAgents 图、Hook 组合、Writer／Reader 契约、状态恢复、Chat HTTP/SSE 与 SQLite |
| `tests/client/test_http_request_identity.py` | SDK 请求级身份、认证模式及共享默认请求头 |
| `tests/ov_context_support/` | 上述测试共用的配置、消息、SDK 和服务替身 |

`test_chat_context_policy.py` 验证 Chat 的预算职责：OV 组使用 Middleware 计数器与配置；无 OV 组使用 DeepAgents 原生摘要的默认计数及压缩。`test_chat_model_profile.py` 验证会话窗口配置与共享模型隔离。

这些测试可控制失败、超时和并发时机；通过结果用于验证代码契约和异常处理。

## 2. ST 系统测试

入口为 [真实服务验收入口](../../../tests/st/ov_context/run.py)，连接独立验证部署：

```bash
bash examples/deepagents_ov_context/scripts/container.sh run python -m \
  tests.st.ov_context.run \
  --ov-url http://127.0.0.1:1933 \
  --api-key-file /config/ov/service.key \
  --model-config /config/models.local.json \
  --output-dir /opt/openviking/result/st/ov_context
```

覆盖 I01–I09：多轮会话、真实进程重启、提交进度恢复、长工具压缩与回读、多身份并发、归档后刷新、跨会话记忆召回、OV 网络故障与取消恢复。

专用容器的 SIGKILL、连续三次容量压缩及持续压力通过 `--extended` 显式启用。测试控制器会校验专用容器标签与端口。

ST 同时覆盖 I01–I09 与 Writer、上下文和多租户检查。组件用例使用已知答案的合成订单与审计日志，连接真实 Middleware、SDK、模型和工具，与其他 ST 统一执行并生成 JUnit、pytest 日志及逐项证据。

| 检查项 | 主要断言 |
|---|---|
| `writer` | 长工具短视图及原文、超过 100 条分批、游标、并行工具关联与原对象保留 |
| `matrix` | 实际模型输入的系统／身份前缀、跨会话隔离、串行／并发耗时与 usage 归属 |
| `compression` | 真实归档、关键信息保留、正文缩减率与完整输入预算 |
| `restart` | SQLite 跨进程恢复、消息及工具关联、已完成工具执行次数 |
| `collision` | 应用拒绝跨 peer 复用 Session，拒绝发生在新增 OV 请求前 |
| `logs` | commit、召回、压缩与异常事件可关联到对应会话 |
| `index` | 记忆正文与可检索候选的一致性 |

压缩与索引一致性在同一用例中先后检查，共用该次归档产生的记忆。跨进程恢复使用内部 worker，共用 ST 指定的 OV 地址、凭据文件和模型配置。

按检查项筛选时使用同一入口的 `--select`，例如 `--select 'component_acceptance and (writer or compression)'`。并发会话数通过 `--parallelism` 配置，组件每阶段超时通过 `--case-timeout` 配置，默认分别为 4 和 300 秒。所有结果保存在本次 ST 输出目录。

## 3. Chat 接口验收

以下脚本通过实际 Chat HTTP/SSE 发起任务，与页面共用执行链路。

**冒烟验收：**

```bash
bash examples/deepagents_ov_context/scripts/container.sh smoke
```

检查 SSE、工具消息、同会话 checkpoint 复用、从 OV 恢复和工具配对。

**多用户与多 Agent：**

```bash
bash examples/deepagents_ov_context/scripts/container.sh run python -m \
  tests.st.ov_context.validate_multi_session \
  --url http://127.0.0.1:1934 --parallelism 2 \
  --output /opt/openviking/result/office/multi-session/results.json
```

新建 5 个会话并执行工具、checkpoint 续聊和 OV 恢复三个阶段。每次检查实际模型输入、消息 ID、工具调用与结果配对及随机会话标记；续聊问题省略旧标记，以验证历史延续。多 account 的隔离由第 2 节真实服务集成测试覆盖。

**跨会话记忆：**

```bash
bash examples/deepagents_ov_context/scripts/container.sh run python -m \
  tests.st.ov_context.validate_memory \
  --chat-url http://127.0.0.1:1934 --ov-url http://127.0.0.1:1933 \
  --api-key-file /config/ov/service.key --timeout 300 \
  --output /opt/openviking/result/office/memory/results.json
```

通过 Chat 写入偏好、提交提取，等待真实记忆可检索，再验证原会话、新会话、同 user 的另一 Agent 和另一 user。通过条件包含实际召回注入、回答使用偏好、归档边界及身份隔离。

## 4. 业务案例与记忆对比

- 日程、报工、邮件、PPT 更新和资料生成文档：按[数据集部署与执行](DEPLOYMENT.md#7-办公案例数据集部署)运行。
- 正式提示词统一位于 `benchmark/office/tasks/`，手工复制和执行脚本共用同一模板。H 为准备轮，含执行方法指导；Q 为测试轮，原生与 OV 组使用同一份任务。

| 案例 | H | Q |
|---|---|---|
| 邮件正文导出 | [H.md](../../../benchmark/office/tasks/mail/H.md) | [Q.md](../../../benchmark/office/tasks/mail/Q.md) |
| 六项报工 | [H.md](../../../benchmark/office/tasks/timesheet/H.md) | [Q.md](../../../benchmark/office/tasks/timesheet/Q.md) |
| 六场会议 | [H.md](../../../benchmark/office/tasks/schedule/H.md) | [Q.md](../../../benchmark/office/tasks/schedule/Q.md) |
| PPT 更新 | [H.md](../../../benchmark/office/tasks/ppt_update/H.md) | [Q.md](../../../benchmark/office/tasks/ppt_update/Q.md) |
| 资料生成文档 | [H.md](../../../benchmark/office/tasks/reqdoc_generation/H.md) | [Q.md](../../../benchmark/office/tasks/reqdoc_generation/Q.md) |

通过 `python -m benchmark.office.tasks --case mail --phase H --workspace mail-h --page-url http://127.0.0.1:19482` 输出填好入口的任务。邮件 H 使用三个准备目标，Q 使用六个验收目标，数据集版本为 `office-mail-h3-q6-v2`；H/Q 使用独立的主题、日期、正文及标识。准备后按工作区中的 case/phase 执行，旧单项工作区使用新的 namespace 重新准备。

### 4.1 配置与评分预算

普通记忆 Reader 候选数为 25，实际注入数量及召回 Token／字符预算仍由 Reader 配置控制。Session 保存创建时的配置快照，调整配置后用新 Session 验证。

五个案例共用 `benchmark/office/cases/scoring.json` 的 Q 轮评分规则：

| 案例 | 每项任务评分预算／秒 |
|---|---:|
| 日程会议 | 450 |
| 报工登记 | 300 |
| 邮件整理 | 550 |
| PPT 更新 | 120 |
| 资料生成文档 | 550 |

1. 每轮包含六个等权子任务。某项全部断言通过，且从 Chat Run 开始到该项保存、回查及核验完成的时间 `t_i ≤ T`，计一次预算内成功。三次测试成功率为成功项总数除以 18。
2. 前 k 项累计时延为 `max(t_1, …, t_k)`，三次均有合格时点时取平均。每项成功率使用自身的 `t_i`；较早任务的延迟修复只影响累计时延。
3. 完整 Run 的终态、最终交付、总耗时及 Agent＋OV LLM Token 单列。执行保护超时由 `execution.timeout_seconds` 控制；评分截止后继续执行至终态，失败样本保留。
4. 正式 runner 自动创建带 `office-auto--案例名--H/Q` 标记的 Session。标准 Chat 在工具结束时保存业务状态、产物哈希与中间版本；Run 结束后，runner 自动请求 `POST /api/sessions/{session_id}/runs/{run_id}/benchmark` 完成验收和计分。

### 4.2 自动验收与分步统计

| 案例 | 自动验收依据 |
|---|---|
| 日程 | 六场会议字段、预约资源、背景记录保持性及各项详情回读审计 |
| 报工 | 各卡片新增记录、日期和工时增量、背景记录保持性及对应 PNG 截图 |
| 邮件 | 六份指定文件的正文与冻结参考逐项比对 |
| PPT | 各中间版本的两处修改、页数及包内容保持性 |
| 资料生成文档 | 六章结构、计算脚本／JSON 解析，以及对照原资料的 27 项自动语义评审 |

- 每项完成时间取最终有效版本序列中首次通过全部验收的快照时点。后续版本破坏已完成项时，清除该项完成时点，修复后重新计时；其他项目独立保留。
- 快照位于 Chat 数据目录 `benchmark/<session_id>/<run_id>/`，包含 `manifest.json`、`snapshots.jsonl` 和按 SHA-256 去重的 `blobs/`。时间统一使用 Chat Run 的单调时钟，产物内容与其哈希一同校验。普通 Chat 会话沿用原执行链路。
- 文档评审在 Run 结束后使用部署的模型配置调用模型，保留原文引用、资料引用、模型标识、每项判定和 usage。相同章节及证据复用缓存；模型请求、格式或引用校验失败时记录“未核验”。评审 Token 单列为 `judge_usage`，与 Agent／OV 用量分别统计。
- 缺失快照、文件损坏或其他证据错误记录到评分结果的 `errors`。`unverified_subtasks` 为零时输出完整 `success_rate`；存在未核验项目时保留已确认成功项数及空成功率。
- 日程、报工与邮件的工作区默认从 Chat 数据目录相邻的 `cases/registry` 读取。独立部署可通过 `OV_BENCHMARK_REGISTRY` 指定同一数据集部署的 registry。Session 的 user 必须与工作区标识相同，case／phase 与已准备的工作区一致。

正式 runner 自动保存 `grade.json`。PPT 和文档另存 `automatic-grade.json`，其中含分步结果与评审证据。标准 Chat 的 `/api/execution-profiles` 返回 `automatic_benchmark: true` 后，runner 才发起任务。

三次测试完成后，从对应 Q 运行目录自动汇总：

```bash
python -m benchmark.office.suites.common.report \
  --runs result/ppt/pair-1/Q-baseline result/ppt/pair-2/Q-baseline result/ppt/pair-3/Q-baseline \
         result/ppt/pair-1/Q-memory result/ppt/pair-2/Q-memory result/ppt/pair-3/Q-memory \
  --output result/ppt/summary.json
```

汇总按案例和实验组检查三次独立 Run、评分预算及模型／输入配置，输出分步累计时延均值、完整耗时、预算内任务成功率、Token 分项与优化百分比。失败及超时 Run 保留在样本中；OV Token 未采集时保持为空，总 LLM Token 的优化率等待用量齐全后计算。已有历史人工时点记录仍可用 `common.timing` 复核，历史成绩保留其原始条件。

## 5. 执行记录

真实测试会创建新 Session 并消耗模型及 Embedding 用量。每次使用独立输出目录；失败、超时与召回降级样本保留，调整配置后重新执行并分别记录。

报告、实际模型输入、SSE、SQLite、下载文件和日志保存在挂载的数据目录，作为本地证据管理。功能正确、业务成功、耗时、估算 Token 与模型 usage 分别统计。容器构建通过、离线测试通过和真实服务验收通过分别记录。
