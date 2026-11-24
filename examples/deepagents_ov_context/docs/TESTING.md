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
  python -m pytest -o addopts='' -q \
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

### 4.1 配置与验收预算

普通记忆Reader候选数为25，实际注入数量及Token／字符预算由Reader配置控制。Session保存创建时的有效配置，调整配置后创建新Session。

验收预算和重复次数由 `benchmark/office/evaluation/policy.json` 固定，任务成功率按[AI复核提示词](../../../benchmark/office/evaluation/AI_REVIEW_PROMPT.md)结合实际证据评审。

| 案例 | Q轮预算／秒 | 质量统计 |
|---|---:|---|
| 日程会议 | 450 | 六项业务子任务通过率 |
| 报工登记 | 300 | 六项业务子任务通过率 |
| 邮件整理 | 550 | 六项业务子任务通过率，允许附带对应准确主题 |
| PPT更新 | 120 | 六项修改及完整回读的预算内通过率 |
| 资料生成文档 | 550 | 六章等权、27个检查点的综合得分 |

每组执行三轮，固定分母为18。AI逐项核验内容、完成时点与交付条件，给出通过、失败或无法核定，引用快照行号、call ID和产物内容。完成时点只有预算后的观测上界时，预算内结果列为待核定。两组均100%时按满分规则判定达成目标，提升仍记录0个百分点。

### 4.2 证据采集与AI复核

runner创建带 `office-auto--案例名--H/Q` 标记的Session，标准Chat在工具结束和Run结束时采集证据。`/api/execution-profiles` 返回 `benchmark_evidence: true` 表示支持此能力。

- Chat数据目录的 `benchmark/<session_id>/<run_id>/` 保存 `manifest.json`、`snapshots.jsonl` 和按SHA-256去重的 `blobs/`，供核对原始文件版本及业务状态。
- runner通过 `GET /api/sessions/{session_id}/runs/{run_id}/benchmark/evidence` 保存 `evidence.json`，包含原始快照、案例参考、Run统计和证据目录位置。缺失文件或采集错误作为待复核证据保留。
- SSE、模型输入、工具参数与返回、最终答复、下载文件及回执随运行目录保存。完整Run耗时和实际模型usage继续由执行链路统计。
- 运行结束后，向AI提供正式Run清单、冻结任务及输入、证据目录和复核提示词。AI产出逐项复核账本和分案例报告，核对证据后形成最终验收结果。

PPT普通记忆Q使用已完成的H目录和复核记录。先执行H并检查真实提交、提取及记忆指纹，再按提示词复核H产物；复核通过后记录H的Run ID、案例契约指纹及证据路径。执行方式见[PPT案例说明](../../../benchmark/office/cases/ppt_update/README.md)。

### 4.3 汇总AI复核材料

在部署的Chat容器中运行统一收集入口：

```bash
bash examples/deepagents_ov_context/scripts/container.sh run python -m \
  benchmark.office.evaluation.collect \
  --runs /opt/openviking/result/ppt/pair-{1,2,3}/Q-{baseline,memory} \
  --chat-data /data/chat --output /opt/openviking/result/ai-review/ppt
```

目录中包含AI提示词、正式Run清单、实际任务、模型及工具轨迹、冻结参考、文件版本和输入／产物。将整份目录交给AI，从根目录 `README.md` 开始。`issues.json`列出缺失或损坏证据，脚本保留失败Run。归档目录映射及输出结构见[材料汇总说明](../../../benchmark/office/evaluation/README.md#一条命令汇总材料)。

### 4.4 性能汇总

三次Q执行完成后汇总完整任务耗时与Token：

```bash
python -m benchmark.office.suites.common.report \
  --runs result/ppt/pair-1/Q-baseline result/ppt/pair-2/Q-baseline result/ppt/pair-3/Q-baseline \
         result/ppt/pair-1/Q-memory result/ppt/pair-2/Q-memory result/ppt/pair-3/Q-memory \
  --output result/ppt/performance.json
```

汇总检查三次独立Run、案例预算和模型／输入条件，保留失败及超时样本，输出完整耗时、Agent／OV LLM用量及优化百分比。OV用量尚未采集时保留为空。质量结果、分步完成时点及累计时延由AI复核账本提供，性能脚本只统计原始测量数据。

## 5. 执行记录

真实测试会创建新 Session 并消耗模型及 Embedding 用量。每次使用独立输出目录；失败、超时与召回降级样本保留，调整配置后重新执行并分别记录。

报告、实际模型输入、SSE、SQLite、下载文件和日志保存在挂载的数据目录，作为本地证据管理。功能正确、业务成功、耗时、估算 Token 与模型 usage 分别统计。容器构建通过、离线测试通过和真实服务验收通过分别记录。
