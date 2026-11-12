# OpenViking 记忆抽取优化最佳实践

这份最佳实践介绍如何从 OpenViking **v0.5.0 官方基线**应用记忆抽取优化补丁，启用确定性抽取、原文证据融合检索及来源生命周期管理，并通过 LoCoMo 对照测试检查问答正确率和 Token 消耗。

## 1. 特性简介

原有 ReAct 抽取通过 LLM 从会话中生成记忆候选。确定性抽取使用语言分析和规则生成 facts、entities、events 候选，再交给共享记忆更新器写入，减少候选生成阶段的模型调用。

- **确定性候选生成**：纯文本候选阶段不调用 LLM，保留数量、时间范围、人物归属和原文来源；可通过配置切回 ReAct。
- **原文证据融合**：新增 `find_evidence`，在 `coverage` 模式中融合词法、语义、记忆、事件及可选图和时间通道，核验来源并按预算返回证据。
- **来源生命周期**：删除会话时撤销对应支持记录；修正已归档消息时更新原文、清理旧支持并重新抽取，共享记忆保留其他来源。
- **向量失败补偿**：原文向量生成暂时失败时，持久化补偿状态并有限重试；删除或修正后的来源不会被旧补偿结果覆盖。

本特性优化的是候选生成及证据组织。摘要、多模态处理、部分记忆合并、embedding 和最终回答仍可能调用模型，不能将“候选阶段零 LLM”理解为整个写入流程零 Token。数量事实保存原始总量、增量或观测值，不自动把多次总量相加，也不保证所有自然语言表达均能正确识别。

## 2. 部署示例

可以先使用单机部署：OpenViking 通过本地文件系统保存会话、记忆和原文索引，使用已有 LLM 与 embedding 服务。

```text
写入消息 -> commit -> 原文归档及 evidence 索引
                 \-> 后台确定性抽取 -> 共享记忆更新器 -> 记忆卡及向量

问题 -> find_evidence coverage -> 来源核验、融合与证据选择 -> 应用调用 LLM 回答
```

### 2.1 从 v0.5.0 基线应用补丁

准备 Python 3.11+、Git、Rust/Cargo（至少 1.91.1）、CMake 和 C/C++ 构建工具。补丁位于本交付仓库的 [patches/OpenViking_V0.5.0/deterministic-memory-extraction.patch](../../../../patches/OpenViking_V0.5.0/deterministic-memory-extraction.patch)，对应 [PR #23](https://gitcode.com/boostkit/OpenViking/pull/23)，源提交为 `d76f7c580ddf92a3b1b2f88219a46192b8e07c50`。

补丁包含 PR 的最终实现、测试、可选 NLP 依赖、LoCoMo 导入调整及特性文档，另补齐官方基线 benchmark 与 SDK 的两处兼容性修复，共 53 个文件：导入通过 `options.created_at` 保留消息时间，QA 同时支持字典和对象形式的检索结果，避免已召回记忆被跳过。目标为官方 `v0.5.0` 标签，commit 为 `20f9c11362ad16193b0daa4a414e1095061af427`。补丁不包含 CacheFS、QueueFS 或 PathLock 实现修改。将补丁保存在待安装源码目录之外，再执行：

```bash
export BENCH_ROOT="$HOME/ov-extraction-bench"
export OV_REPO="$BENCH_ROOT/OpenViking"
export EXTRACTION_PATCH='<本交付仓库绝对路径>/patches/OpenViking_V0.5.0/deterministic-memory-extraction.patch'
mkdir -p "$BENCH_ROOT"
git clone --branch v0.5.0 --single-branch https://github.com/volcengine/OpenViking.git "$OV_REPO"
git -C "$OV_REPO" switch -c v0.5.0-deterministic-extraction
git -C "$OV_REPO" rev-parse HEAD

git -C "$OV_REPO" apply --check "$EXTRACTION_PATCH"
git -C "$OV_REPO" apply "$EXTRACTION_PATCH"
git -C "$OV_REPO" diff --check
git -C "$OV_REPO" diff --stat
```

确认 HEAD 与上述基线一致，且所有检查成功后再安装。该文件是合并后的差异补丁，使用 `git apply`；不需要逐个 cherry-pick PR 提交。保存基线 commit 和补丁 SHA-256；已经包含 PR #23 的代码不能再次应用同一补丁。

### 2.2 安装依赖和 NLP 模型

```bash
python3 -m venv "$BENCH_ROOT/venv"
source "$BENCH_ROOT/venv/bin/activate"
python -m pip install -U pip
OV_REQUIRE_RAGFS_BUILD=1 python -m pip install -v -e "${OV_REPO}[deterministic]" 'spacy==3.8.18'

python -m pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl \
  https://github.com/explosion/spacy-models/releases/download/zh_core_web_lg-3.8.0/zh_core_web_lg-3.8.0-py3-none-any.whl

python -c 'import openviking; print(openviking.__file__)'
python -c 'import spacy; print(spacy.__version__); print(spacy.load("en_core_web_lg").meta["version"]); print(spacy.load("zh_core_web_lg").meta["version"])'
openviking-server init
```

这里固定设计文档中的 spaCy 3.8.18、英文/中文模型 3.8.0。模型缺失时实现会告警并降级到规则，性能验收前应确认两种模型都能加载。源码安装需要构建官方基线的原生扩展；`OV_REQUIRE_RAGFS_BUILD=1` 要求构建成功，不要设置 `OV_SKIP_RAGFS_BUILD=1`。

### 2.3 准备模型服务

设计文档使用 `deepseek-v4.1-flash`、temperature=0，以及 bge-m3、1024 维 embedding。可以连接已有服务，两组测试应使用相同模型、版本、输入处理及重试配置。下面使用 OpenAI 兼容接口，地址和凭据按实际环境填写，不复用文档中的占位值。

## 3. 如何配置

### 3.1 服务端配置

在初始化生成的 `ov.conf` 中合并以下配置，保留所需的其他设置。`<...>` 必须替换为实际值；JSON 不自动展开 shell 环境变量。`storage.workspace` 使用独立的绝对路径。示例采用仅本机访问的 trusted 测试服务。

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 1933,
    "auth_mode": "trusted",
    "root_api_key": "<本地测试服务密钥>"
  },
  "storage": {
    "workspace": "<本地测试数据目录的绝对路径>",
    "vectordb": {"backend": "local"},
    "agfs": {"backend": "local"}
  },
  "vlm": {
    "provider": "openai",
    "api_base": "<LLM服务的OpenAI兼容地址，包含/v1>",
    "api_key": "<LLM服务凭据>",
    "model": "deepseek-v4.1-flash",
    "temperature": 0
  },
  "embedding": {
    "dense": {
      "provider": "openai",
      "api_base": "<embedding服务的OpenAI兼容地址，包含/v1>",
      "api_key": "<embedding服务凭据，无需认证时按服务约定填写>",
      "model": "bge-m3",
      "dimension": 1024
    }
  },
  "memory": {
    "extraction_enabled": true,
    "deterministic_extract_enabled": true,
    "spacy_model_en": "en_core_web_lg",
    "spacy_model_zh": "zh_core_web_lg",
    "evidence_graph_write_enabled": true
  },
  "agent_evolution": {"enabled": false}
}
```

| 配置项 | 作用 |
| --- | --- |
| `memory.extraction_enabled` | 是否执行会话长期记忆抽取；关闭后不能用于比较两种抽取方式 |
| `memory.deterministic_extract_enabled=true` | 启用确定性候选抽取；显式设为 `false` 使用 ReAct |
| `memory.spacy_model_en` / `spacy_model_zh` | 英文/中文 NLP 模型名称；实现默认使用 `*_sm`，此示例改为 `*_lg` |
| `memory.evidence_graph_write_enabled` | 后续归档时建立实体和时间窗口图；默认关闭，不自动补建历史图 |
| `agent_evolution.enabled` | 是否启用 Agent Evolution；此示例关闭，避免混入其他处理路径 |

未显式设置 `deterministic_extract_enabled` 时，读取 `OV_DETERMINISTIC_EXTRACT`，仅字符串 `0` 关闭，默认启用。对照测试应始终显式填写开关。确定性抽取出错不会自动切到 ReAct。模型配置、NLP 模型或开关改变后重启服务；切换抽取模式不会自动重建已有记忆。

将配置保存为 `$BENCH_ROOT/ov.conf`，限制文件访问权限，然后启动：

```bash
chmod 600 "$BENCH_ROOT/ov.conf"
export OPENVIKING_CONFIG_FILE="$BENCH_ROOT/ov.conf"
export OV_URL='http://127.0.0.1:1933'
openviking-server doctor
openviking-server --config "$OPENVIKING_CONFIG_FILE" --host 127.0.0.1 --port 1933
```

在另一个终端激活同一 venv、设置相同路径，执行 `curl -fsS "$OV_URL/ready"`。服务就绪后再写入。已有生产鉴权环境应使用对应的用户凭据和身份，不照搬本机 trusted 配置。

### 3.2 写入和提交示例

以下命令需要 `jq`。测试用户与输出目录保持独立；密钥与上文服务配置一致，不写入报告或版本库。

```bash
export OV_API_KEY='<与ov.conf相同的本地测试服务密钥>'
export TEST_USER="extraction_$(date +%s)"
export CASE_OUT="$BENCH_ROOT/functional/$TEST_USER"
mkdir -p "$CASE_OUT"
ov_api() {
  curl -fsS -H 'Content-Type: application/json' \
    -H "X-API-Key: $OV_API_KEY" \
    -H 'X-OpenViking-Account: default' \
    -H "X-OpenViking-User: $TEST_USER" "$@"
}

SID=$(ov_api -X POST "$OV_URL/api/v1/sessions" \
  --data '{"memory_policy":{"memory_types":["facts","entities","events"],"working_memory":{"enabled":false}},"auto_commit_policy":null}' \
  | jq -r '.result.session_id')
ov_api -X POST "$OV_URL/api/v1/sessions/$SID/messages" \
  --data '{"role":"user","content":"I went to a community support group yesterday.","created_at":"2023-05-08T13:56:02"}'
ov_api -X POST "$OV_URL/api/v1/sessions/$SID/commit" \
  --data '{"telemetry":true}' > "$CASE_OUT/commit.json"
TASK_ID=$(jq -r '.result.task_id' "$CASE_OUT/commit.json")
ov_api "$OV_URL/api/v1/tasks/$TASK_ID" > "$CASE_OUT/task.json"
jq '.result.status, .result.result.token_usage' "$CASE_OUT/task.json"
```

每秒查询任务，直至 `status=completed`；`failed`、`cancelled` 或等待超过 20 分钟时保存结果并处理失败。`commit` 接收成功不等于后台抽取及向量索引已经完成。`memory_policy` 控制允许的记忆类型、self/peer 范围及工作记忆，原有策略仍然生效。

### 3.3 开启 coverage 证据融合

在同一用户下查询已归档数据：

```bash
jq -n --arg uri "viking://user/$TEST_USER" \
  '{query:"When did I go to the community support group?",target_uri:$uri,
    strategy:"coverage",top_k:10,candidate_limit:50,max_supplementary_queries:2,
    use_semantic:true,use_memories:true,use_events:true,use_graph:true,use_time:true}' \
  > "$CASE_OUT/query.json"
ov_api -X POST "$OV_URL/api/v1/search/find_evidence" \
  --data-binary @"$CASE_OUT/query.json" > "$CASE_OUT/evidence.json"
jq '.result | {n_evidence,evidence,facts,diagnostics}' "$CASE_OUT/evidence.json"
```

| 参数 | 作用与边界 |
| --- | --- |
| `strategy` | 默认 `legacy`；使用辅助记忆、事件、图或时间通道时必须设为 `coverage` |
| `top_k` / `candidate_limit` | 最终原文条数预算 / 候选上限，均为 1–50；数量相同不代表两种接口返回相同内容量 |
| `max_supplementary_queries` | 规则补查询次数，0–3，默认 2；补查询不调用 LLM |
| `use_semantic` | 使用当前账户 embedding，默认关闭；需要原文向量可用 |
| `use_memories` / `use_events` | 读取记忆与事件支持；均默认关闭，事实必须回查授权原文 |
| `use_graph` / `use_time` | 使用归档图及时间信号；均默认关闭，写入前还需开启图构建 |
| `event_top_k` | 事件候选数，1–20，默认 2 |

检查证据包含输入原文、来源与用户一致，事实中的事件日期为 2023-05-07，返回原文数不超过 `top_k`。开关开启不代表每个问题都有相应通道贡献，应同时核对 `diagnostics`。接口返回证据及事实，调用方负责构建提示并请求回答模型，不会由接口直接生成答案。

原文索引位于 `viking://user/{user_id}/.index/evidence.json`。`diagnostics.embeddings` 提供总数、就绪数、缺失数、覆盖率及 pending/failed 数。暂时向量错误后台每 30 秒扫描并退避补偿，包括首次调用最多尝试 4 次；永久输入、认证或额度错误保留失败状态。历史缺失向量没有补偿状态时不会自动回填。

删除会话沿用 `DELETE /api/v1/sessions/{session_id}`；修正单个纯文本消息使用 `PUT /api/v1/sessions/{session_id}/messages/{message_id}`，请求体为 `{"content":"修正后的内容"}`。清理失败时按响应重试同一请求，不把部分完成当作成功；会话正在提交时，修正可能返回冲突。

### 3.4 已知配置与 benchmark 兼容性问题

以下问题在所列条件下可稳定复现，部署或测试前应检查。

| 问题与触发条件 | 表现 | 处理方式 |
| --- | --- | --- |
| 客户端鉴权字段混用：服务端配置了 `server.root_api_key`，但 `ovcli.conf` 仅填写 `root_api_key`，且未通过客户端参数或 `OPENVIKING_API_KEY` 提供密钥 | Python SDK 不将客户端配置中的 `root_api_key` 作为请求密钥，业务请求返回鉴权失败；`/ready` 成功不能证明业务请求鉴权正常 | 按第 4.2 节生成配置，将密钥写入客户端的 `api_key`；本机 trusted 示例中，其值与服务端 `server.root_api_key` 一致。HTTP 请求通过 `X-API-Key` 传入 |
| 导入时间参数与 SDK 不兼容：未适配的 `import_to_ov.py` 直接调用 `add_message(..., created_at=...)` | SDK 不接受该关键字参数，导入报 `TypeError`，无法保留数据集中的消息时间 | 改为 `options={"created_at": msg_created_at}`，无时间时传 `None`；不要通过丢弃时间参数绕过错误，否则相对日期解析和时序检索会受影响 |
| QA 检索结果解析与 SDK 不兼容：`find` 返回字典，但未适配的 `run_eval.py` 只按对象属性读取结果及条目 | 服务端有召回，QA 却跳过记忆，形成空上下文回答，正确率不能用于评价检索效果 | 同时适配外层 `memories`、`resources`、`skills` 字典字段和内层条目的字典字段；正式运行前核对召回 URI 与实际回答上下文 |

第 2.1 节的交付补丁已包含两处 benchmark SDK 兼容性修复，应用完整补丁后无需再次手工修改脚本。鉴权字段按本文示例配置即可，无需修改服务端鉴权逻辑。若此前已产生导入失败或未解析召回结果的 QA 产物，应在修复并核验后重新执行受影响步骤，不将其混入正式统计。

## 4. 如何做性能测试

### 4.1 确定对照口径和数据范围

设计文档的主流程只切换抽取方式，两组都使用 `find` 检索。这样观察到的是候选抽取变化及其生成记忆对问答的影响。

| 项目 | A：基线 | B：优化组 |
| --- | --- | --- |
| 代码及补丁 | 相同的 v0.5.0 + PR #23 补丁 | 相同 |
| `deterministic_extract_enabled` | `false` | `true` |
| `extraction_enabled` | `true` | `true` |
| `evidence_graph_write_enabled` | `false` | `false` |
| 检索 | `find`，limit=50 | 相同 |
| rerank 保留数 / 上下文上限 | 10 / 30,000 字符 | 相同 |
| 导入 / QA 并发 | sample 串行 / 4 | 相同 |
| 数据空间 | 独立空 workspace、向量库和队列 | 另一套独立空空间 |
| 模型及策略 | 相同 LLM、embedding、rerank、提示、记忆策略、重试配置 | 相同 |

不使用 rerank 时，两组都将命令中的 `--single-search-rerank-limit` 改为 `0`。不要让另一组单独关闭摘要、改变记忆类型或跳过失败题。全量正式验收以 CPU 920B 为目标，记录实际 CPU、核数、内存及运行时版本；其他硬件的结果只代表对应环境。

准备 LoCoMo `locomo10.json`，记录数据 SHA-256、样本数、非空会话数、各 category 题数。现有脚本跳过 category 5，以下命令覆盖全部样本的 **category 1–4**，不能称为全题验收；全题验收需另补不可回答题的执行、判分和 Token。首次可用独立空间预跑 sample0，正式运行不复用预跑记忆或用量。

### 4.2 准备两组配置

```bash
export LOCOMO_DATA='<locomo10.json的绝对路径>'
export ACCEPT_ROOT="$BENCH_ROOT/acceptance/$(date +%Y%m%d-%H%M%S)"
export OV_URL='http://127.0.0.1:1933'
mkdir -p "$ACCEPT_ROOT"

python - <<'PY'
import copy
import json
import os
from pathlib import Path

root = Path(os.environ["ACCEPT_ROOT"]).resolve()
base = json.loads((Path(os.environ["BENCH_ROOT"]) / "ov.conf").read_text())
for group, enabled in (("A", False), ("B", True)):
    out = root / group
    (out / "result").mkdir(parents=True)
    cfg = copy.deepcopy(base)
    cfg["memory"].update(extraction_enabled=True,
                         deterministic_extract_enabled=enabled,
                         evidence_graph_write_enabled=False)
    cfg["storage"]["workspace"] = str(out / "workspace")
    cli = {"url": os.environ["OV_URL"], "account": "default", "user": "default",
           "api_key": cfg["server"]["root_api_key"], "timeout": 1200}
    for name, value in (("ov.conf", cfg), ("ovcli.conf", cli)):
        path = out / name
        with path.open("x") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        path.chmod(0o600)
PY
```

以上配置生成器适用于前述本机 trusted 服务。服务端凭据字段是 `server.root_api_key`，客户端 `ovcli.conf` 对应字段是 `api_key`。使用外置存储时，还需为两组分配独立 bucket/prefix、向量集合与任务存储，仅改变 workspace 不足以隔离。保存脱敏配置、数据/补丁哈希和实际依赖版本，原始凭据文件留在受控环境。

先停止部署示例服务，启动 A；完成 A 的导入、QA 与判分后停止 A，将 `GROUP=B`，按相同步骤执行 B：

```bash
export GROUP=A
export OPENVIKING_CONFIG_FILE="$ACCEPT_ROOT/$GROUP/ov.conf"
export OPENVIKING_CLI_CONFIG_FILE="$ACCEPT_ROOT/$GROUP/ovcli.conf"
openviking-server --config "$OPENVIKING_CONFIG_FILE" --host 127.0.0.1 --port 1933
```

在另一个终端激活相同 venv，并设置相同的 `GROUP`、服务端/客户端配置变量。回答脚本在客户端读取 `ov.conf` 创建模型，两组客户端也必须指向本组配置。

### 4.3 导入全部会话

```bash
cd "$ACCEPT_ROOT/$GROUP"
python "$OV_REPO/benchmark/locomo/openviking/import_to_ov.py" \
  --input "$LOCOMO_DATA" --openviking-url "$OV_URL" --parallel-samples 1 \
  --success-csv "$ACCEPT_ROOT/$GROUP/import.csv" \
  --error-log "$ACCEPT_ROOT/$GROUP/import_errors.log"
```

脚本在当前目录的 `result/.ingest_record.json` 保存进度，必须在仓库外的各组目录运行。一个原始会话的多个分块复用同一 session，保留消息顺序和时间；不要将分块变成彼此独立的会话。中断后保留目录和存储继续导入，不对已成功会话重复使用 `--force-ingest`。

开始 QA 前检查 `import.csv` 按 `sample_id + session` 覆盖全部非空会话且无重复，错误已处理，后台任务和索引队列均完成。通过读取各 `sample_{i}` 用户的 evidence 索引或检查 `find_evidence` 的向量诊断，确认语义测试所需向量就绪、覆盖率 100%、pending/failed 均为 0。无法达到时报告具体缺失及原因，不能默认为已完成。

`vlm_tokens` 是写入阶段 LLM 用量；`embedding_tokens` 单列，`total_tokens` 包含 embedding，不能直接代替 LLM 主指标。补偿及失败重试的实际模型用量也需归入本组，不能因主任务已返回而漏计。

### 4.4 检索、回答与判分

```bash
python "$OV_REPO/benchmark/locomo/openviking/run_eval.py" "$LOCOMO_DATA" \
  --openviking-url "$OV_URL" --output "$ACCEPT_ROOT/$GROUP/qa.csv" --threads 4 \
  --single-search-context-limit 50 --single-search-rerank-limit 10 \
  --single-search-max-context-chars 30000

export JUDGE_BASE_URL='<判分模型的OpenAI兼容地址>'
export JUDGE_MODEL='deepseek-v4.1-flash'
export OPENAI_API_KEY='<判分模型凭据>'
python "$OV_REPO/benchmark/locomo/openviking/judge.py" \
  --input "$ACCEPT_ROOT/$GROUP/qa.csv" --provider openai \
  --base-url "$JUDGE_BASE_URL" --model "$JUDGE_MODEL" \
  --strict-prompt --parallel 4
python "$OV_REPO/benchmark/locomo/openviking/stat_judge_result.py" \
  --input "$ACCEPT_ROOT/$GROUP/qa.csv" > "$ACCEPT_ROOT/$GROUP/stat.txt"
```

核对 `question_id` 完整且不重复，`response`、`token_usage`、`memory_chars` 可追溯，检索上下文应非空。默认 CSV 不写来源轨迹；可用外置 runner 记录 URI 和请求 ID，或按需增加 `--debug-print-model-input`（同时保存完整提示，注意数据访问范围）查看 `retrieved_uris_by_iteration`。不能把未解析召回结果或空上下文的回答当作有效检索测试。`memory_prompt_tokens` 只表示记忆文本量，不是回答模型完整用量。

当前答题脚本再次执行会重新处理输入题目。补跑时按 `--sample`、`--question-index` 将缺失题写入独立 CSV，再按 `question_id` 合并。判分应全部为 `CORRECT` / `WRONG`；判分服务错误补判，业务空回答、超时及重试耗尽计不成功，仍保留在固定题目分母中。Judge 读取 gold evidence 只用于判分，不提供给回答模型。

### 4.5 汇总正确率和 Token

设每组固定题目数为 N、正确题数为 C、业务 LLM 总 Token 为 T：

| 指标 | 计算方式 |
| --- | --- |
| 问答正确率 | P = C / N |
| 业务 LLM Token | T = 写入 Token + 回答 Token + 尚未计入的业务重试/补偿等 LLM 用量 |
| 每成功任务 Token | E = T / C |
| Token 降低率 | 1 − E_B / E_A |
| 成功率相对劣化 | (P_A − P_B) / P_A；同时报告 P_B − P_A 的百分点变化 |

下面按现成脚本的 category 1–4 范围核对题目和 CSV 用量字段。v0.5.0 的任务跟踪器可能返回累计快照，多次 commit 的 `vlm_tokens` 逐行相加会重复计数，因此这些字段只用于诊断；业务 Token 主指标应使用模型网关或代理记录的逐调用 usage，按调用 ID 去重后按本节公式计算。不能直接用 CSV 计数器之和计算 Token 降低率。

```bash
python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["ACCEPT_ROOT"])
data = json.loads(Path(os.environ["LOCOMO_DATA"]).read_text())
expected = {f"sample_{i}_qa{j}" for i, sample in enumerate(data)
            for j, qa in enumerate(sample["qa"]) if qa.get("category") in (1, 2, 3, 4)}
summary = {}
for group in ("A", "B"):
    with (root / group / "import.csv").open() as f:
        imports = list(csv.DictReader(f))
    with (root / group / "qa.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(expected) and {r["question_id"] for r in rows} == expected
    assert all(r["result"] in ("CORRECT", "WRONG") for r in rows)
    assert len({(r["sample_id"], r["session"]) for r in imports}) == len(imports)
    written = sum(int(r["vlm_tokens"]) for r in imports)
    usage = [json.loads(r["token_usage"]) for r in rows]
    answered = sum(int(u["total_tokens"]) for u in usage)
    correct = sum(r["result"] == "CORRECT" for r in rows)
    assert answered > 0
    summary[group] = dict(N=len(rows), C=correct, P=correct / len(rows),
                          csv_write_counter_sum=written, csv_qa_counter_sum=answered,
                          provider_usage_verified=False)
(root / "csv_checks_cat1_4.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
```

设计目标为每成功任务 Token 降低至少 10%、成功率相对劣化小于 10%，属于验收标准，不能预先认定已达到。正确题数或基线成功率为零时，不计算相应比值。报告总量和各 category 结果，不能只挑正确题统计 Token。

共享导入成本每组只计一次，答错及失败尝试中实际发生的用量仍计入 T。Embedding、非生成式 rerank、judge 分别列示；缺少 usage 的调用标记未知，已观测总量只能作为下界。模型返回的 total 已包含其计入的缓存输入或 reasoning 时，不再次相加。CSV 外的调用用 provider 记录补齐，按调用 ID 去重，不把累计快照当单次用量。

### 4.6 验证检索融合与保留测试产物

现有 `run_eval.py` 没有 `find_evidence` 或 coverage 切换参数。上面的 A/B 流程检验抽取优化，不代表已评测检索融合。若评估整体特性，另建独立对照：A 使用 ReAct + `find`，B 使用确定性抽取 + 第 3.3 节的 `find_evidence`，B 在导入前开启图构建；由外置 QA runner 将返回的 `evidence` / `facts` 组装到回答提示。两组固定题目、回答模型、提示结构、字符预算及判分规则，显式记录 rerank 是否一致。

整体对照可固定 `top_k=50`、`candidate_limit=50`、补查询最多 2 次及全部 `use_*` 开关开启。`find` 的 limit 以记忆卡计，`find_evidence` 的 top_k 以原文消息计，不能只因数值相同就认为上下文等价。报告需说明这类结果同时包含抽取与检索变化，不能单独归因于抽取。

按设计文档建议至少执行 3 轮成对测试，每轮新建目录与空存储，交替 A/B 顺序。合并指标按总 Token / 总正确题数计算，不选择最佳一轮。功能用例在独立空间执行，不混入 LoCoMo 用量。

产物保存在仓库外，至少包括：脱敏配置、代码与补丁/数据哈希、固定题目清单、导入成功与失败记录、逐题回答/判分、来源与通道诊断、真实 provider usage、汇总结果。外置 runner 还应保存请求 ID，将检索、回答和判分对应到同一道题；采集耗时时分别记录客户端耗时、模型耗时和服务内部阶段，不能仅凭 Token 降低宣称时延或硬件收益。
