# 确定性记忆抽取与证据融合检索

本特性提供可配置的确定性记忆抽取、原文证据融合及来源生命周期管理。纯文本候选生成由规则和 NLP 完成，减少该阶段的 LLM 调用；记忆写入仍复用共享更新器、既有模板和账户隔离。

## 快速开始

先按[最佳实践](best_practices.md)部署，安装可选依赖及语言模型，完成基础配置：

```bash
pip install '.[deterministic]'
python -m spacy download en_core_web_lg
python -m spacy download zh_core_web_lg
openviking-server init
```

在 `ov.conf` 的 `memory` 中合并以下设置，保留已有存储、模型和鉴权配置：

```json
{
  "memory": {
    "extraction_enabled": true,
    "deterministic_extract_enabled": true,
    "spacy_model_en": "en_core_web_lg",
    "spacy_model_zh": "zh_core_web_lg",
    "evidence_graph_write_enabled": true
  }
}
```

```bash
openviking-server doctor
openviking-server --config ~/.openviking/ov.conf
```

正常添加消息并提交会话即可触发归档和后台抽取，无需新的抽取接口。提交返回 `task_id` 后，通过 `GET /api/v1/tasks/{task_id}` 等待完成，再检查记忆和索引。

## 抽取对象

| 类型 | 内容 |
| --- | --- |
| facts | 个人陈述、数量变化与精选助手建议；保留总量/增量/观测类型、范围及原始时间表达 |
| entities | 已确认实体、相关事实及有方向的关系；并发合并保留其他来源记录 |
| events | 有明确主体且已发生的事件、来源范围和事件时间；计划不直接视为完成事件 |

记忆保留 `fact_records`、`source_turn_ids` 和 `source_quotes`，可以回到原始消息核验。具名人物用于确认陈述主体，通信角色 `user` / `assistant` 不代替人物身份。会话的 `memory_policy`、允许类型以及 self/peer 隔离继续生效。

## 证据融合检索

新接口 `POST /api/v1/search/find_evidence` 返回已归档原文及来源核验后的事实。下例使用 coverage，并开启所有辅助通道；认证信息和用户需与写入时一致。

```bash
export OV_API_KEY='<当前用户的服务凭据>'
export TEST_USER='<写入时的用户ID>'
curl -fsS -X POST http://127.0.0.1:1933/api/v1/search/find_evidence \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $OV_API_KEY" -H "X-OpenViking-User: $TEST_USER" \
  -d '{
    "query": "我之前完成了多少篇文章？",
    "strategy": "coverage",
    "top_k": 15,
    "candidate_limit": 50,
    "max_supplementary_queries": 2,
    "use_semantic": true,
    "use_memories": true,
    "use_events": true,
    "use_graph": true,
    "use_time": true
  }'
```

`target_uri` 默认为当前用户根目录；显式设置时也必须为有权限的 `viking://user/{user_id}` 根 URI。调用方使用 `evidence` 和 `facts` 构建回答提示，接口本身不生成答案。开关开启不代表所有问题都会命中对应通道，实际贡献、截断和降级情况见 `diagnostics`。

## 配置项

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `memory.deterministic_extract_enabled` | 未设置 | 显式 `true` 启用，`false` 切回 ReAct；未设置时读取 `OV_DETERMINISTIC_EXTRACT`，仅 `0` 关闭，默认开启 |
| `memory.spacy_model_en` / `spacy_model_zh` | `en_core_web_sm` / `zh_core_web_sm` | 可改用示例中的 lg 模型；模型缺失时告警并规则降级 |
| `memory.evidence_graph_write_enabled` | `false` | 在后续归档时建立实体和时间窗口图 |
| `strategy` | `legacy` | `coverage` 增加有界补查询、来源核验及融合证据选择 |
| `top_k` / `candidate_limit` | 15 / 50 | 最终原文预算 / 候选上限，取值均为 1–50 |
| `max_supplementary_queries` | 2 | 规则补查询次数，取值 0–3，不调用 LLM |
| `use_semantic` | `false` | 使用当前账户的 embedding 检索原文 |
| `use_memories` / `use_events` / `use_graph` / `use_time` | 均为 `false` | 辅助通道，要求 `strategy=coverage` |
| `event_top_k` | 2 | 事件候选数，取值 1–20 |

## 整体架构

```text
会话归档 -> 用户原文索引（含来源、向量和可选图）
        \-> 确定性抽取 / ReAct -> 共享更新器 -> 记忆及向量索引

find_evidence coverage
  -> 词法、语义和规则补查询
  -> 记忆、事件、图及时间信号
  -> 授权原文核验、去重、相邻问答组选择
  -> evidence / facts / diagnostics -> 应用侧回答模型
```

原文索引保存在当前账户的 `viking://user/{user_id}/.index/evidence.json`，经 VikingFS 按请求身份读写。词法与语义在候选层融合，重复来源不会因多次命中而重复加票；有界的相邻问答组保留同话题上下文，不通过时间接近拼接不同会话。

## 来源生命周期与可靠性

- **删除会话**：沿用 `DELETE /api/v1/sessions/{session_id}`，撤销原文索引、图成员及受影响记忆支持，保留共享卡片的其他来源；清理失败可重试。
- **修正消息**：使用 `PUT /api/v1/sessions/{session_id}/messages/{message_id}`，请求体为 `{"content":"修正后的内容"}`。已归档消息更新原文并重新抽取；未归档消息在后续提交时抽取。仅支持单个文本 part，任务运行中可能冲突。
- **向量补偿**：原文向量暂时失败时持久化 pending 状态，后台退避补偿，包括首次调用最多 4 次；永久错误保留 failed 状态。写回前复核来源，避免覆盖已删除或修正的数据。
- **诊断**：`diagnostics.embeddings` 显示向量覆盖率及缺失、pending、failed 数；权限、存储和解析错误不会伪装为“未命中”。

## 使用边界

纯文本候选阶段零 LLM 不代表会话全流程零模型调用。摘要、多模态处理、部分合并、embedding 和回答仍使用现有模型链路；确定性抽取失败不会自动回退到 ReAct。

原文索引使用用户级 JSON 快照，尚未改为面向超大历史的增量数据库。历史会话索引和图不会自动补建，没有补偿状态的历史缺失向量不自动回填。规则和 NLP 仍可能误判，例如把歌曲名 Yesterday 识别为时间，需按实际业务验证质量。

本特性不依赖 CacheFS，也不修改 QueueFS 或 PathLock 实现。质量和 Token 收益按[最佳实践的 LoCoMo 对照流程](best_practices.md#4-如何做性能测试)验证；现有 `run_eval.py` 使用 `find`，没有 coverage 开关，整体检索融合评测需由外置 runner 接入新接口。
