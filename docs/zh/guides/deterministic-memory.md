# 确定性记忆抽取与原文证据检索

本特性基于 `extractor-refactor` 分支的实现，适配当前项目的账户隔离、共享记忆更新器、异步 embedding 和 VikingFS。包括 facts / entities / events 抽取、`find_evidence` 检索融合，以及来源删除和消息修正。

## 安装与配置

在仓库根目录安装可选 NLP 依赖及所需语言模型：

```bash
pip install '.[deterministic]'
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```

将以下字段合并到现有 OpenViking 配置的 `memory` 中；原有存储、embedding、VLM 和鉴权配置仍需保留：

```json
{
  "memory": {
    "extraction_enabled": true,
    "deterministic_extract_enabled": true,
    "spacy_model_zh": "zh_core_web_sm",
    "spacy_model_en": "en_core_web_sm",
    "evidence_graph_write_enabled": false
  }
}
```

- `deterministic_extract_enabled=true` 使用确定性候选抽取，`false` 使用原有 ReAct 抽取。未配置时，读取 `OV_DETERMINISTIC_EXTRACT`，仅值为 `0` 时关闭，默认启用。显式配置优先于环境变量。
- spaCy 模型按名称在进程内复用。模型缺失时记录告警并使用规则降级，实体识别和语法分析能力会减少；安装模型后需要重启进程。
- `evidence_graph_write_enabled=true` 在后续归档时建立实体和时间窗口图，默认关闭。查询时还需显式传入 `use_graph` / `use_time`。
- 会话记忆策略、允许的记忆类型和 self / peer 隔离仍然生效。

纯文本候选生成不调用 LLM，确定性候选直接通过共享更新器写入。会话摘要、图像描述、其他抽取路径、向量生成及最终回答仍可能调用模型；该开关不代表整个会话处理流程不再需要模型配置。

## 抽取与来源

正常写入消息并提交会话即可触发归档和后台抽取，无需调用新的抽取接口。提交接口返回后台任务后，应继续查询任务状态；归档成功不等于抽取和索引已经完成。

三类记忆分别保存：

- **facts**：个人陈述、数量更新及精选助手建议；无需先识别出命名实体。数量保留 total / increment / observation、范围及原始时间表达，不自动相加或裁决最新总量。
- **entities**：已确认实体及其来源事实；并发写入时在卡片锁内重新读取并合并，保留其他来源记录。
- **events**：有明确主体且已发生的事件及原文范围；计划不会直接作为已完成事件保存。

记忆卡保留 `fact_records`、`source_turn_ids` 和 `source_quotes`，可回到原始消息核验。`speaker` 保留 user / assistant 角色，具名说话人用于确定事实主体，避免将助手建议归为用户经历。长消息处理片段会重新关联原消息 ID，多个文本 part 都会参与处理。

记忆仍使用当前模板目录 `openviking/prompts/templates/memory/` 和既有 self / peer 路径。原文检索索引保存在当前账户下的：

```text
viking://user/{user_id}/.index/evidence.json
```

索引经 VikingFS 按请求身份读写，账户之间沿用既有物理路径隔离；不会通过仅以 URI 为键的全局索引缓存共享数据。

## 检索融合

调用新增接口检索已归档原文。以下示例假设服务运行在本机 1933 端口；启用鉴权时，请沿用部署环境的身份和认证请求头。

```bash
curl -X POST http://127.0.0.1:1933/api/v1/search/find_evidence \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "我之前完成了多少篇文章？",
    "strategy": "coverage",
    "top_k": 15,
    "candidate_limit": 50,
    "max_supplementary_queries": 2,
    "use_semantic": true,
    "use_memories": true,
    "use_events": true,
    "use_graph": false,
    "use_time": false
  }'
```

`target_uri` 默认为当前用户根目录；如显式指定，也必须是有读取权限的 `viking://user/{user_id}` 根 URI。

- 默认 `strategy="legacy"` 提供词法、实体、窗口及可选语义锚点检索。
- `strategy="coverage"` 增加有界补查询、来源核验、融合去重和相邻问答组选择。
- `use_semantic` 默认关闭，启用后使用当前账户的异步 embedding 接口；向量生成失败时降级到非语义通道。
- `use_memories` / `use_events` / `use_graph` / `use_time` 默认关闭，要求 `strategy="coverage"`，组合不合法时请求被拒绝。
- `top_k` 和 `candidate_limit` 范围为 1–50；`max_supplementary_queries` 为 0–3；`event_top_k` 为 1–20，默认 2。补查询按规则生成，不调用 LLM。

返回的 `result` 包含 `evidence` 原文、`facts` 精简事实、`index_uri` 和 `diagnostics`。调用方负责根据证据生成答案；该接口不会生成自然语言回答。记忆事实需通过当前授权原文重新抽取核验，无法核验的旧摘要不会作为可信事实返回。

coverage 检索按具名人物和陈述主体选择记忆，通信角色 `user` / `assistant` 不代替人物身份。第一人称个人记忆仍沿用用户来源边界；查询助手建议仍要求精选事实元数据。返回事实中的 `source_speaker` / `source_role` 标明原始说话人与角色，并且每条事实都由本次入选原文支持。

词法与语义已在候选层融合，同一来源在补查询中重复命中时保留最强查询支持；记忆卡可补充遗漏来源，不会为已召回原文重复加票。分组选择不因同一会话已选过证据而降低后续相关原文的优先级。问答组最多保留四条消息，包含有界的同话题反问、指代追问及其回复，预算不足时整组跳过。不会通过时间接近推断不同会话之间的邻接关系。

分块导入对话时，应复用同一个 session 多次归档，保留消息全局顺序和递增时间；`benchmark/locomo/openviking/import_to_ov.py` 已采用此方式。不同原始会话仍使用不同 session。此前已导入为多个独立 session 的数据不会自动拼接，需要按原会话关系重新导入才能恢复跨块上下文。

索引不存在时返回空证据；权限、存储和解析错误会传播。来源修正尚未完成时返回冲突，避免读取不一致快照。辅助记忆读取有数量和时间预算，降级及截断信息记录在 `diagnostics` 中。

### 原文向量补偿

原文向量调用失败时仍保存原文，并在索引消息的 `embedding_state` 中保存状态、尝试次数和错误类别。超时、连接故障等临时错误进入 `pending`，后台每30秒扫描并按退避时间补偿；包含首次调用在内最多尝试4次。永久输入错误、认证或额度错误保留 `failed` 状态，不自动重试。服务重启后继续读取持久状态，不重复追加原文。

补偿复用当前账户的 embedding 配置，模型调用在索引锁外执行，写回前重新核对来源及补偿版本；删除或修正过的消息不会被旧补偿结果覆盖。`diagnostics.embeddings` 提供总数、就绪数、缺失数、覆盖率和待补全/失败数。没有补偿状态的历史缺失向量不会被自动回填；未配置 embedding 时仍沿用非语义检索。

## 来源生命周期

### 删除会话

沿用现有接口：

```bash
curl -X DELETE http://127.0.0.1:1933/api/v1/sessions/SESSION_ID
```

删除先撤销该会话来源并持久化索引，再清理 self 和 peer 记忆中受影响的支持记录、链接及向量，最后删除会话原文。共享卡片保留其他来源，正文从剩余引句重建。正在运行的旧抽取结果会在写入边界再次校验来源，过期向量任务也会比对卡片版本，避免重新写回已删除的内容。

清理失败会保留待处理 URI 记录，重试同一删除请求可继续完成清理，不会将部分清理报告为成功。会话已经不存在时，接口仍会尝试修复遗留来源，之后返回未找到。

### 修正消息

新增纯文本消息修正接口：

```bash
curl -X PUT http://127.0.0.1:1933/api/v1/sessions/SESSION_ID/messages/MESSAGE_ID \
  -H 'Content-Type: application/json' \
  -d '{"content": "我今年已完成 7 篇文章。"}'
```

接口修改实际 JSONL 原文、更新原文索引、撤销旧记忆支持并按会话策略重新抽取已归档消息。未归档消息只更新当前内容，后续提交时抽取。仅支持单个文本 part 的消息，不接受空内容；会话提交任务运行中返回冲突。

失败响应会提供处理阶段和 `retry_same_correction` 信息。应使用相同会话 ID、消息 ID 和修正内容重试；来源写入未完成时，检索会拒绝读取该用户的中间索引。成功响应中 `status="completed"`，并包含清理和重抽取结果。

## 兼容性与验证边界

- 此移植未迁移模板目录，也未修改 CacheFS、queuefs 或 pathlock 的实现和缓存策略；一致性控制调用既有锁接口。
- 未自动回填历史会话的原文索引。旧索引仍保留原文检索能力；缺失或损坏的图会降级并标记需要重建。`EvidenceIndex.rebuild_graph()` 是内部 Python 辅助函数，没有新增管理 HTTP 接口。
- 原文索引目前采用用户级 JSON 快照，读写和重建需要加载索引；尚未改为面向超大历史规模的增量数据库。
- 当前验证包括规则和真实英文 spaCy 模型测试、检索预算与来源核验、并发索引追加、清理重试，以及真实文件系统上的抽取 → 检索 → 修正 → 删除服务流程。集成测试使用测试模型提供者，不代表线上模型质量评测。
- 尚未执行设计文档中的完整 LoCoMo、CPU 920B、Token 成本和成功率验收，不据此承诺质量或性能指标。
