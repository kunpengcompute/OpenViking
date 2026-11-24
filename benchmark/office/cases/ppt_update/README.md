# PPT文本修改记忆对照

同一份14页PPT，六项独立任务、每项两个修改点；H和Q的目标值不同。业务提示词、辅助文件、输入哈希、Reader配置及执行档由 `spec.json` 固定。原生组使用DeepAgents，普通记忆组使用OV Middleware。

## 证据与验收

标准Chat保存模型输入、工具调用与回读、PPT版本、最终文件及下载回执。按[AI复核提示词](../../evaluation/AI_REVIEW_PROMPT.md)逐项核验目标内容、结构和样式、完整回读及120秒预算，六项修改全部完成后的统一打包和最终下载单列。视觉核验按本案例规则排除。

H使用Middleware自动提交和提取，脚本观察真实回执与任务结果，保存 `memory-bodies.json`、`memory-acceptance.json` 和 `frozen-memory.json`。冻结文件中的 `memory_integrity` 仅证明记忆池非空、正文哈希和池状态一致；H的业务验收由AI复核记录提供。

## 执行H并复核

在仓库根目录、运行Chat的Python环境中执行。Chat启用pptx Skill，选择的环境、account和OV地址保持一致，凭据由文件提供。

```bash
python -m benchmark.office.suites.ppt_update.run \
  --phase H --mode current --output result/ppt-H \
  --chat-url "$CHAT_URL" \
  --environment-id "$CHAT_ENVIRONMENT" \
  --ov-url "$OV_URL" --ov-key "$OV_KEY_FILE" \
  --account "$OV_ACCOUNT" --user "$FRESH_MEMORY_USER" --peer ppt-six-agent
```

使用复核提示词检查H的实际产物和轨迹。复核账本保留逐项证据，通过后在 `result/ppt-H/preparation-review.json` 保存：

```json
{
  "run_id": "填写H的实际run_id",
  "contract_sha256": "填写frozen-memory.json中memory_integrity.contract_sha256",
  "decision": "passed",
  "evidence": ["填写H逐项复核账本的实际路径"]
}
```

以上是复核结论的引用记录，`passed` 仅在H真实业务验收通过后填写。run_id从H的 `result.json` 读取。

## 执行三组Q对照

```bash
python -m benchmark.office.suites.ppt_update.compare \
  --preparation result/ppt-H \
  --preparation-review result/ppt-H/preparation-review.json \
  --pairs 3 --output result/ppt-comparison \
  --chat-url "$CHAT_URL" \
  --environment-id "$CHAT_ENVIRONMENT" \
  --ov-url "$OV_URL" --ov-key "$OV_KEY_FILE" \
  --account "$OV_ACCOUNT" --user "$FRESH_MEMORY_USER" --peer ppt-six-agent
```

命令核对H的Run和案例契约，交替运行原生／普通记忆，每轮使用新Session。Q前后校验冻结记忆指纹；首个模型输入检查任务保留、执行规则与真实召回。`comparison.json` 保存运行清单和性能统计，逐轮 `evidence.json` 指向原始快照。运行后依据AI复核提示词填写Q的逐项结论。

所有运行的轨迹、产物、失败记录和复核结论默认保存在 `result/`。
