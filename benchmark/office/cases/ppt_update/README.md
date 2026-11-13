# PPT 文本修改记忆对照

版本 `ppt-six-shared-helper-hq-v3`。同一份 14 页 PPT，六项独立任务、每项两个修改点；H 和 Q 的目标值不同，验证同一模板上的方法复用。原生组使用 DeepAgents，普通记忆组使用 OV Middleware。

## 固定资产

- `spec.json`：任务、12 个修改点、输入及 prompt 哈希、每项 120 秒预算和评分口径。
- `execution.json`：256K 上下文、32K 输出、thinking、16000 字符 / 8000 Token 召回预算。
- `benchmark/office/tasks/ppt_update/H.md`：准备任务，提供辅助文件的详细使用方法。
- `benchmark/office/tasks/ppt_update/Q.md`：测试任务，提供业务要求和辅助文件路径。
- `benchmark/office/tasks/ppt_update/system_prompt.md`：逐项执行、局部校验、最终打包及关闭视觉核验的规则。
- `benchmark/office/datasets/ppt_update/modify_test.pptx`：各 Run 分别上传的原稿。
- `benchmark/office/datasets/ppt_update/edit_slide.txt`：H 和两组 Q 均上传的相同 Python 辅助源码；使用文本后缀适配 Chat 文件输入。

## 验收与时延

1. 六项任务各计一分，两处修改都正确才能得分，总分 6；同时记录 12 个修改点的结果。
2. 最终检查完整文字、14 页与页序、原有媒体，以及真实下载和答复中的文件链接。文件名允许变化。
3. Q 轮六项修改分别按 **120 秒**计分：从 Chat Run 开始，到该项全部修改及回查核验完成，120 秒整计为预算内成功。每项独立判断；前 k 项累计时延取前 k 项合格完成时点的最大值。
4. `grade.json.task_pass` 表示最终交付验收；`grade.json.latency.success_count`／`total` 表示预算内子任务成功数及分母。完成时点由标准 Chat 自动保存的产物版本逐项验收得到，证据不足的时点标记为未核验。超预算继续执行至终态，保留完整耗时和 Agent＋OV LLM Token。执行保护超时为 1800 秒，模型无流式活动超时为 180 秒。
5. H 通过 Middleware 自动提交和提取，完成记忆产物验收后形成冻结记忆池；Q 使用全新 Session，提取关闭。记忆 Q 的首次模型输入必须包含真实召回；函数源码由各 Run 的输入文件提供。
6. 顺序执行通过工具 / 修订轨迹单独核验；严格 XML 保持性单列。视觉核验按本案例规则排除。

## H 记忆产物验收

H 的业务验收通过、自动提取任务完成后，读取 OV 中的记忆正文并检查：

1. 记忆池和正文非空，正文 SHA256 与指纹一致，验收前后的记忆池保持一致。
2. 记忆与当前 H/Q 提示词及共享辅助文件的契约指纹关联。
3. 方法记忆可以只保留目录、参数及操作经验；函数源码由共享输入文件提供。

`memory-acceptance.json` 记录正文和指纹检查；`memory-bodies.json` 保存实际提取正文。通过后生成带验收标记的 `frozen-memory.json`；Q 核对冻结清单和实际召回注入。H 使用 Middleware 自动提交，测试脚本只观察结果。

每次运行保存 `input-receipts.json`，核对 PPT 和辅助文件的实际上传哈希；两组共享辅助文件保持字节一致。首次模型输入检查保留业务任务和执行规则，并确认普通记忆组实际收到召回、原生组没有召回。方法是否被采用及其行为收益由工具轨迹分析，不要求模型输入包含函数源码。

## 执行

从仓库根目录执行，使用运行 Chat 的 Python 环境。Chat 启用 pptx Skill，模型沿用该服务配置；Reader 使用 context / openclaw、候选 25、召回 5。选择的 Chat 环境、account 和 OV 地址保持一致，`--ov-key` 指向受保护的凭据文件。

```bash
python -m benchmark.office.suites.ppt_update.compare \
  --pairs 3 --output result/ppt-helper-comparison \
  --chat-url "$CHAT_URL" --chat-data "$CHAT_DATA" \
  --environment-id "$CHAT_ENVIRONMENT" \
  --ov-url "$OV_URL" --ov-key "$OV_KEY_FILE" \
  --account "$OV_ACCOUNT" --user "$FRESH_MEMORY_USER" --peer ppt-six-agent
```

`FRESH_MEMORY_USER` 使用记忆为空的新 user。命令先执行一次 H 自动提取，再交替运行“原生→记忆 / 记忆→原生”的 Q 对照，每次使用独立工作区。结果目录结构如下：

```text
result/ppt-helper-comparison/
├── manifest.json
├── comparison.json
├── H/                         # 准备任务、自动提交回执、冻结池
├── pair-1/
│   ├── Q-baseline/             # 无 OV Middleware
│   └── Q-memory/               # 自然召回 H 记忆
├── pair-2/
└── pair-3/
```

各运行目录保存条件、SSE、真实模型输入、工具轨迹、完整用量、下载文件和评分。`comparison.json` 汇总每次的任务结果及六项任务各自的 120 秒预算结果。H 没有自动提交回执或提取失败时，保留记录并停止；Q 的业务失败正常纳入结果，环境或冻结条件变化单独报错。单次 H/Q 可通过同目录的 `run` 模块执行，参数见 `--help`。

实验产生的记忆正文、执行轨迹与交付文件保留在 `result/`，模型凭据由部署环境管理。
