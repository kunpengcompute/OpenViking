# 版本说明书

## 版本配套说明

### 产品版本信息

| 项目   | 内容                               |
| ---- | -------------------------------- |
| 产品名称 | Kunpeng BoostKit                 |
| 产品版本 | 26.3.0                           |
| 软件名称 | OpenViking 接入 DeepAgents 实现上下文管理 |
| 软件版本 | v1.0.0                           |

### 操作系统、编译器和CPU配套说明

| 操作系统                    | CPU类型          | 编译器        |
| ----------------------- | -------------- | ---------- |
| openEuler 24.03 LTS SP3 | 鲲鹏 920B 7282C（ARM64） | GCC 12.3.1 |

## 版本使用注意事项

### 使用注意事项

请参见《[部署指南](./deployment_guide.md)》。

## v1.0.0

### 更新说明

新增 OpenViking 接入 DeepAgents 的上下文管理能力，支持消息增量写入、工具结果压缩与回读、输入预算、提交归档、记忆召回及 checkpoint 恢复。具体请参见《[OpenViking 接入 DeepAgents 实现上下文管理](./deepagents_context_management.md)》。

### 验收问题跟踪

- [Issue #11](https://gitcode.com/boostkit/OpenViking/issues/11) — 真实网关在输出上限为 32 和 64 token 时均返回 `finish_reason=length`，OV 两次均报告 `VLMOutputLimitError`。输出上限为 8192 token 的对照请求返回 `stop`。
- [Issue #12](https://gitcode.com/boostkit/OpenViking/issues/12) — [开发 PR #28](https://gitcode.com/boostkit/OpenViking/pull/28) 修复了 Chat 在手动记忆归档恢复完成前发起模型调用的问题。在归档任务处于 `pending` 和 `running` 时分别中断 OV 进程，正常 Chat Session 恢复后，首次模型请求均包含原归档摘要。该修复位于开发仓的 Chat 集成，公共补丁交付范围见[补丁说明](../../../patches/OpenViking_V0.4.1/README.md)。

### 遗留问题

无。

## 版本配套文档

### v1.0.0 版本配套文档

| 文档名称 | 内容简介 | 交付形式 |
|---|---|---|
| [《版本说明书》](./release_notes.md) | 本文档提供鲲鹏 OpenViking 的版本发布信息。 | 开源仓 |
| [《OpenViking 接入 DeepAgents 实现上下文管理》](./deepagents_context_management.md) | 本文档提供 OpenViking 接入 DeepAgents 的上下文管理特性使用指导。 | 开源仓 |
| [《部署指南》](./deployment_guide.md) | 本文档提供鲲鹏 OpenViking 的安装使用指导。 | 开源仓 |
| [《快速入门》](./quick_start.md) | 本文档提供 OpenViking 接入 DeepAgents 的上下文管理快速入门指导。 | 开源仓 |

### 获取文档的方法

您可以通过访问[开源仓](https://gitcode.com/boostkit/OpenViking)浏览和获取相关文档。
