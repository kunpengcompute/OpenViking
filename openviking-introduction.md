# OpenViking 介绍材料

## 1. 简介

**OpenViking** 是一个开源的、专为 AI Agent 设计的上下文数据库。OpenViking 通过文件系统范式统一管理 Agent 所需要的上下文（记忆、资源和技能），并实现上下文的分层供给与自我迭代，最终目标是降低 Agent 开发门槛，让开发者更专注于业务创新而非底层上下文管理。

它不是传统意义上只存向量的 RAG 组件，而是把智能体需要的**记忆、资源、技能、会话归档**统一组织到一个可检索、可追踪、可演化的上下文系统中。OpenViking 采用文件系统范式管理上下文，让智能体像访问本地文件一样访问长期记忆和外部知识。

## 2. 为什么需要 OpenViking

在 AI 时代，数据易得，但高质量的上下文却难求。构建 AI Agent 时，开发者经常遇到这些挑战：

- **上下文碎片化**：记忆在代码里，资源在向量库，技能散落各处，难以统一管理
- **所需上下文猛增**：Agent 的长程任务在每次执行时都会产出上下文，简单的截断或压缩会导致信息损失
- **检索效果不佳**：传统 RAG 是平铺式存储，缺乏全局视野，难以理解信息的完整语境
- **上下文不可观测**：传统 RAG 隐式的检索链路如同黑箱，出错时难以调试
- **记忆迭代有限**：目前记忆只是用户记忆的记录，缺乏 Agent 相关的任务记忆

OpenViking 正是为解决这些痛点而设计的上下文数据库。

## 3. 核心概念

### 文件系统范式

OpenViking 使用 `viking://` 命名空间统一组织上下文。记忆、资源和技能不再散落在不同系统里，而是像目录和文件一样被管理、定位和检索。

### 分层上下文

每份上下文会被组织为三个层级：

- `L0`：摘要，用于快速判断是否相关
- `L1`：概览，用于理解结构和核心内容
- `L2`：原文详情，仅在必要时深入读取

### 会话演化

OpenViking 不只存对话，还会将会话中的高价值内容持续沉淀为摘要、归档和长期记忆，使智能体具备“越用越懂上下文”的能力。

### 可观测检索

OpenViking 支持追踪检索路径、目录递归过程和命中结果，方便定位召回问题并持续调优。

## 4. 应用场景

OpenViking 适合以下场景：

- **编程 Agent**：管理代码库、文档、历史决策、技能和调试经验
- **企业知识助手**：统一接入文档、FAQ、知识库和组织内部流程
- **长期对话 Agent**：在多轮、多天、多任务对话中保持上下文连续性
- **研究与分析 Agent**：管理资料、引用关系、阶段性结论和归档结果
- **多工具协作 Agent**：把工具调用结果、资源引用和任务经验沉淀为可复用上下文

如果你的 Agent 需要处理“长期记忆 + 外部知识 + 多步任务”，OpenViking 会比单纯的向量数据库更适合。

## 5. 使用说明

### 基础使用

在开始前，建议先阅读 OpenViking 官网和仓库 README，快速建立整体认知：

- [OpenViking 官网](https://openviking.ai/)
- [OpenViking GitHub README](https://github.com/volcengine/OpenViking/blob/main/README_CN.md)
- [OpenViking 中文文档入口](./01-introduction.md)

安装 OpenViking：

```bash
pip install openviking --upgrade --force-reinstall
```

可选安装 CLI：

```bash
npm i -g @openviking/cli
```

初始化并检查环境：

```bash
openviking-server init
openviking-server doctor
```

启动服务：

```bash
openviking-server
```

服务启动后，你可以通过 SDK、CLI 或上层 Agent 插件接入 OpenViking，把资源、记忆和技能逐步导入到统一上下文空间中。

### OpenClaw 接入

如果你希望把 OpenViking 作为 OpenClaw 的长期上下文引擎，推荐直接使用官方 `openviking` context engine 插件。

前置条件：

- Node.js `>= 22`
- OpenClaw `>= 2026.4.8`
- 已有可访问的 OpenViking HTTP 服务

快速检查：

```bash
node -v
openclaw --version
```

安装与接入：

```bash
openclaw plugins install clawhub:@openviking/openclaw-plugin
openclaw openviking setup --base-url http://your-server:1933 --api-key sk-xxx --json
openclaw gateway restart
openclaw openviking status --json
```

这四步分别完成：

- 安装 OpenClaw 插件
- 写入 OpenViking 连接配置并激活 `contextEngine` 槽位
- 重启 OpenClaw Gateway
- 验证插件注册、连通性和状态

接入后，OpenClaw 会自动使用 OpenViking 处理：

- 会话召回
- 上下文写入
- 长期记忆提取

配置写入位置：

- `setup` 会把配置写入 `$OPENCLAW_STATE_DIR/openclaw.json`
- 默认目录通常为 `~/.openclaw/openclaw.json`

常用验证命令：

```bash
openclaw config get plugins.slots.contextEngine
openclaw config get plugins.entries.openviking.config
openclaw logs --follow
```

如果 `contextEngine` 返回 `openviking`，且 `status` 检查通过，说明接入已经生效。

如果 ClawHub 不可用，官方文档还提供 `ov-install` 作为备用安装路径。

如果需要手动排查，优先检查：

- OpenViking 服务是否可达
- `baseUrl` 与 `apiKey` 是否正确
- `plugins.slots.contextEngine` 是否已切换为 `openviking`
- Gateway 重启后日志里是否出现 `openviking: registered context-engine`

在需要做全链路验证时，还可以使用插件文档中的 health check 脚本，对 Gateway → OpenViking → session/commit/archive/memory extract 这一整条路径做检查。

相关文档：

- [OpenClaw 插件完整说明](./../agent-integrations/03-openclaw.md)
- [OpenClaw 插件仓库文档](../../../examples/openclaw-plugin/README_CN.md)
- [OpenClaw 插件设计与安装 README](https://github.com/volcengine/OpenViking/tree/main/examples/openclaw-plugin)

## 6. Token 节省说明

OpenViking 的 token 节省并不是简单压缩上下文，而是通过**分层 + 按需 + 归档**三种机制一起实现：

- **分层读取**：优先读取 `L0/L1`，避免默认加载 `L2` 全文
- **按需注入**：只把当前任务真正相关的上下文放进模型窗口
- **自动归档**：长会话中的旧内容会转化为摘要和 archive，而不是每轮重复注入

这带来的直接收益是：

- 降低单轮 prompt 的输入 token
- 降低长会话的累计调用成本
- 减少模型窗口被无效上下文占满
- 让上下文预算优先留给当前任务真正需要的信息

对于长对话、多文档、多工具调用的 Agent，这种节省通常比单次 prompt 优化更有价值。

## 7. 检索加速

鲲鹏检索算子库KRL（Kunpeng Retrieval Library）是华为提供的基于鲲鹏平台优化的用于加速向量检索的算子库。KRL针对鲲鹏处理器的指令集架构与内存访问机制进行底层优化，使用低精度量化+高精度重排等方法有效提升召回算法的计算效率与吞吐量，同时保证算法精度，适用于高并发召回场景的性能需求。OpenViking基于KRL加速实现检索性能大幅提升，在保证召回质量的前提下，提高检索吞吐、降低延迟。