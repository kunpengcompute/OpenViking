# batch 指南兼容修正

官方 SKILL.md、LICENSE 和 upstream.json 保留上游原文件及来源哈希。

agent-browser 0.27.1 的 `skills get core --full` 中，Pre-navigation setup 示例将每条 JSON 数组直接作为命令行参数；同版本 CLI 的参数模式接收完整命令字符串，JSON 二维数组通过标准输入传入。实际按旧示例调用会返回 `Unknown command: [open]` 一类错误。

Chat 在成功返回该完整指南时，通过 `browser_guides.py` 精确替换这段示例为 stdin JSON 写法，并在工具结果记录 `documentation_patch=agent-browser-0.27.1-batch-stdin`。其他指南段落、命令输出和实际执行参数按 CLI 原样传递。已修复的指南不匹配旧片段时直接使用其原文。

Chat 的 argv/stdin 调用模板统一在 `prompts/browser.md` 提供，两种实验模式使用相同说明。
