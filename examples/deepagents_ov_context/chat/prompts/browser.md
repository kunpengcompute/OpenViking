[Browser 接入说明]
浏览器操作使用官方 agent-browser Skill 与当前安装版本的 CLI 说明。先读取 /skills/agent-browser/SKILL.md，再通过 browser(argv=["skills", "get", "core"]) 加载官方使用指南；完整参考使用 browser(argv=["skills", "get", "core", "--full"])，具体命令可用 browser(argv=["命令", "--help"]) 查询。

官方示例中的 agent-browser 命令通过 browser 工具执行：去掉 agent-browser 前缀，将命令及参数逐项放入 argv，保留每个参数的原值。例如 agent-browser fill "#title" "会议标题" 对应 browser(argv=["fill", "#title", "会议标题"])。需要标准输入时用 stdin 参数，例如 browser(argv=["eval", "--stdin"], stdin="document.title")。命令按参数数组直接执行。

Chat 按 Run 管理浏览器会话、启动参数、连接、代理和登录状态初始化，Run 结束时关闭本轮浏览器。会话及连接参数由应用提供，页面导航使用 HTTP(S)。官方命令的输出保留在 stdout/stderr，并返回 exit_code。

同一表单的多字段操作可通过 batch 批量执行。Chat 中推荐通过 stdin 传入 JSON 二维数组，每个内层数组是一条命令。例如 browser(argv=["batch", "--bail", "--json"], stdin='[["fill","@e2","示例标题"],["fill","@e8","示例备注"]]')。参数模式的每个参数是一条完整命令字符串，例如 browser(argv=["batch", "fill @e2 示例标题", "fill @e8 示例备注"])。示例值按本轮用户任务填写，refs 使用当前表单快照；检查 batch 各子命令结果和保存后的业务字段。

Chat 文件交付通过独立的 browser_files 工具完成：download_pdf 使用 target 和 filename 下载 PDF；pdf_text 按 filename 提取本轮已下载 PDF；screenshot 按 filename 将截图保存到本轮 /output/，full=true 保存整页。文件名使用单个文件名，PDF 使用 .pdf，截图使用 .png。官方 browser 命令保持原语义，截图交付示例为 browser_files(action="screenshot", filename="verification.png", full=true)。
