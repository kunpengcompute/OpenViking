# DeepAgents OVContext 调试与测试套

提供 Chat HTTP/SSE 调试入口、OV 容器部署与办公案例测试接入。

```text
deepagents_ov_context/
├── chat/                 # Chat 服务与 DeepAgents 接入
│   ├── server.py         # HTTP/SSE 服务入口
│   ├── agent.py          # Agent 创建与运行上下文
│   ├── config.py         # 配置加载及 OV 配置生成
│   ├── session.py        # 会话状态与 SQLite 存储
│   ├── stream.py         # 流式事件与模型观测
│   ├── static/           # Chat 页面
│   ├── prompts/          # 系统提示词
│   └── skills/           # 随测试套提供的 Skill
├── config/               # 环境变量、模型及多环境配置模板
├── scripts/              # 部署、检查与冒烟命令
├── docker/               # 镜像、Compose 与容器运行支撑
└── docs/                 # 部署与测试指南
```

- [部署指南](docs/DEPLOYMENT.md)：配置模型与 OV，构建并启动容器，部署案例数据。
- [测试指南](docs/TESTING.md)：UT、ST、Chat 验收及办公案例。
- [环境变量模板](config/chat.example.env)、[模型配置模板](config/models.example.json)、[多环境配置模板](config/chat.environments.example.json)。

从仓库根目录查看入口参数：

```bash
bash examples/deepagents_ov_context/scripts/container.sh --help
python -m examples.deepagents_ov_context.chat.server --help
python -m examples.deepagents_ov_context.scripts.check --help
python -m examples.deepagents_ov_context.scripts.smoke --help
```

业务案例、任务提示词与数据位于 `benchmark/office/`；单元与隔离集成测试位于 `tests/`，真实服务验收入口为 `tests.st.ov_context.run`。运行配置、SQLite、日志和测试结果保存在部署挂载目录。
