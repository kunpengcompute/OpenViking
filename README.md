# 鲲鹏 OpenViking

基于 OpenViking 的 Agent 上下文与长期记忆扩展，支持 DeepAgents 接入。

## 功能与版本

| 版本 | 功能 | 使用资料 |
|---|---|---|
| OpenViking 0.4.1 | DeepAgents 上下文与长期记忆管理 | [特性说明](docs/zh/OpenViking_V0.4.1/deepagents_context_management.md)、[版本说明](docs/zh/OpenViking_V0.4.1/release_notes.md)、[部署指南](docs/zh/OpenViking_V0.4.1/deployment_guide.md)、[Quick Start](docs/zh/OpenViking_V0.4.1/quick_start.md) |
| OpenViking 0.5.0 | CacheFS 查询缓存 | [特性介绍](docs/zh/OpenViking_V0.5.0/feature_introduction.md)、[使用说明](docs/zh/OpenViking_V0.5.0/best_practices.md) |

## 目录结构

```text
OpenViking/
├── README.md                         # 项目与资料入口
├── docs/zh/
│   ├── OpenViking_V0.4.1/
│   │   ├── LICENSE                    # 文档许可证
│   │   ├── deepagents_context_management.md # DeepAgents 接入与特性说明
│   │   ├── deployment_guide.md        # OV 服务部署
│   │   ├── quick_start.md             # DeepAgents 接入与运行
│   │   └── release_notes.md           # 版本配套、更新与使用限制
│   └── OpenViking_V0.5.0/             # CacheFS 查询缓存资料
└── patches/
    ├── OpenViking_V0.4.1/             # 版本补丁、清单与管理工具
    └── OpenViking_V0.5.0/             # CacheFS 查询缓存补丁
```

0.4.1 补丁以官方 v0.4.1 为固定基线，使用[补丁工具说明](patches/OpenViking_V0.4.1/README.md)中的命令校验、生成和准备源码。

## 参与贡献

[BoostKit 社区贡献指南](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md) · [许可证](LICENSE)
