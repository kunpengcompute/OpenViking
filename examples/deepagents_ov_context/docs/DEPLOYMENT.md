# OVContext Chat 容器部署指南

**基础部署：** OV Server 与 Chat 测试套分别运行在独立容器中。镜像包含源码、Python 及原生依赖；Chat 镜像同时包含浏览器、Skill 和办公渲染工具。部署步骤见第 1–6 节。

**数据集部署：** 完整案例代码随仓库提供，案例服务依赖包含在 Chat 镜像中。办公工作台、Webmail 和 IMAP 在 Chat 容器内按需启动，H/Q 数据持久化到案例目录。PPT 更新使用分阶段输入文件。步骤见第 7 节。

部署入口为 [container.sh](../scripts/container.sh)，镜像定义为 [Dockerfile](../docker/Dockerfile)，服务定义为 [compose.yaml](../docker/compose.yaml)。所有命令从交付源码的仓库根目录执行。

## 1. 运行环境

- **目标平台**：920B 7282C，ARM64，openEuler 24.03 LTS。
- **宿主机依赖**：Docker Engine、Docker Compose 插件、Buildx 插件、Git 和 curl。
- **容器系统**：OV 和 Chat 基于 `openeuler/openeuler:24.03-lts-sp3`。[官方镜像说明](https://github.com/openeuler-mirror/openeuler-docker-images/blob/master/README.en.md)
- **模型服务**：Chat 及 OV 的摘要 / 记忆提取连接外部模型服务；向量化连接 Embedding 服务。
- **网络**：两个容器使用 Linux 宿主机网络，OV、Chat、办公页面、Webmail 和 IMAP 均监听回环地址；模型请求通过 HTTPS 发出。

### 1.1 安装 Docker 与插件

以下命令以部署账号执行，源码目录为当前工作目录。

**安装 Docker Engine 并授权部署账号：**

```bash
sudo dnf install -y docker git curl ca-certificates
sudo systemctl enable --now docker
sudo groupadd -f docker
sudo usermod -aG docker "$USER"
```

退出 SSH 后重新登录，使用户组权限生效，再确认引擎可用：

```bash
docker info
```

**安装 Compose 和 Buildx 插件：**

Compose 与 Buildx 通过仓库脚本单独安装。遇到 `docker: 'compose' is not a docker command` 或 `docker: 'buildx' is not a docker command` 时，执行同一安装命令：

```bash
bash examples/deepagents_ov_context/scripts/install-docker-plugins.sh
docker compose version
docker buildx version
```

脚本为缺少的插件安装 Compose `v2.39.4`、Buildx `v0.17.1`，自动选择 ARM64 或 x86_64 二进制，校验官方发布文件的 SHA-256，并设置可执行权限。已有可用插件沿用当前版本。安装目录为 `${DOCKER_CONFIG:-$HOME/.docker}/cli-plugins/`，文件名分别为 `docker-compose`、`docker-buildx`。部署命令统一使用 `docker compose`。

安装和部署使用同一账号。需要通过 `sudo` 检查 Docker 时，显式传入该账号的插件配置目录：

```bash
sudo env DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" docker compose version
sudo env DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}" docker buildx version
```

插件下载使用当前 Shell 的网络及 `HTTPS_PROXY` 配置；下载失败时，确认 GitHub 发布文件可达后重新执行安装脚本。插件发现路径遵循 [Docker CLI 插件安装规则](https://docs.docker.com/compose/install/linux/)。

### 1.2 容器规格

以下为功能验证环境的起配规格，持久化目录使用本地 SSD / NVMe：

| 容器 | CPU | 内存上限 | 持久化空间 |
|---|---|---:|---:|
| OV Server | 1 个物理核及其逻辑线程 | 2 GiB | 20 GiB 起 |
| Chat 测试套 | 4 个物理核及其逻辑线程 | 8 GiB | 50 GiB 起 |

镜像构建需要额外的编译内存、磁盘和下载空间，建议在独立构建环境完成。案例页面、IMAP 和 Webmail 共享 Chat 容器的 CPU、内存与数据卷。

物理核绑定根据机器拓扑配置：

```bash
lscpu -e=CPU,CORE,SOCKET
```

在第 3 节的 `compose.env` 中将 `OV_CPUSET` 填为同一物理核的逻辑 CPU，将 `OV_CHAT_CPUSET` 填为另四个物理核的逻辑 CPU。默认 CPU 时间配额为 OV 2、Chat 8 个逻辑 CPU；物理核隔离通过 cpuset 配置实现。

## 2. 准备镜像

### 2.1 构建镜像

获取交付源码并进入仓库根目录。镜像构建和部署均使用同一套文件：

```bash
bash examples/deepagents_ov_context/scripts/container.sh --help
BUILD_JOBS=4 bash examples/deepagents_ov_context/scripts/container.sh build
```

构建需要访问 Rust、Python、Node、浏览器和 Skill 的下载源。使用 HTTP 代理时，将代理显式传给构建阶段。代理监听宿主机回环地址时，构建网络选择 `host`：

```bash
export HTTP_PROXY="http://127.0.0.1:实际代理端口"
export HTTPS_PROXY="$HTTP_PROXY"
COMPOSE_BAKE=false OV_BUILD_NETWORK=host BUILD_JOBS=4 \
  bash examples/deepagents_ov_context/scripts/container.sh build \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY
```

`OV_BUILD_NETWORK` 仅控制镜像构建网络，默认值为 `default`。`COMPOSE_BAKE=false` 使用 Compose 标准构建路径。代理参数只传给本次构建；运行服务的模型连接使用其实际网络配置。

Chromium 使用 Playwright 官方 CDN `https://cdn.playwright.dev` 下载；通过 `build --build-arg PLAYWRIGHT_DOWNLOAD_HOST=实际镜像地址` 可指定配套下载源。安装器读取固定 Playwright 版本的浏览器包信息，使用 curl 断点续传，并在解压前检查 ZIP 完整性。镜像先编译原生组件，再复制部署脚本、案例与测试代码；部署资料的修改复用已有原生编译缓存。

LibreOffice 固定为 26.8.1.1，安装包取自[官方版本归档](https://downloadarchive.documentfoundation.org/libreoffice/old/26.8.1.1/rpm/aarch64/)，并校验脚本中固定的 SHA-256。浏览器、渲染器和 Skill 分层安装，构建重试复用已完成的依赖层。

构建生成：

| 镜像 | 默认名称 | 包含内容 |
|---|---|---|
| OV Server | `ov-context-server:local` | OV 服务、CLI、原生存储 / 向量引擎、Python 3.11 |
| Chat 测试套 | `ov-context-chat:local` | OV 运行依赖、DeepAgents、Chat、Node 24、agent-browser、Chromium、pptx Skill、办公 Skill、LibreOffice、Poppler、中文字体、完整案例代码、Dovecot、PHP、Roundcube 和测试依赖 |

Roundcube 的 PHP 与静态资源在构建时从官方发行镜像复制到 Chat，运行时由 Chat 内的 PHP 提供 Webmail 页面。

源码与虚拟环境位于镜像内 `/opt/openviking`，工具位于 `/opt/ov-tools`。安装操作只发生在镜像构建阶段。构建上下文过滤本地配置、凭据、数据库、宿主机原生二进制及开发缓存；原生组件针对 ARM64 从源码构建。

构建阶段执行原生 TEXT 编码检查、浏览器检查和 Skill / 渲染依赖检查。任一安装或检查失败即终止镜像构建。Python 实际依赖清单保存在镜像内 `/opt/openviking/image-requirements.txt`，pptx Skill 来源版本保存在 `/opt/ov-tools/pptx-skill-revision.txt`。

### 2.2 使用交付镜像

部署机器可直接加载配套镜像包：

```bash
docker load -i ov-context-images.tar
```

镜像包包含上表两个镜像。镜像名称不同或使用私有仓库时，通过第 3 节的 `compose.env` 设置 `OV_IMAGE`、`OV_CHAT_IMAGE`；仓库镜像可先执行 `docker pull`。镜像对应的 `container.sh`、Compose 文件和配置模板一起交付。

## 3. 配置与数据目录

DeepAgents 固定为 `0.7.3`，由源码的 `.[deepagents]` 依赖安装。部署检查校验实际安装版本，检查结果中的 `deepagents` 应为 `0.7.3`。

宿主机只需指定一个部署根目录：

```bash
export OV_DEPLOY_HOME="$HOME/ov-context-chat"
bash examples/deepagents_ov_context/scripts/container.sh init
```

初始化通过临时管理容器创建模板及数据目录，已有配置值保留。目录结构如下：

```text
$OV_DEPLOY_HOME/
├── compose.env                  # 镜像、端口和容器规格
├── config/
│   ├── models.local.json        # 模型及 Embedding 凭据
│   ├── .env                     # Reader、Commit 和上下文配置
│   └── ov/
│       ├── ov.conf              # OV 配置，configure 生成
│       └── service.key          # trusted 服务凭据
└── data/
    ├── ov/                      # OV 会话、归档、记忆及索引
    ├── chat/                    # 应用会话、checkpoint、日志、下载和产物
    └── cases/                   # 案例 registry、输入、管理员参考及结果
```

`config/` 在服务容器中只读挂载到 `/config`。OV 挂载 `/data/ov`，Chat 挂载 `/data/chat` 和 `/data/cases`。管理命令复用 Chat 镜像，按需读写配置及数据。OV、Chat 执行进程、办公页面和 PHP 使用部署账号的 UID / GID；Chat 容器入口负责邮件守护进程的启动与进程组管理。

### 3.1 模型和 Embedding

编辑 `$OV_DEPLOY_HOME/config/models.local.json`，填入实际 API Key 和模型配置，例如：

```json
{
  "api_key": "填写自己的 API Key",
  "api_base": "https://ark.cn-beijing.volces.com/api/plan/v3",
  "chat_model": "deepseek-v4-flash",
  "embedding_model": "doubao-embedding-vision",
  "embedding_dimension": 1024,
  "embedding_provider": "volcengine",
  "embedding_input": "multimodal"
}
```

Chat 使用 `chat_model` 调用对话模型；配置生成器默认使用相同模型生成 OV 摘要与记忆提取配置，向量检索使用 Embedding 模型。生成器将 `api_base` 和 `api_key` 同时用于 OV 的生成模型与 Embedding 初始配置。接口地址、模型权限和向量维度须配套；各服务使用独立地址或凭据时，在生成后的 `ov.conf` 中分别配置。

### 3.2 OV 服务配置

**生成配置文件**

填写 `models.local.json` 后执行：

```bash
bash examples/deepagents_ov_context/scripts/container.sh configure
```

该命令生成宿主机文件 `$OV_DEPLOY_HOME/config/ov/ov.conf`，OV 容器通过 `/config/ov/ov.conf` 读取。数据目录固定为容器内 `/data/ov`，对应宿主机 `$OV_DEPLOY_HOME/data/ov`。

OV 使用 `trusted` 认证。首次生成配置时创建 `config/ov/service.key`，并将同一凭据写入 `server.root_api_key`；Chat 从凭据文件读取后访问 OV。`ov.conf` 和 `service.key` 的文件权限均为 600，服务容器只读挂载配置目录。

**配置结构与模型参数**

需要为 OV 设置独立生成模型、Embedding 地址或调整摘要参数时，编辑已生成的 `ov.conf`。以下为配置示例，模型与凭据字段替换为实际值，`server.root_api_key` 沿用生成的服务凭据：

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 1933,
    "auth_mode": "trusted",
    "root_api_key": "沿用与 service.key 一致的服务凭据"
  },
  "storage": {
    "workspace": "/data/ov",
    "agfs": {"backend": "local"},
    "vectordb": {"backend": "local"},
    "transaction": {"lock_expire": 30}
  },
  "vlm": {
    "provider": "openai",
    "api_base": "填写生成模型的 OpenAI 兼容接口地址",
    "api_key": "填写生成模型 API Key",
    "model": "填写摘要和记忆提取模型名称",
    "max_tokens": 8192,
    "timeout": 300,
    "max_retries": 1
  },
  "embedding": {
    "dense": {
      "provider": "volcengine",
      "api_base": "填写 Embedding 接口地址",
      "api_key": "填写 Embedding API Key",
      "model": "填写 Embedding 模型名称",
      "dimension": 1024,
      "input": "multimodal"
    }
  },
  "oauth": {"enabled": false}
}
```

| 配置项 | 配置方法 |
|---|---|
| `server.auth_mode` / `root_api_key` | 使用 `trusted`，服务凭据与 `config/ov/service.key` 保持一致；account、user、actor peer 由 Chat 每次请求传入 |
| `server.host` / `port` | 容器入口通过启动参数指定回环地址和端口；端口统一在 `compose.env` 的 `OV_CHAT_OV_PORT` 中配置，再执行 `configure` |
| `storage.workspace` / `agfs` / `vectordb` | 使用 `/data/ov` 及本地存储后端，消息、工具原文、归档、记忆和向量索引保存到宿主机数据目录 |
| `storage.transaction.lock_expire` | 生成配置使用 30 秒租期，存活任务每 15 秒续租；进程退出后等待租期届满再恢复归档，为 300 秒提交等待留出处理时间 |
| `vlm.provider` / `api_base` / `api_key` / `model` | 配置 OV 执行摘要和记忆提取的模型；OpenAI 兼容接口使用 `provider="openai"` |
| `vlm.max_tokens` / `timeout` / `max_retries` | 分别控制 OV 模型调用的最大输出 Token、单次请求超时秒数和重试次数；示例使用 8192、300 秒和 1 次，按模型能力与任务规模调整 |
| `embedding.dense` | provider、接口地址、模型、维度和输入类型与 Embedding 服务能力配套；示例为 1024 维多模态接口，文本接口填写对应 provider 和 `input="text"` |

`vlm` 参数用于 OV 服务内的摘要与记忆提取。Chat 对话模型使用 `models.local.json` 及 Chat 的执行参数；Middleware 等待归档的 `OV_CONTEXT__COMMIT_WAIT_TIMEOUT` 在 `.env` 中配置，分别生效。

`configure` 每次根据 `models.local.json` 重写 `ov.conf`，保留已有 `service.key`。OV 专用模型参数在生成后设置；后续重新生成配置时同步应用这些设置。切换 Embedding 模型或维度时，使用独立部署目录初始化 OV 数据，并重新写入、提取记忆和建立索引。

**应用配置修改**

首次部署按第 4 节启动。运行中修改模型或 OV 配置后，执行以下命令重建容器并等待就绪，数据继续使用原挂载目录：

```bash
bash examples/deepagents_ov_context/scripts/container.sh compose up -d --wait --force-recreate ov chat
bash examples/deepagents_ov_context/scripts/container.sh status
bash examples/deepagents_ov_context/scripts/container.sh check
```

`check` 检查依赖和服务连通；实际模型访问、摘要生成、提交及召回通过后续 Chat 任务验证。

### 3.3 Reader、提交与上下文

编辑 `$OV_DEPLOY_HOME/config/.env`。完整字段取自 [chat.example.env](../config/chat.example.env)，关键配置示例如下：

```dotenv
OV_CONTEXT__CONTEXT_WINDOW=256000
OV_CONTEXT__COMMIT_WAIT_TIMEOUT=300

OV_READER__SEARCH_MODE=context
OV_READER__ENABLED=true
OV_READER__TARGET_URIS=null
OV_READER__RECALL_LIMIT=5
OV_READER__CONTEXT_TOKEN_BUDGET=256000
OV_READER__EXPERIENCE_ENABLED=false

OV_COMMIT__TOKEN_BUDGET=256000
OV_COMMIT__COMMIT_TOKEN_THRESHOLD_RATIO=0.7
OV_COMMIT__COMPLETED_RUN_THRESHOLD=5
OV_COMMIT__KEEP_RECENT_COUNT=8
```

- 模型总窗口、Commit 源消息预算、Reader 默认读取预算分别配置。模型调用前，Reader 的历史额度由组装器计算。
- `TARGET_URIS=null` 默认检索 user 与当前 Agent 记忆；配置类别配额后按类别解析目录。显式 URI 配置决定实际检索范围。
- Commit 的四项配置必填；保留数按消息条数计，比例范围为 `(0, 0.7]`。
- 同步归档等待上限为 300 秒；单次 OV I/O 受 Chat 的等待配置约束。读取、写入及降级状态通过执行事件观测。
- Session 创建时合并并保存有效配置，每 Run 重建 Agent 时复用该快照。修改 `.env` 后重启 Chat，新建 Session 使用新配置。

### 3.4 模型调用与执行环境

`.env` 的 `CHAT_MODEL__CALL_TIMEOUT_SECONDS` 控制模型响应等待超时，示例配置为 180 秒。流式调用分别限制首个片段和后续相邻片段的等待时间，收到文本、reasoning、工具参数或 usage 片段后重新计时；持续有片段返回的回复可以超过 180 秒。非流式调用限制等待完整响应的时间。整轮任务总时限由 `timeout_seconds` 单独控制。`CHAT_AGENT__ENABLE_TODOS` 控制任务计划工具。创建 Session 时保存有效值，后续 Run 复用；页面可在新会话中覆盖。

Chat 支持通过 `--environments-file` 加载额外 OV 环境，配置文件放在挂载的 `/config` 下。各环境指定独立的 OV 地址、account、凭据文件及环境文件；页面和批量实验通过 `environment_id` 选择。Session 保存环境绑定，`experiment_id` 用于关联一组实验。默认环境为 `daily`。

### 3.5 镜像、端口与资源

创建 `$OV_DEPLOY_HOME/compose.env`，填写镜像、端口和资源参数。已有部署修改对应配置项。例如：

```bash
cat > "$OV_DEPLOY_HOME/compose.env" <<'EOF'
COMPOSE_PROJECT_NAME=ov-context
OV_IMAGE=ov-context-server:local
OV_CHAT_IMAGE=ov-context-chat:local
OV_CASES_ENABLED=false
OV_CASE_PAGE_PORT=19482
OV_CASE_IMAP_PORT=19493
OV_CASE_WEBMAIL_PORT=19480
OV_CHAT_OV_PORT=1933
OV_CHAT_PORT=1934
OV_MEMORY=2g
OV_CHAT_MEMORY=8g
OV_CPUS=2
OV_CHAT_CPUS=8
# 根据 lscpu 的 CPU、CORE、SOCKET 对应关系填写实际 CPU 列表。
OV_CPUSET=
OV_CHAT_CPUSET=
EOF
chmod 600 "$OV_DEPLOY_HOME/compose.env"
```

`OV_CPUSET`、`OV_CHAT_CPUSET` 按第 1.2 节选择的实际逻辑 CPU 列表配置。`container.sh` 自动传入部署账号 UID / GID。每个终端使用相同的 `OV_DEPLOY_HOME`；不同部署使用不同目录、项目名和端口。

## 4. 启动与检查

```bash
bash examples/deepagents_ov_context/scripts/container.sh up
bash examples/deepagents_ov_context/scripts/container.sh status
bash examples/deepagents_ov_context/scripts/container.sh check
```

`up` 启动 OV 和 Chat 两个常驻容器，并等待健康检查。OV `/ready` 就绪后启动 Chat。`OV_CASES_ENABLED=true` 时，Chat 同时启动案例页面、Webmail 和 IMAP；健康检查覆盖这些服务，辅助进程退出后由容器入口统一停止进程组并交给 Docker 重启。`check` 在管理容器中检查原生组件、Skill / 渲染依赖及服务连通；它只检查环境，不调用聊天模型。

默认入口：

| 地址 | 用途 |
|---|---|
| `http://127.0.0.1:1933` | OV HTTP API |
| `http://127.0.0.1:1934` | Chat 页面 |
| `http://127.0.0.1:1934/docs` | Chat API 文档 |
| `http://127.0.0.1:1934/api/health` | Chat 健康及能力配置 |

Chat 默认启用浏览器、Thinking、pptx 和办公 Skill。模型输出上限为 16384，图步数上限为 1000；图步数包含 Middleware 和工具节点。健康接口展示实际启用的能力。

部署验收覆盖镜像依赖、原生组件、配置权限、服务就绪及案例数据初始化。完成第 7 节的数据集部署后，再执行一次 `container.sh check`。模型调用、UT、ST 和业务案例按 [TESTING.md](TESTING.md) 单独执行，并分别记录结果。

查看日志：

```bash
bash examples/deepagents_ov_context/scripts/container.sh logs -f ov chat
```

远端调试时，在本地电脑建立页面转发：

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:1934:127.0.0.1:1934 用户名@服务器地址
```

## 5. 验证 Chat 与 OV 链路

### 5.1 冒烟验收

```bash
bash examples/deepagents_ov_context/scripts/container.sh smoke
```

该命令实际调用模型，检查 SSE、工具消息写入、checkpoint 跨 Run 复用以及从 OV 恢复。结果写入宿主机 `$OV_DEPLOY_HOME/data/chat/smoke.json`。

页面新建会话后，可观察模型输入、Hook 事件、OVContext、实际 Token、首段正文时延和完成耗时。SSE `done` 表示执行结束；业务任务成功还需核对实际结果。

### 5.2 脚本调用与观测

脚本和页面使用相同 HTTP/SSE 接口：

```bash
curl --fail http://127.0.0.1:1934/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"debug-user","agent_id":"chat-agent"}'

curl --fail -N http://127.0.0.1:1934/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"实际会话ID","message":"请调用debug_echo回显hello，并回答结果。"}'
```

用返回的 Session ID 替换占位值。客户端按 SSE 事件接收 `token`、`message`、`statistics` 和终态；完整 `message` 用于校正显示。失败或取消事件也应保存。

统一执行日志位于 `data/chat/logs/runs.jsonl`，按 Session / Run / model_call_id 关联模型输入、预算和实际 usage。调试记录保存在 `sessions.sqlite3`，图状态保存在 `checkpoints.sqlite3`。

### 5.3 真实服务集成测试

通过测试套镜像运行已部署服务的集成测试：

```bash
bash examples/deepagents_ov_context/scripts/container.sh run python -m \
  tests.st.ov_context.run \
  --ov-url http://127.0.0.1:1933 \
  --api-key-file /config/ov/service.key \
  --model-config /config/models.local.json \
  --output-dir /opt/openviking/result/st/ov_context
```

使用独立验证部署运行，测试会创建合成会话并消耗模型 / Embedding 用量。测试入口及覆盖范围见[测试指南](TESTING.md)。

### 5.4 浏览器与附件核验

浏览器 CLI 固定安装 agent-browser 0.41.2，包含动态页面旧 ref 的失效检查与重新定位修复。仓库中的 Skill 发现入口沿用 0.27.1 官方原文及其 LICENSE、来源指纹；模型通过 `browser(argv=["skills", "get", "core"])` 获取当前已安装 CLI 对应的使用指南，普通命令和选项直接传给 CLI。Chat 的传输、Run 资源和文件交付说明位于 `prompts/browser.md`；新建 Session 使用当前模板。

batch 批量操作在 Chat 中使用 `argv=["batch", "--bail", "--json"]`，通过 `stdin` 传入命令二维 JSON 数组。该版本完整指南中的一处 JSON 参数模式示例由 Chat 精确修正，并在工具结果标记 `documentation_patch`；补丁来源与范围见 `skills/agent-browser/COMPATIBILITY.md`。

模型先从页面找到附件链接，再调用 `browser_files(action="download_pdf", target="已发现的URL或SELECTOR", filename="notice.pdf")` 和 `browser_files(action="pdf_text", filename="notice.pdf")`。该扩展通过浏览器读取附件字节，保存到 `downloads/<session_id>/<run_id>/`，再使用 pdfplumber 解析。单份 PDF 上限为 20 MiB、50 页，文本超过 60,000 字符时返回 `truncated: true`。

附件验收核对两个操作均为 `exit_code: 0`、解析未截断、落盘 PDF 存在，以及答案字段与原页一致。直接打开 PDF 返回 `ERR_ABORTED` 时，仍可通过上述附件交付能力读取。截图使用 `browser_files(action="screenshot", filename="verification.png", full=true)`，保存到本轮 `output/` 并由 Chat 发布下载入口。

## 6. 持久化、重启与镜像更新

| 宿主机位置 | 内容 |
|---|---|
| `config/` | 模型、Reader / Commit / 预算配置、trusted 凭据 |
| `data/ov/` | OV 会话、工具原文、归档、记忆和向量索引 |
| `data/chat/sessions.sqlite3` | 会话身份、配置快照、执行记录及统计 |
| `data/chat/checkpoints.sqlite3` | 原始消息、OVContext、写入游标及图状态 |
| `data/chat/logs/` | 统一 JSONL 日志 |
| `data/chat/downloads/`、`data/chat/artifacts/` | 下载附件、每轮工作区和交付文件 |
| `data/cases/` | 案例业务数据库、邮箱、输入、参考及结果 |
| `data/cases/logs/` | 页面、Webmail、IMAP 进程日志 |

停止及重新启动：

```bash
bash examples/deepagents_ov_context/scripts/container.sh stop
bash examples/deepagents_ov_context/scripts/container.sh up
```

备份前停止服务，备份整个部署目录，包含 SQLite 的 WAL / SHM 文件及 OV 完整数据。恢复时沿用目录并使用匹配的镜像。

更新镜像后重建服务容器，保留绑定的数据目录：

```bash
bash examples/deepagents_ov_context/scripts/container.sh compose up -d --wait --force-recreate ov chat
```

同一 Session 的配置快照保持固定。应用配置改变后，新建 Session 使用新值。服务保持单 worker；同一 Session 的 Run 串行执行。

## 7. 办公案例数据集部署

### 7.1 案例代码与依赖

完整案例代码在仓库的 `benchmark/office/` 普通目录中，随 Chat 镜像构建：

```text
benchmark/
└── office/
    ├── tasks/                    # 办公案例的 H/Q 提示词
    ├── datasets/                 # H/Q 种子、PPT 和文档输入及验收参考
    ├── cases/                    # 办公页面、业务规则与案例定义
    ├── suites/                   # 执行、评分和功能验收
    ├── admin/                    # 数据集和邮箱初始化
    └── tests/                    # 案例离线回归
```

完成第 2 节的 Chat 镜像构建后，检查案例资产：

```bash
bash examples/deepagents_ov_context/scripts/container.sh cases case-data check
```

Chat 镜像包含案例代码、Dovecot、PHP、Roundcube 及办公工具，构建阶段核验案例资产和依赖。PPT 原稿、修改清单及参考答案随案例数据包提供；参考答案留在管理员目录。

### 7.2 邮箱与办公页面

端口在初始化前确定，生成配置与运行参数保持一致。初始化私有邮箱配置，再启用 Chat 容器内的案例服务：

```bash
bash examples/deepagents_ov_context/scripts/container.sh datasets-init
bash examples/deepagents_ov_context/scripts/container.sh datasets-up
```

| 服务 | 地址 | 数据 |
|---|---|---|
| 办公工作台 | `http://127.0.0.1:19482/` | SQLite 日程、报工及登录状态 |
| Roundcube | `http://127.0.0.1:19480/` | 邮件页面与正文 iframe |
| Dovecot | `127.0.0.1:19493` | IMAP 邮箱及邮件原文 |

`datasets-init` 生成测试账号、Dovecot 账号文件和 Roundcube 的 SQLite 配置，私有文件权限为 600。邮件守护进程的配置由容器入口根据代码模板生成在 `/run/ov-case-dovecot/`。`datasets-up` 将 `OV_CASES_ENABLED=true` 保存到 `compose.env`，重建 Chat 容器并等待全部服务就绪。后续 `up` 沿用该设置。邮件发送和删除操作由夹具配置关闭。

### 7.3 初始化 H/Q 工作区

每个案例的 H/Q 使用独立 namespace。准备脚本根据 `--case` 生成对应的六目标夹具，根据 `--phase` 选择任务与数据。

准备日程与报工工作区：

```bash
for case in schedule timesheet; do
  for phase in H Q; do
    suffix=$(printf '%s' "$phase" | tr '[:upper:]' '[:lower:]')
    namespace="$case-$suffix"
    bash examples/deepagents_ov_context/scripts/container.sh cases case-data prepare \
      --case "$case" --phase "$phase" --namespace "$namespace" \
      --registry /data/cases/registry --page-url http://127.0.0.1:19482 \
      --output "/data/cases/admin/$namespace-reference.json"
  done
done
```

邮件 H/Q 分别创建独立账号，再导入各自的 64 封合成邮件，其中包含 6 个目标及干扰邮件：

```bash
for phase in H Q; do
  suffix=$(printf '%s' "$phase" | tr '[:upper:]' '[:lower:]')
  namespace="mail-$suffix"
  bash examples/deepagents_ov_context/scripts/container.sh cases case-data mail-account \
    --mail-root /data/cases/mailbox --namespace "$namespace" \
    --output "/data/cases/admin/$namespace-mail.local.json"
  bash examples/deepagents_ov_context/scripts/container.sh cases case-data prepare \
    --case mail --phase "$phase" --namespace "$namespace" \
    --registry /data/cases/registry --page-url http://127.0.0.1:19482 \
    --mail-settings "/data/cases/admin/$namespace-mail.local.json" \
    --output "/data/cases/admin/$namespace-reference.json"
done
```

日程使用固定的会议时间与会议室占用；报工 H 登记 11 月 3 日、Q 登记 11 月 4 日；邮件 H/Q 的主题、日期、标识和正文独立。脚本采用模板对应的固定时钟，准备结果和管理员参考文件记录实际 case、phase、任务版本及执行任务。

打开 `http://127.0.0.1:19482/mock/login?workspace=schedule-h`，输入合成工号 `100001`，选择候选账号登录。其余案例替换 workspace。邮件工作区必须完成对应账号的数据导入后再执行。

手工验证使用单独演示工作区。正式运行使用新的 namespace，原生组和 OV 组使用内容一致、存储独立的数据。初始化和参考输出具有防覆盖检查。

### 7.4 执行日程、报工和邮件案例

测试输出默认保存到仓库 `result/office/`，通过 `--output` 可指定新目录。容器将宿主机 `$OV_DEPLOY_HOME/result/` 挂载到 `/opt/openviking/result/`，容器重建后保留结果。

三个案例使用同一 Chat HTTP/SSE 执行脚本，从 `benchmark/office/tasks/{case}/{phase}.md` 读取提示词并填入入口和工作区；`--phase` 与准备工作区的阶段一致。例如：

```bash
bash examples/deepagents_ov_context/scripts/container.sh cases python -m \
  benchmark.office.suites.run \
  --case schedule --phase H --workspace schedule-h --registry /data/cases/registry \
  --chat-url http://127.0.0.1:1934 --page-url http://127.0.0.1:19482 \
  --mode current
```

手工在 Chat 页面执行时，用同一入口打印任务后复制：

```bash
bash examples/deepagents_ov_context/scripts/container.sh cases python -m \
  benchmark.office.tasks --case mail --phase Q --workspace mail-q \
  --page-url http://127.0.0.1:19482
```

`current` 使用 OV，`baseline` 使用原生方案。修改部署端口后，同步调整 `--chat-url`。保存操作后快照：

```bash
bash examples/deepagents_ov_context/scripts/container.sh cases case-data inventory \
  --registry /data/cases/registry --namespace schedule-h \
  --output /data/cases/admin/schedule-h-after.json
```

- **日程**：六场会议的标题、日历、日期、起止时间及备注符合要求，会议室满足容量、设备、权限、优先级和占用约束；每项新增一条日程及对应预约。
- **报工**：六张卡片的日期、内容和时长正确，累计时长变化符合预期；交付六张截图及前后时长汇总文本。
- **邮件**：按主题、发件人、日期、最新版本及原件条件定位，H 交付 `history_01.txt` 至 `history_03.txt`，Q 交付 `mail_01.txt` 至 `mail_06.txt`；全文完整、文件可下载，保留原有邮件。

业务状态、文件和实际工具轨迹共同决定任务成功。Token 和时延读取 Chat 的统计与统一日志。准备 H 的成本单列，正式 Q 的成功、失败和超时样本均保留。

### 7.5 PPT 更新与资料生成文档

输入夹具随源码进入 Chat 镜像，工具和 Skill 已包含在镜像中。两类文件案例的默认定义如下，正式执行时保存实际选择的定义文件及哈希：

| 案例 | 定义文件 | 输入夹具 |
|---|---|---|
| PPT 更新 | `benchmark/office/cases/ppt_update/spec.json` | `benchmark/office/datasets/ppt_update/modify_test.pptx` |
| 资料生成文档 | `benchmark/office/cases/reqdoc_generation/spec.json` | `benchmark/office/datasets/reqdoc_generation/inputs/`，包含 Markdown、Word、JSON、Excel 和 PDF 六份业务资料，以及 `record_helpers.py` 辅助文件 |

**PPT 更新**：H/Q 分别从原稿副本开始，按定义文件中的页码、位置和目标值完成修改。模型通过 pptx Skill 更新文件，保持受保护内容与版式，并交付定义文件指定的 PPT。

**资料生成文档**：按定义文件向 H、原生 Q 和记忆 Q 上传相同的六份业务资料及 `record_helpers.py`，使用对应 H/Q 提示词生成文档。Q 提示可使用辅助函数，H 额外说明导入方式、事件处理顺序与原始记录核验方法；核对资料事实、指定章节、文件完整性和下载入口。辅助文件和提示词的 SHA-256 纳入案例指纹，更新后重新准备 H 记忆。

脚本执行时从镜像内夹具路径读取文件，通过 Chat 上传接口装载到本次 Run 的 `inputs/`。页面手工执行时，先从配套源码选择同一份输入材料并上传。管理员参考与评分文件保存在独立目录。宿主机 `data/cases/inputs/` 可保存按 H/Q 分开的输入副本，数据版本与定义文件保持一致。

### 7.6 记忆对照与数据管理

准备轮 H 业务通过后，使用 Middleware 的写入、Commit 和提取链路生成记忆，等待归档及索引就绪。Q 使用新 Session、独立工作区及 checkpoint，每个变体从相同记忆初态开始。

`data/cases/registry` 保存业务 SQLite，`admin` 保存参考与快照，宿主机 `result/` 保存执行证据。案例数据、Chat 状态和 OV 记忆分别持久化，整体备份在停止对应服务后执行。

远端人工查看案例页面时，增加两个端口转发：

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:19480:127.0.0.1:19480 \
  -L 127.0.0.1:19482:127.0.0.1:19482 用户名@服务器地址
```

## 8. 常见问题

| 现象 | 检查方式 |
|---|---|
| 构建下载失败 | 检查镜像源、openEuler 软件源、Python / Rust / Node / Skill 下载地址的网络；保留构建日志后重试 |
| 原生存储检查失败 | 确认镜像为 ARM64、源码构建成功，重新构建后执行 `container.sh check` |
| Chat 缺少浏览器或 Skill | 执行 `container.sh check` 检查镜像依赖；确认使用 Chat 目标镜像 |
| 挂载目录权限错误 | 使用数据目录所属账号启动，核对 UID / GID 与目录权限 |
| OV 不就绪 | 查看 `container.sh logs ov`，检查模型、Embedding、凭据和存储路径 |
| 端口被占用 | 初始化前通过 `compose.env` 确定 OV、Chat、IMAP 和 Webmail 端口；办公页面预留 19482 |
| 页面有回复但OV未同步 | 查看 `ov_degraded`、`ov_sync_pending` 和错误类型，核对远端及本地消息后显式恢复 |
| 文件校验或发布失败 | 保存执行记录；`artifact_publish` 错误保留 checkpoint，可在原会话重新生成文件 |
| 容器重建后缺少会话 | 确认沿用同一 `OV_DEPLOY_HOME`，且两份 SQLite 与 OV 数据挂载完整 |
| 案例资产校验失败 | 按 `case-data check` 输出核对仓库内完整案例目录及输入夹具 |
| 案例服务启动失败 | 检查 `data/cases/logs/`，核对初始化配置、端口和目录权限 |


### 浏览器接口与历史会话

新会话记录 `browser_contract_version=2`，普通命令使用官方 CLI，文件交付使用 browser_files。旧会话的执行配置和 checkpoint 保持原样；每 Run 只替换应用生成的浏览器说明，保留业务规则和用户自定义要求。旧接口会话的 download/pdf_text 和原截图调用由定向适配处理；当前官方接口保持原命令语义。运行事件 browser.compatibility 可核对接口版本与适配状态。

按会话测试方法记忆时，可在 execution 中指定 memory_detail（abstract、overview、full）、memory_budget_tokens 和 target_uris，并结合已有 memory_budget_chars 控制实际注入。配置随会话快照冻结，省略字段时沿用服务 Reader 配置。target_uris 使用业务需要的记忆类别目录；记忆类型、预算和实际注入内容应随实验条件保存。
