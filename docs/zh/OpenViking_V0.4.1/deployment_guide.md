# OpenViking 0.4.1 部署指南

本文说明 OpenViking 0.4.1 服务及 DeepAgents 接入所需运行环境的部署方法，通过单个容器安装 OV 服务、HTTP SDK 和接入依赖。SDK 初始化、Middleware 配置及 Agent 调用详见 [Quick Start](quick_start.md)。

## 1. 环境与配套

- 运行平台：鲲鹏 ARM64，openEuler 24.03 LTS SP3。
- OV 服务起配：1 个物理核、2 GiB 内存、20 GiB 本地 SSD 持久化空间；按消息、工具原文和记忆增长量扩容。
- 镜像构建建议：4 个物理核、8 GiB 内存、30 GiB 可用空间。构建过程使用 C/C++、Rust 和 Python 工具链。
- 网络：部署环境可访问源码、镜像及软件包源；运行环境可访问配套的生成模型与 Embedding 服务。
- 软件：宿主机 Docker Engine、Git、Python 3.10+；容器内 Python 3.11、Rust 1.91.1。交付源码的 `pyproject.toml` 在 `[project.optional-dependencies]` 下定义了名为 `deepagents` 的可选依赖组，包含 DeepAgents、LangChain、LangGraph 和 SQLite checkpoint 支持。
- 认证：OV 使用 `trusted`，应用校验 account、user 和 actor peer 后，通过每次请求携带身份。

安装宿主机依赖并启用 Docker：

```bash
sudo dnf install -y docker git python3 curl ca-certificates
sudo systemctl enable --now docker
sudo groupadd -f docker
sudo usermod -aG docker "$USER"
```

退出当前登录会话并重新登录，使用当前用户执行以下检查：

```bash
systemctl is-active docker
id -nG
docker info --format 'Server={{.ServerVersion}} Arch={{.Architecture}} OS={{.OSType}}'
docker run --rm openeuler/openeuler:24.03-lts-sp3 uname -m
```

确认结果：

1. `systemctl` 输出 `active`，表示 Docker 服务运行正常。
2. `id -nG` 输出包含 `docker`，表示当前登录会话已加载 Docker 用户组。
3. `docker info` 输出服务端版本、`Arch=aarch64`（或 `arm64`）及 `OS=linux`，表示当前用户可以访问 Docker 服务。
4. 容器命令输出 `aarch64` 且正常退出，表示基础镜像可获取、容器可正常运行。首次执行时会拉取该镜像。

## 2. 拉取开源基线并应用补丁

从 GitHub 官方仓库拉取 OpenViking `v0.4.1` 基线，从 GitCode 交付仓库获取补丁、管理脚本和文档：

```bash
git clone --depth 1 --branch v0.4.1 \
  https://github.com/volcengine/OpenViking.git openviking-upstream
git clone --branch master \
  https://gitcode.com/boostkit/OpenViking.git openviking-delivery
cd openviking-delivery
```

校验补丁及其在开源基线上的适用性，再以该基线创建源码目录并应用补丁：

```bash
python3 patches/OpenViking_V0.4.1/patch_manager.py verify \
  --source-tree ../openviking-upstream
python3 patches/OpenViking_V0.4.1/prepare_source.py \
  --source-tree ../openviking-upstream \
  --feature-set full-default --output-dir ../openviking-source
cd ../openviking-source
```

后续构建命令均在 `openviking-source` 目录执行。

## 3. 构建 OV 镜像

补丁源码包含 `Dockerfile.ov` 和配套的 `Dockerfile.ov.dockerignore`。在源码目录执行：

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.ov -t openviking-context:0.4.1 .
```

`Dockerfile.ov` 中的 `.[deepagents]` 表示安装当前目录的 OpenViking 源码，以及 `pyproject.toml` 中 `deepagents` 可选依赖组声明的第三方包。`langchain-openai` 和 `python-dotenv` 分别用于 Quick Start 的模型调用和环境文件加载。镜像因此包含 OV 服务、HTTP SDK 及完整的 DeepAgents 接入运行依赖；构建阶段同时检查 RAGFS 与 VectorDB 原生扩展。

## 4. 配置模型、认证和持久化

```bash
export OV_HOME="$HOME/openviking-runtime"
mkdir -p "$OV_HOME/config" "$OV_HOME/data"
chmod 700 "$OV_HOME/config"
```

创建 `$OV_HOME/config/runtime.env`，填写部署环境提供的模型名称、地址和凭据：

```dotenv
OV_SERVICE_KEY=填写服务凭据
OV_LLM_API_BASE=填写生成模型的OpenAI兼容接口地址
OV_LLM_MODEL=填写生成模型名称
OV_LLM_API_KEY=填写生成模型凭据
OV_EMBEDDING_API_BASE=填写Embedding接口地址
OV_EMBEDDING_MODEL=填写Embedding模型名称
OV_EMBEDDING_API_KEY=填写Embedding凭据
```

创建 `$OV_HOME/config/ov.conf`：

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 1933,
    "auth_mode": "trusted",
    "root_api_key": "$OV_SERVICE_KEY"
  },
  "storage": {
    "workspace": "/data",
    "agfs": {"backend": "local"},
    "vectordb": {"backend": "local"}
  },
  "vlm": {
    "provider": "openai",
    "api_base": "$OV_LLM_API_BASE",
    "model": "$OV_LLM_MODEL",
    "api_key": "$OV_LLM_API_KEY"
  },
  "embedding": {
    "dense": {
      "provider": "volcengine",
      "api_base": "$OV_EMBEDDING_API_BASE",
      "model": "$OV_EMBEDDING_MODEL",
      "api_key": "$OV_EMBEDDING_API_KEY",
      "dimension": 1024,
      "input": "multimodal"
    }
  }
}
```

Embedding 示例对应火山方舟多模态接口；模型、维度和输入类型按服务能力配套填写。生成模型使用 OpenAI 兼容接口。应用对话模型可以与 OV 的摘要模型分别配置。

```bash
chmod 600 "$OV_HOME/config/runtime.env" "$OV_HOME/config/ov.conf"
```

OV 加载配置时将环境变量展开到 JSON。模型凭据由容器运行时注入；配置目录只读挂载，消息、归档、工具原文和索引写入数据目录。

## 5. 启动与健康检查

```bash
docker run -d --name openviking-context --restart unless-stopped \
  --memory=2g --cpus=2 \
  --user "$(id -u):$(id -g)" \
  -p 127.0.0.1:1933:1933 \
  --env-file "$OV_HOME/config/runtime.env" \
  -v "$OV_HOME/config:/config:ro" \
  -v "$OV_HOME/data:/data" \
  openviking-context:0.4.1
curl --fail --silent http://127.0.0.1:1933/health
docker logs --tail 100 openviking-context
```

`--cpus=2` 为逻辑 CPU 时间配额。按物理核分配时，通过 `lscpu -e=CPU,CORE,SOCKET` 选择同一物理核对应的逻辑 CPU，并增加 `--cpuset-cpus=所选CPU列表`。跨主机访问时，将发布地址设置为应用可达的内网地址，并通过网络访问控制和 HTTPS 入口提供服务。

健康检查成功只说明服务可用；模型访问、写入、提交和召回由 Quick Start 验证。
