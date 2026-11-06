# OpenViking 文件系统缓存最佳实践

这份最佳实践介绍如何从 OpenViking **v0.5.0 官方基线**应用 CacheFS 特性补丁，开启 Redis 文件与查询缓存，并做一次缓存开启前后的性能对比。后端可以选择 LocalFS（本地文件系统），也可以使用本机部署的 MinIO。先在测试环境跑通，再应用到实际业务。

## 1. 特性简介

Agent 记忆系统会反复读取小文件、摘要和目录。缓存层把这些可重复使用的数据放到 Redis，减少重复访问后端存储的开销。

- **小文件读取加速**：全量读取命中缓存时，直接返回缓存内容；未命中则读取后端并填充缓存。
- **目录和搜索加速**：复用原始目录列表缓存，并缓存 stat、tree、glob、grep 的参数化查询结果；开启 `cached_traversal` 后，遍历还可以复用目录和文件缓存。
- **写后更新**：经过 OpenViking 的写入、删除和移动会更新或失效相关缓存，后端仍是真实数据来源。
- **减少重复回源**：同一进程内，多个请求同时读取同一个未缓存对象时，会尝试合并回源请求。
- **可替换缓存系统**：Redis 为内置 Provider，也可以通过 DynamicProvider 接入其他缓存系统。

当前使用边界：同一个缓存 namespace 只由一个 OpenViking/RAGFS 进程写入，所有文件变更都经过 OpenViking，不要绕过它直接修改本地数据文件或 MinIO 对象。分布式路径锁不能解除这一限制。

收益取决于命中率和网络开销。默认只缓存不超过 1 MiB 的完整文件；超大文件、范围读取和锁等控制文件不适合这类缓存。补丁使用二进制 envelope v2 保存文件与目录；grep 每批最多 MGET 256 个正文 key，与 `node_limit` 无关，并复用 `Bytes` 载荷、整批解码及去重后的 generation 校验；tree 在单次请求内复用 generation。query bucket 与正文分开保存，按受影响路径失效，失效查询结果不会顺带删除正文缓存。

这些缓存只在 `storage.agfs.cachefs.backend=cache` 且顶层 CacheRuntime 配置有效时启用。查询缓存未命中时执行原查询流程；`cached_traversal` 仍可能命中文件或目录缓存，不等于每次都直接访问存储后端。缓存不作为事实数据源，返回内容仍需满足原有权限和一致性检查。

## 2. 部署示例

入门时使用一台机器即可：本机运行 OpenViking、Locust 和 Redis，后端二选一。使用 MinIO 时，MinIO 也部署在本机，通过 S3 API 访问本地持久化目录。

```text
Locust -> OpenViking -> 文件系统缓存 -> Redis（缓存数据）
                                  \-> LocalFS（本地磁盘，原始数据）
                                  或 S3FS -> MinIO（本机对象存储，原始数据）
```

### 2.1 启动 Redis

安装 Docker 后，在独立测试环境执行。以下密码和镜像版本应按实际环境设置，示例只允许本机连接 Redis。

```bash
export REDIS_PASSWORD='<Redis测试密码>'
export REDIS_DATA_DIR="$HOME/ov-cache-redis"
mkdir -p "$REDIS_DATA_DIR"

docker run -d --name ov-cache-redis -p 127.0.0.1:6379:6379 \
  -v "$REDIS_DATA_DIR:/data" -e REDIS_PASSWORD \
  redis:7.2-alpine \
  sh -c 'exec redis-server --requirepass "$REDIS_PASSWORD"'
```

已有同名容器或端口被占用时，换用本轮独立名称和端口。也可以使用已有 Redis；如果部署在其他机器上，需要调整 Redis 监听地址、网络访问和后文的连接地址。

### 2.2 从 v0.5.0 基线应用补丁并安装

准备 Python 3.11+、Git、Rust/Cargo（至少 1.91.1）、CMake 和 C/C++ 构建工具。特性补丁位于本交付仓库的 [patches/OpenViking_V0.5.0/cachefs-query-cache.patch](../../../patches/OpenViking_V0.5.0/cachefs-query-cache.patch)，对应 [PR #20](https://gitcode.com/boostkit/OpenViking/pull/20) 的 CacheFS 修改，源提交为 `0c7153840f625305486092a05d522b76a610bf23`。

补丁以官方 `v0.5.0` 标签（commit `20f9c11362ad16193b0daa4a414e1095061af427`）为目标，合并了 9 个特性文件的最终差异，不包含 PR 中的基线导入、GitCode CI 和文档集成改动。先将补丁保存在待安装源码目录之外，再执行：

```bash
export BENCH_ROOT="$HOME/ov-cache-bench"
export CACHE_PATCH='<本交付仓库绝对路径>/patches/OpenViking_V0.5.0/cachefs-query-cache.patch'
mkdir -p "$BENCH_ROOT/data"
git clone --branch v0.5.0 --single-branch https://github.com/volcengine/OpenViking.git "$BENCH_ROOT/OpenViking"
git -C "$BENCH_ROOT/OpenViking" switch -c v0.5.0-cachefs
git -C "$BENCH_ROOT/OpenViking" rev-parse HEAD

# 先检查能否干净应用，再应用补丁；任何一步失败时先处理错误，不继续安装。
git -C "$BENCH_ROOT/OpenViking" apply --check "$CACHE_PATCH"
git -C "$BENCH_ROOT/OpenViking" apply "$CACHE_PATCH"
git -C "$BENCH_ROOT/OpenViking" diff --check
git -C "$BENCH_ROOT/OpenViking" diff --stat

python3 -m venv "$BENCH_ROOT/venv"
source "$BENCH_ROOT/venv/bin/activate"
python -m pip install -U pip
OV_REQUIRE_RAGFS_BUILD=1 python -m pip install -v -e "$BENCH_ROOT/OpenViking"
python -m pip install 'locust>=2,<3'

openviking-server init
```

确认打印的 HEAD 与上述基线 commit 一致。按初始化提示完成模型等基础配置，保存基线 commit、补丁文件及其 SHA-256；两组测试使用相同补丁和构建。源码安装会重新编译 Rust binding，默认包含 S3 和 CacheFS 所需的 `s3`、`ragfs/cache` 特性；`OV_REQUIRE_RAGFS_BUILD=1` 要求 binding 构建成功，不能只替换 Python 文件而保留旧 binding，也不要设置 `OV_SKIP_RAGFS_BUILD=1`。应用补丁不会自动修改配置，仍需按第 3 节启用缓存。

### 2.3 本地部署 MinIO（选择 S3 后端时）

选择 LocalFS 时跳过本节。以下示例使用单机 MinIO，API 和管理端口只绑定 loopback，数据保存到本地目录。镜像与测试实例独立，MinIO 凭据至少使用 8 位随机密码，不复用业务凭据：

```bash
export MINIO_ROOT_USER='ov-cache-bench'
export MINIO_ROOT_PASSWORD='<至少8位的MinIO测试密码>'
export MINIO_DATA_DIR="$BENCH_ROOT/minio-data"
export MINIO_BUCKET='ov-cache-bench'
mkdir -p "$MINIO_DATA_DIR"

docker run -d --name ov-cache-minio \
  -p 127.0.0.1:9000:9000 -p 127.0.0.1:9001:9001 \
  -e MINIO_ROOT_USER -e MINIO_ROOT_PASSWORD \
  -v "$MINIO_DATA_DIR:/data" \
  minio/minio:RELEASE.2025-04-22T22-12-26Z \
  server /data --address ':9000' --console-address ':9001'

curl -fsS http://127.0.0.1:9000/minio/health/ready
```

MinIO 就绪后，在前面创建的 venv 中创建专用 bucket。启动 OpenViking 前完成此步骤，S3FS 不会替你创建 bucket：

```bash
python -m pip install boto3
python - <<'PY'
import os
import boto3
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:9000",
    aws_access_key_id=os.environ["MINIO_ROOT_USER"],
    aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
    region_name="us-east-1",
    config=Config(s3={"addressing_style": "path"}),
)
bucket = os.environ["MINIO_BUCKET"]
s3.create_bucket(Bucket=bucket)
s3.head_bucket(Bucket=bucket)
print("MinIO bucket ready:", bucket)
PY
```

容器名或端口已被占用时使用独立名称和端口，并同步调整第 3 节的 endpoint。MinIO 管理界面地址为 `http://127.0.0.1:9001`；正式文件数据后续通过 OpenViking API 写入，避免直接修改对象导致缓存失效机制被绕过。

## 3. 如何配置

在生成的 `~/.openviking/ov.conf` 中合并下面的配置，保留已有模型配置。将 `<...>` 替换为实际值，JSON 不会自动展开 shell 环境变量。`workspace` 填写上一步创建的本地数据目录的绝对路径，避免更换启动目录后读到另一份数据。

```json
{
  "service_name": "ov-cache-bench",
  "server": {
    "host": "127.0.0.1",
    "port": 1933,
    "auth_mode": "trusted",
    "root_api_key": "<本轮测试API密钥>",
    "observability": {"metrics": {"enabled": true}}
  },
  "cache": {
    "provider": "redis",
    "params": {
      "mode": "standalone",
      "endpoints": ["redis://127.0.0.1:6379"],
      "password_env": "REDIS_PASSWORD",
      "pool_size": 32,
      "connect_timeout_ms": 1000,
      "command_timeout_ms": 1000,
      "default_ttl_seconds": 86400
    }
  },
  "storage": {
    "workspace": "<本地数据目录的绝对路径>",
    "vectordb": {"backend": "local"},
    "agfs": {
      "backend": "local",
      "cachefs": {
        "backend": "local",
        "max_file_size_bytes": 1048576,
        "traversal_mode": "cached_traversal"
      },
      "queuefs": {"backend": "sqlite"},
      "pathlock": {"provider": "filesystem"}
    }
  }
}
```

这里先关闭缓存，便于建立测试基线：

| 设置 | 作用 |
| --- | --- |
| `storage.agfs.backend=local` | 使用 LocalFS，将文件保存到 workspace 管理的本地目录 |
| `storage.agfs.cachefs.backend=local` | 关闭文件系统缓存，直通选定的 LocalFS 或 S3FS 后端 |
| `storage.agfs.cachefs.backend=cache` | 开启文件系统缓存，使用顶层 Redis Provider |
| `traversal_mode=cached_traversal` | 查询结果未命中时，让 tree、glob、grep 尝试复用缓存遍历 |
| `max_file_size_bytes=1048576` | 完整文件的入缓存上限为 1 MiB |
| `service_name` | 区分独立部署使用的缓存命名空间 |

这两个 `backend` 配置含义不同：前者选择文件存储，后者控制是否启用缓存。缓存开关不改变事实数据源，原始文件保存在选定的 LocalFS 或 MinIO 中。

若使用第 2.3 节的 MinIO，将上面配置中的 `storage.agfs.backend` 改为 `s3`，并增加以下 `s3` 配置；保留已有 `cachefs`、`queuefs`、`pathlock` 及模型配置：

```json
{
  "storage": {
    "agfs": {
      "backend": "s3",
      "s3": {
        "bucket": "ov-cache-bench",
        "region": "us-east-1",
        "access_key": "<与MINIO_ROOT_USER相同的值>",
        "secret_key": "<与MINIO_ROOT_PASSWORD相同的值>",
        "endpoint": "http://127.0.0.1:9000",
        "prefix": "cache-bench",
        "use_ssl": false,
        "use_path_style": true
      }
    }
  }
}
```

JSON 中填写实际凭据，不能直接写 `${MINIO_ROOT_PASSWORD}` 期待自动展开；凭据配置不要提交到代码仓库。MinIO 的对象数据保存在 `MINIO_DATA_DIR`，`storage.workspace` 仍用于本地向量库、队列和运行状态，因此切换到 S3 后也必须保留独立、可写的 workspace。上例 QueueFS 仍使用 SQLite，PathLock 仍使用 filesystem。

设置与 Redis 容器相同的密码后启动服务：

```bash
export REDIS_PASSWORD='<Redis测试密码>'
export OV_HOST='http://127.0.0.1:1933'
export OPENVIKING_API_KEY='<与ov.conf相同的测试API密钥>'
openviking-server doctor
openviking-server --config ~/.openviking/ov.conf
```

在另一个终端激活同一 venv、设置相同环境变量，执行 `curl -fsS "$OV_HOST/ready"`。服务就绪后再测试；存储连接或模型初始化失败时先检查服务日志。

## 4. 如何做性能测试

### 4.1 准备脚本与数据

准备配套目录中的 `dataset.py`、`locust_cache.py` 和 `summarize.py`。后续命令都在运行 OpenViking 的本机 venv 内执行：

```bash
export BENCH_SCRIPTS='<上述脚本所在目录>'
export DATASET_ROOT_URI='viking://resources/cache_bench_001'
mkdir -p "$BENCH_ROOT/manifests" "$BENCH_ROOT/results"

python "$BENCH_SCRIPTS/dataset.py" manifest --root-uri "$DATASET_ROOT_URI" \
  --out "$BENCH_ROOT/manifests" --files 100000 --per-dir 4000
python "$BENCH_SCRIPTS/dataset.py" fill --root-uri "$DATASET_ROOT_URI" --host "$OV_HOST" \
  --files 100000 --per-dir 4000
python "$BENCH_SCRIPTS/dataset.py" verify --root-uri "$DATASET_ROOT_URI" --host "$OV_HOST" \
  --files 100000 --per-dir 4000
```

正式数据集为 **10 万个 1 KiB 文件，分布在 25 个目录，每目录 4000 个数据文件**。脚本通过 OpenViking API 填充并校验，由选定的 LocalFS 或 MinIO 保存；不要手工向数据目录或 bucket 写文件。第一次使用时可先把三个命令都改成 `--files 100 --per-dir 100`，用独立根 URI 跑通，再准备正式数据集。

### 4.2 先测 cache off，再测 cache on

| Workload | 测试内容 |
| --- | --- |
| `content_read` | 按固定清单读取文件正文 |
| `fs_ls` | 列出单层目录 |
| `fs_tree` | 查询目录树；本脚本限定为一层 |
| `grep` | 用不同的稀疏 token 搜索文件内容 |
| `glob` | 按文件名模式匹配，验证查询结果缓存及遍历复用 |

下面以 1 用户的 `content_read` 为例，每次运行 5 分钟：

```bash
export CACHE_MODE=off WORKLOAD=content_read LOCUST_USERS=1
export INPUT_MANIFEST="$BENCH_ROOT/manifests/${WORKLOAD}.jsonl"
export RUN_MODE=cycle TARGET_QPS=0 VERIFY_EVERY=100
export FINAL_STATS_FILE="$BENCH_ROOT/results/${CACHE_MODE}_${WORKLOAD}_u${LOCUST_USERS}_final.json"

locust -f "$BENCH_SCRIPTS/locust_cache.py" --headless \
  -u "$LOCUST_USERS" -r "$LOCUST_USERS" -t 300s --host "$OV_HOST" \
  --csv "$BENCH_ROOT/results/${CACHE_MODE}_${WORKLOAD}_u${LOCUST_USERS}"
```

将 `WORKLOAD` 换成表中的其他项目，将 `LOCUST_USERS` 分别设为 **1、8、32**，每次重新执行整个命令块。这样得到 cache off 的 15 组结果。`CACHE_MODE` 只控制结果文件名，实际开关必须修改 `ov.conf`。

然后按以下顺序测试 cache on：

1. 停止本轮 OpenViking，将 `cachefs.backend` 从 `local` 改成 `cache`，重启并确认就绪。保留相同版本、补丁、workspace、后端数据和 URI 清单；使用 MinIO 时也保留同一个 bucket/prefix，不在两组之间切换文件后端。
2. 先将所有测试文件和目录读取一遍，让缓存进入热态，命令见下方。
3. 将上面命令块中的 `CACHE_MODE` 改成 `on`，用相同的五类 workload 和三个并发档位再测一遍。

```bash
export RUN_MODE=once TARGET_QPS=0 VERIFY_EVERY=1000
export WORKLOAD=content_read LOCUST_USERS=32
export INPUT_MANIFEST="$BENCH_ROOT/manifests/content_read.jsonl"
export FINAL_STATS_FILE="$BENCH_ROOT/results/warm_content_read_final.json"
locust -f "$BENCH_SCRIPTS/locust_cache.py" --headless -u 32 -r 32 --host "$OV_HOST" \
  --csv "$BENCH_ROOT/results/warm_content_read"

export WORKLOAD=fs_ls LOCUST_USERS=1
export INPUT_MANIFEST="$BENCH_ROOT/manifests/fs_ls.jsonl"
export FINAL_STATS_FILE="$BENCH_ROOT/results/warm_fs_ls_final.json"
locust -f "$BENCH_SCRIPTS/locust_cache.py" --headless -u 1 -r 1 --host "$OV_HOST" \
  --csv "$BENCH_ROOT/results/warm_fs_ls"
```

预热命令会在清单读完后退出。检查预热失败数为 0、read 数为 100000、ls 数为 25，再进入正式窗口；预热数据不参与 A/B 汇总，也不要在预热后重启服务。

### 4.3 看结果和缓存指标

跑齐 **5 workload × 3 并发 × 2 模式 = 30 组**结果后汇总：

```bash
python "$BENCH_SCRIPTS/summarize.py" "$BENCH_ROOT/results"

curl -fsS -H "X-API-Key: $OPENVIKING_API_KEY" "$OV_HOST/metrics" \
  | grep '^openviking_ragfs_cache_'
```

先确认返回内容正确、请求基本无失败，再比较同 workload、同并发的平均延迟和 p95：

```text
延迟收益 = (关闭缓存时延 - 开启缓存时延) / 关闭缓存时延 × 100%
```

在测试窗口前后各抓一次 `/metrics`：`requests_total` 的 `result="hit"` 增加，表示命中文件或目录缓存；`backend_fallbacks_total` 增加，表示缓存未命中后成功读取所选后端。它不覆盖所有后端操作，也不等于底层磁盘 I/O 次数。Redis 自身的命中数包含正文、目录、generation 和查询 bucket，不能直接代替文件缓存或查询缓存命中率。

LocalFS 本身可能很快，尤其文件已经进入操作系统页缓存时，访问 Redis 的开销可能抵消收益，甚至增加时延，应如实记录。本地 MinIO 的数据也可能命中 OS page cache，关闭 CacheFS 后仍需经过 S3 HTTP 与 MinIO 处理；S3FS 自带的目录/stat 进程缓存同样可能生效，cache off 不等于完全无缓存。两组保持相同后端、数据和 S3FS 配置，不额外清理某一组的系统页缓存。

上述流程比较未全量预热的 off 与已预热的 on，应记录这种差异，不将全部收益归因于 Redis；tree/glob/grep 的首轮填充与重复查询命中也应分开分析。本地 MinIO 的结果不直接代表远程 S3 的网络时延。测试时还要留意 Redis 内存及淘汰、本机 CPU 和网络。目录可能包含隐藏元数据，总 entry 超过 4096 时不会缓存原始目录列表。
