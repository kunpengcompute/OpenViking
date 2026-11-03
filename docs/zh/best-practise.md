# OpenViking 文件系统缓存最佳实践

这份最佳实践以 LocalFS（本地文件系统）为例，介绍如何给 OpenViking 开启 Redis 文件缓存，并做一次缓存开启前后的性能对比。先在测试环境跑通，再应用到实际业务。

## 1. 特性简介

Agent 记忆系统会反复读取小文件、摘要和目录。缓存层把这些可重复使用的数据放到 Redis，减少重复访问后端存储的开销。

- **小文件读取加速**：全量读取命中缓存时，直接返回缓存内容；未命中则读取后端并填充缓存。
- **目录和搜索加速**：缓存目录列表；开启 `cached_traversal` 后，tree 和 grep 可以复用目录、文件缓存。
- **写后更新**：经过 OpenViking 的写入、删除和移动会更新或失效相关缓存，后端仍是真实数据来源。
- **减少重复回源**：同一进程内，多个请求同时读取同一个未缓存对象时，会尝试合并回源请求。
- **可替换缓存系统**：Redis 为内置 Provider，也可以通过 DynamicProvider 接入其他缓存系统。

当前使用边界：同一个缓存 namespace 只由一个 OpenViking/RAGFS 进程写入，所有文件变更都经过 OpenViking，不要绕过它直接修改本地数据文件。分布式路径锁不能解除这一限制。

收益取决于命中率和网络开销。默认只缓存不超过 1 MiB 的完整文件；超大文件、范围读取和锁等控制文件不适合这类缓存。当前 `stat` 直接访问后端，`glob` 也直接转发后端，不应预设它们有 Redis 缓存收益。

## 2. 部署示例

入门时使用一台机器即可：本机运行 OpenViking、Locust 和 Redis，文件由 LocalFS 保存到本地磁盘。

```text
Locust -> OpenViking -> 文件系统缓存 -> Redis（缓存数据）
                                  \-> LocalFS（本地磁盘，原始数据）
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

### 2.2 安装 OpenViking

准备 Python 3.11+、Git、Rust/Cargo 和 C/C++ 构建工具，安装同一份源码及测试依赖：

```bash
export BENCH_ROOT="$HOME/ov-cache-bench"
mkdir -p "$BENCH_ROOT/data"
git clone --branch main https://github.com/volcengine/OpenViking.git "$BENCH_ROOT/OpenViking"
python3 -m venv "$BENCH_ROOT/venv"
source "$BENCH_ROOT/venv/bin/activate"
python -m pip install -U pip
python -m pip install -e "$BENCH_ROOT/OpenViking"
python -m pip install 'locust>=2,<3'
git -C "$BENCH_ROOT/OpenViking" rev-parse HEAD

openviking-server init
```

按初始化提示完成模型等基础配置，并记录 commit；两组测试都使用这个版本。源码安装会编译 Rust binding，不要只替换 Python 文件而保留旧 binding。

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
| `storage.agfs.cachefs.backend=local` | 关闭文件系统缓存，读取 LocalFS |
| `storage.agfs.cachefs.backend=cache` | 开启文件系统缓存，使用顶层 Redis Provider |
| `traversal_mode=cached_traversal` | 让 tree、grep 尝试复用缓存遍历 |
| `max_file_size_bytes=1048576` | 完整文件的入缓存上限为 1 MiB |
| `service_name` | 区分独立部署使用的缓存命名空间 |

这两个 `backend` 配置含义不同：前者选择文件存储，后者控制是否启用缓存。本例无论缓存开关如何，原始文件始终保存在 LocalFS。

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

正式数据集为 **10 万个 1 KiB 文件，分布在 25 个目录，每目录 4000 个数据文件**。脚本通过 OpenViking API 填充并校验，由 LocalFS 保存；不要手工向数据目录写文件。第一次使用时可先把三个命令都改成 `--files 100 --per-dir 100`，用独立根 URI 跑通，再准备正式数据集。

### 4.2 先测 cache off，再测 cache on

| Workload | 测试内容 |
| --- | --- |
| `content_read` | 按固定清单读取文件正文 |
| `fs_ls` | 列出单层目录 |
| `fs_tree` | 查询目录树；本脚本限定为一层 |
| `grep` | 用不同的稀疏 token 搜索文件内容 |
| `glob` | 按文件名模式匹配，作为后端直通对照项 |

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

1. 停止本轮 OpenViking，将 `cachefs.backend` 从 `local` 改成 `cache`，重启并确认就绪。保留相同版本、workspace、本地文件和 URI 清单。
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

在测试窗口前后各抓一次 `/metrics`：`requests_total` 的 `result="hit"` 增加，表示命中文件或目录缓存；`backend_fallbacks_total` 增加，表示缓存未命中后成功读取 LocalFS。它不覆盖所有后端操作，也不等于底层磁盘 I/O 次数。Redis 自身的命中数同样不能代替文件缓存命中率。

LocalFS 本身可能很快，尤其文件已经进入操作系统页缓存时，访问 Redis 的开销可能抵消收益，甚至增加时延，应如实记录。两轮使用相同磁盘和数据，不额外清理某一组的系统页缓存。测试时还要留意 Redis 内存及淘汰、本机 CPU 和网络。目录可能包含隐藏元数据，总 entry 超过 4096 时不会缓存该目录。

配套测试脚本位于 `ov_cache_test` 目录。更多指标操作可参考 [/metrics 实测说明](ov_cache_test/metrics-cache-guide.md)，缓存配置说明见 [RAGFS 缓存指南](https://github.com/volcengine/OpenViking/blob/main/docs/zh/guides/14-ragfs-cache.md)。
