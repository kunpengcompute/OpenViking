# OpenViking Filesystem Cache Best Practices

This guide uses LocalFS (the local filesystem) to show how to enable Redis-backed file caching in OpenViking and compare performance with and without caching. Validate the setup in a test environment before applying it to production workloads.

## 1. Feature Overview

Agent memory systems repeatedly read small files, summaries, and directories. The cache layer stores this reusable data in Redis to reduce the overhead of repeated access to backend storage.

- **Faster small-file reads**: When a full-file read hits the cache, the cached content is returned directly. On a miss, the file is read from the backend and added to the cache.
- **Faster directory listing and search**: Directory listings are cached. With `cached_traversal` enabled, tree and grep operations can reuse directory and file caches.
- **Updates after writes**: Writes, deletions, and moves performed through OpenViking update or invalidate the relevant cache entries. The backend remains the source of truth.
- **Fewer duplicate backend reads**: Within a single process, concurrent requests for the same uncached object attempt to coalesce their backend reads.
- **Replaceable cache systems**: Redis is the built-in provider. Other cache systems can be integrated through DynamicProvider.

Current usage constraints: Only one OpenViking/RAGFS process may write to a given cache namespace, and all file changes must go through OpenViking. Do not modify local data files directly. Distributed path locks do not remove this restriction.

Benefits depend on the hit rate and network overhead. By default, only complete files of up to 1 MiB are cached. Very large files, range reads, and control files such as lock files are unsuitable for this type of caching. Currently, `stat` accesses the backend directly, and `glob` is also forwarded directly to the backend. Do not assume these operations benefit from Redis caching.

## 2. Deployment Example

A single machine is sufficient to get started: Run OpenViking, Locust, and Redis locally, with LocalFS storing files on the local disk.

```text
Locust -> OpenViking -> Filesystem cache -> Redis (cached data)
                                       \-> LocalFS (local disk, source data)
```

### 2.1 Start Redis

After installing Docker, run the following in an isolated test environment. Set the password and image version to suit your environment. This example allows only local connections to Redis.

```bash
export REDIS_PASSWORD='<Redis test password>'
export REDIS_DATA_DIR="$HOME/ov-cache-redis"
mkdir -p "$REDIS_DATA_DIR"

docker run -d --name ov-cache-redis -p 127.0.0.1:6379:6379 \
  -v "$REDIS_DATA_DIR:/data" -e REDIS_PASSWORD \
  redis:7.2-alpine \
  sh -c 'exec redis-server --requirepass "$REDIS_PASSWORD"'
```

If a container with the same name already exists or the port is in use, choose a separate name and port for this test run. You can also use an existing Redis instance. If Redis runs on another machine, adjust its listening address, network access, and the connection address used below.

### 2.2 Install OpenViking

Prepare Python 3.11+, Git, Rust/Cargo, and C/C++ build tools, then install the source code and test dependencies:

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

Follow the initialization prompts to configure models and other basic settings, and record the commit. Use this same version for both sets of tests. Installing from source compiles the Rust bindings; do not replace only the Python files while retaining old bindings.

## 3. Configuration

Merge the following configuration into the generated `~/.openviking/ov.conf`, keeping your existing model settings. Replace `<...>` with actual values; JSON does not automatically expand shell environment variables. Set `workspace` to the absolute path of the local data directory created above to avoid accessing a different dataset when the server is launched from another directory.

```json
{
  "service_name": "ov-cache-bench",
  "server": {
    "host": "127.0.0.1",
    "port": 1933,
    "auth_mode": "trusted",
    "root_api_key": "<API key for this test run>",
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
    "workspace": "<absolute path to the local data directory>",
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

Caching is initially disabled to establish a test baseline:

| Setting | Purpose |
| --- | --- |
| `storage.agfs.backend=local` | Use LocalFS to store files in the local directory managed by the workspace |
| `storage.agfs.cachefs.backend=local` | Disable filesystem caching and read from LocalFS |
| `storage.agfs.cachefs.backend=cache` | Enable filesystem caching using the top-level Redis provider |
| `traversal_mode=cached_traversal` | Allow tree and grep to attempt traversal using cached data |
| `max_file_size_bytes=1048576` | Limit caching to complete files of up to 1 MiB |
| `service_name` | Distinguish cache namespaces used by separate deployments |

The two `backend` settings serve different purposes: The first selects file storage, while the second controls whether caching is enabled. In this example, source files are always stored in LocalFS, regardless of the cache setting.

Set the same password as the Redis container, then start the server:

```bash
export REDIS_PASSWORD='<Redis test password>'
export OV_HOST='http://127.0.0.1:1933'
export OPENVIKING_API_KEY='<same test API key as in ov.conf>'
openviking-server doctor
openviking-server --config ~/.openviking/ov.conf
```

In another terminal, activate the same virtual environment, set the same environment variables, and run `curl -fsS "$OV_HOST/ready"`. Start testing only after the server is ready. If storage connections or model initialization fail, check the server logs first.

## 4. Performance Testing

### 4.1 Prepare Scripts and Data

Prepare `dataset.py`, `locust_cache.py`, and `summarize.py` from the accompanying directory. Run all subsequent commands in the virtual environment on the machine running OpenViking:

```bash
export BENCH_SCRIPTS='<directory containing the scripts above>'
export DATASET_ROOT_URI='viking://resources/cache_bench_001'
mkdir -p "$BENCH_ROOT/manifests" "$BENCH_ROOT/results"

python "$BENCH_SCRIPTS/dataset.py" manifest --root-uri "$DATASET_ROOT_URI" \
  --out "$BENCH_ROOT/manifests" --files 100000 --per-dir 4000
python "$BENCH_SCRIPTS/dataset.py" fill --root-uri "$DATASET_ROOT_URI" --host "$OV_HOST" \
  --files 100000 --per-dir 4000
python "$BENCH_SCRIPTS/dataset.py" verify --root-uri "$DATASET_ROOT_URI" --host "$OV_HOST" \
  --files 100000 --per-dir 4000
```

The full dataset contains **100,000 files of 1 KiB each, distributed across 25 directories with 4,000 data files per directory**. The scripts populate and verify the dataset through the OpenViking API, with LocalFS handling storage. Do not write files into the data directory manually. For an initial trial, change all three commands to `--files 100 --per-dir 100` and use a separate root URI. Prepare the full dataset after that trial succeeds.

### 4.2 Test with the Cache Off, Then On

| Workload | Test operation |
| --- | --- |
| `content_read` | Read file contents using a fixed manifest |
| `fs_ls` | List a single directory level |
| `fs_tree` | Query a directory tree; this script limits traversal to one level |
| `grep` | Search file contents using different sparse tokens |
| `glob` | Match filename patterns as a control operation that passes directly through to the backend |

The following example runs `content_read` with one user for five minutes:

```bash
export CACHE_MODE=off WORKLOAD=content_read LOCUST_USERS=1
export INPUT_MANIFEST="$BENCH_ROOT/manifests/${WORKLOAD}.jsonl"
export RUN_MODE=cycle TARGET_QPS=0 VERIFY_EVERY=100
export FINAL_STATS_FILE="$BENCH_ROOT/results/${CACHE_MODE}_${WORKLOAD}_u${LOCUST_USERS}_final.json"

locust -f "$BENCH_SCRIPTS/locust_cache.py" --headless \
  -u "$LOCUST_USERS" -r "$LOCUST_USERS" -t 300s --host "$OV_HOST" \
  --csv "$BENCH_ROOT/results/${CACHE_MODE}_${WORKLOAD}_u${LOCUST_USERS}"
```

Repeat the entire command block for each workload in the table, setting `LOCUST_USERS` to **1, 8, and 32** in turn. This produces 15 sets of results with the cache off. `CACHE_MODE` only controls result filenames; the actual cache setting must be changed in `ov.conf`.

Then test with the cache on in this order:

1. Stop the OpenViking instance for this test run, change `cachefs.backend` from `local` to `cache`, restart it, and confirm readiness. Keep the same version, workspace, local files, and URI manifests.
2. Read all test files and directories once to warm the cache, using the commands below.
3. Change `CACHE_MODE` in the command block above to `on`, then repeat the tests using the same five workloads and three concurrency levels.

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

The warm-up commands exit after processing the entire manifest. Before starting the measurement window, confirm that warm-up failures are zero, the read count is 100,000, and the ls count is 25. Exclude warm-up data from the A/B summary, and do not restart the server after warming the cache.

### 4.3 Review Results and Cache Metrics

After completing **5 workloads × 3 concurrency levels × 2 modes = 30 runs**, summarize the results:

```bash
python "$BENCH_SCRIPTS/summarize.py" "$BENCH_ROOT/results"

curl -fsS -H "X-API-Key: $OPENVIKING_API_KEY" "$OV_HOST/metrics" \
  | grep '^openviking_ragfs_cache_'
```

First confirm that responses are correct and requests have few or no failures. Then compare average latency and p95 for the same workload and concurrency level:

```text
Latency improvement = (cache-off latency - cache-on latency) / cache-off latency × 100%
```

Capture `/metrics` once before and once after the test window. An increase in `requests_total` with `result="hit"` indicates file or directory cache hits. An increase in `backend_fallbacks_total` indicates successful reads from LocalFS after cache misses. This metric does not cover all backend operations and does not equal the number of underlying disk I/O operations. Redis's own hit count is also no substitute for the file cache hit rate.

LocalFS may already be fast, especially when files are in the operating system's page cache. Redis access overhead may offset the benefits or even increase latency; record these results accurately. Use the same disk and data for both rounds, and do not clear the system page cache for only one group. Also monitor Redis memory usage and evictions, local CPU usage, and network activity. Directories may contain hidden metadata; a directory is not cached if its total entry count exceeds 4,096.

The accompanying test scripts are in the `ov_cache_test` directory. For more on working with metrics, see the [/metrics Measurement Guide](ov_cache_test/metrics-cache-guide.md). For cache configuration, see the [RAGFS Cache Guide](https://github.com/volcengine/OpenViking/blob/main/docs/zh/guides/14-ragfs-cache.md).
