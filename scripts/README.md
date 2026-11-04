# OpenViking CI 脚本

运行环境使用 `master` 分支 `.gitcode/dockerfiles/Dockerfile` 构建的镜像，或直接拉取：

```bash
docker pull swr.cn-north-4.myhuaweicloud.com/cloud_boostkit/openeuler24.03_lts_sp3:OpenViking_26.3.0.04
docker run --rm --cpus=8 --memory=24g \
  -v "$PWD:/work" -w /work \
  swr.cn-north-4.myhuaweicloud.com/cloud_boostkit/openeuler24.03_lts_sp3:OpenViking_26.3.0.04 \
  bash -lc 'bash build_ci.sh && OV_SKIP_BUILD=1 bash ut_ci.sh'
```

在 dev 分支的流水线容器内执行：

```bash
bash build_ci.sh
OV_SKIP_BUILD=1 bash ut_ci.sh
```

`build_ci.sh` 按 `uv.lock` 同步依赖，通过项目构建后端编译 Rust CLI、RAGFS 绑定和 C++ VectorDB，生成 wheel，并验证原生模块加载及 wheel 内的必需产物。`ut_ci.sh` 默认自行构建后执行 `tests/unit` 全集；同一作业已完成编译时通过 `OV_SKIP_BUILD=1` 复用产物。

| 参数 | 默认值 | 用途 |
|---|---|---|
| `OV_SOURCE_DIR` | 脚本所在仓库 | 指定待验证源码目录 |
| `UV_PROJECT_ENVIRONMENT` | 镜像 `/opt/ov-venv`，其他环境使用仓库 `.venv` | 复用 Python 虚拟环境 |
| `OV_SKIP_UV_SYNC` | `0` | 设为 `1` 复用流水线已按 lock 同步的依赖 |
| `CI_BUILD_JOBS` | `4` | Rust 和 C++ 编译并行度 |
| `OV_SKIP_STUDIO_BUILD` | `1` | 设为 `0` 同时构建 Web Studio，需要 npm |
| `OPENVIKING_VERSION` | 项目 SCM 版本 | 源码包验证时指定版本，并同步 Python 和 Rust CLI 的版本 |
| `REPORT_DIR` | `build/ci-reports` | 编译日志、构建摘要、wheel、UT 日志、JUnit、覆盖率 XML 和 HTML |
| `UT_SCOPE` | `unit` | `gate` 追加已有隔离集成与 SDK 身份用例；`full` 执行 `tests` 全集 |
| `UT_WORKERS` | `1` | 大于 `1` 时使用 pytest-xdist 按模块分配测试 |

指定用例及 pytest 参数：

```bash
OV_SKIP_BUILD=1 bash ut_ci.sh tests/unit/ov_context -x
```

测试失败、收集失败或零用例均保留 pytest 退出码，作为门禁失败处理。`full` 范围使用对应集成测试所需的服务与配置。流水线调用 dev 分支仓库根目录的 `build_ci.sh` 和 `ut_ci.sh`。
