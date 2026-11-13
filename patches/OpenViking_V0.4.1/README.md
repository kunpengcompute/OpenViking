# OpenViking 0.4.1 补丁

## 基线与范围

基线为官方 `v0.4.1` 提交 `740b7fcbe7890f442fa8b7ad775d74ae1bc6b084`。`manifest.json` 记录补丁分组、交付源码 tree 与开发来源，`SOURCE_COMMIT` 记录开发提交，`SHA256SUMS` 覆盖交付文件。

`context-integration` 包含 OpenViking 库、原生 HTTP SDK、配套存储实现和构建依赖。导出规则显式选择 `openviking/`、`openviking_cli/`、`src/`、RAGFS 及构建文件（含 `Dockerfile.ov` 和专用构建忽略文件）；内部示例、办公案例、开发测试和设计资料由 `excluded_paths` 排除。未分类改动使审计失败。

开发提交的完整 tree 记为 `development_tree`；实际交付的“官方基线＋公开改动”记为 `source_tree`。校验和源码重建均以 `source_tree` 为准。

## 接收方：准备源码

Python 3.10+ 和 Git 即可运行，从本交付仓库根目录执行：

```bash
python3 patches/OpenViking_V0.4.1/patch_manager.py verify
python3 patches/OpenViking_V0.4.1/prepare_source.py \
  --feature-set full-default --output-dir ../openviking-source
```

工具下载固定 tag，核对基线后应用补丁，并验证交付 tree。源码目录的 HEAD 为官方基线，补丁处于 Git 暂存区；`.git/openviking-patches.json` 保存来源回执。输出路径使用新目录。

已有包含基线对象的本地源码库时，可复用 Git 对象：

```bash
python3 patches/OpenViking_V0.4.1/patch_manager.py verify --source-tree ../ov-dev
python3 patches/OpenViking_V0.4.1/prepare_source.py \
  --source-tree ../ov-dev --feature-set full-default --output-dir ../openviking-source
```

服务部署见[部署指南](../../docs/zh/OpenViking_V0.4.1/deployment_guide.md)。

## 维护者：导出新版本

来源源码工作区保持干净，包含官方基线对象；显式指定已提交的开发版本：

```bash
python3 patches/OpenViking_V0.4.1/patch_manager.py audit \
  --source-tree ../ov-dev --source-ref HEAD
python3 patches/OpenViking_V0.4.1/patch_manager.py generate \
  --source-tree ../ov-dev --source-ref HEAD --output-dir ../ov-release-0.4.1
```

`generate` 在新目录生成补丁、来源记录及校验值，校验所有 Profile 并核对公开源码 tree。将验证后的版本交付文件更新至本目录，保留 `tests/` 工具测试。

```bash
python3 patches/OpenViking_V0.4.1/patch_manager.py checksums
python3 patches/OpenViking_V0.4.1/patch_manager.py verify
python3 -m unittest discover -s patches/OpenViking_V0.4.1/tests -v
```
