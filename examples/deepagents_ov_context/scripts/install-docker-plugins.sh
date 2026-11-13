#!/usr/bin/env bash
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
set -euo pipefail
command -v docker >/dev/null || { echo 'Install Docker Engine first.' >&2; exit 1; }
case "$(uname -m)" in
  aarch64) compose_arch=aarch64; buildx_arch=arm64 ;;
  x86_64) compose_arch=x86_64; buildx_arch=amd64 ;;
  *) echo 'Unsupported Linux architecture.' >&2; exit 1 ;;
esac
[[ "$(uname -s)" == Linux ]] || { echo 'This installer targets Linux.' >&2; exit 1; }
plugin_dir="${DOCKER_CONFIG:-$HOME/.docker}/cli-plugins"
mkdir -p "$plugin_dir"
task_tmp="$(mktemp -d)"
trap 'rm -rf "$task_tmp"' EXIT
install_plugin() {
  local project="$1" version="$2" asset="$3" destination="$4"
  local base="https://github.com/docker/$project/releases/download/$version"
  curl -fL --connect-timeout 15 --max-time 300 --retry 2 "$base/$asset" -o "$task_tmp/$asset"
  curl -fL --connect-timeout 15 --max-time 60 --retry 2 "$base/checksums.txt" -o "$task_tmp/checksums.txt"
  (
    cd "$task_tmp"
    awk -v name="$asset" '$2 == name || $2 == "*" name {print}' checksums.txt > selected.sha256
    [[ -s selected.sha256 ]] || { echo 'Release checksum missing.' >&2; exit 1; }
    sha256sum -c selected.sha256
  )
  install -m 755 "$task_tmp/$asset" "$plugin_dir/$destination"
}
if ! docker compose version >/dev/null 2>&1; then
  install_plugin compose v2.39.4 "docker-compose-linux-$compose_arch" docker-compose
fi
if ! docker buildx version >/dev/null 2>&1; then
  install_plugin buildx v0.17.1 "buildx-v0.17.1.linux-$buildx_arch" docker-buildx
fi
docker compose version
docker buildx version
