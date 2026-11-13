#!/usr/bin/env bash
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
set -euo pipefail

OV_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OV_DEPLOY_REPO="$(cd -- "$OV_DEPLOY_DIR/../.." && pwd)"
export OV_CHAT_HOME="${OV_CHAT_HOME:-$HOME/ov-context-chat}"
export OV_CHAT_TOOLS="${OV_CHAT_TOOLS:-$HOME/ov-chat-tools}"
export OV_CHAT_OV_PORT="${OV_CHAT_OV_PORT:-1933}"
export OV_CHAT_PORT="${OV_CHAT_PORT:-1934}"
export OV_CHAT_OV_URL="${OV_CHAT_OV_URL:-http://127.0.0.1:$OV_CHAT_OV_PORT}"
export OV_CHAT_URL="${OV_CHAT_URL:-http://127.0.0.1:$OV_CHAT_PORT}"
export PATH="$OV_CHAT_TOOLS/bin:$OV_CHAT_TOOLS/node/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONPATH="$OV_DEPLOY_REPO${PYTHONPATH:+:$PYTHONPATH}"
export LITELLM_LOCAL_MODEL_COST_MAP=true
cd "$OV_DEPLOY_REPO"

usage() {
  cat <<'HELP'
Usage: bash examples/deepagents_ov_context/scripts/deploy.sh COMMAND [OPTIONS]
  install-python       Install editable source and native components; BUILD_JOBS defaults to 8
  backup-build         Move the repository build cache to a timestamped backup
  smoke                Run the existing live Chat smoke test (calls the model)
  check-install        Verify source, Chat imports and native storage
  init-config          Create private config templates and fill missing .env fields
  configure            Generate OV config after filling models.local.json
  start-ov             Run OV in the foreground
  start-chat           Run ordinary Chat in the foreground
  install-browser      Install Node 24, agent-browser and Chromium on openEuler/Linux
  install-pptx-skill   Install the official Anthropic pptx Skill and dependencies
  install-pptx-renderer Install LibreOffice, Poppler and fonts on openEuler ARM64
  check-pptx-skill     Verify Skill, Node, Python and rendering dependencies
  install-office-skills Install Word, Excel, PDF and data analysis dependencies
  check-office-skills Verify bundled office Skills and dependencies
  check-browser        Verify Node, Chromium and PDF parser
  start-browser-chat   Run browser Chat with thinking and PDF support in the foreground
  check-services       Check OV health/readiness and Chat health (no model calls)
  check-session ID     Verify a completed session's checkpoint and write cursor
  check-downloads RUN  Verify downloaded PDFs in this Run directory
  check-artifacts ID   Verify downloadable PPTX files and optional --slides N
  case-data            Check/prepare optional office-case datasets (see --help)

Environment: OV_CHAT_HOME, OV_CHAT_TOOLS, OV_CHAT_CHROME, OV_CHAT_OV_PORT,
OV_CHAT_PORT, OV_CHAT_OV_URL, OV_CHAT_URL, OV_CHAT_BROWSER_PROXY, OV_CHAT_PPTX_SKILL.
Services remain in the foreground; Ctrl+C stops only that process.
HELP
}

activate_python() {
  if [[ ! -x "$OV_DEPLOY_REPO/.venv/bin/python" ]]; then
    echo 'Missing .venv; run install-python first.' >&2
    exit 1
  fi
  source "$OV_DEPLOY_REPO/.venv/bin/activate"
  # Activating a venv changes PATH; keep the selected Node ahead of system Node.
  export PATH="$OV_CHAT_TOOLS/node/bin:$PATH"
}

browser_paths() {
  export PLAYWRIGHT_BROWSERS_PATH="$OV_CHAT_TOOLS/playwright"
  if [[ -z "${OV_CHAT_CHROME:-}" ]]; then
    if [[ ! -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
      echo 'Missing browser directory; run install-browser first.' >&2
      exit 1
    fi
    OV_CHAT_CHROME="$(find "$PLAYWRIGHT_BROWSERS_PATH" -type f -name headless_shell -print -quit)"
  fi
  export OV_CHAT_CHROME
}

install_browser() {
  [[ "$(uname -s)" == Linux ]] || { echo 'Browser installer supports Linux only.' >&2; exit 1; }
  local arch archive
  case "$(uname -m)" in
    aarch64) arch=arm64 ;;
    x86_64) arch=x64 ;;
    *) echo 'Unsupported architecture.' >&2; exit 1 ;;
  esac
  mkdir -p "$OV_CHAT_TOOLS"
  (
    cd "$OV_CHAT_TOOLS"
    echo "Downloading Node 24 manifest ($arch)"
    curl -fsSL --connect-timeout 15 --max-time 300 --retry 2 https://nodejs.org/dist/latest-v24.x/SHASUMS256.txt -o SHASUMS256.txt
    archive="$(awk -v arch="$arch" '$2 ~ ("^node-v24\\..*-linux-" arch "\\.tar\\.xz$") {print $2; exit}' SHASUMS256.txt)"
    [[ -n "$archive" ]]
    echo "Downloading $archive"
    curl -fsSL --connect-timeout 15 --max-time 300 --retry 2 "https://nodejs.org/dist/latest-v24.x/$archive" -o "$archive"
    awk -v name="$archive" '$2 == name' SHASUMS256.txt | sha256sum --check -
    tar -xJf "$archive"
    if [[ -e node && ! -L node ]]; then
      echo 'Refusing to replace a real node directory; choose a new OV_CHAT_TOOLS.' >&2
      exit 1
    fi
    ln -sfn "${archive%.tar.xz}" node
    npm install --prefix "$OV_CHAT_TOOLS" agent-browser@0.41.2 playwright@1.58.2
    export PLAYWRIGHT_BROWSERS_PATH="$OV_CHAT_TOOLS/playwright"
    # Use Playwright's pinned artifact metadata with resumable curl downloads.
    local metadata archive_url browser_dir browser_archive attempt
    metadata="$(node -e '
      const {registry} = require(process.argv[1]);
      const browser = registry.findExecutable("chromium-headless-shell");
      console.log(browser.downloadURLs[0]);
      console.log(browser.directory);
    ' "$OV_CHAT_TOOLS/node_modules/playwright-core/lib/server/registry/index.js")"
    archive_url="${metadata%%$'\n'*}"
    browser_dir="${metadata#*$'\n'}"
    browser_archive="$OV_CHAT_TOOLS/chromium-headless-shell.zip.part"
    for attempt in 1 2 3 4 5; do
      if curl -fL --connect-timeout 15 --max-time 600 --continue-at - \
        "$archive_url" -o "$browser_archive"; then break; fi
      echo "Chromium download attempt $attempt interrupted; resuming."
    done
    unzip -tq "$browser_archive"
    mkdir -p "$browser_dir"
    unzip -qo "$browser_archive" -d "$browser_dir"
    touch "$browser_dir/INSTALLATION_COMPLETE"
    rm "$browser_archive"
  )
}

install_pptx_renderer() {
  [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || {
    echo 'This renderer installer targets openEuler ARM64.' >&2; exit 1;
  }
  local privilege=() archive archive_dir rpm_directory
  if [[ "$EUID" -ne 0 ]]; then privilege=(sudo); fi
  "${privilege[@]}" dnf --disablerepo='*source*' --disablerepo='*debuginfo*' \
    install -y curl tar poppler-utils google-noto-sans-cjk-ttc-fonts
  if ! command -v soffice >/dev/null && [[ ! -x /opt/libreoffice26.8/program/soffice ]]; then
    archive_dir="$OV_CHAT_TOOLS/libreoffice/26.8.1"
    archive="$archive_dir/LibreOffice_26.8.1_Linux_aarch64_rpm.tar.gz"
    mkdir -p "$archive_dir"
    if [[ ! -f "$archive" ]]; then
      curl -fsSL --connect-timeout 15 --max-time 600 --retry 2 \
        "${OV_CHAT_LIBREOFFICE_URL:-https://downloadarchive.documentfoundation.org/libreoffice/old/26.8.1.1/rpm/aarch64/LibreOffice_26.8.1.1_Linux_aarch64_rpm.tar.gz}" \
        -o "$archive.part"
      mv "$archive.part" "$archive"
    fi
    printf '%s  %s\n' '29ff4be12d0d9ce000fe3088e4e8ca6e5f88a892e8253457166c32a03d8917de' "$archive" | sha256sum --check -
    tar -xzf "$archive" -C "$archive_dir"
    rpm_directory="$(find "$archive_dir" -mindepth 2 -maxdepth 2 -type d -name RPMS -print -quit)"
    [[ -n "$rpm_directory" ]] || { echo 'LibreOffice RPM directory missing.' >&2; exit 1; }
    "${privilege[@]}" dnf --disablerepo='*source*' --disablerepo='*debuginfo*' \
      install -y "$rpm_directory"/*.rpm
  fi
  if ! command -v soffice >/dev/null; then
    [[ -x /opt/libreoffice26.8/program/soffice ]] || { echo 'LibreOffice executable missing.' >&2; exit 1; }
    mkdir -p "$OV_CHAT_TOOLS/bin"
    ln -sfn /opt/libreoffice26.8/program/soffice "$OV_CHAT_TOOLS/bin/soffice"
  fi
  soffice --version
  pdftoppm -v
}

command="${1:---help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  --help|-h|help) usage; exit 0 ;;
  install-pptx-renderer) install_pptx_renderer; exit 0 ;;
  install-python)
    unset OV_SKIP_OV_BUILD OV_SKIP_RAGFS_BUILD
    export OV_REQUIRE_RAGFS_BUILD=1 OV_SKIP_STUDIO_BUILD=1
    export CARGO_BUILD_JOBS="${BUILD_JOBS:-8}" CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-8}"
    if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.11; fi
    activate_python
    python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required; use a new deployment directory for another venv"'
    uv pip install -v -e '.[deepagents]' 'langchain-openai>=1,<2' "$@"
    uv pip check
    exec python -m examples.deepagents_ov_context.scripts.check install ;;
  backup-build)
    if [[ -d build ]]; then mv build "build.previous-$(date +%Y%m%d%H%M%S)"; fi
    exit 0 ;;
  init-config)
    umask 077
    mkdir -p "$OV_CHAT_HOME"
    chmod 700 "$OV_CHAT_HOME"
    if [[ ! -e "$OV_CHAT_HOME/models.local.json" ]]; then
      cp "$OV_DEPLOY_DIR/config/models.example.json" "$OV_CHAT_HOME/models.local.json"
    fi
    if [[ ! -e "$OV_CHAT_HOME/.env" ]]; then
      cp "$OV_DEPLOY_DIR/config/chat.example.env" "$OV_CHAT_HOME/.env"
    else
      # Add new template fields without replacing existing values or evaluating shell text.
      while IFS= read -r entry || [[ -n "$entry" ]]; do
        case "$entry" in ''|\#*) continue ;; esac
        key="${entry%%=*}"
        if ! grep -Eq "^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=" "$OV_CHAT_HOME/.env"; then
          printf '\n%s\n' "$entry" >> "$OV_CHAT_HOME/.env"
        fi
      done < "$OV_DEPLOY_DIR/config/chat.example.env"
    fi
    chmod 600 "$OV_CHAT_HOME/.env"
    chmod 600 "$OV_CHAT_HOME/models.local.json"
    echo "Fill API Key in $OV_CHAT_HOME/models.local.json, then run configure."
    exit 0 ;;
  check-install|configure|start-ov|start-chat|start-browser-chat|install-browser|install-pptx-skill|check-pptx-skill|install-office-skills|check-office-skills|check-browser|check-services|check-session|check-downloads|check-artifacts|case-data|smoke) activate_python ;;
  *) usage >&2; exit 2 ;;
esac

case "$command" in
  smoke) exec python -m examples.deepagents_ov_context.scripts.smoke --url "$OV_CHAT_URL" --output "$OV_CHAT_HOME/chat-smoke.json" "$@" ;;
  check-install) exec python -m examples.deepagents_ov_context.scripts.check install "$@" ;;
  configure)
    exec python -m examples.deepagents_ov_context.chat.config \
      --model-config "$OV_CHAT_HOME/models.local.json" --output "$OV_CHAT_HOME/ov" \
      --port "$OV_CHAT_OV_PORT" "$@" ;;
  start-ov)
    export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost,ark.cn-beijing.volces.com"
    export no_proxy="$NO_PROXY"
    exec python -m openviking.server.bootstrap --config "$OV_CHAT_HOME/ov/ov.conf" \
      --host 127.0.0.1 --port "$OV_CHAT_OV_PORT" "$@" ;;
  start-chat|start-browser-chat)
    args=(--model-config "$OV_CHAT_HOME/models.local.json" --data-dir "$OV_CHAT_HOME/chat"
          --ov-url "$OV_CHAT_OV_URL" --port "$OV_CHAT_PORT")
    if [[ -f "$OV_CHAT_HOME/.env" ]]; then args+=(--env-file "$OV_CHAT_HOME/.env"); fi
    if [[ "$command" == start-browser-chat ]]; then
      browser_paths
      python -m examples.deepagents_ov_context.scripts.check browser \
        --tools "$OV_CHAT_TOOLS" --chrome "$OV_CHAT_CHROME"
      args+=(--agent-browser "$OV_CHAT_TOOLS/node_modules/.bin/agent-browser"
             --browser-executable "$OV_CHAT_CHROME" --thinking --max-tokens 16384 --recursion-limit 1000)
      if [[ -n "${OV_CHAT_BROWSER_PROXY:-}" ]]; then args+=(--browser-proxy "$OV_CHAT_BROWSER_PROXY"); fi
    else
      args+=(--max-tokens 2048)
    fi
    skill_path="${OV_CHAT_PPTX_SKILL:-$OV_CHAT_TOOLS/anthropic-skills/skills/pptx}"
    if [[ -f "$skill_path/SKILL.md" ]]; then
      python -m examples.deepagents_ov_context.scripts.check pptx-skill --skill "$skill_path" --tools "$OV_CHAT_TOOLS"
      args+=(--pptx-skill "$skill_path" --skill-node-modules "$OV_CHAT_TOOLS/node_modules")
    fi
    exec python -m examples.deepagents_ov_context.chat.server "${args[@]}" "$@" ;;
  install-pptx-skill)
    command -v node >/dev/null || { echo 'Install Node first (install-browser).' >&2; exit 1; }
    command -v git >/dev/null || { echo 'Install Git first.' >&2; exit 1; }
    skill_repo="$OV_CHAT_TOOLS/anthropic-skills"
    if [[ ! -d "$skill_repo" ]]; then
      git clone --depth 1 --filter=blob:none --sparse https://github.com/anthropics/skills.git "$skill_repo"
      git -C "$skill_repo" sparse-checkout set skills/pptx
    fi
    git -C "$skill_repo" rev-parse HEAD > "$OV_CHAT_TOOLS/pptx-skill-revision.txt"
    npm install --prefix "$OV_CHAT_TOOLS" pptxgenjs@4.0.1 react react-dom react-icons sharp
    python -m pip install 'markitdown[pptx]' Pillow defusedxml lxml
    exec python -m examples.deepagents_ov_context.scripts.check pptx-skill --skill "$skill_repo/skills/pptx" --tools "$OV_CHAT_TOOLS" ;;
  install-office-skills)
    python -m pip install -r "$OV_DEPLOY_DIR/docker/requirements-office.txt"
    exec python -m examples.deepagents_ov_context.scripts.check office-skills ;;
  check-office-skills)
    exec python -m examples.deepagents_ov_context.scripts.check office-skills ;;
  check-pptx-skill)
    exec python -m examples.deepagents_ov_context.scripts.check pptx-skill --skill "${OV_CHAT_PPTX_SKILL:-$OV_CHAT_TOOLS/anthropic-skills/skills/pptx}" --tools "$OV_CHAT_TOOLS" "$@" ;;
  install-browser) install_browser; browser_paths
    exec python -m examples.deepagents_ov_context.scripts.check browser --tools "$OV_CHAT_TOOLS" --chrome "$OV_CHAT_CHROME" ;;
  check-browser) browser_paths
    exec python -m examples.deepagents_ov_context.scripts.check browser --tools "$OV_CHAT_TOOLS" --chrome "$OV_CHAT_CHROME" "$@" ;;
  check-services) exec python -m examples.deepagents_ov_context.scripts.check services --ov-url "$OV_CHAT_OV_URL" --url "$OV_CHAT_URL" "$@" ;;
  check-session) exec python -m examples.deepagents_ov_context.scripts.check session --url "$OV_CHAT_URL" "$@" ;;
  check-downloads) exec python -m examples.deepagents_ov_context.scripts.check downloads --data-dir "$OV_CHAT_HOME/chat" "$@" ;;
  check-artifacts) exec python -m examples.deepagents_ov_context.scripts.check artifacts --url "$OV_CHAT_URL" "$@" ;;
  case-data) exec python -m benchmark.office.admin.datasets "$@" ;;
  *) usage >&2; exit 2 ;;
esac
