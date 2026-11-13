#!/usr/bin/env bash
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  cat <<'HELP'
Usage: bash examples/deepagents_ov_context/scripts/container.sh COMMAND [ARGS]
  build           Build OV and Chat images with all Chat tools baked in
  init            Create mounted config templates and data directories
  configure       Generate trusted OV configuration from models.local.json
  up              Start the two main services
  check           Check image dependencies and service readiness
  smoke           Run the real Chat/SSE smoke test (uses the model)
  run ...         Run an administrator command in the Chat image
  datasets-init   Create private Dovecot/Roundcube configuration
  datasets-up     Enable case services inside the Chat container and restart it
  cases ...       Run an administrator command using the Chat image
  status|logs     Show deployment status or logs
  stop            Stop this deployment's containers
  down            Remove this deployment's containers; keep bind-mounted data
  compose ...     Pass explicit arguments to Docker Compose
Environment: OV_DEPLOY_HOME (default $HOME/ov-context-chat), COMPOSE_PROJECT_NAME,
OV_IMAGE, OV_CHAT_IMAGE, OV_CPUSET, OV_CHAT_CPUSET, BUILD_JOBS.
HELP
}
command="${1:---help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in --help|-h|help) usage; exit 0 ;; esac
command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 1; }
docker compose version >/dev/null || { echo 'Docker Compose v2 is required.' >&2; exit 1; }
export OV_DEPLOY_HOME="${OV_DEPLOY_HOME:-$HOME/ov-context-chat}"
export OV_UID="${OV_UID:-$(id -u)}" OV_GID="${OV_GID:-$(id -g)}"
[[ "$OV_UID" =~ ^[1-9][0-9]*$ && "$OV_GID" =~ ^[1-9][0-9]*$ ]] || {
  echo 'Set OV_UID and OV_GID to the non-root deployment account.' >&2; exit 1;
}
umask 077
mkdir -p "$OV_DEPLOY_HOME/config" "$OV_DEPLOY_HOME/data/ov" \
  "$OV_DEPLOY_HOME/data/chat" "$OV_DEPLOY_HOME/data/cases" "$OV_DEPLOY_HOME/result"
export OV_DEPLOY_HOME="$(cd "$OV_DEPLOY_HOME" && pwd -P)"
compose=(docker compose -f "$task_dir/docker/compose.yaml")
if [[ -f "$OV_DEPLOY_HOME/compose.env" ]]; then
  compose+=(--env-file "$OV_DEPLOY_HOME/compose.env")
fi
admin=("${compose[@]}" -f "$task_dir/docker/compose.admin.yaml" run --rm --no-deps chat)
case "$command" in
  build)
    [[ -d "$task_dir/../../benchmark/office" && ! -L "$task_dir/../../benchmark/office" ]] || {
      echo 'Chat requires the complete benchmark/office source directory.' >&2; exit 1;
    }
    exec "${compose[@]}" build ov chat "$@" ;;
  init|configure) exec "${admin[@]}" "$command" "$@" ;;
  up) exec "${compose[@]}" up -d --wait ov chat "$@" ;;
  check)
    "${admin[@]}" check
    exec "${admin[@]}" check-services "$@" ;;
  smoke) exec "${admin[@]}" smoke "$@" ;;
  run|cases) exec "${admin[@]}" "$@" ;;
  datasets-init) exec "${admin[@]}" init-mailbox "$@" ;;
  datasets-up)
    settings="$OV_DEPLOY_HOME/compose.env"
    updated="$(mktemp "$OV_DEPLOY_HOME/compose.env.XXXXXX")"
    if [[ -f "$settings" ]]; then
      awk '!/^[[:space:]]*(export[[:space:]]+)?OV_CASES_ENABLED[[:space:]]*=/' "$settings" > "$updated"
    fi
    printf '\nOV_CASES_ENABLED=true\n' >> "$updated"
    mv "$updated" "$settings"
    # The file may have been created after the command array was assembled.
    export OV_CASES_ENABLED=true
    exec docker compose -f "$task_dir/docker/compose.yaml" --env-file "$settings" \
      up -d --wait --force-recreate chat "$@" ;;
  status) exec "${compose[@]}" ps "$@" ;;
  logs) exec "${compose[@]}" logs "$@" ;;
  stop|down) exec "${compose[@]}" "$command" "$@" ;;
  compose) exec "${compose[@]}" "$@" ;;
  *) usage >&2; exit 2 ;;
esac
