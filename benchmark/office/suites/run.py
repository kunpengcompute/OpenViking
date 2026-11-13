"""Run one office benchmark case through the existing Chat HTTP/SSE API."""

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from benchmark.office.suites.common.timing import scoring_policy
from benchmark.office.tasks import CASES, PHASES, render_task
from benchmark.paths import RESULT_ROOT


def task_for_workspace(case, phase, workspace, page_url, config):
    """Reject stale/smoke workspaces before creating a model session."""
    if config.get("task_case") != case:
        raise ValueError(f"Prepare a matching workload with case-data prepare --case {case}")
    prepared_phase = config.get("task_phase")
    if phase is not None and phase != prepared_phase:
        raise ValueError("Requested phase differs from the prepared workspace")
    if case == "mail" and not config.get("mail_ready"):
        raise ValueError("Prepare this mail workspace with --mail-settings before running")
    return prepared_phase, render_task(
        case, prepared_phase, workspace=workspace, page_url=page_url, anchor=config["anchor"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--phase", choices=PHASES, help="Defaults to the prepared workspace phase")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "office/cases")
    parser.add_argument("--chat-url", default="http://127.0.0.1:19448")
    parser.add_argument("--page-url", default="http://127.0.0.1:19482")
    parser.add_argument("--mode", choices=["baseline", "current"], default="baseline")
    args = parser.parse_args()
    from benchmark.office.cases.portal.app import OfficePortal

    config = OfficePortal(args.registry).config(args.workspace)
    phase, query = task_for_workspace(args.case, args.phase, args.workspace, args.page_url, config)

    def api(path, body=None):
        request = urllib.request.Request(
            args.chat_url.rstrip("/") + path,
            data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=960)

    with api("/api/sessions") as response:
        sessions = json.load(response)
    if any((s.get("last_run") or {}).get("status") == "running" for s in sessions):
        raise RuntimeError("Chat is busy")
    with api("/api/execution-profiles") as response:
        if not json.load(response).get("automatic_benchmark"):
            raise RuntimeError("Chat must support automatic benchmark snapshots")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    out = args.output / (args.case + "-" + args.mode + "-" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    entry = args.page_url.rstrip("/") + "/mock/login?workspace=" + args.workspace
    context = (
        f"这是合成办公测试。统一入口：{entry}。测试工号{config['employee']['number']}，"
        f"账号{config['employee']['name']}（{config['employee']['department']}）。"
        f"任务发起时刻固定为{config['anchor']}，时区Asia/Shanghai。请先从模拟登录页登录，再从已订阅应用办理任务。"
        "仅通过可见网页操作本工作区合成资源；邮箱只读。业务结果以页面保存后回查为准。"
        "最终交付文件放到本轮output/目录；所有业务账号凭据均由测试平台管理。"
    )
    execution = {
        "mode": args.mode,
        "system_prompt": context,
        "timeout_seconds": 900,
        "recursion_limit": 300,
        "max_tokens": 16384,
    }
    with api(
        "/api/sessions",
        {
            "user_id": args.workspace,
            "agent_id": "office-" + args.case + "-agent",
            "experiment_id": f"office-auto--{args.case}--{phase}",
            "execution": execution,
        },
    ) as response:
        session = json.load(response)
    sid = session["session_id"]
    (out / "condition.json").write_text(
        json.dumps(
            {
                "case": args.case,
                "phase": phase,
                "task_version": config["task_version"],
                "task_template": f"benchmark/office/tasks/{args.case}/{phase}.md",
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "workspace": args.workspace,
                "mode": args.mode,
                "session_id": sid,
                "entry": entry,
                "execution": execution,
                "scoring_policy": scoring_policy(args.case),
                "query": query,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    terminal = None
    run_id = None
    event = "message"
    chunks = []
    with (
        api("/api/chat", {"session_id": sid, "message": query}) as response,
        (out / "events.jsonl").open("w") as log,
    ):
        for raw in response:
            line = raw.decode().rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                chunks.append(line[5:].lstrip())
            elif not line and chunks:
                data = json.loads("\n".join(chunks))
                chunks = []
                log.write(json.dumps({"event": event, "data": data}, ensure_ascii=False) + "\n")
                log.flush()
                if event == "start":
                    run_id = data["run_id"]
                if event in ("done", "error", "cancelled"):
                    terminal = {"event": event, "data": data}
                    print(
                        json.dumps({"terminal": event, "session_id": sid}, ensure_ascii=False),
                        flush=True,
                    )
    (out / "terminal.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2))
    if run_id is not None and terminal is not None:
        with api(f"/api/sessions/{sid}/runs/{run_id}") as response:
            trace = json.load(response)
        (out / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2))
        (out / "result.json").write_text(
            json.dumps(
                {
                    "session_id": sid,
                    "run_id": run_id,
                    "statistics": trace["statistics"],
                    "terminal": terminal,
                    "phase": phase,
                    "case": args.case,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        with api(f"/api/sessions/{sid}/runs/{run_id}/benchmark", {}) as response:
            grade = json.load(response)
        (out / "grade.json").write_text(json.dumps(grade, ensure_ascii=False, indent=2))
    if not terminal or terminal["event"] != "done":
        raise RuntimeError("Run did not complete; evidence retained")
    print("Evidence:", out)


if __name__ == "__main__":
    main()
