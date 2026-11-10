# DeepAgents Quick Start

通过 Python 直接调用 DeepAgents，完成消息写入、同 Session 的 checkpoint 恢复，以及新 Session 的记忆召回。运行环境和 OV 服务按[部署指南](deployment_guide.md)准备。特性说明见《[OpenViking 接入 DeepAgents 实现上下文管理](deepagents_context_management.md)》。

## 1. 准备应用配置

以下命令在部署机器执行，复用 OV 镜像中的 Python 与接入依赖：

```bash
export OV_HOME="$HOME/openviking-runtime"
mkdir -p "$OV_HOME/app"
cp "$OV_HOME/config/runtime.env" "$OV_HOME/app/.env"
chmod 600 "$OV_HOME/app/.env"
```

在 `$OV_HOME/app/.env` 中补充下列参数。`OV_LLM_*` 沿用部署时填写的 OpenAI 兼容模型配置，也可以替换为应用自己的对话模型：

```dotenv
OV_URL=http://127.0.0.1:1933
OV_ACCOUNT=quickstart
OV_USER=demo-user
OV_PEER=assistant
OV_CONTEXT_WINDOW=256000
OV_OUTPUT_RESERVE=4096
OV_COMMIT_WAIT_TIMEOUT=300
OV_COMMIT_TOKEN_BUDGET=256000
OV_COMMIT_TOKEN_THRESHOLD_RATIO=0.7
OV_COMMIT_COMPLETED_RUN_THRESHOLD=1
OV_COMMIT_KEEP_RECENT_COUNT=8
OV_RECALL_MAX_TOKENS=1600
```

模型窗口和输出预留按实际模型能力填写。示例将完成轮数阈值设为 1，便于观察提交；应用使用自己的配置值。应用在创建 Session 时确定配置，同 Session 后续 Run 复用该配置。

## 2. 接入代码

将以下代码保存为 `$OV_HOME/app/quick_start.py`：

```python
import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openviking.integrations.langchain import OpenVikingContextMiddleware
from openviking.integrations.langchain.ov_context import (
    CommitConfig,
    MemoryReaderConfig,
    Metrics,
    OVContextConfig,
    OVContextReader,
    OVMessageWriter,
    RunIdentity,
)
from openviking_cli.client.http import AsyncHTTPClient

load_dotenv(Path(__file__).with_name(".env"))


@dataclass(frozen=True)
class RequestContext:
    identity: RunIdentity


class OVSummaryOwner(AgentMiddleware):
    # 由 OV 管理上下文；替换框架的默认摘要槽位。
    @property
    def name(self):
        return "SummarizationMiddleware"


def setting(name):
    return os.environ[name]


async def wait_commit(sdk, saved, identity):
    pending = saved.get("pending_commit") or {}
    task_id = pending.get("task_id")
    if not task_id:
        raise RuntimeError("本轮没有提交任务，请检查提交配置与运行日志")
    deadline = asyncio.get_running_loop().time() + float(setting("OV_COMMIT_WAIT_TIMEOUT"))
    while asyncio.get_running_loop().time() < deadline:
        task = await sdk.get_task(task_id, identity=identity.sdk_identity)
        status = (task or {}).get("status")
        if status == "completed":
            view = await sdk.get_session_context(
                identity.session_id, identity=identity.sdk_identity
            )
            archive = view.get("latest_finished_archive_id")
            if not archive:
                raise RuntimeError("提交任务完成，但上下文尚无已完成归档")
            print("归档完成:", archive)
            return
        if status == "failed":
            raise RuntimeError("OV 提交任务失败，请检查 OV 日志")
        await asyncio.sleep(1)
    raise TimeoutError("等待归档超时，请检查 OV 模型与记忆提取日志")


async def invoke(sdk, saver, model, session_id, text):
    # 每个请求生成新 run_id；account、user、peer 与 Session 保持绑定。
    identity = RunIdentity(
        setting("OV_ACCOUNT"),
        setting("OV_USER"),
        setting("OV_PEER"),
        session_id,
        "r-" + uuid.uuid4().hex,
    )
    events = []

    def observe(event, **fields):
        if event == "memory.recall":
            events.append(fields)
            print("召回:", json.dumps(fields, ensure_ascii=False, default=str))

    window = int(setting("OV_CONTEXT_WINDOW"))
    middleware = OpenVikingContextMiddleware(
        client=sdk,
        message_writer=OVMessageWriter(sdk),
        context_reader=OVContextReader(
            sdk,
            MemoryReaderConfig(
                search_mode="context",
                context_token_budget=window,
                recall_max_tokens=int(setting("OV_RECALL_MAX_TOKENS")),
            ),
        ),
        context_config=OVContextConfig(
            context_window=window,
            output_reserve=int(setting("OV_OUTPUT_RESERVE")),
            commit_wait_timeout=float(setting("OV_COMMIT_WAIT_TIMEOUT")),
        ),
        commit_config=CommitConfig(
            token_budget=int(setting("OV_COMMIT_TOKEN_BUDGET")),
            commit_token_threshold_ratio=float(setting("OV_COMMIT_TOKEN_THRESHOLD_RATIO")),
            completed_run_threshold=int(setting("OV_COMMIT_COMPLETED_RUN_THRESHOLD")),
            keep_recent_count=int(setting("OV_COMMIT_KEEP_RECENT_COUNT")),
        ),
        memory_policy={"self": {"enabled": True}, "peer": {"enabled": False}},
        metrics=Metrics(observe),
    )
    graph = create_deep_agent(
        model=model,
        system_prompt="你是简洁的助理，根据用户要求回答，并使用与当前任务相关的记忆。",
        middleware=[
            OVSummaryOwner(),
            FilesystemMiddleware(
                tool_token_limit_before_evict=None,
                human_message_token_limit_before_evict=None,
            ),
            middleware,
        ],
        context_schema=RequestContext,
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": json.dumps(identity.session_key)}}
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config=config,
            context=RequestContext(identity),
        )
        saved = (await graph.aget_state(config)).values["ov_context"]
        print("回答:", result["messages"][-1].content)
        print(
            "会话状态:",
            json.dumps(
                {
                    "session_id": saved["session_id"],
                    "write_index": saved["write_index"],
                    "cached_messages": len(saved["messages"]),
                    "archive_id": saved.get("archive_id"),
                },
                ensure_ascii=False,
            ),
        )
        assert saved["write_index"] > 0, "消息写入游标未推进"
        await wait_commit(sdk, saved, identity)
        return events
    finally:
        await middleware.aclose()


async def main():
    model = ChatOpenAI(
        model=setting("OV_LLM_MODEL"),
        base_url=setting("OV_LLM_API_BASE"),
        api_key=setting("OV_LLM_API_KEY"),
        max_tokens=int(setting("OV_OUTPUT_RESERVE")),
        timeout=180,
    )
    # 应用持有连接池和 Saver；每轮创建的 Middleware 借用 SDK。
    async with httpx.AsyncClient(timeout=60) as pool:
        sdk = AsyncHTTPClient(
            url=setting("OV_URL"),
            api_key=setting("OV_SERVICE_KEY"),
            auth_mode="trusted",
            http_client=pool,
        )
        await sdk.initialize()
        try:
            accounts = await sdk.admin_list_accounts()
            if not any(item["account_id"] == setting("OV_ACCOUNT") for item in accounts):
                await sdk.admin_create_account(setting("OV_ACCOUNT"), setting("OV_USER"))
            async with AsyncSqliteSaver.from_conn_string("checkpoint.sqlite") as saver:
                session = "s-" + uuid.uuid4().hex
                await invoke(
                    sdk,
                    saver,
                    model,
                    session,
                    "请记住我的长期偏好：回答先给结论，再列步骤，统一使用中文。",
                )
                await invoke(sdk, saver, model, session, "请复述上一轮我提出的回答格式。")
                events = await invoke(
                    sdk,
                    saver,
                    model,
                    "s-" + uuid.uuid4().hex,
                    "我偏好怎样的回答格式？请根据已有记忆回答。",
                )
                if not any(event.get("count", 0) > 0 for event in events):
                    raise RuntimeError("新会话未召回记忆，请检查提取结果和检索配置")
                print("Quick Start 完成：写入、checkpoint 恢复和跨 Session 召回已验证")
        finally:
            await sdk.close()


if __name__ == "__main__":
    asyncio.run(main())
```

运行代码：

```bash
docker run --rm --network host --user "$(id -u):$(id -g)" \
  -v "$OV_HOME/app:/app" -w /app \
  --entrypoint /opt/venv/bin/python \
  openviking-context:0.4.1 quick_start.py
```

应用运行于单独的 Python 进程，复用同一 OV 镜像中的依赖；持久化的 checkpoint 位于 `$OV_HOME/app/checkpoint.sqlite`。应用进程可用同一文件重建 Saver。

## 3. 查看结果

以下为三轮成功执行的示意输出，仅展示核对所需字段。实际 Session ID、消息数量、召回条数和回答内容随运行结果变化。

### 3.1 第一轮：写入偏好并提交归档

用户输入：

```text
请记住我的长期偏好：回答先给结论，再列步骤，统一使用中文。
```

输出示例：

```text
召回: {"session_id": "s-11111111111111111111111111111111", "status": "empty", "count": 0}
回答: 已记住：回答先给结论，再列步骤，统一使用中文。
会话状态: {"session_id": "s-11111111111111111111111111111111", "write_index": 2, "cached_messages": 2, "archive_id": null}
归档完成: archive_001
```

1. `write_index` 和 `cached_messages` 大于 0，表示原始消息已同步到 OV，模型缓存已有消息。
2. `归档完成: archive_001` 表示程序已确认对应提交任务完成，并从 OV 读到已结束的归档边界。
3. `会话状态` 在等待归档之前打印，因此其中的 `archive_id` 可以为 `null`。首次使用空记忆空间时，召回条数为 0。

### 3.2 第二轮：重建 Agent 并延续同一 Session

用户输入：

```text
请复述上一轮我提出的回答格式。
```

输出示例：

```text
召回: {"session_id": "s-11111111111111111111111111111111", "status": "success", "count": 1}
回答: 你的回答格式偏好是：先给结论，再列步骤，统一使用中文。
会话状态: {"session_id": "s-11111111111111111111111111111111", "write_index": 4, "cached_messages": 3, "archive_id": "archive_001"}
归档完成: archive_002
```

1. `session_id` 与第一轮相同；代码每轮创建新的 Agent，沿用相同 `thread_id` 和 Saver，恢复此前保存的图状态。
2. `write_index` 在第一轮基础上继续推进，表示写入进度跨请求延续。
3. `cached_messages` 可以因加载历史摘要而减少。回答应能复述第一轮偏好；本轮的召回条数以实际结果为准。

### 3.3 第三轮：在新 Session 召回长期记忆

用户输入：

```text
我偏好怎样的回答格式？请根据已有记忆回答。
```

输出示例：

```text
召回: {"session_id": "s-22222222222222222222222222222222", "status": "success", "count": 1}
回答: 你偏好使用中文，先给结论，再列出具体步骤。
会话状态: {"session_id": "s-22222222222222222222222222222222", "write_index": 2, "cached_messages": 2, "archive_id": null}
归档完成: archive_001
Quick Start 完成：写入、checkpoint 恢复和跨 Session 召回已验证
```

1. `session_id` 与前两轮不同，account、user 和 actor peer 保持一致，使用独立的 checkpoint 线程。
2. `memory.recall` 事件的 `status` 为 `success`、`count` 大于 0，表示新 Session 命中了长期记忆。
3. 结合回答核对是否包含“中文、先结论、后步骤”三项偏好。程序检查召回条数，回答语义由运行者核对。
4. 新 Session 的写入游标独立累计，归档编号也从该 Session 的首个归档开始。

记忆提取结果取决于模型与内容。第三轮零命中时，依次核对提交任务、OV 记忆提取日志、Embedding 配置和 Reader 检索范围；程序保留 checkpoint，便于查看会话状态。

## 4. 应用接入约定

- **身份**：从应用认证与会话记录取得 account、user、actor peer 和 session_id，通过 `ainvoke(..., context=RequestContext(identity))` 传入；`run_id` 标识本次请求。
- **Session**：固定绑定 account、user 和 actor peer；身份组合变化时创建独立 Session。同 Session 使用固定配置，并由应用串行执行各轮请求。
- **checkpoint**：`thread_id` 使用完整 Session 身份生成，区分用户与 Agent；应用复用持久化 Saver。图自动保存 `ov_context` 字段。
- **资源**：应用初始化并关闭 HTTP 连接池和 SDK；Reader、Writer 与 Middleware 借用同一 SDK，通过每次调用的 identity 访问 OV。
- **上下文**：示例由 OV 提供历史摘要与工具短视图，框架的文件系统工具沿用原生能力。应用可按需要增加业务工具和 Skill。
