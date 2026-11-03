"""Native message samples shared by Writer unit and integration tests."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def exchange(outputs, *, names=None):
    names = names or [f"lookup_{i}" for i in range(len(outputs))]
    calls = [
        {"id": f"call_{i}", "name": name, "args": {"record_id": i}} for i, name in enumerate(names)
    ]
    return [
        HumanMessage(id="user", content="查询数据"),
        AIMessage(id="ai", content="", tool_calls=calls),
        *[
            ToolMessage(
                id=f"tool_{i}",
                tool_call_id=f"call_{i}",
                content=output,
                name=names[i],
                artifact={"source": i},
                additional_kwargs={"trace": i},
                response_metadata={"elapsed": i},
            )
            for i, output in enumerate(outputs)
        ],
    ]
