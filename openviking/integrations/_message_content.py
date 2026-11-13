# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Framework-independent message normalization and tool content codecs."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

MessageContent = str | list[str | dict[str, Any]]
LANGCHAIN_CONTENT_MIME = "application/vnd.langchain.message-content+json"


def has_image_blocks(content: MessageContent) -> bool:
    """Recognize multimodal blocks without inspecting or copying image bytes."""
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in {"image", "image_url", "input_image"}
        for block in content
    )


def tool_output_has_images(output: str, mime: str) -> bool:
    """Exclude images from text externalization, including encoded tool blocks."""
    media_type = mime.partition(";")[0].strip().lower()
    if media_type.startswith("image/"):
        return True
    if media_type not in {LANGCHAIN_CONTENT_MIME, "application/json"}:
        return False
    try:
        content = json.loads(output)
    except (TypeError, ValueError):
        return False
    return has_image_blocks(content)


def normalize_image_block(block: dict[str, Any]) -> dict[str, Any]:
    """Use the same URL/data-URL representation without modifying source blocks."""
    kind = block.get("type")
    if kind == "image":
        url = block.get("url")
        if not url and block.get("base64") and block.get("mime_type"):
            url = f"data:{block['mime_type']};base64,{block['base64']}"
        detail = block.get("detail")
    elif kind == "image_url":
        payload = block.get("image_url")
        url = payload.get("url") if isinstance(payload, dict) else payload
        detail = payload.get("detail") if isinstance(payload, dict) else None
    else:
        return deepcopy(block)
    if not isinstance(url, str) or not url:
        raise ValueError("Image block has no supported URL/base64 representation")
    payload = {"url": url}
    if detail is not None:
        payload["detail"] = detail
    return {"type": "image_url", "image_url": payload}


def comparable_content(content: str | list[Any]) -> list[Any]:
    """Normalize text and images for comparison, preserving order and source data."""
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    result: list[Any] = []
    for block in blocks:
        if isinstance(block, str):
            block = {"type": "text", "text": block}
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if result and isinstance(result[-1], dict) and result[-1].get("type") == "text":
                result[-1]["text"] += text
            else:
                result.append({"type": "text", "text": text})
        elif isinstance(block, dict) and block.get("type") in {"image", "image_url"}:
            result.append(normalize_image_block(block))
        else:
            result.append(deepcopy(block))
    return result


def encode_tool_content(content: MessageContent) -> tuple[str, str]:
    if isinstance(content, str):
        return content, "text/plain"
    if not isinstance(content, list):
        raise ValueError("Tool content must be a string or content-block list")
    return json.dumps(content, ensure_ascii=False, allow_nan=False), LANGCHAIN_CONTENT_MIME


def decode_tool_content(part: dict[str, Any]) -> MessageContent:
    """Decode full content; preview metadata's MIME describes the stored original."""
    output = part.get("tool_output", "")
    if not isinstance(output, str):
        raise ValueError("Tool output must be a string")
    short_view = (
        part.get("tool_output_truncated")
        or part.get("tool_output_source_ref")
        or (part.get("tool_output_ref") and part.get("tool_output_externalized_reason"))
    )
    if short_view or part.get("tool_output_mime_type") != LANGCHAIN_CONTENT_MIME:
        return output
    value = json.loads(output)
    if not isinstance(value, list) or any(not isinstance(block, (str, dict)) for block in value):
        raise ValueError("Invalid structured tool content")
    return deepcopy(value)
