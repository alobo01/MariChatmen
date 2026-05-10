"""Public row contracts for MariChatmen datasets and evaluation files."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class CPTRow(TypedDict):
    text: str
    metadata: NotRequired[dict[str, Any]]


class SFTRow(TypedDict):
    messages: list[ChatMessage]
    metadata: NotRequired[dict[str, Any]]


class ORPORow(TypedDict):
    prompt: list[ChatMessage]
    chosen: list[ChatMessage]
    rejected: list[ChatMessage]
    metadata: NotRequired[dict[str, Any]]


class BenchmarkRow(TypedDict):
    id: str
    prompt: list[ChatMessage]
    reference: str
    category: NotRequired[str]
    input_style: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


def _require_mapping(row: Any, name: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError(f"{name} must be a mapping")
    return row


def validate_chat_message(message: Any) -> ChatMessage:
    item = _require_mapping(message, "chat message")
    role = item.get("role")
    content = item.get("content")
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"Invalid chat role: {role!r}")
    if not isinstance(content, str):
        raise TypeError("Chat message content must be a string")
    return {"role": role, "content": content}  # type: ignore[return-value]


def validate_messages(messages: Any, *, name: str = "messages") -> list[ChatMessage]:
    if not isinstance(messages, list):
        raise TypeError(f"{name} must be a list")
    return [validate_chat_message(message) for message in messages]


def validate_cpt_row(row: Any) -> CPTRow:
    item = _require_mapping(row, "CPT row")
    if not isinstance(item.get("text"), str):
        raise TypeError("CPT row text must be a string")
    result: CPTRow = {"text": item["text"]}
    if isinstance(item.get("metadata"), dict):
        result["metadata"] = item["metadata"]
    return result


def validate_sft_row(row: Any) -> SFTRow:
    item = _require_mapping(row, "SFT row")
    result: SFTRow = {"messages": validate_messages(item.get("messages"))}
    if isinstance(item.get("metadata"), dict):
        result["metadata"] = item["metadata"]
    return result


def validate_orpo_row(row: Any) -> ORPORow:
    item = _require_mapping(row, "ORPO row")
    result: ORPORow = {
        "prompt": validate_messages(item.get("prompt"), name="prompt"),
        "chosen": validate_messages(item.get("chosen"), name="chosen"),
        "rejected": validate_messages(item.get("rejected"), name="rejected"),
    }
    if isinstance(item.get("metadata"), dict):
        result["metadata"] = item["metadata"]
    return result


def validate_benchmark_row(row: Any) -> BenchmarkRow:
    item = _require_mapping(row, "benchmark row")
    if not isinstance(item.get("id"), str):
        raise TypeError("Benchmark row id must be a string")
    if not isinstance(item.get("reference"), str):
        raise TypeError("Benchmark row reference must be a string")
    result: BenchmarkRow = {
        "id": item["id"],
        "prompt": validate_messages(item.get("prompt"), name="prompt"),
        "reference": item["reference"],
    }
    for key in ("category", "input_style"):
        if isinstance(item.get(key), str):
            result[key] = item[key]  # type: ignore[literal-required]
    if isinstance(item.get("metadata"), dict):
        result["metadata"] = item["metadata"]
    return result
