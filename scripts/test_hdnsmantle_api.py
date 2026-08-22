#!/usr/bin/env python3
"""Interactively test an OpenAI-compatible provider without storing the API key."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "https://api.hdnsmantle.com/v1"
DEFAULT_MODEL = "gpt-5.6-luna"
TIMEOUT_SECONDS = 60
BODY_PREVIEW_LIMIT = 1200


@dataclass
class TestResult:
    name: str
    endpoint: str
    ok: bool
    status: int | None
    body: str
    parsed: Any = None


def prompt_with_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Chat Completions, function calling, and Responses API compatibility."
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Read OPENAI_API_KEY, OPENAI_BASE_URL, and MODEL_NAME from the environment.",
    )
    return parser.parse_args()


def normalize_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    for endpoint in ("/chat/completions", "/responses", "/models"):
        if base_url.endswith(endpoint):
            base_url = base_url[: -len(endpoint)]
            print(f"提示：已移除末尾的 {endpoint}，Base URL 使用 {base_url}")
            break
    return base_url


def preview(text: str, api_key: str) -> str:
    safe_text = text.replace(api_key, "***REDACTED***") if api_key else text
    if len(safe_text) > BODY_PREVIEW_LIMIT:
        return safe_text[:BODY_PREVIEW_LIMIT] + "\n...（响应已截断）"
    return safe_text


def request_json(
    name: str,
    method: str,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> TestResult:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "daily-arxiv-api-diagnostic/1.0",
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        endpoint,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed = None
            return TestResult(name, endpoint, True, response.status, raw_body, parsed)
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            parsed = None
        return TestResult(name, endpoint, False, exc.code, raw_body, parsed)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return TestResult(name, endpoint, False, None, str(exc))


def show_result(result: TestResult, api_key: str) -> None:
    marker = "✅" if result.ok else "❌"
    status = result.status if result.status is not None else "无 HTTP 状态码"
    print(f"\n{marker} {result.name}")
    print(f"   地址：{result.endpoint}")
    print(f"   状态：{status}")
    if result.body:
        print("   响应：")
        for line in preview(result.body, api_key).splitlines():
            print(f"     {line}")


def has_chat_content(result: TestResult) -> bool:
    if not result.ok or not isinstance(result.parsed, dict):
        return False
    choices = result.parsed.get("choices")
    return isinstance(choices, list) and bool(choices)


def has_tool_call(result: TestResult) -> bool:
    if not has_chat_content(result):
        return False
    try:
        tool_calls = result.parsed["choices"][0]["message"].get("tool_calls")
    except (KeyError, IndexError, TypeError, AttributeError):
        return False
    return isinstance(tool_calls, list) and bool(tool_calls)


def has_responses_output(result: TestResult) -> bool:
    if not result.ok or not isinstance(result.parsed, dict):
        return False
    return bool(result.parsed.get("output") or result.parsed.get("output_text"))


def main() -> int:
    args = parse_args()
    print("HDNS Mantle / OpenAI 兼容接口诊断")
    print("API Key 将隐藏输入，只保存在本进程内存中，不会写入文件或打印。")
    print("测试会发送 3 个很短的模型请求，可能产生极少量费用。\n")

    if args.non_interactive:
        base_url = normalize_base_url(os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
        model = os.environ.get("MODEL_NAME", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        print(f"非交互模式：Base URL={base_url}，模型={model}")
    else:
        base_url = normalize_base_url(prompt_with_default("Base URL", DEFAULT_BASE_URL))
        model = prompt_with_default("模型名称", DEFAULT_MODEL)
        api_key = getpass.getpass("API Key（隐藏输入）: ").strip()

    if not api_key:
        source = "OPENAI_API_KEY Secret" if args.non_interactive else "API Key"
        print(f"错误：{source} 不能为空。", file=sys.stderr)
        return 2

    tests = [
        request_json("读取模型列表", "GET", f"{base_url}/models", api_key),
        request_json(
            "普通 Chat Completions",
            "POST",
            f"{base_url}/chat/completions",
            api_key,
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: API_OK"}],
                "stream": False,
            },
        ),
        request_json(
            "Chat Completions function calling",
            "POST",
            f"{base_url}/chat/completions",
            api_key,
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Call return_test_status with status API_OK.",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "return_test_status",
                            "description": "Return the API diagnostic status.",
                            "parameters": {
                                "type": "object",
                                "properties": {"status": {"type": "string"}},
                                "required": ["status"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "return_test_status"},
                },
                "stream": False,
            },
        ),
        request_json(
            "Responses API（用于判断是否只支持 Responses 协议）",
            "POST",
            f"{base_url}/responses",
            api_key,
            {
                "model": model,
                "input": "Reply with exactly: API_OK",
                "stream": False,
            },
        ),
    ]

    for result in tests:
        show_result(result, api_key)

    models_result, chat_result, tool_result, responses_result = tests
    print("\n诊断结论")
    if models_result.status in {401, 403}:
        print("- 模型列表鉴权失败：优先检查 API Key、账号权限或运营商访问策略。")
    elif models_result.ok:
        print("- API Key 至少能够访问模型列表。")
    else:
        print("- 模型列表不可用；部分兼容服务不开放此接口，需结合生成请求判断。")

    if has_chat_content(chat_result):
        print("- 普通 Chat Completions 成功：Base URL、Key、模型和基础聊天协议可用。")
    else:
        print("- 普通 Chat Completions 未成功：当前仓库无法正常生成 AI 内容。")

    if has_tool_call(tool_result):
        print("- Function calling 成功：兼容本仓库的结构化详情生成方式。")
    elif tool_result.ok:
        print("- 请求返回 2xx，但没有 tool_calls：function calling 兼容性仍有问题。")
    else:
        print("- Function calling 失败：即使普通聊天成功，论文结构化处理也可能回退。")

    if has_responses_output(responses_result):
        print("- Responses API 成功。")
        if not has_chat_content(chat_result):
            print("- 只有 Responses 成功：运营商可能不兼容本仓库当前使用的 ChatOpenAI 调用方式。")
    else:
        print("- Responses API 未成功或未返回标准 output。")

    if any("blocked" in result.body.lower() for result in tests):
        print("- 服务明确返回 blocked：这是运营商/上游拦截，不是 Base URL 少写 /chat/completions。")

    return 0 if has_chat_content(chat_result) and has_tool_call(tool_result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
