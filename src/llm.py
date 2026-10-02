# -*- coding: utf-8 -*-
"""统一 LLM 客户端（与 task-agent 保持一致的实现风格）。

只用标准库 urllib，任何 OpenAI 兼容接口可用；无 Key 时降级为 mock，
保证整个仓库 clone 下来、不配置任何东西也能跑通。

环境变量：LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
"""
import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")


def mode() -> str:
    return "mock" if not API_KEY else "api"


def chat(messages, tools=None, tool_choice="auto", temperature=0.0, retries=2):
    if mode() == "mock":
        return {"role": "assistant", "content": "", "tool_calls": []}

    url = BASE_URL.rstrip("/") + "/chat/completions"
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    last_err = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM 调用失败: {last_err}")
