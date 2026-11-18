"""LLM access through any OpenAI-compatible endpoint (Groq, Ollama, OpenRouter, Gemini...).

Configure with LLM_API_KEY, LLM_BASE_URL and LLM_MODEL, in the environment or in a .env file.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

import requests


def _load_dotenv() -> None:
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_dotenv()

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")  # default: local Ollama
# One model, or several separated by commas. Free tiers meter each model's daily quota separately,
# so when one runs out the next is used.
MODELS = [m.strip() for m in os.environ.get("LLM_MODEL", "openai/gpt-oss-120b").split(",") if m.strip()]
MODEL = MODELS[0]
_exhausted: set[str] = set()


def _active_idx() -> int | None:
    for i, m in enumerate(MODELS):
        if m not in _exhausted:
            return i
    return None
REASONING = os.environ.get("LLM_REASONING_EFFORT", "low")  # only sent to gpt-oss models; low saves tokens
# Free tiers cap tokens per minute, and a single request over the cap is rejected outright.
# Input estimate + max_tokens must fit inside it, so every call is sized against this budget.
TPM = int(os.environ.get("LLM_TPM_LIMIT", "8000"))
RPM = int(os.environ.get("LLM_RPM_LIMIT", "0"))  # 0 = no request-count pacing
SAFETY = 500

# Optional callback so the UI can show "waiting for rate limit" instead of looking frozen.
wait_hook: Callable[[str], None] | None = None


class TooLarge(RuntimeError):
    pass


def est_tokens(text: str) -> int:
    return int(len(text) / 3.0) + 1  # deliberately pessimistic


def out_budget(input_text: str, want: int) -> int:
    room = TPM - SAFETY - est_tokens(input_text)
    if room < 600:
        raise TooLarge(f"prompt (~{est_tokens(input_text)} tokens) leaves no room for an answer under the {TPM}-token limit")
    return min(want, room)


class Refused(RuntimeError):
    pass


class DailyQuotaExhausted(RuntimeError):
    pass


def api_key_set() -> bool:
    # Local servers such as Ollama need no key.
    return bool(os.environ.get("LLM_API_KEY")) or "localhost" in BASE_URL or "127.0.0.1" in BASE_URL


_window: list[tuple[float, int]] = []  # (time, tokens) of recent requests in this process
_window_lock = threading.Lock()


def _reserve(tokens: int) -> None:
    """Block until `tokens` fit in the last 60 seconds' budget, so threads in this process don't starve each other."""
    while True:
        with _window_lock:
            now = time.time()
            _window[:] = [(t, n) for t, n in _window if now - t < 60]
            fits = sum(n for _, n in _window) + tokens <= TPM - SAFETY and (not RPM or len(_window) < RPM)
            if not _window or fits:
                _window.append((now, tokens))
                return
            wait = 60 - (now - _window[0][0]) + 0.5
        if wait_hook:
            wait_hook(f"      pacing for the free-tier limit, waiting {wait:.0f}s")  # wait is for the oldest request to age out
        time.sleep(min(max(wait, 1), 30))


def _post(payload: dict, retries: int = 12) -> dict:
    # Count the prompt plus most of the allowed output against the shared per-minute budget.
    _reserve(est_tokens(json.dumps(payload.get("messages", []))) + int(payload.get("max_tokens", 0) * 0.7))
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LLM_API_KEY']}"
    idx = _active_idx()
    if idx is None:
        raise DailyQuotaExhausted("Every configured model has used its free daily quota.")
    payload["model"] = MODELS[idx]  # the first model that still has quota
    if "gpt-oss" in payload["model"] or "gemini" in payload["model"]:  # reasoning models: keep hidden thinking small
        payload.setdefault("reasoning_effort", REASONING)
    last = ""
    for attempt in range(retries):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=300)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 and re.search(r"per ?day|PerDay|\bRPD\b|\bTPD\b", r.text, re.I):
            # A daily quota will not clear by waiting a minute: move to the next model, or stop with a clear message.
            _exhausted.add(payload["model"])
            nxt = _active_idx()
            if nxt is None:
                raise DailyQuotaExhausted(
                    "Every configured model has used its free daily quota. Wait for the reset (the provider's "
                    f"message says when: {r.text[-300:].strip()}), add models to LLM_MODEL (comma separated), "
                    "or switch provider.")
            if wait_hook:
                wait_hook(f"      daily quota used up on {payload['model']}; switching to {MODELS[nxt]}")
            payload["model"] = MODELS[nxt]
            continue
        if r.status_code == 429 or r.status_code >= 500:
            # Free tiers rate-limit by tokens per minute: wait as long as the server asks.
            wait = min(float(r.headers.get("retry-after") or min(2 ** attempt * 3, 60)) + 1, 90)
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if wait_hook:
                wait_hook(f"      rate limit reached, waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"LLM request failed (HTTP {r.status_code}): {r.text[:500]}")
        return r.json()
    raise RuntimeError(f"LLM request kept failing: {last}")


def _choice(data: dict) -> dict:
    choice = data["choices"][0]
    if choice.get("finish_reason") == "content_filter":
        raise Refused("the model declined the request")
    if choice.get("finish_reason") == "length":
        raise RuntimeError("response was cut off at the token limit; shorten the input or raise the limit")
    return choice


def complete(system: str, user: str, max_tokens: int = 4_000, effort: str | None = None) -> str:
    # Reasoning models spend part of max_tokens on hidden thinking, so a cut-off is retried with more room.
    for attempt in range(2):
        data = _post({
            "model": MODEL, "max_tokens": out_budget(system + user, max_tokens * (attempt + 1)),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        choice = data["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            raise Refused("the model declined the request")
        content = (choice["message"].get("content") or "").strip()
        if choice.get("finish_reason") != "length":
            return content
        if content and attempt == 1:
            if wait_hook:
                wait_hook("      warning: a model answer was cut off at the token limit")
            return content
    raise RuntimeError("the model used its whole token budget thinking and produced no answer; try a shorter topic")


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools]


def _shrink(messages: list[dict], budget_tokens: int) -> None:
    """Keep the conversation under budget by collapsing the oldest tool results to their first lines.

    Every paper the tools return is already in the library, so the model loses only abstract text.
    """
    def size() -> int:
        return est_tokens(json.dumps(messages))

    for m in messages:
        if size() <= budget_tokens:
            return
        if m["role"] == "tool" and len(m["content"]) > 400:
            lines = [l[:150] for l in m["content"].splitlines() if l.startswith("[@")]
            m["content"] = "\n".join(lines) or m["content"][:300]
            m["content"] += "\n(abstracts dropped to save space; use get_paper for details)"
    if size() > budget_tokens:  # still too big: drop the oldest assistant/tool exchanges entirely
        while size() > budget_tokens and len(messages) > 4:
            del messages[2]
        # a tool message must follow its assistant tool_calls message, so remove orphans
        while len(messages) > 2 and messages[2]["role"] == "tool":
            del messages[2]


def run_tool_agent(
    system: str,
    user: str,
    tools: list[dict],
    dispatch: Callable[[str, dict], str],
    max_turns: int = 20,
    max_tokens: int = 3_000,
    on_tool: Callable[[str, dict], None] | None = None,
) -> str:
    """Agent loop: call the model, run its tool calls, feed results back until it answers in text."""
    oa_tools = _to_openai_tools(tools)
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    tools_text = json.dumps(oa_tools)
    for turn in range(max_turns):
        final_turn = turn == max_turns - 1
        if final_turn:
            messages.append({"role": "user", "content": "Turn budget reached. Write your final notes now; make no more tool calls."})
        _shrink(messages, TPM - SAFETY - 2_200 - est_tokens(tools_text))  # leave room for the answer
        payload = {"model": MODEL, "messages": messages,
                   "max_tokens": out_budget(json.dumps(messages) + tools_text, max_tokens)}
        if not final_turn:
            payload["tools"] = oa_tools
        msg = _choice(_post(payload))["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip()
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": "ERROR: arguments were not valid JSON"})
                continue
            if on_tool:
                on_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": dispatch(name, args)})
    return ""
