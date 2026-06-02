import json
import os
import re
import time
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _proxy_env_str() -> str:
    hp = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or ""
    sp = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or ""
    return f"HTTP_PROXY={hp or '-'} | HTTPS_PROXY={sp or '-'}"


def _make_http_client(timeout: float) -> httpx.Client:
    proxies = {
        "http://": os.getenv("HTTP_PROXY") or None,
        "https://": os.getenv("HTTPS_PROXY") or None,
    }
    proxies = {k: v for k, v in proxies.items() if v}
    try:
        return httpx.Client(proxies=proxies or None, timeout=timeout, verify=True)
    except TypeError:
        return httpx.Client(timeout=timeout, verify=True)


class LLMPolicy:
    def __init__(
        self,
        model: str = "qwen-plus",
        temperature: float = 0.2,
        max_tokens: int = 512,
        debug: bool = True,
        timeout: float = 45.0,
        max_retries: int = 3,
        retry_backoff: float = 1.8,
    ):
        self.api_key = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or "").strip()
        self.enabled = bool(self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.http_client = _make_http_client(timeout=self.timeout)
        self.client = None

        if self.enabled:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=DASHSCOPE_BASE,
                    http_client=self.http_client,
                    timeout=self.timeout,
                )
            except TypeError:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=DASHSCOPE_BASE,
                )

        if self.debug:
            state = "enabled" if self.enabled else "disabled"
            print(
                f"[LLMPolicy] initialized ({state}), proxy={_proxy_env_str()}, "
                f"timeout={self.timeout}s, httpx={getattr(httpx, '__version__', 'unknown')}"
            )

    def before_simulation(self):
        if not self.enabled:
            print("[LLMPolicy] no DASHSCOPE_API_KEY/QWEN_API_KEY found; using noop decisions.")
            return
        try:
            self._quick_ping()
            print("[LLMPolicy] DashScope connectivity check passed.")
        except Exception as exc:
            print(f"[LLMPolicy] connectivity check failed, safe fallback remains active: {exc}")

    def after_simulation(self, result_paths: Dict[str, str]):
        print(f"[LLMPolicy] after_simulation(): result paths -> {result_paths}")

    def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.client is None:
            return {"action_type": "noop", "decisions": [], "notes": "no API key configured"}

        prompt = self._build_prompt(context)
        messages = [
            {"role": "system", "content": "You are a construction scheduling assistant. Return JSON only."},
            {"role": "user", "content": prompt},
        ]

        attempt = 0
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = completion.choices[0].message.content
                self._print_success(text)
                action = self._parse_action(text)
                if action is None:
                    return {"action_type": "noop", "decisions": [], "notes": "fallback: parse failed"}
                return action
            except Exception as exc:
                attempt += 1
                print(f"[LLMPolicy] call failed: {exc} | proxy={_proxy_env_str()}")
                if attempt >= self.max_retries:
                    return {"action_type": "noop", "decisions": [], "notes": "fallback: call failed"}
                time.sleep(self.retry_backoff ** (attempt - 1))

    def _build_prompt(self, ctx: Dict[str, Any]) -> str:
        brief = {
            "task": ctx.get("current_task"),
            "time": round(float(ctx.get("time", 0)), 3),
            "resource": ctx.get("resource"),
            "in_progress_top": (ctx.get("in_progress") or [])[:12],
        }
        schema = {
            "action_type": "resolve_conflicts / reprioritize / noop",
            "decisions": [
                {"conflict_of": "task or resource id", "choose": "selected option", "reason": "brief reason"}
            ],
            "notes": "optional comment",
        }
        return (
            "Return a valid JSON scheduling decision using this schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}\n"
            f"Context: {json.dumps(brief, ensure_ascii=False)}"
        )

    def _print_success(self, text: str):
        if self.debug:
            preview = (text or "").strip().replace("\n", " ")
            if len(preview) > 300:
                preview = preview[:300] + " ..."
            print(f"[LLMPolicy] successful response: {preview}")

    def _parse_action(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _quick_ping(self):
        if not self.enabled:
            raise RuntimeError("API key is not configured")
        url = f"{DASHSCOPE_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "pong"},
                {"role": "user", "content": "ping"},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        response = self.http_client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
