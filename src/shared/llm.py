"""
Single choke point for all LLM calls. Every agent goes through here so that
(a) swapping providers is a one-line change, and (b) token accounting for
the benchmark's efficiency metric is never missed.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from src.shared.config import CONFIG


class LLMResponse:
    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _call_anthropic(system: str, prompt: str, max_tokens: int) -> LLMResponse:
    import anthropic

    client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)
    resp = client.messages.create(
        model=CONFIG.anthropic_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return LLMResponse(
        text=text,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )


def _call_openai(system: str, prompt: str, max_tokens: int) -> LLMResponse:
    from openai import OpenAI

    client = OpenAI(api_key=CONFIG.openai_api_key)
    resp = client.chat.completions.create(
        model=CONFIG.openai_model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    choice = resp.choices[0].message.content or ""
    usage = resp.usage
    return LLMResponse(
        text=choice,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


def _call_mock(_system: str, _prompt: str, _max_tokens: int) -> LLMResponse:
    """No API key configured — deterministic stub so the pipeline still
    runs end to end for local dev / dry runs. Replace by setting
    ANTHROPIC_API_KEY or OPENAI_API_KEY."""
    stub = (
        '{"action": "answer", "action_input": {}, '
        '"rationale": "MOCK MODE: no LLM key configured", '
        '"answer": "[mock answer — configure an API key]", "confidence": 0.0}'
    )
    return LLMResponse(text=stub, input_tokens=0, output_tokens=0)


def _call_groq(system: str, prompt: str, max_tokens: int) -> LLMResponse:
    """
    Uses Groq via its OpenAI-compatible endpoint.
    Avoids the native groq SDK which has CLR memory issues on Python 3.13 / Windows.
    Automatically handles rate limits with exponential backoff and cascading
    across models (e.g. gpt-oss-120b -> qwen3.8-27b -> gpt-oss-20b).
    """
    from openai import OpenAI, RateLimitError, APIError
    import logging
    import time

    client = OpenAI(
        api_key=CONFIG.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    primary_model = CONFIG.groq_model or "qwen/qwen3.8-27b"
    candidate_models = [primary_model, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    seen = set()
    models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

    last_error = None
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                msg = resp.choices[0].message
                choice = msg.content or getattr(msg, "reasoning", None) or ""
                usage = resp.usage
                return LLMResponse(
                    text=choice,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                )
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                status_code = getattr(e, "status_code", None)
                if (status_code == 429 or "rate_limit" in err_msg or "429" in err_msg) and attempt == 0:
                    sleep_time = 1.5
                    logging.getLogger("llm").warning(
                        "Groq rate limit on '%s' (attempt 1/2). Pausing %.1fs...",
                        model_name,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue
                logging.getLogger("llm").warning(
                    "Groq issue on '%s': %s. Trying next candidate model...",
                    model_name,
                    e,
                )
                break

    # If all remote models failed/rate-limited, graceful fallback so pipeline never crashes
    logging.getLogger("llm").error("All Groq models exhausted/rate-limited. Falling back to mock generator: %s", last_error)
    return _call_mock(system, prompt, max_tokens)


import hashlib
import os
import sqlite3
import threading
import time

_CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "llm_cache.db")
_CACHE_LOCK = threading.Lock()
_CACHE_CON = None
_LAST_REQUEST_TIME = 0.0


def _get_cache_db():
    global _CACHE_CON
    if _CACHE_CON is None:
        with _CACHE_LOCK:
            if _CACHE_CON is None:
                os.makedirs(os.path.dirname(_CACHE_DB_PATH), exist_ok=True)
                con = sqlite3.connect(_CACHE_DB_PATH, check_same_thread=False)
                con.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        text TEXT,
                        input_tokens INT,
                        output_tokens INT,
                        created_at REAL
                    );
                """)
                con.commit()
                _CACHE_CON = con
    return _CACHE_CON


def _lookup_cache(key: str) -> Optional[LLMResponse]:
    if os.getenv("DISABLE_LLM_CACHE") == "1":
        return None
    try:
        con = _get_cache_db()
        with _CACHE_LOCK:
            row = con.execute("SELECT text, input_tokens, output_tokens FROM cache WHERE key = ?;", (key,)).fetchone()
            if row:
                return LLMResponse(text=row[0], input_tokens=row[1], output_tokens=row[2])
    except Exception:
        pass
    return None


def _save_cache(key: str, resp: LLMResponse):
    if os.getenv("DISABLE_LLM_CACHE") == "1":
        return
    try:
        con = _get_cache_db()
        with _CACHE_LOCK:
            con.execute(
                "INSERT OR REPLACE INTO cache(key, text, input_tokens, output_tokens, created_at) VALUES (?, ?, ?, ?, ?);",
                (key, resp.text, resp.input_tokens, resp.output_tokens, time.time())
            )
            con.commit()
    except Exception:
        pass


def _rate_limit_throttle(min_interval: float = 1.0):
    """Gentle throttle to stay comfortably below Groq's 30 RPM limit."""
    global _LAST_REQUEST_TIME
    with _CACHE_LOCK:
        now = time.time()
        elapsed = now - _LAST_REQUEST_TIME
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _LAST_REQUEST_TIME = time.time()


def complete(system: str, prompt: str, max_tokens: int = 1024) -> LLMResponse:
    provider = CONFIG.llm_provider
    cache_key = hashlib.sha256(f"{provider}:{CONFIG.groq_model}:{system}:{prompt}:{max_tokens}".encode()).hexdigest()

    cached = _lookup_cache(cache_key)
    if cached is not None:
        return cached

    _rate_limit_throttle(min_interval=1.0)

    if provider == "groq" and CONFIG.groq_api_key:
        resp = _call_groq(system, prompt, max_tokens)
    elif provider == "anthropic" and CONFIG.anthropic_api_key:
        resp = _call_anthropic(system, prompt, max_tokens)
    elif provider == "openai" and CONFIG.openai_api_key:
        resp = _call_openai(system, prompt, max_tokens)
    elif CONFIG.groq_api_key:
        resp = _call_groq(system, prompt, max_tokens)
    elif CONFIG.anthropic_api_key:
        resp = _call_anthropic(system, prompt, max_tokens)
    elif CONFIG.openai_api_key:
        resp = _call_openai(system, prompt, max_tokens)
    else:
        resp = _call_mock(system, prompt, max_tokens)

    _save_cache(cache_key, resp)
    return resp
def complete_json(system: str, prompt: str, max_tokens: int = 1024) -> tuple[dict[str, Any], LLMResponse]:
    """Ask for strict JSON, parse defensively (strip code fences, retry-safe)."""
    json_instruction = (
        "\n\nRespond with ONLY a single valid JSON object. No prose before or "
        "after it, no markdown code fences."
    )
    resp = complete(system, prompt + json_instruction, max_tokens)
    text = resp.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # last-resort: find the outermost {...}
        start, end = text.find("{"), text.rfind("}")
        parsed = json.loads(text[start:end + 1]) if start != -1 and end != -1 else {}
    return parsed, resp

