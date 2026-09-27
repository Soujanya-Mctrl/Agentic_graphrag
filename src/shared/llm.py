"""
Single choke point for all LLM calls. Every agent goes through here so that
(a) swapping providers is a one-line change, and (b) token accounting for
the benchmark's efficiency metric is never missed.
"""
from __future__ import annotations

import json
from typing import Any

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


def _call_mock(system: str, prompt: str, max_tokens: int) -> LLMResponse:
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
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=CONFIG.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model=CONFIG.groq_model,
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


def complete(system: str, prompt: str, max_tokens: int = 1024) -> LLMResponse:
    provider = CONFIG.llm_provider
    if provider == "groq" and CONFIG.groq_api_key:
        return _call_groq(system, prompt, max_tokens)
    if provider == "anthropic" and CONFIG.anthropic_api_key:
        return _call_anthropic(system, prompt, max_tokens)
    if provider == "openai" and CONFIG.openai_api_key:
        return _call_openai(system, prompt, max_tokens)
    # Auto-detect by available key (fallback when LLM_PROVIDER is unset or unavailable).
    if CONFIG.groq_api_key:
        return _call_groq(system, prompt, max_tokens)
    if CONFIG.anthropic_api_key:
        return _call_anthropic(system, prompt, max_tokens)
    if CONFIG.openai_api_key:
        return _call_openai(system, prompt, max_tokens)
    return _call_mock(system, prompt, max_tokens)
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
