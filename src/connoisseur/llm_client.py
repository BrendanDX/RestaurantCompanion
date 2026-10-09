from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI

from connoisseur import config


def get_client() -> OpenAI:
    return OpenAI(api_key=config.get_api_key(), base_url=config.DEEPSEEK_BASE_URL)


def _extra_kwargs() -> dict:
    """Optional thinking-mode switch (DEEPSEEK_THINKING=enabled|disabled). Unset = send nothing."""
    mode = os.environ.get("DEEPSEEK_THINKING", "").strip().lower()
    if mode in ("enabled", "disabled"):
        return {"extra_body": {"thinking": {"type": mode}}}
    return {}


def chat(
    system_msg: str,
    user_msg: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """System + user prompt in, reply text out (drop-in for the old `llm_model`)."""
    kwargs: dict = {
        "model": model or config.TEXT_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        **_extra_kwargs(),
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = get_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content


def vision_chat(
    system_msg: str,
    prompt_txt: str,
    image_path: str | Path,
    *,
    max_tokens: int = 300,
    model: str | None = None,
) -> str:
    """Caption/describe a local image (drop-in for the old `vision_llm`)."""
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_txt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ],
        },
    ]
    response = get_client().chat.completions.create(
        model=model or config.VISION_MODEL,
        max_tokens=max_tokens,
        messages=messages,
        **_extra_kwargs(),
    )
    return response.choices[0].message.content


def get_langchain_chat_model(temperature: float = 0.7):
    """LangChain chat model pointed at DeepSeek (used by Labs 9 and 12)."""
    from langchain_openai import ChatOpenAI  # lazy: only needed for LangChain labs

    return ChatOpenAI(
        model=config.TEXT_MODEL,
        api_key=config.get_api_key(),
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )
