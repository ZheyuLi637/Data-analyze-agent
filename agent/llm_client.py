from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMConfig:
    enabled: bool
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        enabled = os.getenv("LLM_ENABLED", "true").lower() not in {"0", "false", "no"}
        api_key = os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return cls(enabled=enabled, api_key=api_key, base_url=base_url, model=model)

    @property
    def ready(self) -> bool:
        return self.enabled and bool(self.api_key and self.base_url and self.model)


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig | None = None, timeout: int = 30) -> None:
        self.config = config or LLMConfig.from_env()
        self.timeout = timeout

    @property
    def ready(self) -> bool:
        return self.config.ready

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        if not self.ready:
            raise RuntimeError("LLM is not configured. Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

