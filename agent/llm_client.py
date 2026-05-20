from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class LLMConfig:
    enabled: bool
    api_key: str
    base_url: str
    model: str
    site_url: str = ""
    app_name: str = "COMPSCI 767 Data Analysis Agent"

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "LLMConfig":
        local_env = _read_env_file(env_file)

        def value(name: str, default: str = "") -> str:
            return os.getenv(name, local_env.get(name, default))

        enabled = value("LLM_ENABLED", "true").lower() not in {"0", "false", "no"}
        return cls(
            enabled=enabled,
            api_key=value("LLM_API_KEY"),
            base_url=value("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=value("LLM_MODEL", "gpt-4o-mini"),
            site_url=value("LLM_SITE_URL"),
            app_name=value("LLM_APP_NAME", "COMPSCI 767 Data Analysis Agent"),
        )

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
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url
        if self.config.app_name:
            headers["X-Title"] = self.config.app_name

        response = requests.post(
            url,
            headers=headers,
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


def _read_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        value = raw_value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values
