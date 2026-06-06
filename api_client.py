from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from PIL import Image


def encode_image_path_to_data_url(
    image_path: Path,
    *,
    max_edge: int = 1536,
    jpeg_quality: int = 90,
) -> str:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        if max_edge and max(rgb.size) > max_edge:
            rgb.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=int(jpeg_quality), optimize=True)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@dataclass
class ChatResult:
    content: str
    reasoning_content: str
    finish_reason: str
    raw_response: Dict[str, Any]
    thinking_disabled: bool = False

    @property
    def primary_text(self) -> str:
        return (self.content or "").strip() or (self.reasoning_content or "").strip()

    @property
    def primary_source(self) -> str:
        if (self.content or "").strip():
            return "content"
        if (self.reasoning_content or "").strip():
            return "reasoning_content"
        return "empty"


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_s: int = 180,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = int(timeout_s)
        self.max_retries = int(max_retries)
        self.session = requests.Session()
        self.session.trust_env = False

    @staticmethod
    def _candidate_chat_urls(base_url: str) -> List[str]:
        normalized = (base_url or "").strip().rstrip("/")
        if not normalized:
            return []
        if normalized.endswith("/chat/completions"):
            return [normalized]

        parsed = urlparse(normalized)
        path = parsed.path.rstrip("/")
        candidates = [f"{normalized}/chat/completions"]
        if not path.endswith("/v1"):
            candidates.append(f"{normalized}/v1/chat/completions")

        deduped: List[str] = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _should_disable_thinking(
        *,
        model: str,
        response_format: Optional[Dict[str, str]],
    ) -> bool:
        normalized = (model or "").strip().lower()
        if not response_format:
            return False
        return "qwen3.6" in normalized

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatResult:
        last_error: Exception | None = None
        candidate_urls = self._candidate_chat_urls(self.base_url)
        if not candidate_urls:
            raise RuntimeError("Missing chat completion endpoint URL.")
        base_payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload_variants: List[Dict[str, Any]] = []
        primary_payload = dict(base_payload)
        thinking_disabled = False
        if response_format:
            primary_payload["response_format"] = response_format
        if self._should_disable_thinking(model=model, response_format=response_format):
            primary_payload["enable_thinking"] = False
            thinking_disabled = True
        payload_variants.append(primary_payload)

        if response_format:
            fallback_payload = dict(base_payload)
            if thinking_disabled:
                fallback_payload["enable_thinking"] = False
            payload_variants.append(fallback_payload)

        for attempt in range(self.max_retries):
            for payload in payload_variants:
                for url in candidate_urls:
                    try:
                        response = self.session.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=self.timeout_s,
                        )
                        if response.status_code >= 400:
                            raise RuntimeError(
                                f"HTTP {response.status_code} for {url}: {response.text[:1000]}"
                            )
                        data = response.json()
                        choice = data["choices"][0]
                        message = choice.get("message", {}) or {}
                        return ChatResult(
                            content=str(message.get("content") or ""),
                            reasoning_content=str(message.get("reasoning_content") or ""),
                            finish_reason=str(choice.get("finish_reason") or ""),
                            raw_response=data,
                            thinking_disabled=thinking_disabled,
                        )
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
            if attempt < self.max_retries - 1:
                time.sleep(1.0)
        assert last_error is not None
        raise last_error
