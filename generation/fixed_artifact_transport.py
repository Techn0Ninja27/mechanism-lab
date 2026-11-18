"""Explicit HTTP adapter; construction neither reads credentials nor sends a request."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Any, Callable

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def openrouter_transport(key_loader: Callable[[], str], *, opener: Callable[..., Any] = urllib.request.urlopen,
                         timeout: float = 180) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a single-attempt transport. Only invocation reads the key and sends.

    Authentication never enters the returned record. This adapter has no retry,
    logging, budgeting, or campaign authority; the fixed client owns those checks.
    """
    if not callable(key_loader) or not callable(opener):
        raise TypeError("credential loader and opener must be callable")

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        key = key_loader()
        if not isinstance(key, str) or not key.strip():
            raise ValueError("OpenRouter credential is unavailable")
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(ENDPOINT, data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
        with opener(request, timeout=timeout) as response:
            raw = response.read()
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or len(data["choices"]) != 1:
            raise ValueError("provider response must contain exactly one choice")
        choice = data["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("provider choice/message malformed")
        return {"response_id": data.get("id"), "resolved_model": data.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "content": choice["message"].get("content"), "usage": data.get("usage"),
                "provider": data.get("provider"),
                "provider_response_sha256": hashlib.sha256(raw).hexdigest()}

    return send
