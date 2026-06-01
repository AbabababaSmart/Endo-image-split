from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent


def load_env_file(env_path: Optional[Path] = None, override: bool = False) -> Path:
    candidates = [env_path] if env_path is not None else [PROJECT_ROOT / ".env"]
    for path in candidates:
        if path is None or not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            if override or key not in os.environ:
                os.environ[key] = value
        return path
    raise FileNotFoundError(f"Env file not found. Tried: {candidates}")


def get_api_config(env_path: Optional[Path] = None) -> Tuple[str, str, Path]:
    loaded_env = load_env_file(env_path=env_path, override=False)
    base_url = (
        os.getenv("BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("VLLM_BASE_URL")
        or ""
    ).strip()
    api_key = (
        os.getenv("DEER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
        or ""
    ).strip()
    if not base_url:
        raise RuntimeError("Missing BASE_URL in env file.")
    if not api_key:
        raise RuntimeError("Missing API key in env file.")
    return base_url, api_key, loaded_env
