from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


class CodexExecError(RuntimeError):
    pass


def _summarize_codex_failure(stdout: str, stderr: str, returncode: int) -> str:
    detail = stderr or stdout or f"exit code {returncode}"
    channel_errors = [
        line.strip()
        for line in detail.splitlines()
        if "No available channel" in line or "Service Unavailable" in line
    ]
    if channel_errors:
        return "\n".join(dict.fromkeys(channel_errors[-4:]))
    important = [
        line.strip()
        for line in detail.splitlines()
        if "No available channel" in line
        or "unexpected status" in line
        or "Service Unavailable" in line
        or "ERROR:" in line
        or "error:" in line.lower()
    ]
    if important:
        return "\n".join(dict.fromkeys(important[-8:]))
    return detail[-4000:]


def run_codex_exec(
    *,
    prompt: str,
    image_path: Optional[Path] = None,
    image_paths: Optional[List[Path]] = None,
    result_json_path: Path,
    work_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    sandbox: str = "workspace-write",
    timeout_s: int = 180,
) -> Dict[str, Any]:
    if image_paths is None:
        if image_path is None:
            raise ValueError("Either image_path or image_paths is required.")
        image_paths = [image_path]
    image_paths = [path.expanduser().resolve() for path in image_paths]
    result_json_path = result_json_path.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()

    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found for codex exec: {path}")

    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    result_json_path.unlink(missing_ok=True)

    cmd = [
        "codex",
        "exec",
        "--sandbox",
        sandbox,
        "--skip-git-repo-check",
    ]
    for path in image_paths:
        cmd.extend(["-i", str(path)])
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)

    env = os.environ.copy()
    env["CODEX_API_KEY"] = api_key
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = base_url
    env["BASE_URL"] = base_url

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_s)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CodexExecError(f"codex exec timed out after {timeout_s}s") from exc

    try:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            detail = _summarize_codex_failure(stdout, stderr, completed.returncode)
            raise CodexExecError(f"codex exec failed: {detail}")
        if not result_json_path.exists():
            detail = _summarize_codex_failure(stdout, stderr, completed.returncode)
            raise CodexExecError(f"codex exec did not produce the expected JSON result file: {detail}")
        return json.loads(result_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raw = result_json_path.read_text(encoding="utf-8") if result_json_path.exists() else stdout
        raise CodexExecError(f"codex exec returned invalid JSON: {raw[:1000]}") from exc
