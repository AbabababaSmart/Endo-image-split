from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield obj


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def robust_json_loads(text: str) -> Any:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.S)
    if match:
        raw = match.group(1)
    return json.loads(raw)


def make_sample_id(image_path: str) -> str:
    return hashlib.sha1(image_path.encode("utf-8")).hexdigest()


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def clamp_box_xyxy(box: List[int], width: int, height: int) -> List[int]:
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = clamp(x0, 0, width - 1)
    y0 = clamp(y0, 0, height - 1)
    x1 = clamp(x1, x0 + 1, width)
    y1 = clamp(y1, y0 + 1, height)
    return [x0, y0, x1, y1]


def area_xyxy(box: List[int]) -> int:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)
