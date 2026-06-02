from __future__ import annotations

from os import path
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from multi_modal_rag.split_image.io_utils import make_sample_id, read_jsonl, write_jsonl


def build_manifest(
    *,
    input_jsonl: Path,
    manifest_path: Path,
    progress: bool = False,
    progress_every: int = 1000,
    logger: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    def _log(message: str) -> None:
        if logger is not None:
            logger(message)
        elif progress:
            print(message)

    seen = set()
    rows: List[Dict[str, Any]] = []
    scanned = 0

    for row in read_jsonl(input_jsonl):
        scanned += 1
        if progress and progress_every > 0 and scanned % progress_every == 0:
            _log(f"[manifest] scanned={scanned} kept={len(rows)}")
        image_path_str = str(row.get("image_path", "")).strip()
        if not image_path_str or image_path_str in seen:
            continue
        seen.add(image_path_str)

        try:
            image_path = Path(image_path_str).expanduser().resolve()
        except Exception:
            continue
        if not path.exists(image_path):
            continue

        rows.append(
            {
                "sample_id": make_sample_id(image_path_str),
                "image_path": str(image_path),
                "source_final_description": str(row.get("final_description", "") or "").strip(),
            }
        )

    write_jsonl(manifest_path, rows)
    _log(f"[manifest] done scanned={scanned} kept={len(rows)} out={manifest_path}")
    return rows
