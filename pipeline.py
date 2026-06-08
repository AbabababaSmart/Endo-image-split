from __future__ import annotations

import concurrent.futures
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from .api_client import ChatResult, OpenAICompatibleClient, encode_image_path_to_data_url
    from .codex_exec_client import run_codex_exec
    from .env_utils import get_provider_config
    from .io_utils import append_jsonl, area_xyxy, clamp_box_xyxy, read_jsonl, robust_json_loads
    from .prompts import (
        build_stage1_messages,
        build_stage2_codex_prompt,
        build_stage2_vlm_messages,
    )
    from .provenance import build_manifest
except ImportError:
    from api_client import ChatResult, OpenAICompatibleClient, encode_image_path_to_data_url
    from codex_exec_client import run_codex_exec
    from env_utils import get_provider_config
    from io_utils import append_jsonl, area_xyxy, clamp_box_xyxy, read_jsonl, robust_json_loads
    from prompts import (
        build_stage1_messages,
        build_stage2_codex_prompt,
        build_stage2_vlm_messages,
    )
    from provenance import build_manifest

CLASSIFIED_OK = "classified_ok"
SPLIT_ERROR = "error"


class ModelResponseParseError(ValueError):
    def __init__(self, *, stage: str, response: ChatResult, cause: Exception) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.stage = stage
        self.response = response
        self.cause = cause


class TimedStageError(RuntimeError):
    def __init__(self, *, elapsed_s: float, cause: Exception) -> None:
        super().__init__(str(cause))
        self.elapsed_s = elapsed_s
        self.cause = cause


def _load_row_map(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            rows[sample_id] = row
    return rows


def _load_processed_split_ids(split_results_path: Path) -> set[str]:
    processed = set()
    if not split_results_path.exists():
        return processed
    for row in read_jsonl(split_results_path):
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id and "error_type" not in row:
            processed.add(sample_id)
    return processed


def _build_timing_row(
    *,
    stage: str,
    row: Dict[str, Any],
    elapsed_s: float,
    status: str,
    model: str = "",
    split_backend: str = "",
    split_count: Optional[int] = None,
    error_type: str = "",
    error_message: str = "",
) -> Dict[str, Any]:
    timing_row: Dict[str, Any] = {
        "stage": stage,
        "sample_id": str(row.get("sample_id", "") or ""),
        "image_path": str(row.get("image_path", "") or ""),
        "elapsed_s": round(float(elapsed_s), 6),
        "status": status,
    }
    if model:
        timing_row["model"] = model
    if split_backend:
        timing_row["split_backend"] = split_backend
    if split_count is not None:
        timing_row["split_count"] = int(split_count)
    if error_type:
        timing_row["error_type"] = error_type
    if error_message:
        timing_row["error_message"] = error_message
    return timing_row


def _normalize_manifest_rows(rows: List[Dict[str, Any]], *, work_dir: Path) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        image_path_str = str(row.get("image_path", "")).strip()
        if not image_path_str:
            continue
        image_path = Path(image_path_str).expanduser()
        if not image_path.is_absolute():
            image_path = (work_dir / image_path).resolve()
        else:
            image_path = image_path.resolve()
        normalized.append(
            {
                "sample_id": str(row.get("sample_id", "")).strip(),
                "image_path": str(image_path),
                "source_final_description": str(row.get("source_final_description", "") or "").strip(),
            }
        )
    return normalized


def _parse_stage1(response: ChatResult) -> Dict[str, Any]:
    obj = robust_json_loads(response.primary_text)
    if not isinstance(obj, dict):
        raise ValueError("Stage1 response is not a JSON object")
    return {
        "is_endoscopic": bool(obj.get("is_endoscopic", False)),
        "is_composite": bool(obj.get("is_composite", False)),
        "estimated_subfigure_count": max(0, int(obj.get("estimated_subfigure_count", 0) or 0)),
        "confidence": max(0.0, min(1.0, float(obj.get("confidence", 0.0) or 0.0))),
        "reason": str(obj.get("reason", "") or "").strip(),
    }


def _parse_stage2_vlm(response: ChatResult) -> List[Dict[str, Any]]:
    obj = robust_json_loads(response.primary_text)
    if not isinstance(obj, dict):
        raise ValueError("Stage2 response is not a JSON object")
    items = obj.get("subfigures", [])
    if not isinstance(items, list):
        raise ValueError("Stage2 response missing subfigures list")
    parsed: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_norm1000_xyxy")
        description = str(item.get("description", "") or "").strip()
        if not isinstance(bbox, list) or len(bbox) != 4 or not description:
            continue
        parsed.append(
            {
                "bbox_norm1000_xyxy": [int(round(float(v))) for v in bbox],
                "description": description,
            }
        )
    return parsed


def _make_client(args: Any, *, base_url: str, api_key: str) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        timeout_s=int(args.timeout_s),
        max_retries=int(args.max_retries),
    )


def _sanitize_norm1000_subfigures(subfigures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in subfigures:
        box = clamp_box_xyxy(item["bbox_norm1000_xyxy"], width=1000, height=1000)
        if area_xyxy(box) <= 0:
            continue
        key = (tuple(box), item["description"])
        if key in seen:
            continue
        seen.add(key)
        out.append({**item, "bbox_norm1000_xyxy": box})
    out.sort(key=lambda x: (x["bbox_norm1000_xyxy"][1], x["bbox_norm1000_xyxy"][0]))
    return out


def _sanitize_source_subfigures(
    subfigures: List[Dict[str, Any]],
    *,
    source_image_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    src_w, src_h = source_image_size
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in subfigures:
        bbox = item.get("bbox_source_xyxy")
        description = str(item.get("description", "") or "").strip()
        if not isinstance(bbox, list) or len(bbox) != 4 or not description:
            continue
        box = clamp_box_xyxy([int(round(float(v))) for v in bbox], width=src_w, height=src_h)
        if area_xyxy(box) <= 0:
            continue
        key = (tuple(box), description)
        if key in seen:
            continue
        seen.add(key)
        out.append({"bbox_source_xyxy": box, "description": description})
    out.sort(key=lambda x: (x["bbox_source_xyxy"][1], x["bbox_source_xyxy"][0]))
    return out


def _resolve_codex_output_path(path_value: Any, *, work_dir: Path) -> Path:
    path_str = str(path_value or "").strip()
    if not path_str:
        raise ValueError("Codex output path is empty")
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = work_dir / path
    path = path.resolve()
    try:
        path.relative_to(work_dir)
    except ValueError as exc:
        raise ValueError(f"Codex output path is outside work_dir: {path}") from exc
    return path


def _prepare_codex_input_image(*, image_path: Path, sample_id: str, input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        suffix = ".jpg"
    local_path = input_dir / f"{sample_id}{suffix}"
    shutil.copy2(image_path, local_path)
    return local_path.resolve()


def _sanitize_codex_subfigures(
    subfigures: List[Dict[str, Any]],
    *,
    sample_id: str,
    source_image_size: Tuple[int, int],
    work_dir: Path,
    splits_dir: Path,
) -> List[Dict[str, Any]]:
    src_w, src_h = source_image_size
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in subfigures:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_source_xyxy")
        description = str(item.get("description", "") or "").strip()
        if not isinstance(bbox, list) or len(bbox) != 4 or not description:
            continue
        box = clamp_box_xyxy([int(round(float(v))) for v in bbox], width=src_w, height=src_h)
        if area_xyxy(box) <= 0:
            continue

        split_path = _resolve_codex_output_path(item.get("split_image_path"), work_dir=work_dir)
        try:
            split_path.relative_to(splits_dir)
        except ValueError as exc:
            raise ValueError(f"Codex split image path is outside splits_dir: {split_path}") from exc
        raw_index = int(item.get("subfigure_index") or len(out) + 1)
        expected_path = (splits_dir / f"{sample_id}__{raw_index:02d}.jpg").resolve()
        if split_path != expected_path:
            raise ValueError(f"Codex split image path must be {expected_path}, got {split_path}")
        if not split_path.exists():
            raise FileNotFoundError(f"Codex split image not found: {split_path}")
        with Image.open(split_path) as crop_image:
            crop_w, crop_h = crop_image.size
            if crop_w <= 0 or crop_h <= 0:
                raise ValueError(f"Codex split image has invalid size: {split_path}")

        key = (tuple(box), str(split_path), description)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "subfigure_index": raw_index,
                "bbox_source_xyxy": box,
                "split_image_path": str(split_path),
                "description": description,
            }
        )
    out.sort(key=lambda x: x["subfigure_index"])
    expected_indices = list(range(1, len(out) + 1))
    actual_indices = [int(item["subfigure_index"]) for item in out]
    if actual_indices != expected_indices:
        raise ValueError(f"Codex subfigure_index values must be sequential from 1: {actual_indices}")
    return out


def _is_split_candidate(classification_row: Dict[str, Any]) -> bool:
    return bool(classification_row.get("is_endoscopic", False)) and bool(
        classification_row.get("is_composite", False)
    )


def _selected_stage2_model(args: Any) -> str:
    split_backend = str(args.split_backend).strip().lower()
    if split_backend == "vlm":
        return str(args.stage2_vlm_model).strip()
    return str(args.stage2_codex_model).strip()


def _map_norm1000_box_to_source_image(
    *,
    box_norm1000_xyxy: List[int],
    source_image_size: Tuple[int, int],
) -> List[int]:
    src_w, src_h = source_image_size
    x0, y0, x1, y1 = [int(v) for v in box_norm1000_xyxy]
    mapped = [
        int(round(src_w * x0 / 1000.0)),
        int(round(src_h * y0 / 1000.0)),
        int(round(src_w * x1 / 1000.0)),
        int(round(src_h * y1 / 1000.0)),
    ]
    return clamp_box_xyxy(mapped, width=src_w, height=src_h)


def _save_crop(
    *,
    image_path: Path,
    crop_box_xyxy: List[int],
    out_path: Path,
    image_format: str = "JPEG",
    jpeg_quality: int = 95,
) -> Tuple[Path, Tuple[int, int]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        crop = rgb.crop(tuple(crop_box_xyxy))
        save_kwargs = {"format": image_format}
        if image_format.upper() == "JPEG":
            save_kwargs["quality"] = int(jpeg_quality)
            save_kwargs["optimize"] = True
        crop.save(out_path, **save_kwargs)
        return out_path, crop.size


def _build_split_decision(
    *,
    sample_id: str,
    image_path: Path,
    subfigures: Optional[List[Dict[str, Any]]] = None,
    error_type: str = "",
    error_message: str = "",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "subfigures": subfigures or [],
    }
    if error_type:
        row["error_type"] = error_type
    if error_message:
        row["error_message"] = error_message
    return row


def _build_split_rows(
    row: Dict[str, Any],
    *,
    source_subfigures: List[Dict[str, Any]],
    artifacts_dir: Path,
) -> List[Dict[str, Any]]:
    sample_id = str(row["sample_id"])
    image_path = Path(row["image_path"]).expanduser().resolve()
    split_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(source_subfigures, start=1):
        split_path = artifacts_dir / "splits" / f"{sample_id}__{idx:02d}.jpg"
        _save_crop(
            image_path=image_path,
            crop_box_xyxy=item["bbox_source_xyxy"],
            out_path=split_path,
            image_format="JPEG",
            jpeg_quality=95,
        )
        split_rows.append(
            {
                "sample_id": sample_id,
                "source_image_path": str(image_path),
                "split_image_path": str(split_path),
                "subfigure_index": idx,
                "bbox_source_xyxy": item["bbox_source_xyxy"],
                "description": item["description"],
                "source_final_description": str(row.get("source_final_description", "") or "").strip(),
            }
        )
    return split_rows


def _build_codex_split_rows(
    row: Dict[str, Any],
    *,
    source_subfigures: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sample_id = str(row["sample_id"])
    image_path = Path(row["image_path"]).expanduser().resolve()
    split_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(source_subfigures, start=1):
        split_rows.append(
            {
                "sample_id": sample_id,
                "source_image_path": str(image_path),
                "split_image_path": item["split_image_path"],
                "subfigure_index": idx,
                "bbox_source_xyxy": item["bbox_source_xyxy"],
                "description": item["description"],
                "source_final_description": str(row.get("source_final_description", "") or "").strip(),
            }
        )
    return split_rows


def _classify_one(
    row: Dict[str, Any],
    *,
    client: OpenAICompatibleClient,
    stage1_model: str,
    api_image_max_edge: int,
    api_image_jpeg_quality: int,
) -> Dict[str, Any]:
    sample_id = str(row["sample_id"])
    image_path = Path(row["image_path"]).expanduser().resolve()
    target_image_url = encode_image_path_to_data_url(
        image_path,
        max_edge=api_image_max_edge,
        jpeg_quality=api_image_jpeg_quality,
    )
    stage1_response = client.chat(
        model=stage1_model,
        messages=build_stage1_messages(target_image_url),
        max_tokens=220,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        stage1 = _parse_stage1(stage1_response)
    except Exception as exc:
        raise ModelResponseParseError(stage="stage1", response=stage1_response, cause=exc) from exc
    return {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "status": CLASSIFIED_OK,
        "is_endoscopic": stage1["is_endoscopic"],
        "is_composite": stage1["is_composite"],
        "composite_confidence": stage1["confidence"],
        "estimated_subfigure_count": stage1["estimated_subfigure_count"],
        "reason": stage1["reason"],
    }


def _split_one_vlm(
    row: Dict[str, Any],
    *,
    client: OpenAICompatibleClient,
    stage2_vlm_model: str,
    api_image_max_edge: int,
    api_image_jpeg_quality: int,
    artifacts_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sample_id = str(row["sample_id"])
    image_path = Path(row["image_path"]).expanduser().resolve()
    target_image_url = encode_image_path_to_data_url(
        image_path,
        max_edge=api_image_max_edge,
        jpeg_quality=api_image_jpeg_quality,
    )
    stage2_response = client.chat(
        model=stage2_vlm_model,
        messages=build_stage2_vlm_messages(
            target_image_data_url=target_image_url,
            source_final_description=str(row.get("source_final_description", "") or "").strip(),
        ),
        max_tokens=2600,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        parsed_subfigures = _sanitize_norm1000_subfigures(_parse_stage2_vlm(stage2_response))
    except Exception as exc:
        raise ModelResponseParseError(stage="stage2_vlm", response=stage2_response, cause=exc) from exc

    with Image.open(image_path) as src_image:
        src_size = src_image.size

    source_subfigures = [
        {
            "bbox_source_xyxy": _map_norm1000_box_to_source_image(
                box_norm1000_xyxy=item["bbox_norm1000_xyxy"],
                source_image_size=src_size,
            ),
            "description": item["description"],
        }
        for item in parsed_subfigures
    ]

    if len(source_subfigures) < 2:
        return _build_split_decision(
            sample_id=sample_id,
            image_path=image_path,
        ), []

    split_rows = _build_split_rows(row, source_subfigures=source_subfigures, artifacts_dir=artifacts_dir)
    return _build_split_decision(
        sample_id=sample_id,
        image_path=image_path,
        subfigures=source_subfigures,
    ), split_rows


def _split_one_codex(
    row: Dict[str, Any],
    *,
    classification_row: Dict[str, Any],
    args: Any,
    artifacts_dir: Path,
    work_dir: Path,
    base_url: str,
    api_key: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sample_id = str(row["sample_id"])
    image_path = Path(row["image_path"]).expanduser().resolve()
    splits_dir = artifacts_dir / "splits"
    projection_path = artifacts_dir / "bbox_projections" / f"{sample_id}.jpg"
    codex_result_path = artifacts_dir / "codex_results" / f"{sample_id}.json"
    splits_dir.mkdir(parents=True, exist_ok=True)
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    codex_result_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"codex_input_{sample_id}_", dir="/tmp") as temp_dir:
        codex_image_path = _prepare_codex_input_image(
            image_path=image_path,
            sample_id=sample_id,
            input_dir=Path(temp_dir),
        )
        prompt = build_stage2_codex_prompt(
            sample_id=sample_id,
            image_path=str(codex_image_path),
            work_dir=str(work_dir),
            splits_dir=str(splits_dir),
            projection_image_path=str(projection_path),
            result_json_path=str(codex_result_path),
            source_final_description=str(row.get("source_final_description", "") or "").strip(),
            stage1_reason=str(classification_row.get("reason", "") or "").strip(),
            estimated_subfigure_count=int(classification_row.get("estimated_subfigure_count", 0) or 0),
        )
        result = run_codex_exec(
            prompt=prompt,
            image_path=codex_image_path,
            result_json_path=codex_result_path,
            work_dir=work_dir,
            base_url=base_url,
            api_key=api_key,
            model=str(args.stage2_codex_model).strip(),
            sandbox=str(args.codex_sandbox).strip(),
            timeout_s=int(args.timeout_s),
        )

        if str(result.get("sample_id", "") or "").strip() != sample_id:
            raise ValueError(f"Codex returned mismatched sample_id: {result.get('sample_id')}")

        with Image.open(image_path) as src_image:
            src_size = src_image.size
        source_subfigures = _sanitize_codex_subfigures(
            list(result.get("subfigures", []) or []),
            sample_id=sample_id,
            source_image_size=src_size,
            work_dir=work_dir,
            splits_dir=splits_dir.resolve(),
        )

        if not source_subfigures:
            return _build_split_decision(
                sample_id=sample_id,
                image_path=image_path,
            ), []

        projection_value = str(result.get("projection_image_path", "") or "").strip()
        if not projection_value:
            raise ValueError("Codex split result missing projection_image_path")
        returned_projection_path = _resolve_codex_output_path(projection_value, work_dir=work_dir)
        if returned_projection_path != projection_path.resolve():
            raise ValueError(f"Codex projection image path must be {projection_path.resolve()}, got {returned_projection_path}")
        if not returned_projection_path.exists():
            raise FileNotFoundError(f"Codex projection image not found: {returned_projection_path}")
        with Image.open(returned_projection_path) as projection_image:
            proj_w, proj_h = projection_image.size
            if proj_w <= 0 or proj_h <= 0:
                raise ValueError(f"Codex projection image has invalid size: {returned_projection_path}")

    split_rows = _build_codex_split_rows(row, source_subfigures=source_subfigures)
    return _build_split_decision(
        sample_id=sample_id,
        image_path=image_path,
        subfigures=source_subfigures,
    ), split_rows


def _ensure_manifest(args: Any, manifest_path: Path) -> List[Dict[str, Any]]:
    work_dir = Path(args.work_dir).expanduser().resolve()
    manifest_test_path = manifest_path.with_name("source_manifest_test.jsonl")

    if manifest_path.exists():
        if getattr(args, "progress", False):
            print(f"[startup] loading manifest: {manifest_path}")
        return _normalize_manifest_rows(list(read_jsonl(manifest_path)), work_dir=work_dir)

    if manifest_test_path.exists():
        if getattr(args, "progress", False):
            print(f"[startup] loading test manifest fallback: {manifest_test_path}")
        return _normalize_manifest_rows(list(read_jsonl(manifest_test_path)), work_dir=work_dir)

    if getattr(args, "progress", False):
        print("[startup] building source manifest from input jsonl...")
    return build_manifest(
        input_jsonl=Path(args.input_jsonl).expanduser().resolve(),
        manifest_path=manifest_path,
        progress=getattr(args, "progress", False),
    )


def _run_classification_stage(
    *,
    args: Any,
    manifest_rows: List[Dict[str, Any]],
    classification_path: Path,
    classify_timing_log_path: Path,
    manifest_path: Path,
    base_url: str,
    api_key: str,
    env_path: Path,
) -> Dict[str, Any]:
    existing_map = _load_row_map(classification_path) if args.resume else {}
    todo_rows = [row for row in manifest_rows if row["sample_id"] not in existing_map]
    if args.limit and args.limit > 0:
        todo_rows = todo_rows[: args.limit]
    if getattr(args, "progress", False):
        print(
            f"[classify] manifest_rows={len(manifest_rows)} existing={len(existing_map)} todo={len(todo_rows)}",
            flush=True,
        )

    summary = {
        "mode": "classify-only",
        "stage1_model": str(args.stage1_model).strip(),
        "env_file": str(env_path),
        "manifest_path": str(manifest_path),
        "classification_jsonl": str(classification_path),
        "classify_timing_log": str(classify_timing_log_path),
        "total_manifest_rows": len(manifest_rows),
        "to_process_rows": len(todo_rows),
        "processed_rows": 0,
        "classified_ok_rows": 0,
        "composite_rows": 0,
        "single_rows": 0,
        "error_rows": 0,
    }

    def _progress(message: str) -> None:
        if getattr(args, "progress", False):
            print(message, flush=True)

    def _worker(row: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        started_at = time.perf_counter()
        image_name = os.path.basename(str(row["image_path"]))
        try:
            _progress(f"[classify pending] start image={image_name}")
            client = _make_client(args, base_url=base_url, api_key=api_key)
            _progress(f"[classify pending] request_start image={image_name} model={str(args.stage1_model).strip()}")
            result = _classify_one(
                row,
                client=client,
                stage1_model=str(args.stage1_model).strip(),
                api_image_max_edge=int(args.api_image_max_edge),
                api_image_jpeg_quality=int(args.api_image_jpeg_quality),
            )
            _progress(
                f"[classify pending] request_done image={image_name} "
                f"is_composite={bool(result.get('is_composite', False))}"
            )
            return result, time.perf_counter() - started_at
        except Exception as exc:
            elapsed_s = time.perf_counter() - started_at
            raise TimedStageError(elapsed_s=elapsed_s, cause=exc) from exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.parallelism))) as executor:
        future_map = {executor.submit(_worker, row): row for row in todo_rows}
        for future in concurrent.futures.as_completed(future_map):
            row = future_map[future]
            elapsed_s = 0.0
            try:
                result, elapsed_s = future.result()
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, TimedStageError):
                    elapsed_s = exc.elapsed_s
                    cause = exc.cause
                else:
                    cause = exc
                result = {
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "status": SPLIT_ERROR,
                    "error_type": type(cause).__name__,
                    "error_message": str(cause),
                    "traceback": traceback.format_exc(limit=8),
                }

            append_jsonl(classification_path, [result])
            append_jsonl(
                classify_timing_log_path,
                [
                    _build_timing_row(
                        stage="classify",
                        row=row,
                        elapsed_s=elapsed_s,
                        status=str(result.get("status", "") or ""),
                        model=str(args.stage1_model).strip(),
                        error_type=str(result.get("error_type", "") or ""),
                        error_message=str(result.get("error_message", "") or ""),
                    )
                ],
            )
            summary["processed_rows"] += 1
            if result["status"] == CLASSIFIED_OK:
                summary["classified_ok_rows"] += 1
                if bool(result.get("is_composite", False)):
                    summary["composite_rows"] += 1
                else:
                    summary["single_rows"] += 1
            else:
                summary["error_rows"] += 1

            extra = f"is_composite={bool(result.get('is_composite', False))}" if result["status"] == CLASSIFIED_OK else "is_composite=?"
            _progress(
                f"[classify {summary['processed_rows']}/{summary['to_process_rows']}] "
                f"{result['status']} image={os.path.basename(str(row['image_path']))} {extra}"
            )

    final_map = _load_row_map(classification_path)
    classified_ok = [row for row in final_map.values() if row.get("status") == CLASSIFIED_OK]
    summary["total_classified_ok_rows"] = len(classified_ok)
    summary["total_composite_rows"] = sum(1 for row in classified_ok if bool(row.get("is_composite", False)))
    summary["total_single_rows"] = sum(1 for row in classified_ok if not bool(row.get("is_composite", False)))
    summary["composite_ratio"] = (
        summary["total_composite_rows"] / summary["total_classified_ok_rows"]
        if summary["total_classified_ok_rows"] > 0
        else 0.0
    )
    return summary


def _run_split_backend(
    row: Dict[str, Any],
    *,
    classification_row: Dict[str, Any],
    args: Any,
    artifacts_dir: Path,
    work_dir: Path,
    base_url: str,
    api_key: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    split_backend = str(args.split_backend).strip().lower()
    if split_backend == "vlm":
        client = _make_client(args, base_url=base_url, api_key=api_key)
        return _split_one_vlm(
            row,
            client=client,
            stage2_vlm_model=str(args.stage2_vlm_model).strip(),
            api_image_max_edge=int(args.api_image_max_edge),
            api_image_jpeg_quality=int(args.api_image_jpeg_quality),
            artifacts_dir=artifacts_dir,
        )
    return _split_one_codex(
        row,
        classification_row=classification_row,
        args=args,
        artifacts_dir=artifacts_dir,
        work_dir=work_dir,
        base_url=base_url,
        api_key=api_key,
    )


def _run_split_stage(
    *,
    args: Any,
    manifest_rows: List[Dict[str, Any]],
    classification_path: Path,
    split_results_path: Path,
    output_jsonl: Path,
    split_timing_log_path: Path,
    artifacts_dir: Path,
    work_dir: Path,
    base_url: str,
    api_key: str,
    env_path: Path,
) -> Dict[str, Any]:
    if not classification_path.exists():
        raise FileNotFoundError(
            f"Classification results not found: {classification_path}. "
            "Run classify-only first or use mode=full."
        )

    classification_map = _load_row_map(classification_path)
    split_processed_ids = _load_processed_split_ids(split_results_path) if args.resume else set()
    classified_ok_map = {k: v for k, v in classification_map.items() if v.get("status") == CLASSIFIED_OK}
    composite_ids = {k for k, v in classified_ok_map.items() if _is_split_candidate(v)}
    todo_rows = [
        row
        for row in manifest_rows
        if row["sample_id"] in composite_ids and row["sample_id"] not in split_processed_ids
    ]
    if args.limit and args.limit > 0:
        todo_rows = todo_rows[: args.limit]

    if getattr(args, "progress", False):
        print(
            f"[split] backend={args.split_backend} manifest_rows={len(manifest_rows)} "
            f"classified_ok={len(classified_ok_map)} composite={len(composite_ids)} "
            f"processed={len(split_processed_ids)} todo={len(todo_rows)}"
        )

    summary = {
        "mode": "split-only",
        "split_backend": str(args.split_backend).strip(),
        "stage2_model": _selected_stage2_model(args),
        "env_file": str(env_path),
        "classification_jsonl": str(classification_path),
        "split_results_jsonl": str(split_results_path),
        "output_jsonl": str(output_jsonl),
        "split_timing_log": str(split_timing_log_path),
        "total_manifest_rows": len(manifest_rows),
        "classified_ok_rows": len(classified_ok_map),
        "classified_composite_rows": len(composite_ids),
        "already_split_rows": len(split_processed_ids),
        "to_process_rows": len(todo_rows),
        "processed_split_rows": 0,
        "processed_sample_rows": 0,
        "empty_split_rows": 0,
        "error_rows": 0,
        "unclassified_rows": len(manifest_rows) - len(classified_ok_map),
    }

    if todo_rows:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _worker(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float]:
        started_at = time.perf_counter()
        try:
            decision, split_rows = _run_split_backend(
                row,
                classification_row=classified_ok_map[row["sample_id"]],
                args=args,
                artifacts_dir=artifacts_dir,
                work_dir=work_dir,
                base_url=base_url,
                api_key=api_key,
            )
            return decision, split_rows, time.perf_counter() - started_at
        except Exception as exc:
            raise TimedStageError(elapsed_s=time.perf_counter() - started_at, cause=exc) from exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.parallelism))) as executor:
        future_map = {executor.submit(_worker, row): row for row in todo_rows}
        for future in concurrent.futures.as_completed(future_map):
            row = future_map[future]
            elapsed_s = 0.0
            try:
                decision, split_rows, elapsed_s = future.result()
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, TimedStageError):
                    elapsed_s = exc.elapsed_s
                    cause = exc.cause
                else:
                    cause = exc
                decision = _build_split_decision(
                    sample_id=str(row["sample_id"]),
                    image_path=Path(row["image_path"]).expanduser().resolve(),
                    error_type=type(cause).__name__,
                    error_message=str(cause),
                )
                split_rows = []

            append_jsonl(split_results_path, [decision])
            if split_rows:
                append_jsonl(output_jsonl, split_rows)
            append_jsonl(
                split_timing_log_path,
                [
                    _build_timing_row(
                        stage="split",
                        row=row,
                        elapsed_s=elapsed_s,
                        status="error" if "error_type" in decision else ("split_empty" if not split_rows else "split_done"),
                        model=_selected_stage2_model(args),
                        split_backend=str(args.split_backend).strip(),
                        split_count=len(split_rows),
                        error_type=str(decision.get("error_type", "") or ""),
                        error_message=str(decision.get("error_message", "") or ""),
                    )
                ],
            )

            summary["processed_sample_rows"] += 1
            summary["processed_split_rows"] += len(split_rows)
            if "error_type" in decision:
                summary["error_rows"] += 1
            elif not split_rows:
                summary["empty_split_rows"] += 1

            if getattr(args, "progress", False):
                split_state = "error" if "error_type" in decision else ("split_empty" if not split_rows else "split_done")
                print(
                    f"[split {summary['processed_sample_rows']}/{summary['to_process_rows']}] "
                    f"{split_state} image={os.path.basename(str(row['image_path']))} "
                    f"splits={len(split_rows)}"
                )

    return summary


def _run_full_stage(
    *,
    args: Any,
    manifest_rows: List[Dict[str, Any]],
    classification_path: Path,
    split_results_path: Path,
    output_jsonl: Path,
    classify_timing_log_path: Path,
    split_timing_log_path: Path,
    artifacts_dir: Path,
    work_dir: Path,
    base_url: str,
    api_key: str,
    env_path: Path,
) -> Dict[str, Any]:
    classification_map = _load_row_map(classification_path) if classification_path.exists() else {}
    split_processed_ids = _load_processed_split_ids(split_results_path) if args.resume else set()
    todo_rows: List[Dict[str, Any]] = []
    for row in manifest_rows:
        existing = classification_map.get(row["sample_id"])
        if existing is None or existing.get("status") != CLASSIFIED_OK:
            todo_rows.append(row)
            continue
        if _is_split_candidate(existing) and row["sample_id"] not in split_processed_ids:
            todo_rows.append(row)
    if args.limit and args.limit > 0:
        todo_rows = todo_rows[: args.limit]

    if getattr(args, "progress", False):
        print(
            f"[full] backend={args.split_backend} manifest_rows={len(manifest_rows)} "
            f"classification_cache={len(classification_map)} processed={len(split_processed_ids)} "
            f"todo={len(todo_rows)}"
        )

    summary = {
        "mode": "full",
        "split_backend": str(args.split_backend).strip(),
        "stage1_model": str(args.stage1_model).strip(),
        "stage2_model": _selected_stage2_model(args),
        "env_file": str(env_path),
        "classification_jsonl": str(classification_path),
        "split_results_jsonl": str(split_results_path),
        "output_jsonl": str(output_jsonl),
        "classify_timing_log": str(classify_timing_log_path),
        "split_timing_log": str(split_timing_log_path),
        "total_manifest_rows": len(manifest_rows),
        "to_process_rows": len(todo_rows),
        "classification_reused_rows": 0,
        "classification_new_rows": 0,
        "processed_sample_rows": 0,
        "skipped_single_rows": 0,
        "processed_split_rows": 0,
        "empty_split_rows": 0,
        "error_rows": 0,
    }

    def _progress(message: str) -> None:
        if getattr(args, "progress", False):
            print(message, flush=True)

    def _worker(
        row: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[float]]:
        image_name = os.path.basename(str(row["image_path"]))
        _progress(f"[full pending] start image={image_name}")
        existing = classification_map.get(row["sample_id"])
        if existing is not None and existing.get("status") == CLASSIFIED_OK:
            classification_row = existing
            new_classification_row = None
            classify_elapsed_s = None
            _progress(f"[full pending] reuse_classification image={image_name}")
        else:
            classify_started_at = time.perf_counter()
            try:
                _progress(f"[full pending] classify_start image={image_name}")
                client = _make_client(args, base_url=base_url, api_key=api_key)
                classification_row = _classify_one(
                    row,
                    client=client,
                    stage1_model=str(args.stage1_model).strip(),
                    api_image_max_edge=int(args.api_image_max_edge),
                    api_image_jpeg_quality=int(args.api_image_jpeg_quality),
                )
                _progress(
                    f"[full pending] classify_done image={image_name} "
                    f"is_composite={bool(classification_row.get('is_composite', False))}"
                )
                classify_elapsed_s = time.perf_counter() - classify_started_at
            except Exception as exc:  # noqa: BLE001
                classify_elapsed_s = time.perf_counter() - classify_started_at
                return (
                    {
                        "sample_id": row["sample_id"],
                        "image_path": row["image_path"],
                        "status": SPLIT_ERROR,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    },
                    classify_elapsed_s,
                    None,
                    [],
                    None,
                )
            new_classification_row = classification_row

        if not _is_split_candidate(classification_row):
            return new_classification_row, classify_elapsed_s, None, [], None

        split_started_at = time.perf_counter()
        try:
            _progress(f"[full pending] split_start image={image_name} backend={args.split_backend}")
            decision, split_rows = _run_split_backend(
                row,
                classification_row=classification_row,
                args=args,
                artifacts_dir=artifacts_dir,
                work_dir=work_dir,
                base_url=base_url,
                api_key=api_key,
            )
            _progress(f"[full pending] split_done image={image_name} splits={len(split_rows)}")
            split_elapsed_s = time.perf_counter() - split_started_at
        except Exception as exc:  # noqa: BLE001
            split_elapsed_s = time.perf_counter() - split_started_at
            _progress(f"[full pending] split_error image={image_name} error={type(exc).__name__}: {exc}")
            decision = _build_split_decision(
                sample_id=str(row["sample_id"]),
                image_path=Path(row["image_path"]).expanduser().resolve(),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            split_rows = []
        return new_classification_row, classify_elapsed_s, decision, split_rows, split_elapsed_s

    if todo_rows:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.parallelism))) as executor:
        future_map = {executor.submit(_worker, row): row for row in todo_rows}
        for future in concurrent.futures.as_completed(future_map):
            row = future_map[future]
            classify_elapsed_s = None
            split_elapsed_s = None
            try:
                new_classification_row, classify_elapsed_s, decision, split_rows, split_elapsed_s = future.result()
            except Exception as exc:  # noqa: BLE001
                new_classification_row = None
                decision = _build_split_decision(
                    sample_id=str(row["sample_id"]),
                    image_path=Path(row["image_path"]).expanduser().resolve(),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                split_rows = []

            if new_classification_row is not None:
                append_jsonl(classification_path, [new_classification_row])
                if classify_elapsed_s is not None:
                    append_jsonl(
                        classify_timing_log_path,
                        [
                            _build_timing_row(
                                stage="classify",
                                row=row,
                                elapsed_s=classify_elapsed_s,
                                status=str(new_classification_row.get("status", "") or ""),
                                model=str(args.stage1_model).strip(),
                                error_type=str(new_classification_row.get("error_type", "") or ""),
                                error_message=str(new_classification_row.get("error_message", "") or ""),
                            )
                        ],
                    )
                if new_classification_row.get("status") == CLASSIFIED_OK:
                    classification_map[new_classification_row["sample_id"]] = new_classification_row
                summary["classification_new_rows"] += 1
            else:
                summary["classification_reused_rows"] += 1

            summary["processed_sample_rows"] += 1

            if new_classification_row is not None and new_classification_row.get("status") != CLASSIFIED_OK:
                summary["error_rows"] += 1
                _progress(
                    f"[full {summary['processed_sample_rows']}/{summary['to_process_rows']}] "
                    f"{new_classification_row['status']} image={os.path.basename(str(row['image_path']))} splits=0"
                )
                continue

            if decision is None:
                summary["skipped_single_rows"] += 1
                _progress(
                    f"[full {summary['processed_sample_rows']}/{summary['to_process_rows']}] "
                    f"{CLASSIFIED_OK} image={os.path.basename(str(row['image_path']))} is_composite=False"
                )
                continue

            append_jsonl(split_results_path, [decision])
            if split_rows:
                append_jsonl(output_jsonl, split_rows)
            if split_elapsed_s is not None:
                append_jsonl(
                    split_timing_log_path,
                    [
                        _build_timing_row(
                            stage="split",
                            row=row,
                            elapsed_s=split_elapsed_s,
                            status="error" if "error_type" in decision else ("split_empty" if not split_rows else "split_done"),
                            model=_selected_stage2_model(args),
                            split_backend=str(args.split_backend).strip(),
                            split_count=len(split_rows),
                            error_type=str(decision.get("error_type", "") or ""),
                            error_message=str(decision.get("error_message", "") or ""),
                        )
                    ],
                )

            if "error_type" in decision:
                summary["error_rows"] += 1
            elif not split_rows:
                summary["empty_split_rows"] += 1
            summary["processed_split_rows"] += len(split_rows)

            split_state = "error" if "error_type" in decision else ("split_empty" if not split_rows else "split_done")
            _progress(
                f"[full {summary['processed_sample_rows']}/{summary['to_process_rows']}] "
                f"{split_state} image={os.path.basename(str(row['image_path']))} splits={len(split_rows)}"
            )

    return summary


def run_pipeline(args: Any) -> Dict[str, Any]:
    base_url, api_key, env_path = get_provider_config(Path(args.env_file).expanduser().resolve())
    work_dir = Path(args.work_dir).expanduser().resolve()
    artifacts_dir = work_dir / "artifacts"
    runs_dir = work_dir / "runs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifacts_dir / "source_manifest.jsonl"
    classification_path = Path(args.classification_jsonl).expanduser().resolve()
    split_results_path = Path(args.split_results_jsonl).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    classify_timing_log_path = Path(args.classify_timing_log).expanduser().resolve()
    split_timing_log_path = Path(args.split_timing_log).expanduser().resolve()
    mode = str(args.mode).strip().lower()

    if not args.resume:
        if mode in {"classify-only", "full"} and classification_path.exists():
            classification_path.unlink()
        if mode in {"classify-only", "full"} and classify_timing_log_path.exists():
            classify_timing_log_path.unlink()
        if mode in {"split-only", "full"}:
            if split_results_path.exists():
                split_results_path.unlink()
            if output_jsonl.exists():
                output_jsonl.unlink()
            if split_timing_log_path.exists():
                split_timing_log_path.unlink()

    manifest_rows = _ensure_manifest(args, manifest_path)

    if mode == "classify-only":
        return _run_classification_stage(
            args=args,
            manifest_rows=manifest_rows,
            classification_path=classification_path,
            classify_timing_log_path=classify_timing_log_path,
            manifest_path=manifest_path,
            base_url=base_url,
            api_key=api_key,
            env_path=env_path,
        )
    if mode == "split-only":
        return _run_split_stage(
            args=args,
            manifest_rows=manifest_rows,
            classification_path=classification_path,
            split_results_path=split_results_path,
            output_jsonl=output_jsonl,
            split_timing_log_path=split_timing_log_path,
            artifacts_dir=artifacts_dir,
            work_dir=work_dir,
            base_url=base_url,
            api_key=api_key,
            env_path=env_path,
        )
    return _run_full_stage(
        args=args,
        manifest_rows=manifest_rows,
        classification_path=classification_path,
        split_results_path=split_results_path,
        output_jsonl=output_jsonl,
        classify_timing_log_path=classify_timing_log_path,
        split_timing_log_path=split_timing_log_path,
        artifacts_dir=artifacts_dir,
        work_dir=work_dir,
        base_url=base_url,
        api_key=api_key,
        env_path=env_path,
    )
