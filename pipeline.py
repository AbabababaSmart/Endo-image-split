from __future__ import annotations

import concurrent.futures
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from .api_client import ChatResult, OpenAICompatibleClient, encode_image_path_to_data_url
    from .env_utils import get_api_config
    from .io_utils import append_jsonl, area_xyxy, clamp_box_xyxy, read_jsonl, robust_json_loads
    from .prompts import build_stage1_messages, build_stage2_messages
    from .provenance import build_manifest
except ImportError:
    from api_client import ChatResult, OpenAICompatibleClient, encode_image_path_to_data_url
    from env_utils import get_api_config
    from io_utils import append_jsonl, area_xyxy, clamp_box_xyxy, read_jsonl, robust_json_loads
    from prompts import build_stage1_messages, build_stage2_messages
    from provenance import build_manifest

CLASSIFIED_OK = "classified_ok"
SPLIT_DONE = "split_done"
SPLIT_EMPTY = "split_empty"


class ModelResponseParseError(ValueError):
    def __init__(self, *, stage: str, response: ChatResult, cause: Exception) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.stage = stage
        self.response = response
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


def _load_processed_split_ids(split_decision_log_path: Path) -> set[str]:
    processed = set()
    if not split_decision_log_path.exists():
        return processed
    for row in read_jsonl(split_decision_log_path):
        status = str(row.get("status", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id and status in {SPLIT_DONE, SPLIT_EMPTY}:
            processed.add(sample_id)
    return processed


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


def _parse_stage2(response: ChatResult) -> List[Dict[str, Any]]:
    obj = robust_json_loads(response.primary_text)
    if not isinstance(obj, dict):
        raise ValueError("Stage2 response is not a JSON object")
    items = obj.get("subfigures", [])
    if not isinstance(items, list):
        raise ValueError("Stage2 response missing subfigures list")
    parsed: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_norm1000_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        description = str(item.get("description", "") or "").strip()
        if not description:
            continue
        parsed.append(
            {
                "bbox_norm1000_xyxy": [int(round(float(v))) for v in bbox],
                "description": description,
            }
        )
    return parsed


def _build_response_meta(prefix: str, response: ChatResult) -> Dict[str, Any]:
    return {
        f"{prefix}_output_text": response.primary_text,
        f"{prefix}_output_source": response.primary_source,
        f"{prefix}_content": response.content,
        f"{prefix}_reasoning_content": response.reasoning_content,
        f"{prefix}_finish_reason": response.finish_reason,
        f"{prefix}_thinking_disabled": response.thinking_disabled,
    }


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


def _is_split_candidate(classification_row: Dict[str, Any]) -> bool:
    return bool(classification_row.get("is_endoscopic", False)) and bool(
        classification_row.get("is_composite", False)
    )


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


def _split_one(
    row: Dict[str, Any],
    *,
    classification_row: Dict[str, Any],
    client: OpenAICompatibleClient,
    stage1_model: str,
    stage2_model: str,
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
        model=stage2_model,
        messages=build_stage2_messages(
            target_image_data_url=target_image_url,
            source_final_description=str(row.get("source_final_description", "") or "").strip(),
        ),
        max_tokens=2600,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        parsed_subfigures = _sanitize_norm1000_subfigures(_parse_stage2(stage2_response))
    except Exception as exc:
        raise ModelResponseParseError(stage="stage2", response=stage2_response, cause=exc) from exc

    if len(parsed_subfigures) < 2:
        return {
            "sample_id": sample_id,
            "image_path": str(image_path),
            "status": SPLIT_EMPTY,
            "is_composite": True,
            "estimated_subfigure_count": int(classification_row.get("estimated_subfigure_count", 0) or 0),
            "reason": "model_returned_fewer_than_two_valid_subfigures",
            "stage1_model": stage1_model,
            "stage2_model": stage2_model,
            **_build_response_meta("stage2", stage2_response),
        }, []

    with Image.open(image_path) as src_image:
        src_size = src_image.size

    split_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(parsed_subfigures, start=1):
        box_norm1000 = item["bbox_norm1000_xyxy"]
        box_source = _map_norm1000_box_to_source_image(
            box_norm1000_xyxy=box_norm1000,
            source_image_size=src_size,
        )
        split_path = artifacts_dir / "splits" / f"{sample_id}__{idx:02d}.jpg"
        _save_crop(
            image_path=image_path,
            crop_box_xyxy=box_source,
            out_path=split_path,
            image_format="JPEG",
            jpeg_quality=95,
        )
        split_rows.append(
            {
                "sample_id": sample_id,
                "source_image_path": str(image_path),
                "split_image_path": str(split_path),
                "description": item["description"],
                "source_final_description": str(row.get("source_final_description", "") or "").strip(),
                "subfigure_bbox_norm1000_xyxy": box_norm1000,
                "subfigure_bbox_source_xyxy": box_source,
                "stage1_model": stage1_model,
                "stage2_model": stage2_model,
            }
        )

    return {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "status": SPLIT_DONE,
        "is_composite": True,
        "estimated_subfigure_count": int(classification_row.get("estimated_subfigure_count", 0) or 0),
        "split_count": len(split_rows),
        "reason": str(classification_row.get("reason", "") or "").strip(),
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        **_build_response_meta("stage2", stage2_response),
    }, split_rows


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
        print(f"[classify] manifest_rows={len(manifest_rows)} existing={len(existing_map)} todo={len(todo_rows)}")

    summary = {
        "mode": "classify-only",
        "env_file": str(env_path),
        "manifest_path": str(manifest_path),
        "classification_jsonl": str(classification_path),
        "total_manifest_rows": len(manifest_rows),
        "to_process_rows": len(todo_rows),
        "processed_rows": 0,
        "classified_ok_rows": 0,
        "composite_rows": 0,
        "single_rows": 0,
        "error_rows": 0,
    }

    def _worker(row: Dict[str, Any]) -> Dict[str, Any]:
        client = _make_client(args, base_url=base_url, api_key=api_key)
        return _classify_one(
            row,
            client=client,
            stage1_model=args.stage1_model or args.model,
            api_image_max_edge=int(args.api_image_max_edge),
            api_image_jpeg_quality=int(args.api_image_jpeg_quality),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.parallelism))) as executor:
        future_map = {executor.submit(_worker, row): row for row in todo_rows}
        for future in concurrent.futures.as_completed(future_map):
            row = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
                if isinstance(exc, ModelResponseParseError):
                    result.update(
                        {
                            "stage1_output_text": exc.response.primary_text,
                            "stage1_content": exc.response.content,
                            "stage1_reasoning_content": exc.response.reasoning_content,
                        }
                    )

            append_jsonl(classification_path, [result])
            summary["processed_rows"] += 1
            if result["status"] == CLASSIFIED_OK:
                summary["classified_ok_rows"] += 1
                if bool(result.get("is_composite", False)):
                    summary["composite_rows"] += 1
                else:
                    summary["single_rows"] += 1
            else:
                summary["error_rows"] += 1

            if getattr(args, "progress", False):
                extra = f"is_composite={bool(result.get('is_composite', False))}" if result["status"] == CLASSIFIED_OK else "is_composite=?"
                print(
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


def _run_split_stage(
    *,
    args: Any,
    manifest_rows: List[Dict[str, Any]],
    classification_path: Path,
    split_decision_log_path: Path,
    output_jsonl: Path,
    artifacts_dir: Path,
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
    split_processed_ids = _load_processed_split_ids(split_decision_log_path) if args.resume else set()
    classified_ok_map = {k: v for k, v in classification_map.items() if v.get("status") == CLASSIFIED_OK}
    composite_ids = {k for k, v in classified_ok_map.items() if _is_split_candidate(v)}
    todo_rows = [row for row in manifest_rows if row["sample_id"] in composite_ids and row["sample_id"] not in split_processed_ids]
    if args.limit and args.limit > 0:
        todo_rows = todo_rows[: args.limit]

    if getattr(args, "progress", False):
        print(
            f"[split] manifest_rows={len(manifest_rows)} classified_ok={len(classified_ok_map)} "
            f"composite={len(composite_ids)} processed={len(split_processed_ids)} todo={len(todo_rows)}"
        )

    summary = {
        "mode": "split-only",
        "env_file": str(env_path),
        "classification_jsonl": str(classification_path),
        "split_decision_log_path": str(split_decision_log_path),
        "output_jsonl": str(output_jsonl),
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

    def _worker(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        client = _make_client(args, base_url=base_url, api_key=api_key)
        return _split_one(
            row,
            classification_row=classified_ok_map[row["sample_id"]],
            client=client,
            stage1_model=args.stage1_model or args.model,
            stage2_model=args.stage2_model or args.model,
            api_image_max_edge=int(args.api_image_max_edge),
            api_image_jpeg_quality=int(args.api_image_jpeg_quality),
            artifacts_dir=artifacts_dir,
        )

    if todo_rows:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.parallelism))) as executor:
        future_map = {executor.submit(_worker, row): row for row in todo_rows}
        for future in concurrent.futures.as_completed(future_map):
            row = future_map[future]
            try:
                decision, split_rows = future.result()
            except Exception as exc:  # noqa: BLE001
                decision = {
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                    "stage1_model": args.stage1_model or args.model,
                    "stage2_model": args.stage2_model or args.model,
                }
                if isinstance(exc, ModelResponseParseError):
                    decision.update(_build_response_meta(exc.stage, exc.response))
                split_rows = []

            append_jsonl(split_decision_log_path, [decision])
            if split_rows:
                append_jsonl(output_jsonl, split_rows)

            summary["processed_sample_rows"] += 1
            summary["processed_split_rows"] += len(split_rows)
            if decision["status"] == SPLIT_EMPTY:
                summary["empty_split_rows"] += 1
            elif decision["status"] == "error":
                summary["error_rows"] += 1

            if getattr(args, "progress", False):
                print(
                    f"[split {summary['processed_sample_rows']}/{summary['to_process_rows']}] "
                    f"{decision['status']} image={os.path.basename(str(row['image_path']))} "
                    f"splits={len(split_rows)}"
                )

    return summary


def _run_full_stage(
    *,
    args: Any,
    manifest_rows: List[Dict[str, Any]],
    classification_path: Path,
    split_decision_log_path: Path,
    output_jsonl: Path,
    artifacts_dir: Path,
    base_url: str,
    api_key: str,
    env_path: Path,
) -> Dict[str, Any]:
    classification_map = _load_row_map(classification_path) if classification_path.exists() else {}
    split_processed_ids = _load_processed_split_ids(split_decision_log_path) if args.resume else set()
    todo_rows = [row for row in manifest_rows if row["sample_id"] not in split_processed_ids]
    if args.limit and args.limit > 0:
        todo_rows = todo_rows[: args.limit]
    if getattr(args, "progress", False):
        print(
            f"[full] manifest_rows={len(manifest_rows)} classification_cache={len(classification_map)} "
            f"processed={len(split_processed_ids)} todo={len(todo_rows)}"
        )

    summary = {
        "mode": "full",
        "env_file": str(env_path),
        "classification_jsonl": str(classification_path),
        "split_decision_log_path": str(split_decision_log_path),
        "output_jsonl": str(output_jsonl),
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

    def _worker(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        client = _make_client(args, base_url=base_url, api_key=api_key)
        existing = classification_map.get(row["sample_id"])
        if existing is not None and existing.get("status") == CLASSIFIED_OK:
            classification_row = existing
            new_classification_row = None
        else:
            classification_row = _classify_one(
                row,
                client=client,
                stage1_model=args.stage1_model or args.model,
                api_image_max_edge=int(args.api_image_max_edge),
                api_image_jpeg_quality=int(args.api_image_jpeg_quality),
            )
            new_classification_row = classification_row

        if not _is_split_candidate(classification_row):
            return new_classification_row, classification_row, []

        split_decision, split_rows = _split_one(
            row,
            classification_row=classification_row,
            client=client,
            stage1_model=args.stage1_model or args.model,
            stage2_model=args.stage2_model or args.model,
            api_image_max_edge=int(args.api_image_max_edge),
            api_image_jpeg_quality=int(args.api_image_jpeg_quality),
            artifacts_dir=artifacts_dir,
        )
        return new_classification_row, split_decision, split_rows

    if todo_rows:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.parallelism))) as executor:
        future_map = {executor.submit(_worker, row): row for row in todo_rows}
        for future in concurrent.futures.as_completed(future_map):
            row = future_map[future]
            try:
                new_classification_row, decision, split_rows = future.result()
            except Exception as exc:  # noqa: BLE001
                new_classification_row = {
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                    "stage1_model": args.stage1_model or args.model,
                }
                if isinstance(exc, ModelResponseParseError):
                    new_classification_row.update(_build_response_meta(exc.stage, exc.response))
                decision = new_classification_row
                split_rows = []

            if new_classification_row is not None:
                append_jsonl(classification_path, [new_classification_row])
                if new_classification_row.get("status") == CLASSIFIED_OK:
                    classification_map[new_classification_row["sample_id"]] = new_classification_row
                summary["classification_new_rows"] += 1
            else:
                summary["classification_reused_rows"] += 1

            if decision.get("status") in {SPLIT_DONE, SPLIT_EMPTY, "error"}:
                append_jsonl(split_decision_log_path, [decision])
                if split_rows:
                    append_jsonl(output_jsonl, split_rows)

            summary["processed_sample_rows"] += 1
            status = str(decision.get("status", "")).strip()
            if status == CLASSIFIED_OK and not bool(decision.get("is_composite", False)):
                summary["skipped_single_rows"] += 1
            elif status == SPLIT_EMPTY:
                summary["empty_split_rows"] += 1
            elif status == "error":
                summary["error_rows"] += 1
            summary["processed_split_rows"] += len(split_rows)

            if getattr(args, "progress", False):
                suffix = f"splits={len(split_rows)}"
                if status == CLASSIFIED_OK:
                    suffix = f"is_composite={bool(decision.get('is_composite', False))}"
                print(
                    f"[full {summary['processed_sample_rows']}/{summary['to_process_rows']}] "
                    f"{status} image={os.path.basename(str(row['image_path']))} {suffix}"
                )

    return summary


def run_pipeline(args: Any) -> Dict[str, Any]:
    base_url, api_key, env_path = get_api_config(Path(args.env_file).expanduser().resolve())
    work_dir = Path(args.work_dir).expanduser().resolve()
    artifacts_dir = work_dir / "artifacts"
    runs_dir = work_dir / "runs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifacts_dir / "source_manifest.jsonl"
    classification_path = Path(args.classification_jsonl).expanduser().resolve()
    split_decision_log_path = artifacts_dir / "decision_log.jsonl"
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    mode = str(args.mode).strip().lower()

    if not args.resume:
        if mode in {"classify-only", "full"} and classification_path.exists():
            classification_path.unlink()
        if mode in {"split-only", "full"}:
            if split_decision_log_path.exists():
                split_decision_log_path.unlink()
            if output_jsonl.exists():
                output_jsonl.unlink()

    manifest_rows = _ensure_manifest(args, manifest_path)

    if mode == "classify-only":
        return _run_classification_stage(
            args=args,
            manifest_rows=manifest_rows,
            classification_path=classification_path,
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
            split_decision_log_path=split_decision_log_path,
            output_jsonl=output_jsonl,
            artifacts_dir=artifacts_dir,
            base_url=base_url,
            api_key=api_key,
            env_path=env_path,
        )
    return _run_full_stage(
        args=args,
        manifest_rows=manifest_rows,
        classification_path=classification_path,
        split_decision_log_path=split_decision_log_path,
        output_jsonl=output_jsonl,
        artifacts_dir=artifacts_dir,
        base_url=base_url,
        api_key=api_key,
        env_path=env_path,
    )
