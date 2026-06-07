#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .env_utils import PROJECT_ROOT
    from .pipeline import run_pipeline
except ImportError:
    from env_utils import PROJECT_ROOT
    from pipeline import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split composite image-text pairs into subfigure crops with descriptions."
    )
    parser.add_argument(
        "--input-jsonl",
        default=(
            "/mnt/data_10/mwx/workspace/endo_image_text_pair_construction/"
            "output_image_text_pair_all/output_pairs_all_min_filtered.jsonl"
        ),
    )
    parser.add_argument(
        "--work-dir",
        default=str(PROJECT_ROOT),
        help="Project root used for artifacts/ and runs/ directories.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(PROJECT_ROOT / "runs" / "subfigure_pairs.jsonl"),
    )
    parser.add_argument(
        "--classification-jsonl",
        default=str(PROJECT_ROOT / "artifacts" / "composite_classification.jsonl"),
        help="Classification output reused by split-only/full modes.",
    )
    parser.add_argument(
        "--split-results-jsonl",
        default=str(PROJECT_ROOT / "artifacts" / "split_results.jsonl"),
        help="Per-source-image split decisions for either vlm or codex backends.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "classify-only", "split-only"],
        default="full",
        help="Run full pipeline, only composite classification, or only splitting from prior classification results.",
    )
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument(
        "--split-backend",
        choices=["vlm", "codex"],
        default="vlm",
        help="Second-stage backend: structured VLM output or codex exec agentic workflow.",
    )
    parser.add_argument("--stage1-model", default="", help="Model used for first-stage composite triage.")
    parser.add_argument(
        "--stage2-vlm-model",
        default="",
        help="Model used when --split-backend=vlm.",
    )
    parser.add_argument(
        "--stage2-codex-model",
        default="",
        help="Model used when --split-backend=codex.",
    )
    parser.add_argument(
        "--codex-sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="danger-full-access",
        help="Sandbox mode passed to codex exec for the second-stage codex backend.",
    )
    parser.add_argument("--api-image-max-edge", type=int, default=1536)
    parser.add_argument("--api-image-jpeg-quality", type=int, default=90)
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    mode = str(args.mode).strip().lower()
    split_backend = str(args.split_backend).strip().lower()

    if mode in {"classify-only", "full"} and not str(args.stage1_model).strip():
        parser.error("--stage1-model is required for classify-only and full modes.")

    if mode in {"split-only", "full"}:
        if split_backend == "vlm" and not str(args.stage2_vlm_model).strip():
            parser.error("--stage2-vlm-model is required when --split-backend=vlm.")
        if split_backend == "codex" and not str(args.stage2_codex_model).strip():
            parser.error("--stage2-codex-model is required when --split-backend=codex.")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
