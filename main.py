#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_modal_rag.split_image.env_utils import PROJECT_ROOT
from multi_modal_rag.split_image.pipeline import run_pipeline


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
        "--mode",
        choices=["full", "classify-only", "split-only"],
        default="full",
        help="Run full pipeline, only composite classification, or only splitting from prior classification results.",
    )
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--model", required=True, help="Default VLM model for both stages.")
    parser.add_argument("--stage1-model", default="", help="Optional override model for composite triage.")
    parser.add_argument("--stage2-model", default="", help="Optional override model for split extraction.")
    parser.add_argument("--api-image-max-edge", type=int, default=1536)
    parser.add_argument("--api-image-jpeg-quality", type=int, default=90)
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
