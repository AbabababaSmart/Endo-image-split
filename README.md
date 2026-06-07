# Split Image

Two-stage pipeline for:

1. classifying whether an image is a splittable `composite` figure
2. splitting composite figures into subfigures with descriptions

## Setup

Create `.env` in the project root:

```env
BASE_URL=...
API_KEY=...
```

`BASE_URL` and `API_KEY` are shared by:

- stage 1 composite classification
- stage 2 `vlm` backend
- stage 2 `codex` backend

The code still accepts some legacy aliases internally, but `BASE_URL` + `API_KEY` is the preferred setup.

## Run

```bash
cd /mnt/data_10/mwx/workspace/multi_modal_rag/split_image
```

Run the full pipeline with the structured VLM split backend:

```bash
python3 main.py \
  --mode full \
  --split-backend vlm \
  --stage1-model your_stage1_model \
  --stage2-vlm-model your_stage2_vlm_model \
  --progress
```

Run only stage 2 with the Codex backend:

```bash
python3 main.py \
  --mode split-only \
  --split-backend codex \
  --stage2-codex-model your_stage2_codex_model \
  --progress
```

The pipeline supports parallel processing via `--parallelism` (default: `2`).

## Modes

- `classify-only`: run composite classification only. Output: `artifacts/composite_classification.jsonl`
- `split-only`: read `artifacts/composite_classification.jsonl`, keep rows with `status=classified_ok`, `is_endoscopic=true`, and `is_composite=true`, then run stage 2
- `full`: run classification first, then run stage 2 on the filtered composite rows

## Split Backends

- `vlm`: call a standard VLM endpoint and require structured JSON output for subfigure detection and description generation
- `codex`: call `codex exec` with an output schema and let Codex complete stage 2 agentically

Both backends produce the same final outputs:

- `artifacts/split_results.jsonl`: one split decision per source image
- `artifacts/splits/`: cropped subfigure images
- `runs/subfigure_pairs.jsonl`: one final row per cropped subfigure

When `--split-backend codex` is used, Codex owns the full stage 2 workflow: it locates
subfigures, saves the final crops under `artifacts/splits/`, projects the final boxes
back onto the source image under `artifacts/bbox_projections/`, reopens those files for
self-check, and only then returns the final bbox/description metadata. The host only
validates that the JSON is well formed and the referenced image files exist and open.

Typical staged usage:

```bash
python3 main.py \
  --mode classify-only \
  --stage1-model your_stage1_model \
  --progress

python3 main.py \
  --mode split-only \
  --split-backend vlm \
  --stage2-vlm-model your_stage2_vlm_model \
  --progress
```

or:

```bash
python3 main.py \
  --mode classify-only \
  --stage1-model your_stage1_model \
  --progress

python3 main.py \
  --mode split-only \
  --split-backend codex \
  --stage2-codex-model your_stage2_codex_model \
  --progress
```

## Key Arguments

- `--stage1-model`: model used for stage 1 classification
- `--split-backend`: `vlm` or `codex`
- `--stage2-vlm-model`: model used when `--split-backend=vlm`
- `--stage2-codex-model`: model used when `--split-backend=codex`
- `--codex-sandbox`: sandbox mode passed to `codex exec`, defaults to `danger-full-access` so the nested Codex agent can run Python/PIL and write crop artifacts
- `--resume`: reuse prior classification and split outputs
- `--limit`: only process the first N rows of the current todo set

## Data

Manifest loading order:

1. `artifacts/source_manifest.jsonl`
2. `artifacts/source_manifest_test.jsonl`
3. otherwise rebuild from `--input-jsonl`

The repository can be run directly with the bundled test set:

- `artifacts/source_manifest_test.jsonl`
- `test_data/images/`

For full-scale runs, you still need the external full image corpus and the upstream input JSONL.
