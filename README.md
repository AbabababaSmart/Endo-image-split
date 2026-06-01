# Split Image

Two-stage pipeline for:

1. classifying whether an image is a splittable `composite` figure
2. splitting composite figures into subfigures with descriptions

## Setup

Create `.env` in the project root:

```env
BASE_URL=...
DEER_API_KEY=...
```

## Run

```bash
cd Endo-image-split/
```

Run the full pipeline:

```bash
python3 main.py \
  --mode full \
  --model your_vlm_model \  # 根据API提供的可选模型
  --progress
```

## Modes

- `classify-only`: run composite classification only. Output: `artifacts/composite_classification.jsonl`
- `split-only`: run splitting only for rows with `is_composite=true` in `artifacts/composite_classification.jsonl`
- `full`: run classification first, then split composite figures

Typical staged usage:

```bash
python3 main.py --mode classify-only --model your_vlm_model --progress
python3 main.py --mode split-only --model your_vlm_model --progress
```

## Data

Manifest loading order:

1. `artifacts/source_manifest.jsonl`
2. `artifacts/source_manifest_test.jsonl`
3. otherwise rebuild from `--input-jsonl`

The repository can be run directly with the bundled test set:

- `artifacts/source_manifest_test.jsonl`
- `test_data/images/`

For full-scale runs, you still need the external full image corpus and the upstream input JSONL.
