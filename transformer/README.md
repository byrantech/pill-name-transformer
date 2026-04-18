# pill name transformer

Train a tiny, character-level transformer on real drug names, then generate fake prescription-style names locally on Apple Silicon with a clean path to ESP32-S3 deployment.

## What this does

- Pulls real drug names from openFDA (`generic_name`, `brand_name`, `substance_name`)
- Filters to strict brand-style invented names (single-token, non-English-word bias)
- Merges `data/curated_brand_names.txt` and removes entries in `data/brand_blocklist.txt`
- Trains a small causal transformer for name generation
- Generates fake names on your Mac
- Exports weights to `.npz` plus an optional C header scaffold for firmware integration

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python generate.py --count 20
```

## Tuned for your use case

The defaults intentionally keep model size small:

- `d_model=64`
- `heads=4`
- `layers=2`
- `max_len=24`

This is a practical balance between novelty and eventual embedded portability.

## Important paths

- Real names dataset: `data/drug_names.json`
- Trained model: `artifacts/model.pt`
- Exported weights: `artifacts/model_weights.npz`
- Vocabulary: `artifacts/vocab.json`

## Export for ESP32 work

```bash
python export_c_header.py
```

This produces `artifacts/model_export.h`. It is a bridge format: you can start firmware integration quickly, then switch to quantized weights (`int8`/`int16`) once your runtime is stable.

### On-device inference (ESP32-S3 + ST7789)

Firmware loads weights from `include/model_weights_generated.h`, produced by:

```bash
source .venv/bin/activate
python transformer/export_esp32_weights.py
```

Then from the repo root: `pio run -t upload` (default env `esp32-s3-lcd-1-3`). The board shows a parody RX label; **shake** the device to run the transformer and refresh the name, quantity, SIG, and barcode.

## Notes for ESP32-S3 deployment

- Keep generation short (8-14 chars) and sampling simple (top-k + temperature).
- Quantize weights before final firmware to reduce flash/RAM pressure.
- If full transformer inference is too heavy, keep this training pipeline and distill into a smaller runtime model with the same vocabulary and tokenization.

## Safety

This project generates fictional names only. Use generated outputs strictly as artistic text, not as medical content.
