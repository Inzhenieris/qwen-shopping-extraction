# Qwen Shopping-List Extraction

Fine-tune **Qwen3-0.6B** with 4-bit QLoRA to turn a natural-language shopping request into a canonical JSON list.

## Layout

- `dataset/` — scenario generator, rendered dataset, frozen splits, SFT chat-format files.
- `benchmark/` — sentence-only extraction benchmark against scenario-derived gold.
- `train/` — QLoRA training script and live progress viewer.

## Dataset

- `dataset/data/generated/dataset_2500_live.jsonl` — 2,500 rendered examples.
- `dataset/data/splits_live/{train,val,test}.jsonl` — 2,000 / 200 / 300.
- `dataset/data/sft/{train,val,test}.jsonl` — chat-format `messages` records for SFT.
- `dataset/data/scenarios/scenarios_2500_seed42.jsonl` — reproducible source scenarios and gold.

Generated sentences are marked `unreviewed`; gold labels are scenario-grounded, not independently audited.

## Training

```bash
# one-time model download (not committed)
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

.venv/bin/python train/train_qlora.py \
  --model models/Qwen3-0.6B \
  --epochs 1 \
  --output train/out/qwen3-0.6b-lora
```

4-bit NF4, LoRA r=16/alpha=32, 512-token sequences, batch 1 × gradient accumulation 8. Adapters are written to `train/out/` and are intentionally not committed.

## Evaluation

`benchmark/RESULTS.md` records the prompt-only Qwen baseline. Post-training exact-match results are added after the adapter is evaluated on the frozen 300-example test split.

## Notes

- `.env` is not committed. Data generation reads `OPENROUTER_API_KEY` from the environment or `dataset/.env`.
- Model weights, virtual environments, adapters, logs, and old partial runs are excluded from git.

## Results

Base Qwen3-0.6B vs the QLoRA adapter on the frozen 300-example test set:

| Metric | Base | Adapter |
|---|---:|---:|
| Schema-valid | 2/300 | 299/300 |
| Exact match | 0/300 | **263/300 (87.7%)** |
| Micro F1 | 0.0 | **0.952** |

Full breakdown: [train/RESULTS.md](train/RESULTS.md).
