# QLoRA evaluation results

Base model: Qwen3-0.6B, 4-bit NF4.
Adapter: `train/out/qwen3-0.6b-lora` (LoRA r=16, alpha=32; 10.1M trainable params).

Training: 1 epoch, 2,000 examples, 250 steps.
- train loss: 0.6361
- final eval loss: 0.4345
- eval token accuracy: 87.2%

Frozen test set: 300 examples (`dataset/data/sft/test.jsonl`, SHA-256 `3cc75e53...011b`).

| Metric | Base | Adapter |
|---|---:|---:|
| Raw JSON parse | 4/300 | **300/300** |
| Schema-valid | 2/300 | **299/300** |
| Exact match | 0/300 (0.0%) | **263/300 (87.7%)** |
| Micro precision | 0.0 | **0.955** |
| Micro recall | 0.0 | **0.949** |
| Micro F1 | 0.0 | **0.952** |

Adapter exact match by scenario type:

| Kind | Exact | Accuracy |
|---|---:|---:|
| requests | 75/75 | 100% |
| exclusions | 72/75 | 96% |
| mixed | 69/75 | 92% |
| duplicates | 47/75 | 62.7% |

Remaining weakness: duplicate aggregation. The base model fails the JSON contract almost entirely; the adapter fixes formatting and most extraction behavior.

The 2,500 source sentences remain marked `unreviewed`. These results measure agreement with scenario-derived gold, not audited sentence truth.
