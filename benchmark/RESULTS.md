# Qwen baseline result

Model: qwen3:0.6b (Q4_K_M). Three-example smoke succeeded operationally, then all 100 dataset examples completed locally.

Strict raw JSON parse: 0/100. Schema-valid: 0/100. Exact match: 0/100. Micro precision/recall/F1: 0. Transport/envelope errors: 0. All four scenario buckets: 0/25 exact.

Mean latency: 0.597 seconds; summed inference latency: 59.745 seconds. Dataset source hash unchanged before/after.

100 responses started with Markdown fences. Fences are intentionally not stripped, so these fail the strict raw-JSON contract. Inspected examples also show duplicate aggregation and exclusion errors; zero strict score is not a semantic-only measure. No rescored or rescued answers are included.

Full artifacts: results/baseline_qwen3_06b_100/. Smoke artifacts: results/smoke_qwen3_06b/.

Source labels are scenario-derived and sentences are unreviewed. No model downloads, paid requests, service/config edits or fine-tuning were performed.
