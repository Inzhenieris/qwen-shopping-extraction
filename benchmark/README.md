# Local sentence-only extraction benchmark

Standard-library Python, local Ollama only. No credentials, downloads, paid APIs, fine-tuning, or service/config changes.

## Run

    cd /home/jekabs/QWEN-FINE-TUNING/benchmark
    python3 -m unittest -v
    python3 benchmark.py --limit 3 --run-dir results/new_smoke
    python3 benchmark.py --limit 100 --run-dir results/new_baseline

Use a fresh run directory each time. --dataset can select an explicit file; otherwise the newest-by-mtime stable complete 100-row JSONL under ../dataset/data/generated is selected. Every row must have a nonempty sentence and valid gold. The source is read twice before use, copied into the run directory, and hashed again after evaluation; all inference uses the captured in-memory snapshot. If the source changes during evaluation, the summary flags it and the saved snapshot remains the exact evaluated data.

## Model

Selected smallest installed Qwen model: **qwen3:0.6b**, Ollama Q4_K_M, 522653767 bytes; installed metadata reports 751.63M parameters. Exact digest is saved in each manifest. The other Qwen-family model found was 9B. No installation or download is performed. --model must exactly match an installed Ollama tag.

Loopback endpoint: http://127.0.0.1:11434/api/chat. Environment proxies are bypassed for these local requests only (ordinary proxied localhost discovery returned an HTTP error). No environment or service configuration is modified.

Settings: temperature 0, seed 42, num_predict 1024, num_ctx 4096, think=false, serial nonstreamed calls, default 60-second timeout per example, no retries. No JSON constrained decoding: this baseline measures instruction-following raw output validity as well as extraction. It does not strip Markdown fences or rescue prose. A fenced otherwise-correct answer therefore fails strict JSON validity.

## Input and target contract

The model receives a fixed extraction instruction and **only the sentence**. It never receives the scenario, gold, example ID/type, or a hardcoded vocabulary. The prompt is saved and hashed in the manifest.

Target schema:

    {"items":[{"name":"apples","quantity":3}]}

Gold conversion is purely mechanical: gold item becomes name, quantity is unchanged, and the array is wrapped under items. Item order is ignored. Name spelling, case, plurals, whitespace and packaging are not normalized. Quantities must be positive JSON integers (not booleans or floats). No extra keys, duplicate names or duplicate object keys are accepted. Empty items is valid. Requests should be summed and globally excluded items omitted.

## Metrics

All selected examples are in the denominator, including transport/envelope errors. Reports include raw JSON parse success count, schema success count, strict exact-match accuracy, mismatch/failure count, transport/envelope error count, and mean/total request latency. Counts divided by examples give parse/schema rates. Per-scenario buckets use the same metrics.

Micro precision/recall/F1 compare exact (name, quantity) pairs. Wrong quantity counts as both a false positive and false negative. Invalid outputs contribute no predicted pairs and all expected pairs as false negatives; they always fail exact match, including empty-gold cases. Precision/recall/F1 are defined as zero on zero denominators. Strict item metrics intentionally do not infer items from malformed answers and should be interpreted alongside schema failures, not as semantic-only scores.

## Artifacts

Every new results/run-name directory contains:
- dataset_snapshot.jsonl: complete stable input snapshot, including all 100 rows even for a smoke subset.
- manifest.json: exact model/digest, prompt/hash, dataset/hash, selection and inference settings.
- predictions.jsonl: per-example sentence, expected JSON, raw response, parsed prediction, error, scores, finish reason and latency.
- summary.json: aggregate and per-scenario metrics, source stability flag.

Progress goes to stderr; final summary goes to stdout. Interrupted runs retain completed prediction rows; there is no automatic resume and a partial run without summary must not be treated as completed.

**Label limitation:** the source sentences are still marked unreviewed. Gold is deterministically scenario-grounded, not independently verified sentence truth. Errors may include sentence-generation fidelity issues. This is a baseline against those labels, not a claim of audited semantic accuracy.
