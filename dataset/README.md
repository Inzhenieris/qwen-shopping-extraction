# Shopping dataset generator

Python 3 standard library only. No installation, training, fine-tuning, or configuration changes.

## Run

From /home/jekabs/QWEN-FINE-TUNING/dataset:

```sh
python3 -m unittest discover -s tests -v
python3 scripts/dataset_generator.py --dry-run --count 100 --seed 42 --output data/scenarios/new_scenarios.jsonl
# Paid rendering: requires OPENROUTER_API_KEY already set in your environment.
python3 scripts/dataset_generator.py --count 1 --seed 42 --retries 0 --output data/generated/new_live_sample.jsonl
```

Omit --seed for fresh randomness. A seed reproduces scenarios and gold, not model prose. Use --min-quantity and --max-quantity (default 1..100, inclusive) to adjust quantities. Every occurrence draws an independent integer; duplicates may coincidentally have equal quantities. Gold totals can exceed the per-occurrence maximum. Counts refer to examples, not vocabulary size. The vocabulary always has exactly 100 distinct names; small samples need not contain all 100.

Outputs are created exclusively: existing files are never overwritten. Parent output directories must already exist. Configuration and the first response are validated before output publication. First-request failures create no output file; later failures preserve completed JSONL rows, and the process returns nonzero. Use a different output filename to retry; there is no automatic resume. No automatic bulk run is scheduled. Each live example is a paid API call, potentially repeated by configured retries; decide your count and budget before running.

## Scenario and gold semantics

Four scenario types cycle in order: requests, duplicates, exclusions, mixed. Each samples 2..6 distinct requested names; duplicate scenarios add 1..3 independently quantified repeated requests. Exclusion scenarios cancel one or more requested items and mention additional excluded-only items. Full cancellation and empty gold are valid.

Exclusions apply globally, regardless of mention order, and remove **all** quantities of the excluded item. Remaining duplicate requests are summed. Gold is a list of item/quantity objects sorted by item name, calculated only by Python **before** rendering. The API receives the scenario, never gold; it cannot create or replace labels.

Each JSONL row includes schema_version, id, seed, quantity_bounds, structured scenario, gold, sentence, model catalog evidence, status, and review_required. Dry-run has sentence/model null and status scenario_only. Live output has status unreviewed and review_required true.

**Correct scenario-derived gold does not guarantee that the generated sentence expresses it.** Response validation checks JSON shape, nonempty text, and finish_reason=stop, not semantic fidelity. Review each live sentence against every request occurrence, quantity, duplicate, and exclusion before training use. Nothing here marks a live row approved. The sentence field can contain multiple natural-language sentences.

## Model verification and credentials

On 2026-10-22, the public catalog https://openrouter.ai/api/v1/models listed the exact requested model:

- ID: deepseek/deepseek-v4.1-flash
- Name: DeepSeek: DeepSeek V4.1 Flash

That exact ID is the default. --model permits an explicit alternative selected by the operator. Every live run verifies the exact selected ID against the public catalog and fails if absent; no silent substitute, latest alias, or dated variant is selected. Dry-run does not access the catalog or network. Model availability can change.

The key is read from the process environment or the project .env as described below and transmitted as the HTTPS Authorization header to OpenRouter. It is not printed or saved. Raw error response bodies are not logged. HTTP failures include bounded, allowlisted structured provider diagnostics with key, credential-pattern, and request-content redaction. No secret stores or gateway configuration are read.

## Network limits and errors

Default timeout: 60 seconds per network operation (--timeout, maximum 300). Default maximum output tokens: 1200 (--max-tokens, 256..8192). Default retries: 2 additional attempts (--retries, 0..5). Retryable errors: 429, 500, 502, 503, 504, timeouts and transport errors. Authentication failures, malformed JSON, malformed sentence objects, and truncation fail immediately. Retries can incur extra charges after ambiguous transport failures.

Retry-After supports seconds and HTTP dates. A delay over 120 seconds aborts rather than retrying too early; otherwise the stated delay is honored. Without a valid header, bounded exponential backoff is used. API responses are size-limited. This is serial generation, not concurrency or a scheduler.

## Delivered verification artifacts

- scripts/dataset_generator.py: implementation and CLI.
- tests/test_dataset_generator.py: network-mocked tests; no paid calls in tests.
- data/scenarios/pilot_scenarios_100.jsonl: 100 offline examples with seed 42.
- data/generated/pilot_live_1.jsonl: single live attempt output (empty if the attempt failed).
- reports/verification.json: verification outcomes and public model evidence, without credentials.

The one-example live pilot uses --retries 0 to enforce at most one sentence request. No bulk paid generation is authorized or performed by this setup.

## Organized layout

- scripts/: single generator implementation.
- tests/: network-mocked tests.
- data/scenarios/: offline scenario/gold JSONL.
- data/generated/: live and user-created outputs; original empty/partial dataset.jsonl and dataset_v2.jsonl are preserved.
- reports/: verification, HTTP 400 investigation, hash-verified move manifest, preserved legacy bytecode.

The HTTP 400 investigation used one successful live request; reorganization used no live calls. See reports/DIAGNOSTIC_400.md for unresolved user-specific failure and diagnostic command.

## Environment model selection
Set OPENROUTER_MODEL in .env alongside OPENROUTER_API_KEY. Load with: set -a; source .env; set +a
The --model option overrides OPENROUTER_MODEL; if unset, the built-in DeepSeek model is used. The script does not automatically load .env.

## Project .env and precedence

The generator automatically reads dataset/.env relative to the script, independent of your current directory. Supported settings: OPENROUTER_API_KEY and OPENROUTER_MODEL. Literal KEY=value, optional export, single/double quotes, blank lines and comments are supported. No shell execution, interpolation, or escape expansion occurs. Other variables are ignored. Keys are never printed or copied into generated data.

Default precedence: CLI --model > process environment > project .env > built-in model. For the key: process environment > project .env. To avoid a stale exported key/model, explicitly use --env-file: that file becomes the sole source of key/model settings, with --model still highest priority. An explicit file lacking a key fails instead of falling back to another credential. Dry-run ignores .env entirely.

Verified from an arbitrary working directory (choose a fresh output path):

```sh
python3 /home/jekabs/QWEN-FINE-TUNING/dataset/scripts/dataset_generator.py --env-file /home/jekabs/QWEN-FINE-TUNING/dataset/.env --count 1 --seed 44 --retries 0 --output /home/jekabs/QWEN-FINE-TUNING/dataset/data/generated/verified_terminal_1.jsonl
```

Publication uses a same-directory temporary file and an exclusive atomic hard link after the first complete row. Existing files are never replaced, including empty failures from earlier versions. Later successful rows are flushed as they complete. This requires a filesystem supporting hard links (verified on this computer).

HTTP non-JSON/HTML errors are now classified using fixed safe labels and byte counts, without echoing their body. A real project-.env live request succeeded; see reports/DIAGNOSTIC_400.md.

## Progress logging

New invocations print flushed progress to stderr by default: model/count/mode, model-verification stage, each request starting, and Saved N/total only after the row is flushed and synced (and the first row atomically published). Completion logs include elapsed time and an ETA based on elapsed time per saved example; it is an estimate, not a guarantee. Retries log HTTP status or transport category, next attempt/maximum attempts, and backoff delay. Payloads, sentences, and credentials are not logged. Final success includes saved count, output path, and duration; failures include the actual completed count. JSONL stays free of progress text. Routine output no longer goes to stdout.

Add --quiet to suppress routine progress and the success summary; errors still appear. Redirect stderr with 2>progress.log if desired.

Already-running Python jobs retain the old loaded code: they will not gain these logs automatically. Let an active run finish, then use a new invocation with a fresh output filename. No active job has been stopped or restarted by this update. There is no automatic resume; do not restart against an existing partial output.
