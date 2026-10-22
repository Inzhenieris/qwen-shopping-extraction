#!/usr/bin/env python3
"""Generate shopping scenarios/gold locally; optionally render prose via OpenRouter."""
from contextvars import ContextVar
import argparse
import collections
import math
import email.utils
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import tempfile
import urllib.error
import urllib.request

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL = "deepseek/deepseek-v4.1-flash"
BASE_URL = "https://openrouter.ai/api/v1"
ITEMS = tuple("""apples|bananas|oranges|pears|peaches|plums|grapes|strawberries|blueberries|raspberries|lemons|limes|mangoes|pineapples|watermelons|avocados|tomatoes|potatoes|sweet potatoes|carrots|onions|garlic bulbs|broccoli heads|cauliflower heads|cucumbers|zucchini|eggplants|bell peppers|lettuce heads|cabbage heads|spinach bags|kale bags|mushroom packs|celery bunches|radish bunches|beetroot|pumpkins|corn cobs|green bean bags|pea bags|milk cartons|yogurt tubs|butter packs|cheddar blocks|mozzarella packs|cream cartons|egg cartons|bread loaves|bagels|croissants|tortilla packs|rice bags|pasta packs|oat bags|flour bags|sugar bags|salt packs|black pepper jars|olive oil bottles|vinegar bottles|honey jars|jam jars|peanut butter jars|cereal boxes|granola bags|coffee bags|tea boxes|cocoa tins|water bottles|orange juice cartons|apple juice cartons|sparkling water bottles|chicken breast packs|turkey packs|beef mince packs|salmon fillets|tuna cans|sardine cans|tofu packs|lentil bags|chickpea cans|kidney bean cans|black bean cans|tomato paste cans|coconut milk cans|soup cans|crackers boxes|biscuit packs|chocolate bars|almond bags|walnut bags|cashew bags|raisin bags|popcorn bags|frozen pizza boxes|ice cream tubs|dish soap bottles|laundry detergent bottles|paper towel rolls|toilet paper rolls""".split("|"))
assert len(ITEMS) == len(set(ITEMS)) == 100
KINDS = ("requests", "duplicates", "exclusions", "mixed")


QUIET = ContextVar("quiet", default=False)


def progress(message):
    if not QUIET.get():
        print(message, file=sys.stderr, flush=True)


class GenerationError(Exception):
    pass


def load_env(path, required=False):
    """Read literal supported assignments only, without execution or expansion."""
    if not path.exists() and not required:
        return {}
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r'(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)', line)
        if not match:
            raise GenerationError("Invalid env assignment on line " + str(number))
        name, value = match.groups()
        if name not in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
            continue
        if value.startswith(("'", '"')):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].strip().startswith("#")):
                raise GenerationError("Invalid env quoting on line " + str(number))
            value = value[1:end]
        else:
            value = re.split(r'\s+#', value, maxsplit=1)[0].strip()
        values[name] = value
    return values


def settings(cli_model=None, env_file=None):
    values = load_env(env_file or PROJECT_DIR / ".env", required=env_file is not None)
    effective = values if env_file is not None else {**values, **os.environ}
    key = effective.get("OPENROUTER_API_KEY", "").strip()
    model = (cli_model or effective.get("OPENROUTER_MODEL") or MODEL).strip()
    if not key:
        raise GenerationError("OPENROUTER_API_KEY is required for live mode")
    if any(c.isspace() for c in key) or not key.isascii() or key.lower() in ("your_api_key", "your-key", "changeme"):
        raise GenerationError("API key has invalid whitespace/characters or is a placeholder; check selected environment source")
    return key, model


def gold_for(scenario):
    """Exclusion is global: remove an item entirely, independent of mention order."""
    excluded = set(scenario["exclusions"])
    totals = collections.Counter()
    for request in scenario["requests"]:
        if request["item"] not in excluded:
            totals[request["item"]] += request["quantity"]
    return [{"item": name, "quantity": totals[name]} for name in sorted(totals)]


def make_scenario(rng, minimum=1, maximum=100, kind=None):
    if minimum < 1 or maximum < minimum:
        raise ValueError("quantity range must satisfy 1 <= minimum <= maximum")
    kind = kind or rng.choice(KINDS)
    if kind not in KINDS:
        raise ValueError("unknown scenario kind")
    names = rng.sample(ITEMS, rng.randint(2, 6))
    requests = [{"item": n, "quantity": rng.randint(minimum, maximum)} for n in names]
    if kind in ("duplicates", "mixed"):
        for name in rng.sample(names, rng.randint(1, min(3, len(names)))):
            requests.append({"item": name, "quantity": rng.randint(minimum, maximum)})
    exclusions = []
    if kind in ("exclusions", "mixed"):
        exclusions = rng.sample(names, rng.randint(1, len(names)))
        exclusions += rng.sample([n for n in ITEMS if n not in names], rng.randint(1, 2))
    rng.shuffle(requests)
    rng.shuffle(exclusions)
    return {"kind": kind, "requests": requests, "exclusions": exclusions,
            "exclusion_rule": "Exclude all quantities of excluded items; sum all other repeated requests."}


def retry_delay(header, attempt):
    if header:
        try:
            seconds = float(header)
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(header)
                seconds = when.timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                seconds = 2 ** attempt
        if not math.isfinite(seconds):
            seconds = 2 ** attempt
        if seconds > 120:
            raise GenerationError("Retry-After exceeds 120 seconds; stop and retry later")
        return max(0, seconds)
    return min(2 ** attempt, 30)


def http_error_detail(exc, key=None, payload=None):
    """Allowlist structured diagnostics; never dump raw bodies or request objects."""
    try:
        raw = exc.read(16385)
        if len(raw) > 16384:
            return " (provider error body omitted: size limit)"
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError):
            # Inspect privately; emit only fixed classifications, never raw HTML/text.
            lowered = raw.lower()
            categories = []
            for needle, label in ((b"cloudflare", "edge/cloudflare"), (b"proxy", "proxy"),
                                  (b"invalid api key", "invalid credential"),
                                  (b"invalid header", "invalid header"),
                                  (b"request header", "request header rejected"),
                                  (b"bad request", "bad request"),
                                  (b"response_format", "response_format mentioned")):
                if needle in lowered:
                    categories.append(label)
            return " " + json.dumps({"body_format": "non-JSON", "body_bytes": len(raw),
                                      "classifications": categories})
        error = body.get("error", body) if isinstance(body, dict) else {}
        if isinstance(error, str):
            error = {"message": error}
        if not isinstance(error, dict):
            return " (no structured provider error)"
    except (ValueError, UnicodeError, OSError, AttributeError):
        return " (provider detail unavailable)"

    def clean(value):
        if not isinstance(value, (str, int, float)):
            return None
        value = str(value)
        if key:
            value = value.replace(key, "[REDACTED]")
        for message in (payload or {}).get("messages", []):
            content = message.get("content")
            if isinstance(content, str) and content:
                value = value.replace(content, "[request content omitted]")
                value = value.replace(json.dumps(content)[1:-1], "[request content omitted]")
        # Drop the remainder if diagnostic prose starts echoing request/header data.
        value = re.split(r'(?i)\b(?:authorization|bearer|api[_ -]?key|messages?|prompt|request[_ ]?body|content)\s*[:=]', value)[0]
        value = re.sub(r'(?i)\bBearer\s+\S+', '[REDACTED]', value)
        value = re.sub(r'(?i)\b(?:sk|pk|rk)[-_][a-z0-9_-]+', '[REDACTED]', value)
        value = re.sub(r'[A-Za-z0-9_+/=-]{32,}', '[REDACTED]', value)
        value = re.sub(r'[\x00-\x1f\x7f]', ' ', value)
        return value[:500]

    details = {}
    for field in ("code", "type", "param", "message"):
        value = clean(error.get(field))
        if value:
            details[field] = value
    metadata = error.get("metadata")
    if isinstance(metadata, dict):
        provider = clean(metadata.get("provider_name"))
        if provider:
            details["provider"] = provider
        # Some providers nest their error JSON in metadata.raw; allowlist only.
        nested = metadata.get("raw")
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except ValueError:
                nested = None
        if isinstance(nested, dict):
            nested = nested.get("error", nested)
            if isinstance(nested, dict):
                for field in ("code", "type", "param", "message"):
                    value = clean(nested.get(field))
                    if value:
                        details["provider_" + field] = value
    return " " + json.dumps(details, ensure_ascii=True)[:1800] if details else " (no safe structured provider detail)"


def request_json(url, payload=None, key=None, retries=2, timeout=60):
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if key:
        headers["Authorization"] = "Bearer " + key
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise GenerationError("API response exceeds size limit")
            try:
                return json.loads(raw)
            except (ValueError, UnicodeError):
                raise GenerationError("API returned invalid JSON") from None
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise GenerationError("API HTTP error " + str(exc.code) + http_error_detail(exc, key, payload)) from None
            delay = retry_delay(exc.headers.get("Retry-After"), attempt)
            progress(f"Retry: HTTP {exc.code}; attempt {attempt + 2}/{retries + 1} in {delay:.1f}s")
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == retries:
                raise GenerationError("API connection failed or timed out") from None
            delay = retry_delay(None, attempt)
            progress(f"Retry: connection/timeout; attempt {attempt + 2}/{retries + 1} in {delay:.1f}s")
            time.sleep(delay)
    raise GenerationError("API retries exhausted")


def verify_model(model, retries=2, timeout=60):
    result = request_json(BASE_URL + "/models", retries=retries, timeout=timeout)
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        raise GenerationError("Invalid model catalog response")
    for entry in result["data"]:
        if isinstance(entry, dict) and entry.get("id") == model:
            return {"id": model, "name": entry.get("name"), "source": BASE_URL + "/models"}
    raise GenerationError("Exact model ID absent from OpenRouter catalog: " + model + "; no substitution made")


def render_sentence(scenario, model, key, retries=2, timeout=60, max_tokens=1200):
    instruction = (
        "Render the supplied shopping scenario as natural English shopping instructions. "
        "Preserve every request occurrence and its exact integer quantity, including duplicates "
        "and requests later excluded. Use exact item names. Explicitly state all exclusions apply "
        "globally, canceling every quantity of those items. Do not aggregate, drop, add, or alter "
        "any item or quantity. Multiple sentences are allowed. Return ONLY a JSON object with "
        "one key, sentence, containing a nonempty string. Do not compute or return gold labels."
    )
    result = request_json(BASE_URL + "/chat/completions", {
        "model": model, "messages": [{"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps(scenario)}],
        "temperature": 0.8, "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}}, key, retries, timeout)
    try:
        choices = result["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError()
        if choices[0]["finish_reason"] != "stop":
            raise ValueError()
        content = choices[0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError()
        obj = json.loads(content)
        if not isinstance(obj, dict) or set(obj) != {"sentence"}:
            raise ValueError()
        sentence = obj["sentence"]
        if not isinstance(sentence, str) or not sentence.strip() or len(sentence) > 20000:
            raise ValueError()
    except (KeyError, IndexError, TypeError, ValueError):
        raise GenerationError("Invalid or truncated sentence response; no label inferred") from None
    return sentence.strip()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quiet", action="store_true", help="Suppress routine progress, not errors")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--start", type=int, default=0, help="Zero-based scenario offset; used to render a range of the same seeded sequence")
    p.add_argument("--seed", type=int)
    p.add_argument("--min-quantity", type=int, default=1)
    p.add_argument("--max-quantity", type=int, default=100)
    p.add_argument("--dry-run", action="store_true", help="Scenarios/gold only; no network or key")
    p.add_argument("--model", help="Explicit model overrides environment and default")
    p.add_argument("--env-file", type=Path, help="Explicit file overrides process settings; never executed")
    p.add_argument("--retries", type=int, default=2, help="Additional HTTP attempts, 0..5")
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--max-tokens", type=int, default=1200)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.count < 1 or args.start < 0 or not 1 <= args.min_quantity <= args.max_quantity:
        p.error("positive count, nonnegative start, and 1 <= min-quantity <= max-quantity required")
    if not 0 <= args.retries <= 5 or not 0 < args.timeout <= 300 or not 256 <= args.max_tokens <= 8192:
        p.error("retries must be 0..5, timeout (0,300], max-tokens 256..8192")
    started = time.monotonic()
    completed = 0
    quiet_token = QUIET.set(args.quiet)
    out = None
    temporary = None
    try:
        if args.output.exists() or args.output.is_symlink():
            raise GenerationError("Output already exists; choose a fresh filename")
        key, model = (None, args.model or MODEL) if args.dry_run else settings(args.model, args.env_file)
        safe_model = model if re.fullmatch(r"[A-Za-z0-9_./:~-]{1,160}", model) and not model.startswith("sk-") and model != key else "[redacted]"
        progress(f"Starting: model={safe_model} count={args.count} mode={'dry-run' if args.dry_run else 'live'}")
        progress("Model verification: skipped (dry-run)" if args.dry_run else "Model verification: checking exact ID")
        model_info = None if args.dry_run else verify_model(model, args.retries, args.timeout)
        if not args.dry_run:
            progress("Model verification: confirmed")
        rng = random.Random(args.seed)
        for skipped in range(args.start):
            make_scenario(rng, args.min_quantity, args.max_quantity, KINDS[skipped % 4])
        for index in range(args.start, args.start + args.count):
            progress(f"Example {completed + 1}/{args.count}: {'scenario generation' if args.dry_run else 'request'} starting")
            scenario = make_scenario(rng, args.min_quantity, args.max_quantity, KINDS[index % 4])
            gold = gold_for(scenario)
            sentence = None if args.dry_run else render_sentence(
                scenario, model, key, args.retries, args.timeout, args.max_tokens)
            record = {"schema_version": 1, "id": index + 1, "seed": args.seed,
                      "quantity_bounds": [args.min_quantity, args.max_quantity],
                      "scenario": scenario, "gold": gold, "sentence": sentence,
                      "model": model_info, "review_required": not args.dry_run,
                      "status": "scenario_only" if args.dry_run else "unreviewed"}
            line = json.dumps(record, ensure_ascii=False) + "\n"
            if out is None:
                out = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".dataset-",
                                                  dir=args.output.parent, delete=False)
                temporary = Path(out.name)
                out.write(line)
                out.flush()
                os.fsync(out.fileno())
                os.link(temporary, args.output)
                temporary.unlink()
                temporary = None
            else:
                out.write(line)
                out.flush()
                os.fsync(out.fileno())
            completed += 1
            elapsed = time.monotonic() - started
            eta = elapsed / completed * (args.count - completed)
            progress(f"Saved {completed}/{args.count} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s (observed rate)")
        progress(f"Success: saved {completed}/{args.count} to {args.output} | duration {time.monotonic() - started:.1f}s")
        return 0
    except (GenerationError, OSError) as exc:
        print(f"Error: {exc}. Saved {completed}/{args.count} | duration {time.monotonic() - started:.1f}s. No output is published before the first completed record; later failures preserve partial rows.", file=sys.stderr, flush=True)
        return 1
    finally:
        QUIET.reset(quiet_token)
        if out is not None:
            out.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
