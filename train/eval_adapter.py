#!/usr/bin/env python3
"""Evaluate base and LoRA adapter on the frozen shopping-extraction test split."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_strict(raw):
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return False, False, None
    if not isinstance(obj, dict) or set(obj) != {"items"} or not isinstance(obj["items"], list):
        return True, False, obj
    seen = set()
    pairs = []
    for item in obj["items"]:
        if (not isinstance(item, dict) or set(item) != {"name", "quantity"}
                or not isinstance(item["name"], str) or not item["name"].strip()
                or type(item["quantity"]) is not int or item["quantity"] <= 0
                or item["name"] in seen):
            return True, False, obj
        seen.add(item["name"])
        pairs.append((item["name"], item["quantity"]))
    return True, True, obj


def gold_pairs(content):
    parsed, ok, obj = parse_strict(content)
    if not ok:
        raise ValueError("invalid gold record")
    return {(x["name"], x["quantity"]) for x in obj["items"]}


def metrics(rows):
    n = len(rows)
    if not n:
        return {"examples": 0}
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    exact = sum(r["exact_match"] for r in rows)
    kinds = Counter(r.get("scenario_kind", "unknown") for r in rows)
    by_kind = {}
    for kind in sorted(kinds):
        part = [r for r in rows if r.get("scenario_kind") == kind]
        e = sum(r["exact_match"] for r in part)
        by_kind[kind] = {"examples": len(part), "exact_matches": e,
                         "exact_match_accuracy": e / len(part) if part else 0.0}
    return {
        "examples": n,
        "json_parse_success": sum(r["parse_ok"] for r in rows),
        "schema_success": sum(r["schema_ok"] for r in rows),
        "exact_matches": exact,
        "exact_match_accuracy": exact / n,
        "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "by_scenario_kind": by_kind,
        "mean_latency_seconds": sum(r["latency"] for r in rows) / n,
    }


def chat_prompt(tokenizer, row):
    messages = row["messages"][:2]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)


def load_rows(path, limit=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    for row in rows:
        gold_pairs(row["messages"][2]["content"])
    return rows


def run_eval(model, tokenizer, rows, name, outdir, batch_size=4, max_new_tokens=128):
    out_predictions = outdir / f"predictions_{name}.jsonl"
    records = []
    tokenizer.padding_side = "left"
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [chat_prompt(tokenizer, row) for row in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        import time
        t0 = time.monotonic()
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        elapsed = (time.monotonic() - t0) / len(batch)
        generated = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, raw in zip(batch, texts):
            gold = gold_pairs(row["messages"][2]["content"])
            parsed, schema_ok, obj = parse_strict(raw.strip())
            pred = {(x["name"], x["quantity"]) for x in obj["items"]} if schema_ok else set()
            records.append({
                "id": row.get("id"), "scenario_kind": row.get("scenario_kind"),
                "sentence": row["messages"][1]["content"],
                "gold": sorted(gold), "raw_response": raw,
                "parse_ok": parsed, "schema_ok": schema_ok,
                "exact_match": schema_ok and pred == gold,
                "tp": len(pred & gold), "fp": len(pred - gold), "fn": len(gold - pred),
                "latency": elapsed,
            })
        print(f"{name}: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    with out_predictions.open("x") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = metrics(records)
    summary["model"] = name
    (outdir / f"summary_{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    p.add_argument("--adapter", default=str(ROOT / "train/out/qwen3-0.6b-lora"))
    p.add_argument("--test", default=str(ROOT / "dataset/data/sft/test.jsonl"))
    p.add_argument("--out", default=str(ROOT / "train/eval"))
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.test, args.limit)
    print(f"loaded {len(rows)} test rows", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()

    base_summary = run_eval(model, tokenizer, rows, "base", outdir, args.batch_size)
    print(json.dumps(base_summary, indent=2), flush=True)

    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    adapter_summary = run_eval(model, tokenizer, rows, "adapter", outdir, args.batch_size)
    print(json.dumps(adapter_summary, indent=2), flush=True)

    combined = {
        "test_file": args.test,
        "test_sha256": sha(args.test),
        "model": args.model,
        "adapter": args.adapter,
        "limit": len(rows),
        "base": base_summary,
        "adapter": adapter_summary,
    }
    (outdir / "summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    print("WROTE", outdir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
