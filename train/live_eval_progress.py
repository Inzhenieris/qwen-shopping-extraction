#!/usr/bin/env python3
import re
import sys
import time
from pathlib import Path

LOG = Path("/home/jekabs/QWEN-FINE-TUNING/train/eval.log")
TOTAL = 300


def bar(done: int, total: int, width: int = 42) -> str:
    pct = min(100.0, done / max(1, total) * 100)
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)


def render() -> bool:
    text = LOG.read_text(errors="replace") if LOG.exists() else ""
    base_hits = re.findall(r"^base: (\d+)/300", text, re.M)
    adapter_hits = re.findall(r"^adapter: (\d+)/300", text, re.M)
    base_done = int(base_hits[-1]) if base_hits else 0
    adapter_done = int(adapter_hits[-1]) if adapter_hits else 0
    complete = "WROTE" in text

    if adapter_done:
        phase = "adapter evaluation"
    elif base_done:
        phase = "base evaluation"
    else:
        phase = "loading / starting"

    sys.stdout.write("\033[2J\033[H")
    print("Qwen3-0.6B — frozen test-set evaluation")
    print()
    print(f"base     {bar(base_done, TOTAL)}  {base_done:3d}/{TOTAL}")
    print(f"adapter  {bar(adapter_done, TOTAL)}  {adapter_done:3d}/{TOTAL}")
    print()
    print(f"phase: {phase}" + ("   COMPLETE" if complete else ""))
    print()
    print("recent log:")
    for line in text.splitlines()[-6:]:
        print("  " + line[:160])
    sys.stdout.flush()
    return complete


while True:
    if render():
        break
    time.sleep(1)
