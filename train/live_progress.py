#!/usr/bin/env python3
import re
import sys
import time
from pathlib import Path

LOG = Path("/home/jekabs/QWEN-FINE-TUNING/train/full.log")


def render() -> bool:
    text = LOG.read_text(errors="replace") if LOG.exists() else ""
    clean = text.replace("\r", "\n")

    steps = re.findall(r"(\d+)/(\d+) \[([0-9:]+)<([0-9:]+),", clean)
    if steps:
        step, total, elapsed, eta = steps[-1]
    else:
        step, total, elapsed, eta = "0", "250", "00:00", "?"

    losses = re.findall(r"\{'loss': '([0-9.]+)'.*?'epoch': '([0-9.]+)'\}", clean)
    evals = re.findall(r"\{'eval_loss': '([0-9.]+)'.*?'epoch': '([0-9.]+)'\}", clean)
    done = "SAVED" in text

    try:
        pct = min(100.0, int(step) / max(1, int(total)) * 100)
    except ValueError:
        pct = 0.0
    width = 44
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)

    sys.stdout.write("\033[2J\033[H")
    print("Qwen3-0.6B QLoRA — shopping-list extraction")
    print()
    print(f"{bar}  {pct:5.1f}%")
    print(f"step {step}/{total}   elapsed {elapsed}   ETA {eta}")
    if losses:
        print(f"train loss {losses[-1][0]}   epoch {losses[-1][1]}")
    if evals:
        print("eval loss " + "   ".join(f"{loss} @ epoch {epoch}" for loss, epoch in evals[-3:]))
    print("status: " + ("COMPLETE — adapter saved" if done else "training"))
    print()
    print("recent log:")
    for line in clean.splitlines()[-6:]:
        print("  " + line[:160])
    sys.stdout.flush()
    return done


while True:
    if render():
        break
    time.sleep(1)
