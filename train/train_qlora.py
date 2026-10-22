#!/usr/bin/env python3
"""QLoRA fine-tune Qwen on the shopping-extraction SFT set."""
import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-0.6B")
    p.add_argument("--train", default="dataset/data/sft/train.jsonl")
    p.add_argument("--val", default="dataset/data/sft/val.jsonl")
    p.add_argument("--output", default="train/out/qwen3-0.6b-lora")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    raw = load_dataset("json", data_files={"train": args.train, "validation": args.val})

    def to_text(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = raw.map(to_text, remove_columns=raw["train"].column_names)
    print(ds, flush=True)

    cfg = SFTConfig(
        output_dir=args.output,
        dataset_text_field="text",
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        packing=False,
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("SAVED", args.output, flush=True)

if __name__ == "__main__":
    main()
