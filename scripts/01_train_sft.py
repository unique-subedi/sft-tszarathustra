#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from packaging.version import Version
import transformers as tfv

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}\n<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n{instruction}\n<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

def formatting_func(examples: Dict[str, List[str]]) -> List[str]:
    out = []
    systems = examples.get("system", [""] * len(examples["instruction"]))
    for sys, instr, tgt in zip(systems, examples["instruction"], examples["output"]):
        out.append(PROMPT_TEMPLATE.format(system=sys, instruction=instr) + tgt)
    return out

def load_splits(train_path: str, eval_path: str):
    ds = load_dataset("json", data_files={"train": train_path, "eval": eval_path})
    required = {"instruction", "output"}
    for split in ("train", "eval"):
        cols = set(ds[split].column_names)
        if not required.issubset(cols):
            missing = required - cols
            raise ValueError(f"{split} split missing columns: {missing}")
        if "system" not in cols:
            ds[split] = ds[split].add_column("system", [""] * len(ds[split]))
    return ds["train"], ds["eval"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=os.environ.get("BOOKSFT_BASE", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    ap.add_argument("--train_file", default="data/sft_train.jsonl")
    ap.add_argument("--eval_file", default="data/sft_eval.jsonl")
    ap.add_argument("--out_dir", default="out/booksft-sft")
    ap.add_argument("--max_seq_len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "1024")))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)  # lower LR for full FT
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=500)
    args = ap.parse_args()

    print(f"[info] Base model: {args.base_model}")
    print(f"[info] Max seq len: {args.max_seq_len}")
    print(f"[info] Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # 1) Data
    train_ds, eval_ds = load_splits(args.train_file, args.eval_file)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Model (full fine-tune — no LoRA, no quantization)
    model_kwargs = {"torch_dtype": torch.bfloat16} if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else {}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    # 4) TrainingArguments (version-aware for Transformers 4.57+)
    common = dict(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=25,
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=False,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],        # no wandb by default
        max_grad_norm=1.0,
        gradient_checkpointing=True,
    )
    if Version(tfv.__version__) >= Version("4.57"):
        training_args = TrainingArguments(**common, eval_strategy="steps", save_strategy="steps")
    else:
        training_args = TrainingArguments(**common, evaluation_strategy="steps")

    # 5) SFTTrainer — pass tokenizer + formatting_func (the key fixes)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=formatting_func,   # builds the text from your columns
        max_seq_length=args.max_seq_len,
        packing=True,
        args=training_args,
    )

    trainer.train()
    # Save full fine-tuned model
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("[done] Saved full fine-tuned model to:", args.out_dir)

if __name__ == "__main__":
    main()
