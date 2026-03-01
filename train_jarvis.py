#!/usr/bin/env python3
"""
SFT for Qwen 0.5B on Jarvis instruction->command output.

Data: jarvis_train.jsonl with lines {"instruction": "...", "output": "..."}.
Output: finetuned_quen_jarvis/ in this folder.

Usage:
  cd quen_jarvis && python train_jarvis.py
  python train_jarvis.py --epochs 5 --lr 2e-5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

JARVIS_DIR = Path(__file__).resolve().parent
DATA_PATH = JARVIS_DIR / "jarvis_train.jsonl"
DEFAULT_OUTPUT_DIR = JARVIS_DIR / "jarvis_v1.0"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 2
DEFAULT_LR = 5e-6
MAX_SEQ_LENGTH = 512


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Qwen 0.5B on Jarvis instruction-output data")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="JSONL training data (instruction, output)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output model directory")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: {args.data} not found", file=sys.stderr)
        sys.exit(1)

    examples = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instruction = obj.get("instruction", "").strip()
            output = obj.get("output", "").strip()
            if not instruction and not output:
                continue
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            examples.append({"messages": messages})

    if not examples:
        print("Error: no examples in data", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(examples)} examples from {args.data}")

    try:
        from datasets import Dataset
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        from datasets import Dataset

    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

    model_load_id = MODEL_ID
    load_kw = {"trust_remote_code": True}
    tokenizer = AutoTokenizer.from_pretrained(model_load_id, **load_kw)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = Dataset.from_list(examples)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (no CUDA)")

    model = AutoModelForCausalLM.from_pretrained(model_load_id, **load_kw)
    if args.gradient_checkpointing and getattr(model, "gradient_checkpointing_enable", None):
        model.gradient_checkpointing_enable()
    if device == "cuda":
        model = model.to(device)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def tokenize_example(ex):
        messages = ex["messages"]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_enc = tokenizer(
            full_text,
            return_tensors=None,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
        prompt_enc = tokenizer(prompt_text, return_tensors=None, add_special_tokens=False)
        prompt_len = min(len(prompt_enc["input_ids"]), args.max_seq_length)
        raw_ids = full_enc["input_ids"]
        input_ids = (raw_ids + [pad_id] * args.max_seq_length)[:args.max_seq_length]
        n_content = min(len(raw_ids), args.max_seq_length)
        attention_mask = [1] * n_content + [0] * (args.max_seq_length - n_content)
        labels = [-100] * prompt_len + list(input_ids[prompt_len:])
        labels = (labels + [-100] * args.max_seq_length)[:args.max_seq_length]
        for i in range(prompt_len, len(labels)):
            if labels[i] == pad_id:
                labels[i] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = dataset.map(tokenize_example, remove_columns=dataset.column_names, num_proc=1)
    n_examples = len(examples)
    steps_per_epoch = max(1, (n_examples + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"Steps per epoch: {steps_per_epoch}, total steps: {total_steps} ({args.epochs} epochs)")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 4 // args.batch_size),
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        no_cuda=(device == "cpu"),
        dataloader_pin_memory=(device == "cuda"),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    print("Training...")
    trainer.train()
    print(f"Saving model and tokenizer to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_config.json").write_text(
        json.dumps({
            "model_id": model_load_id,
            "data": str(args.data),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_seq_length": args.max_seq_length,
        }, indent=2)
    )
    print("Done.")


if __name__ == "__main__":
    main()
