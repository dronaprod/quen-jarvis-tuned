#!/usr/bin/env python3
"""
Interactive chat tester for the Jarvis v1.0 fine-tuned model.

Usage:
  cd quen_jarvis && python chat_jarvis.py
  python chat_jarvis.py --model ./jarvis_v1.0
  python chat_jarvis.py --model ./jarvis_v1.0 --max-new-tokens 128

The model translates natural-language instructions into macOS shell commands,
returning JSON like: {"command": "...", "command_number": "last"}

Type 'quit' or 'exit' to stop. Type 'examples' to see sample prompts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

JARVIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = JARVIS_DIR / "jarvis_v1.0"

SAMPLE_PROMPTS = [
    "open chrome",
    "take a screenshot",
    "open terminal",
    "increase volume",
    "lock the screen",
    "open settings",
    "create a new folder on desktop",
    "open the downloads folder",
    "open spotify",
    "turn on dark mode",
]


def load_model(model_dir: Path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        dtype=dtype,
    )
    model.to(device)
    model.eval()
    print(f"Model loaded on {device} ({dtype})\n")
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    device: str,
    instruction: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
) -> str:
    import torch

    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return raw


def pretty_print_response(raw: str):
    """Try to parse as JSON and pretty-print; fall back to raw text."""
    try:
        parsed = json.loads(raw)
        print("\n  Response (parsed JSON):")
        print(f"    command        : {parsed.get('command', 'N/A')}")
        print(f"    command_number : {parsed.get('command_number', 'N/A')}")
        if set(parsed.keys()) - {"command", "command_number"}:
            extras = {k: v for k, v in parsed.items() if k not in ("command", "command_number")}
            print(f"    extra fields   : {extras}")
    except json.JSONDecodeError:
        print(f"\n  Response (raw): {raw}")


def run_chat(model, tokenizer, device: str, max_new_tokens: int):
    print("=" * 60)
    print("  Jarvis v1.0 — Instruction → macOS Command Chat")
    print("=" * 60)
    print("  Type a natural-language instruction to get the")
    print("  corresponding macOS shell command as JSON.")
    print("  Commands: 'examples', 'quit' / 'exit'")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        if user_input.lower() == "examples":
            print("\n  Sample prompts you can try:")
            for p in SAMPLE_PROMPTS:
                print(f"    - {p}")
            continue

        print("  Generating...", end="\r")
        raw = generate_response(model, tokenizer, device, user_input, max_new_tokens)
        print(" " * 20, end="\r")  # clear the "Generating..." line
        pretty_print_response(raw)


def run_batch(model, tokenizer, device: str, max_new_tokens: int):
    """Non-interactive: run sample prompts and print results."""
    print("Running batch evaluation on sample prompts...\n")
    for prompt in SAMPLE_PROMPTS:
        print(f"Instruction: {prompt}")
        raw = generate_response(model, tokenizer, device, prompt, max_new_tokens)
        pretty_print_response(raw)
        print()


def main():
    parser = argparse.ArgumentParser(description="Chat with Jarvis v1.0 fine-tuned model")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_DIR,
        help=f"Path to model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Max tokens to generate per response (default: 128)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature; 0 = greedy (default: 0.1)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Non-interactive: run all sample prompts and exit",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: model directory not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    model, tokenizer, device = load_model(args.model)

    if args.batch:
        run_batch(model, tokenizer, device, args.max_new_tokens)
    else:
        run_chat(model, tokenizer, device, args.max_new_tokens)


if __name__ == "__main__":
    main()
