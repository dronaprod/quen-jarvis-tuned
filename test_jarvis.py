#!/usr/bin/env python3
"""
Test the fine-tuned Jarvis Qwen 0.5B on jarvis_train.jsonl.

Usage:
  cd quen_jarvis && python test_jarvis.py
  python test_jarvis.py --limit 100          # test first 100 only
  python test_jarvis.py --show-failures 20   # show up to 20 failure details
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

JARVIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = JARVIS_DIR / "finetuned_quen_jarvis"
DATA_PATH = JARVIS_DIR / "jarvis_train.jsonl"


def load_tests(data_path: Path) -> list[tuple[str, str]]:
    tests = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instruction = obj.get("instruction", "").strip()
            output = obj.get("output", "").strip()
            if instruction or output:
                tests.append((instruction, output))
    return tests


def compare_output(expected: str, got: str) -> bool:
    """True if got matches expected (by command field or exact string)."""
    try:
        expected_parsed = json.loads(expected)
        got_parsed = json.loads(got) if got else None
        if got_parsed is None:
            return False
        return expected_parsed.get("command") == got_parsed.get("command")
    except Exception:
        return expected.strip() == (got.strip() if got else "")


def main():
    parser = argparse.ArgumentParser(description="Test Jarvis model on jarvis_train.jsonl")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="JSONL test data")
    parser.add_argument("--limit", type=int, default=None, help="Test only first N examples")
    parser.add_argument("--show-failures", type=int, default=15, help="Max number of failure details to print (0 = none)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if not MODEL_DIR.is_dir():
        print(f"Error: model dir not found: {MODEL_DIR}", file=sys.stderr)
        sys.exit(1)
    if not args.data.exists():
        print(f"Error: data not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    tests = load_tests(args.data)
    if args.limit is not None:
        tests = tests[: args.limit]
    n_tests = len(tests)
    print(f"Loaded {n_tests} test cases from {args.data}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    passed = 0
    failures = []

    print("Running inference...")
    for i, (instruction, expected) in enumerate(tests):
        messages = [{"role": "user", "content": instruction}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        got = generated.strip()

        match = compare_output(expected, got)
        if match:
            passed += 1
        else:
            failures.append((instruction, expected, got))

        status = "PASS" if match else "FAIL"
        print(f"  [{i + 1}/{n_tests}] {status}  {instruction!r}", flush=True)

    print()
    print("=" * 70)
    print("Results (trained model vs jarvis_train.jsonl)")
    print("=" * 70)
    print(f"  Total:  {n_tests}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {len(failures)}")
    if n_tests:
        pct = 100.0 * passed / n_tests
        print(f"  Accuracy: {pct:.2f}%")
    print("=" * 70)

    if failures and args.show_failures > 0:
        n_show = min(len(failures), args.show_failures)
        print(f"\nFirst {n_show} failure(s):")
        for instruction, expected, got in failures[: n_show]:
            print(f"\n  instruction: {instruction!r}")
            print(f"  expected: {expected}")
            print(f"  got:      {got}")

    sys.exit(0 if len(failures) == 0 else 1)


if __name__ == "__main__":
    main()
