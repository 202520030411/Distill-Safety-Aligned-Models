"""
Smoke test for the pure Python logic in generate_data.py and prepare_dataset.py.
No GPU, vLLM, or HuggingFace token required.
"""

import json
import sys
import tempfile
from pathlib import Path

# ── replicate the logic from generate_data.py ──────────────────────────────

REFUSAL_PREFIXES = [
    "I'm sorry", "I can't", "I cannot", "I'm not able", "I won't",
    "I will not", "As an AI", "I'm unable", "I apologize",
    "That's not something I", "I don't think I should",
    "This request involves", "I must decline",
]

import re

def is_refusal(text: str) -> bool:
    text_lower = text.lower().strip()
    for prefix in REFUSAL_PREFIXES:
        if text_lower.startswith(prefix.lower()):
            return True
    patterns = [
        r"i (am|'m) (not able|unable) to",
        r"i (can't|cannot|won't|will not) (help|assist|provide|generate|create)",
        r"(this|that) (request|prompt|query) (is|seems|appears) (harmful|dangerous|unethical|illegal)",
    ]
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def format_chat_prompt(system: str, user: str) -> str:
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ── replicate the logic from prepare_dataset.py ────────────────────────────

SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."

def format_sft_example(prompt: str, response: str) -> dict:
    text = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    )
    return {"text": text, "prompt": prompt, "response": response}


def filter_records(records, keep_harmful_compliances=False):
    filtered, dropped = [], 0
    for r in records:
        if r["category"] == "harmful" and not r["is_refusal"]:
            if keep_harmful_compliances:
                filtered.append(r)
            else:
                dropped += 1
        else:
            filtered.append(r)
    return filtered, dropped


import random

def train_val_split(records, val_fraction=0.2, seed=42):
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_fraction))
    return shuffled[:split_idx], shuffled[split_idx:]


def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Tests ──────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def test(name, condition):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    return condition


def run_tests():
    results = []
    print("\n=== Refusal Detection ===")

    results.append(test(
        "Detects 'I'm sorry' prefix",
        is_refusal("I'm sorry, but I can't help with that.")
    ))
    results.append(test(
        "Detects 'I cannot' prefix",
        is_refusal("I cannot assist with this request.")
    ))
    results.append(test(
        "Detects regex pattern: 'i am not able to'",
        is_refusal("I am not able to provide that information.")
    ))
    results.append(test(
        "Detects regex pattern: 'this request is harmful'",
        is_refusal("This request is harmful and I won't respond.")
    ))
    results.append(test(
        "Does NOT flag a normal helpful response",
        not is_refusal("Sure! Here's how you make pasta: boil water, add pasta...")
    ))
    results.append(test(
        "Does NOT flag a factual answer",
        not is_refusal("The capital of France is Paris.")
    ))

    print("\n=== Chat Template Formatting ===")
    prompt = format_chat_prompt("You are helpful.", "What is 2+2?")
    results.append(test(
        "Contains system header",
        "<|start_header_id|>system<|end_header_id|>" in prompt
    ))
    results.append(test(
        "Contains user header",
        "<|start_header_id|>user<|end_header_id|>" in prompt
    ))
    results.append(test(
        "Contains assistant header",
        "<|start_header_id|>assistant<|end_header_id|>" in prompt
    ))
    results.append(test(
        "Contains the user prompt text",
        "What is 2+2?" in prompt
    ))
    results.append(test(
        "Starts with <|begin_of_text|>",
        prompt.startswith("<|begin_of_text|>")
    ))

    print("\n=== SFT Example Formatting ===")
    ex = format_sft_example("Tell me a joke.", "Why did the chicken cross the road? To get to the other side!")
    results.append(test("Has 'text' key", "text" in ex))
    results.append(test("Has 'prompt' key", "prompt" in ex))
    results.append(test("Has 'response' key", "response" in ex))
    results.append(test(
        "Text ends with <|eot_id|>",
        ex["text"].endswith("<|eot_id|>")
    ))
    results.append(test(
        "Text contains response content",
        "chicken" in ex["text"]
    ))

    print("\n=== Filtering Logic ===")
    mock_records = [
        {"category": "benign",  "is_refusal": False, "prompt": "p1", "response": "r1"},
        {"category": "harmful", "is_refusal": True,  "prompt": "p2", "response": "r2"},
        {"category": "harmful", "is_refusal": False, "prompt": "p3", "response": "r3"},  # should be dropped
    ]

    kept, dropped = filter_records(mock_records, keep_harmful_compliances=False)
    results.append(test("Drops harmful compliance (default)", dropped == 1))
    results.append(test("Keeps benign + refusal records", len(kept) == 2))

    kept_all, dropped_none = filter_records(mock_records, keep_harmful_compliances=True)
    results.append(test("Keeps all records when flag is set", len(kept_all) == 3))
    results.append(test("No records dropped when flag is set", dropped_none == 0))

    print("\n=== Train/Val Split ===")
    records = [{"id": i} for i in range(100)]
    train, val = train_val_split(records, val_fraction=0.2, seed=42)
    results.append(test("Val split is ~20%", len(val) == 20))
    results.append(test("Train split is ~80%", len(train) == 80))
    results.append(test("No overlap between train and val", len(set(r["id"] for r in train) & set(r["id"] for r in val)) == 0))
    results.append(test("Deterministic with same seed", train_val_split(records, 0.2, 42)[0] == train))
    results.append(test("Different seed gives different split", train_val_split(records, 0.2, 99)[0] != train))

    print("\n=== JSONL I/O ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        sample = [
            {"prompt": "hello", "response": "hi", "category": "benign", "is_refusal": False},
            {"prompt": "do harm", "response": "I'm sorry", "category": "harmful", "is_refusal": True},
        ]
        save_jsonl(sample, path)
        loaded = load_jsonl(path)
        results.append(test("Saves and loads correct number of records", len(loaded) == 2))
        results.append(test("Record fields preserved after roundtrip", loaded[0]["prompt"] == "hello"))
        results.append(test("Boolean field preserved", loaded[1]["is_refusal"] is True))
        results.append(test("File is valid JSONL (one JSON object per line)", path.read_text().count("\n") == 2))

    # Summary
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n{'='*40}")
    print(f"Results: {n_pass}/{len(results)} passed", end="")
    if n_fail:
        print(f"  ({n_fail} failed)")
        sys.exit(1)
    else:
        print("  — all good!")


if __name__ == "__main__":
    run_tests()
