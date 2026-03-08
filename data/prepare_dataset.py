"""
Prepare the final train/validation splits from the raw teacher output JSONL.

Steps:
  1. Load raw_teacher_outputs.jsonl
  2. Filter out harmful records where the teacher complied (non-refusals),
     since we only want to teach the student to refuse — not replicate harmful content.
  3. Shuffle and split into train / validation sets.
  4. Format each record into the Llama-3 chat template for SFT.
  5. Save as HuggingFace Dataset (Arrow) and as JSONL for inspection.

Output files (under data/):
  processed/train.jsonl
  processed/val.jsonl
  processed/dataset_stats.json
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Llama-3 chat template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."


def format_sft_example(prompt: str, response: str) -> dict:
    """
    Format a single (prompt, response) pair into the Llama-3 chat template.
    Returns a dict with 'text' (the full formatted string) and the original fields.
    """
    text = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    )
    return {"text": text, "prompt": prompt, "response": response}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_records(records: list[dict], keep_harmful_compliances: bool) -> list[dict]:
    """
    By default, drop harmful records where the teacher complied (is_refusal=False),
    since training the student on harmful content is unsafe.

    Set keep_harmful_compliances=True only for research/ablation purposes.
    """
    filtered = []
    dropped = 0
    for r in records:
        if r["category"] == "harmful" and not r["is_refusal"]:
            if keep_harmful_compliances:
                filtered.append(r)
            else:
                dropped += 1
                continue
        else:
            filtered.append(r)

    if dropped:
        print(f"  Dropped {dropped} harmful compliance records (teacher did not refuse).")

    return filtered


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def train_val_split(
    records: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_fraction))
    return shuffled[:split_idx], shuffled[split_idx:]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(train: list[dict], val: list[dict]) -> dict:
    def split_stats(records: list[dict]) -> dict:
        cat_counts = Counter(r["category"] for r in records)
        refusal_count = sum(1 for r in records if r.get("is_refusal", False))
        return {
            "total": len(records),
            "benign": cat_counts.get("benign", 0),
            "harmful_refusal": refusal_count,
            "avg_prompt_len": (
                sum(len(r["prompt"]) for r in records) / max(len(records), 1)
            ),
            "avg_response_len": (
                sum(len(r["response"]) for r in records) / max(len(records), 1)
            ),
        }

    return {"train": split_stats(train), "val": split_stats(val)}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/val splits from raw teacher output JSONL."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_teacher_outputs.jsonl",
        help="Path to raw teacher output JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to write processed splits (default: %(default)s)",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Fraction of data held out for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: %(default)s)",
    )
    parser.add_argument(
        "--keep_harmful_compliances",
        action="store_true",
        default=False,
        help=(
            "If set, keep harmful records where the teacher complied "
            "(unsafe — only use for ablation studies)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print(f"Loading raw records from {input_path} ...")
    records = load_jsonl(input_path)
    print(f"  Loaded {len(records)} records.")

    print("\nFiltering ...")
    records = filter_records(records, args.keep_harmful_compliances)
    print(f"  Kept {len(records)} records after filtering.")

    print("\nFormatting into SFT chat template ...")
    formatted = []
    for r in records:
        example = format_sft_example(r["prompt"], r["response"])
        example["category"] = r["category"]
        example["is_refusal"] = r.get("is_refusal", False)
        formatted.append(example)

    print("\nSplitting into train/val ...")
    train_records, val_records = train_val_split(formatted, args.val_fraction, args.seed)
    print(f"  Train: {len(train_records)}  |  Val: {len(val_records)}")

    print("\nSaving ...")
    save_jsonl(train_records, output_dir / "train.jsonl")
    save_jsonl(val_records, output_dir / "val.jsonl")

    stats = compute_stats(train_records, val_records)
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats → {stats_path}")

    print("\n--- Final Dataset Summary ---")
    for split, s in stats.items():
        print(f"  [{split}]")
        print(f"    Total            : {s['total']}")
        print(f"    Benign           : {s['benign']}")
        print(f"    Harmful (refusal): {s['harmful_refusal']}")
        print(f"    Avg prompt len   : {s['avg_prompt_len']:.0f} chars")
        print(f"    Avg response len : {s['avg_response_len']:.0f} chars")
    print("-----------------------------")


if __name__ == "__main__":
    main()
