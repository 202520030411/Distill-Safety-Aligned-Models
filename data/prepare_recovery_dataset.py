"""
Prepare a small refusal-only patch dataset for post-hoc safety recovery.

By default this samples from data/train.jsonl rather than raw_teacher_outputs.jsonl,
so the held-out eval split in data/val.jsonl remains untouched.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a small harmful-refusal patch set for safety recovery."
    )
    parser.add_argument(
        "--input",
        default="data/train.jsonl",
        help="Source JSONL. Default uses the training split to avoid eval leakage.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/recovery_baseline",
        help="Directory for the sampled train/val recovery files.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Total number of harmful-refusal examples to sample.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of the sampled patch set reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    rng = random.Random(args.seed)

    records = load_jsonl(input_path)
    refusals = [
        record
        for record in records
        if record.get("category") == "harmful" and record.get("is_refusal")
    ]

    if not refusals:
        raise ValueError(f"No harmful refusal records found in {input_path}")

    if args.num_examples <= 0:
        raise ValueError("--num_examples must be positive")

    if args.num_examples > len(refusals):
        raise ValueError(
            f"Requested {args.num_examples} examples but only found {len(refusals)} harmful refusals"
        )

    sampled = refusals[:]
    rng.shuffle(sampled)
    sampled = sampled[: args.num_examples]

    if args.num_examples == 1:
        train_records = sampled
        val_records: list[dict] = []
    else:
        val_size = max(1, int(round(args.num_examples * args.val_fraction)))
        val_size = min(val_size, args.num_examples - 1)
        train_records = sampled[val_size:]
        val_records = sampled[:val_size]

    save_jsonl(output_dir / "train.jsonl", train_records)
    save_jsonl(output_dir / "val.jsonl", val_records)

    stats = {
        "source": str(input_path),
        "num_available_refusals": len(refusals),
        "num_selected": len(sampled),
        "train_size": len(train_records),
        "val_size": len(val_records),
        "seed": args.seed,
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()