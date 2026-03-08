"""
Generate the distillation dataset by running the teacher model (Llama-3.1-8B-Instruct)
on benign instruction prompts and harmful prompts.

Outputs a JSONL file where each record has:
  - prompt:      the user input
  - response:    the teacher's output
  - category:    "benign" | "harmful"
  - is_refusal:  bool (heuristic-detected refusal on harmful prompts)

Sources:
  Benign  : tatsu-lab/alpaca (instruction-following)
  Harmful : walledai/AdvBench (harmful behaviors subset)
"""

import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEACHER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

REFUSAL_PREFIXES = [
    "I'm sorry",
    "I can't",
    "I cannot",
    "I'm not able",
    "I won't",
    "I will not",
    "As an AI",
    "I'm unable",
    "I apologize",
    "That's not something I",
    "I don't think I should",
    "This request involves",
    "I must decline",
]

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest assistant."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_refusal(text: str) -> bool:
    """Heuristic check: does the response look like a refusal?"""
    text_lower = text.lower().strip()
    for prefix in REFUSAL_PREFIXES:
        if text_lower.startswith(prefix.lower()):
            return True
    refusal_patterns = [
        r"i (am|'m) (not able|unable) to",
        r"i (can't|cannot|won't|will not) (help|assist|provide|generate|create)",
        r"(this|that) (request|prompt|query) (is|seems|appears) (harmful|dangerous|unethical|illegal)",
    ]
    for pattern in refusal_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def format_chat_prompt(system: str, user: str) -> str:
    """Format a single-turn chat prompt using Llama-3 chat template."""
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def load_benign_prompts(n: int) -> list[str]:
    """Load n instruction prompts from the Alpaca dataset."""
    print(f"Loading {n} benign prompts from tatsu-lab/alpaca ...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    prompts = []
    for row in ds:
        instruction = row["instruction"].strip()
        inp = row["input"].strip()
        if inp:
            user_turn = f"{instruction}\n\n{inp}"
        else:
            user_turn = instruction
        if len(user_turn) > 20:  # skip trivially short prompts
            prompts.append(user_turn)
        if len(prompts) >= n:
            break

    print(f"  Loaded {len(prompts)} benign prompts.")
    return prompts


def load_harmful_prompts(n: int) -> list[str]:
    """Load n harmful behavior prompts from AdvBench."""
    print(f"Loading {n} harmful prompts from walledai/AdvBench ...")
    ds = load_dataset("walledai/AdvBench", split="train")

    prompts = []
    for row in ds:
        prompt = row["prompt"].strip()
        if prompt:
            prompts.append(prompt)
        if len(prompts) >= n:
            break

    print(f"  Loaded {len(prompts)} harmful prompts.")
    return prompts


def run_inference(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    system: str,
    category: str,
) -> list[dict]:
    """Run teacher inference on a list of prompts and return records."""
    formatted = [format_chat_prompt(system, p) for p in prompts]
    outputs = llm.generate(formatted, sampling_params)

    records = []
    for prompt, output in zip(prompts, outputs):
        response = output.outputs[0].text.strip()
        record = {
            "prompt": prompt,
            "response": response,
            "category": category,
            "is_refusal": is_refusal(response) if category == "harmful" else False,
        }
        records.append(record)

    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def print_stats(records: list[dict]) -> None:
    benign = [r for r in records if r["category"] == "benign"]
    harmful = [r for r in records if r["category"] == "harmful"]
    refusals = [r for r in harmful if r["is_refusal"]]

    print("\n--- Dataset Statistics ---")
    print(f"  Total records  : {len(records)}")
    print(f"  Benign         : {len(benign)}")
    print(f"  Harmful        : {len(harmful)}")
    print(f"  Refusals       : {len(refusals)} / {len(harmful)} "
          f"({100 * len(refusals) / max(len(harmful), 1):.1f}%)")
    print(f"  Compliances    : {len(harmful) - len(refusals)} / {len(harmful)}")
    print("--------------------------\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate distillation dataset using a safety-aligned teacher model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=TEACHER_MODEL,
        help="HuggingFace model ID for the teacher (default: %(default)s)",
    )
    parser.add_argument(
        "--n_benign",
        type=int,
        default=5000,
        help="Number of benign instruction prompts to include (default: %(default)s)",
    )
    parser.add_argument(
        "--n_harmful",
        type=int,
        default=500,
        help="Number of harmful prompts to include (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw_teacher_outputs.jsonl",
        help="Output JSONL file path (default: %(default)s)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate per response (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0.0 = greedy (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM may use (default: %(default)s)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "Warning: HF_TOKEN environment variable not set. "
            "You may need it to access gated Llama models on HuggingFace."
        )

    # Load prompts
    benign_prompts = load_benign_prompts(args.n_benign)
    harmful_prompts = load_harmful_prompts(args.n_harmful)

    # Initialize vLLM engine
    print(f"\nLoading teacher model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=False,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    # Run inference
    print("\nRunning inference on benign prompts ...")
    benign_records = run_inference(
        llm, benign_prompts, sampling_params, SYSTEM_PROMPT, "benign"
    )

    print("\nRunning inference on harmful prompts ...")
    harmful_records = run_inference(
        llm, harmful_prompts, sampling_params, SYSTEM_PROMPT, "harmful"
    )

    all_records = benign_records + harmful_records
    print_stats(all_records)

    save_jsonl(all_records, Path(args.output))


if __name__ == "__main__":
    main()
