"""
Prepare DPO preference pairs for safety-aligned distillation.

For each harmful prompt in raw_teacher_outputs.jsonl:
  - chosen  = teacher's refusal (the safe response we want)
  - rejected = baseline SFT model's response (trained on benign-only, likely to comply)

Output: data/dpo_pairs.jsonl
Each record: { "prompt", "chosen", "rejected" }

Run AFTER outputs/baseline/ checkpoint exists (Member 2's SFT baseline).

Usage:
    python -m train.prepare_dpo_data \
        --raw_file   data/raw_teacher_outputs.jsonl \
        --output     data/dpo_pairs.jsonl \
        --model_id   LLM-Research/Llama-3.2-1B-Instruct \
        --model_local_dir /root/autodl-tmp/LLM/Llama-3.2-1B-Instruct \
        --sft_checkpoint  outputs/baseline
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.dataset import SYSTEM_PROMPT, load_jsonl_records
from train.common import MODEL_ROOT, REPO_ROOT
from train.trainer_utils import save_json

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RAW_FILE = REPO_ROOT / "data" / "raw_teacher_outputs.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "dpo_pairs.jsonl"
DEFAULT_SFT_CHECKPOINT = REPO_ROOT / "outputs" / "baseline"
DEFAULT_STUDENT_MODEL_ID = "LLM-Research/Llama-3.2-1B-Instruct"
DEFAULT_STUDENT_MODEL_DIR = MODEL_ROOT / "Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt_messages(user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    """Run batched greedy inference and return decoded responses."""
    responses = []
    model.eval()

    for i in tqdm(range(0, len(prompts), batch_size), desc="generating rejected"):
        batch = prompts[i : i + batch_size]
        chats = [build_prompt_messages(p) for p in batch]

        inputs_text = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            for chat in chats
        ]
        inputs = tokenizer(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for output_ids in outputs:
            response = tokenizer.decode(
                output_ids[input_len:], skip_special_tokens=True
            ).strip()
            responses.append(response)

    return responses


def is_refusal(text: str) -> bool:
    """Lightweight heuristic — mirrors generate_data.py."""
    import re
    text_lower = text.lower().strip()
    prefixes = [
        "i'm sorry", "i can't", "i cannot", "i'm not able", "i won't",
        "i will not", "as an ai", "i'm unable", "i apologize",
        "that's not something i", "i don't think i should",
        "this request involves", "i must decline",
    ]
    for p in prefixes:
        if text_lower.startswith(p):
            return True
    patterns = [
        r"i (am|'m) (not able|unable) to",
        r"i (can't|cannot|won't|will not) (help|assist|provide|generate|create)",
        r"(this|that) (request|prompt|query) (is|seems|appears) (harmful|dangerous|unethical|illegal)",
    ]
    for pat in patterns:
        if re.search(pat, text_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare DPO preference pairs.")
    p.add_argument("--raw_file",        type=str, default=str(DEFAULT_RAW_FILE))
    p.add_argument("--output",          type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--sft_checkpoint",  type=str, default=str(DEFAULT_SFT_CHECKPOINT),
                   help="Path to baseline SFT LoRA adapter directory (outputs/baseline).")
    p.add_argument("--model_id",        type=str, default=DEFAULT_STUDENT_MODEL_ID,
                   help="HuggingFace / ModelScope model ID for the base student model.")
    p.add_argument("--model_local_dir", type=str, default=str(DEFAULT_STUDENT_MODEL_DIR),
                   help="Local directory where the base student model weights are stored.")
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--max_new_tokens",  type=int, default=256)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve model path: local first, then HF model_id ──────────────────
    local_dir = Path(args.model_local_dir)
    model_path = str(local_dir) if local_dir.exists() else args.model_id
    print(f"[model] Base model: {model_path}")
    print(f"[model] SFT checkpoint: {args.sft_checkpoint}")

    # ── Load tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load base model + baseline LoRA adapter ─────────────────────────────
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, args.sft_checkpoint)
    print("[model] Baseline SFT adapter loaded.")

    # ── Load harmful prompts with teacher refusals ──────────────────────────
    records = load_jsonl_records(args.raw_file)
    harmful_with_refusal = [
        r for r in records
        if r.get("category") == "harmful" and r.get("is_refusal", False)
    ]
    print(f"[data] Harmful refusal records: {len(harmful_with_refusal)}")

    prompts  = [r["prompt"]   for r in harmful_with_refusal]
    chosen   = [r["response"] for r in harmful_with_refusal]  # teacher refusals

    # ── Generate rejected responses from baseline model ─────────────────────
    rejected = generate_responses(
        model, tokenizer, prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # ── Build DPO pairs ─────────────────────────────────────────────────────
    pairs = []
    skipped_both_refusal = 0
    for prompt, ch, rej in zip(prompts, chosen, rejected):
        # If baseline also refused, skip — not a useful preference pair
        if is_refusal(rej):
            skipped_both_refusal += 1
            continue
        pairs.append({"prompt": prompt, "chosen": ch, "rejected": rej})

    print(f"[data] Valid DPO pairs      : {len(pairs)}")
    print(f"[data] Skipped (both refuse): {skipped_both_refusal}")

    # ── Save ────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[data] Saved {len(pairs)} DPO pairs → {output_path}")

    # Save a short stats summary alongside the pairs
    save_json(
        output_path.with_suffix(".stats.json"),
        {
            "total_harmful_refusals": len(harmful_with_refusal),
            "valid_pairs": len(pairs),
            "skipped_both_refuse": skipped_both_refusal,
            "model_path": str(model_path),
            "sft_checkpoint": args.sft_checkpoint,
        },
    )


if __name__ == "__main__":
    main()
