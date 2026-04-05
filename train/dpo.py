"""
DPO (Direct Preference Optimization) training for safety-aligned student model.

Starts from the SFT baseline checkpoint (outputs/baseline) and applies DPO using
preference pairs from data/dpo_pairs.jsonl:
  - chosen  = teacher refusal (safe behaviour to reinforce)
  - rejected = baseline model compliance (unsafe behaviour to suppress)

The reference model is a frozen copy of the same baseline checkpoint.

Usage (AutoDL):
    python -m train.dpo

Usage (Kaggle / HuggingFace):
    python -m train.dpo \
        --model_id   meta-llama/Llama-3.2-1B-Instruct \
        --sft_checkpoint outputs/baseline \
        --dpo_pairs  data/dpo_pairs.jsonl \
        --output_dir outputs/dpo

Design choices:
  beta=0.1      — mild KL penalty; keeps the model close to the SFT baseline
  lr=5e-5       — 4× lower than SFT; DPO destabilises quickly at high LR
  epochs=2      — preference data is small (~400 pairs); 3+ risks over-fitting
  LoRA r=16     — same rank as Member 2's SFT for easy comparison
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

from train.common import (
    MODEL_ROOT,
    REPO_ROOT,
    SaveRuntimeSummaryCallback,
    pick_dtype,
    resolve_target_modules,
)
from train.trainer_utils import save_json

# ---------------------------------------------------------------------------
# Defaults — mirror Member 2's common.py conventions
# ---------------------------------------------------------------------------

DEFAULT_DPO_PAIRS       = REPO_ROOT / "data" / "dpo_pairs.jsonl"
DEFAULT_SFT_CHECKPOINT  = REPO_ROOT / "outputs" / "baseline"
DEFAULT_OUTPUT_DIR      = REPO_ROOT / "outputs" / "dpo"
DEFAULT_LOGGING_DIR     = REPO_ROOT / "logs"    / "dpo"
DEFAULT_STUDENT_MODEL_ID    = "LLM-Research/Llama-3.2-1B-Instruct"
DEFAULT_STUDENT_MODEL_DIR   = MODEL_ROOT / "Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dpo_pairs(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"DPO pairs file not found: {path}\n"
            "Run `python -m train.prepare_dpo_data` first."
        )
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dpo_dataset(
    pairs: list[dict[str, str]],
    tokenizer,
) -> Dataset:
    """
    Format each pair into the messages schema expected by TRL DPOTrainer.

    TRL ≥ 0.9 format:
        prompt   → list[dict]  (system + user turns)
        chosen   → list[dict]  (single assistant turn — the safe refusal)
        rejected → list[dict]  (single assistant turn — the compliance)
    """
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        rows.append({
            "prompt": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": pair["prompt"]},
            ],
            "chosen":   [{"role": "assistant", "content": pair["chosen"]}],
            "rejected": [{"role": "assistant", "content": pair["rejected"]}],
        })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(args: argparse.Namespace, torch_dtype: torch.dtype):
    """
    Load the raw base model weights (no LoRA).
    Tries the local directory first, falls back to the HF model_id.
    """
    local_dir = Path(args.model_local_dir)
    model_path = str(local_dir) if local_dir.exists() else args.model_id
    print(f"[model] Base model: {model_path}")

    quantization_config = None
    if args.use_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        torch_dtype=None if quantization_config else torch_dtype,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, model_path


def load_policy_model(
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
):
    """
    Load the policy model: base weights → apply SFT LoRA → add new trainable DPO LoRA.

    DPO requires a reference model (frozen SFT baseline) and a policy model (trainable).
    TRL DPOTrainer handles the reference model internally when we pass `ref_model=None`
    and set `precompute_ref_log_probs=False` — it uses a frozen copy of the initial policy.

    We add a *fresh* LoRA on top of the merged SFT weights so that DPO only updates
    the new adapter, leaving the SFT knowledge intact.
    """
    base, model_path = load_base_model(args, torch_dtype)

    print(f"[model] Loading SFT adapter from: {args.sft_checkpoint}")
    # Merge SFT LoRA into base weights so the reference logits are stable
    sft_model = PeftModel.from_pretrained(base, args.sft_checkpoint)
    merged = sft_model.merge_and_unload()
    print("[model] SFT weights merged.")

    # Apply a fresh LoRA for DPO training
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=resolve_target_modules(args.target_modules),
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(merged, lora_config)
    policy.config.use_cache = False
    policy.print_trainable_parameters()

    return policy, model_path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DPO safety alignment training.")
    p.add_argument("--dpo_pairs",       type=str,   default=str(DEFAULT_DPO_PAIRS))
    p.add_argument("--sft_checkpoint",  type=str,   default=str(DEFAULT_SFT_CHECKPOINT))
    p.add_argument("--model_id",        type=str,   default=DEFAULT_STUDENT_MODEL_ID)
    p.add_argument("--model_local_dir", type=str,   default=str(DEFAULT_STUDENT_MODEL_DIR))
    p.add_argument("--output_dir",      type=str,   default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--logging_dir",     type=str,   default=str(DEFAULT_LOGGING_DIR))
    p.add_argument("--attn_implementation", type=str, default="sdpa")

    # DPO-specific
    p.add_argument("--beta",            type=float, default=0.1,
                   help="KL penalty coefficient. Higher = stay closer to reference.")
    p.add_argument("--loss_type",       type=str,   default="sigmoid",
                   choices=["sigmoid", "hinge", "ipo"],
                   help="DPO loss variant.")
    p.add_argument("--max_prompt_length", type=int, default=512,
                   help="Max tokens for the prompt portion.")
    p.add_argument("--max_length",      type=int,   default=1024,
                   help="Max total tokens (prompt + response).")

    # Training
    p.add_argument("--num_train_epochs",    type=float, default=2.0)
    p.add_argument("--learning_rate",       type=float, default=5e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--warmup_ratio",        type=float, default=0.05)
    p.add_argument("--weight_decay",        type=float, default=0.0)
    p.add_argument("--lr_scheduler_type",   type=str,   default="cosine")
    p.add_argument("--max_grad_norm",       type=float, default=1.0)
    p.add_argument("--logging_steps",       type=int,   default=10)
    p.add_argument("--save_steps",          type=int,   default=100)
    p.add_argument("--save_total_limit",    type=int,   default=2)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--report_to",           type=str,   default="none")
    p.add_argument("--optim",               type=str,   default="adamw_torch")
    p.add_argument("--dataloader_num_workers", type=int, default=0)

    # LoRA
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=32)
    p.add_argument("--lora_dropout",    type=float, default=0.05)
    p.add_argument("--target_modules",  type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Quantization
    p.add_argument("--use_4bit",        action="store_true", default=True)
    p.add_argument("--no_use_4bit",     action="store_true")

    # Misc
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Cap training pairs (for quick smoke tests).")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.use_4bit = args.use_4bit and not args.no_use_4bit
    args.output_dir  = str(Path(args.output_dir).resolve())
    args.logging_dir = str(Path(args.logging_dir).resolve())
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = normalize_args(build_parser().parse_args())
    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logging_dir).mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    local_dir = Path(args.model_local_dir)
    model_path_str = str(local_dir) if local_dir.exists() else args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_path_str, trust_remote_code=True)
    tokenizer.padding_side = "left"   # DPO uses left-padding for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ─────────────────────────────────────────────────────────────
    pairs = load_dpo_pairs(args.dpo_pairs)
    if args.max_train_samples:
        pairs = pairs[: args.max_train_samples]
    print(f"[data] DPO pairs loaded: {len(pairs)}")
    dpo_dataset = build_dpo_dataset(pairs, tokenizer)

    # ── Model ───────────────────────────────────────────────────────────────
    torch_dtype, torch_dtype_name = pick_dtype()
    policy, model_path = load_policy_model(args, torch_dtype)

    # ── DPO config ──────────────────────────────────────────────────────────
    use_bf16 = torch_dtype == torch.bfloat16
    use_fp16 = torch_dtype == torch.float16

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        beta=args.beta,
        loss_type=args.loss_type,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=True if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
        # Let TRL handle reference logits without a separate ref_model copy
        # by precomputing them before training. Saves ~1B params in VRAM.
        precompute_ref_log_probs=True,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=policy,
        ref_model=None,           # TRL uses a frozen snapshot of `model` at init
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
        callbacks=[SaveRuntimeSummaryCallback(Path(args.logging_dir) / "latest_metrics.json")],
    )

    # ── Train ───────────────────────────────────────────────────────────────
    print("[train] Starting DPO training …")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    # ── Save run summary ────────────────────────────────────────────────────
    save_json(
        Path(args.output_dir) / "final_summary.json",
        {
            "variant": "dpo",
            "model_path": str(model_path),
            "sft_checkpoint": args.sft_checkpoint,
            "dpo_pairs_file": args.dpo_pairs,
            "num_pairs": len(pairs),
            "beta": args.beta,
            "loss_type": args.loss_type,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "torch_dtype": torch_dtype_name,
            "train_metrics": train_metrics,
        },
    )
    print(f"[train] Done. Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
