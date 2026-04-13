"""
Post-hoc safety recovery for a degraded student model.

Default workflow:
1. Start from the unsafe baseline adapter.
2. Merge that adapter into the base Llama-3.2-1B weights.
3. Attach a fresh LoRA adapter.
4. Fine-tune on a tiny refusal-only patch set (50-100 examples).

This keeps the experiment aligned with the milestone report's recovery question:
can safety be cheaply patched back into an unsafe distilled model?
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_prefix(prompt: str) -> str:
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_full(prompt: str, response: str) -> str:
    return f"{build_prefix(prompt)}{response}<|eot_id|>"


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_tokenizer(base_model_id: str, adapter_path: Path):
    for source in (str(adapter_path), base_model_id):
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            break
        except ValueError:
            continue
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_records(tokenizer, records: list[dict], max_length: int) -> Dataset:
    encoded = []
    dropped = 0
    for record in records:
        prefix = build_prefix(record["prompt"])
        full = build_full(record["prompt"], record["response"])
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        tokenized = tokenizer(
            full,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = list(input_ids)
        prefix_length = min(len(prefix_ids), len(labels))
        labels[:prefix_length] = [-100] * prefix_length
        if not any(label != -100 for label in labels):
            dropped += 1
            continue
        encoded.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    print(f"Encoded {len(encoded)} records, dropped {dropped}")
    return Dataset.from_list(encoded)


def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(features: list[dict]) -> dict:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_id] * pad)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad)
            batch_labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    return collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny refusal-only recovery adapter.")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--source_adapter", default="outputs/baseline")
    parser.add_argument("--train_file", default="data/recovery_baseline/train.jsonl")
    parser.add_argument("--val_file", default="data/recovery_baseline/val.jsonl")
    parser.add_argument("--output_dir", default="outputs/recovered_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_path = Path(args.train_file)
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    val_path = Path(args.val_file)
    if not val_path.is_absolute():
        val_path = PROJECT_ROOT / val_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    source_adapter = Path(args.source_adapter)
    if not source_adapter.is_absolute():
        source_adapter = PROJECT_ROOT / source_adapter

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path) if val_path.exists() else []
    if not train_records:
        raise ValueError(f"No training records found in {train_path}")

    tokenizer = load_tokenizer(args.base_model, source_adapter)
    train_dataset = encode_records(tokenizer, train_records, args.max_length)
    val_dataset = encode_records(tokenizer, val_records, args.max_length) if val_records else None

    dtype = choose_dtype()
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    merged = PeftModel.from_pretrained(base_model, str(source_adapter))
    merged = merged.merge_and_unload()
    merged.config.use_cache = False
    if hasattr(merged, "peft_config"):
        delattr(merged, "peft_config")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(merged, lora_cfg)
    model.print_trainable_parameters()

    eval_strategy = "epoch" if val_dataset is not None and len(val_dataset) > 0 else "no"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy=eval_strategy,
        save_total_limit=1,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=make_collator(tokenizer),
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "source_adapter": str(source_adapter),
        "train_file": str(train_path),
        "val_file": str(val_path),
        "num_train_examples": len(train_dataset),
        "num_val_examples": len(val_dataset) if val_dataset is not None else 0,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "metrics": result.metrics,
    }
    with open(output_dir / "recovery_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()