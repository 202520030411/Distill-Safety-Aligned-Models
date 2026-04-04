import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from modelscope.hub.snapshot_download import snapshot_download

from train.dataset import create_or_sample_smoke_files, load_variant_dataset
from train.trainer_utils import SampleWeightedTrainer, SupervisedDataCollator, save_json

REPO_ROOT = Path("/root/autodl-tmp/Distill-Safety-Aligned-Models")
MODEL_ROOT = Path("/root/autodl-tmp/LLM")
DEFAULT_STUDENT_MODEL_ID = "LLM-Research/Llama-3.2-1B-Instruct"
DEFAULT_TEACHER_MODEL_ID = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
DEFAULT_STUDENT_MODEL_DIR = MODEL_ROOT / "Llama-3.2-1B-Instruct"
DEFAULT_TEACHER_MODEL_DIR = MODEL_ROOT / "Meta-Llama-3.1-8B-Instruct"
DEFAULT_TRAIN_FILE = REPO_ROOT / "data/train.jsonl"
DEFAULT_EVAL_FILE = REPO_ROOT / "data/val.jsonl"
DEFAULT_SMOKE_DIR = REPO_ROOT / "data/processed_smoke"


@dataclass
class RuntimeConfig:
    variant: str
    model_path: str
    model_id: str
    train_file: str
    eval_file: str
    output_dir: str
    logging_dir: str
    quantization_mode: str
    torch_dtype: str
    train_summary: dict[str, Any]
    eval_summary: dict[str, Any] | None
    smoke_metadata: dict[str, Any] | None


class SaveRuntimeSummaryCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            save_json(self.path, {"global_step": state.global_step, "latest_logs": logs})


def build_parser(variant: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {variant} student SFT training.")
    parser.add_argument("--train_file", type=str, default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--eval_file", type=str, default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--model_id", type=str, default=DEFAULT_STUDENT_MODEL_ID)
    parser.add_argument("--teacher_model_id", type=str, default=DEFAULT_TEACHER_MODEL_ID)
    parser.add_argument("--model_local_dir", type=str, default=str(DEFAULT_STUDENT_MODEL_DIR))
    parser.add_argument("--teacher_model_dir", type=str, default=str(DEFAULT_TEACHER_MODEL_DIR))
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / variant),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=str(REPO_ROOT / "logs" / variant),
    )
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_use_4bit", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--refusal_weight", type=float, default=3.0)
    parser.add_argument("--benign_weight", type=float, default=1.0)
    parser.add_argument("--use_smoke_data", action="store_true")
    parser.add_argument("--smoke_data_dir", type=str, default=str(DEFAULT_SMOKE_DIR))
    parser.add_argument("--smoke_train_size", type=int, default=24)
    parser.add_argument("--smoke_eval_size", type=int, default=8)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.use_4bit = args.use_4bit and not args.no_use_4bit
    args.gradient_checkpointing = not args.no_gradient_checkpointing
    args.output_dir = str(Path(args.output_dir).resolve())
    args.logging_dir = str(Path(args.logging_dir).resolve())
    args.model_local_dir = str(Path(args.model_local_dir).resolve())
    args.teacher_model_dir = str(Path(args.teacher_model_dir).resolve())
    args.train_file = str(Path(args.train_file).resolve())
    args.eval_file = str(Path(args.eval_file).resolve())
    args.smoke_data_dir = str(Path(args.smoke_data_dir).resolve())
    return args


def resolve_target_modules(target_modules: str) -> list[str]:
    return [module.strip() for module in target_modules.split(",") if module.strip()]


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def ensure_student_model(
    model_id: str,
    model_local_dir: str | Path,
    revision: str | None,
) -> Path:
    model_local_dir = Path(model_local_dir)
    artifact_markers = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    if model_local_dir.exists() and any((model_local_dir / marker).exists() for marker in artifact_markers):
        print(f"[model] Using existing local student model: {model_local_dir}")
        return model_local_dir

    ensure_dirs(MODEL_ROOT, model_local_dir)
    print(f"[model] Downloading student model from ModelScope: {model_id}")
    proxy_backup = {}
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"]:
        if key in os.environ:
            proxy_backup[key] = os.environ.pop(key)
    try:
        snapshot_download(
            model_id=model_id,
            revision=revision,
            cache_dir=str(MODEL_ROOT / "modelscope_cache"),
            local_dir=str(model_local_dir),
            allow_patterns=[
                "config.json",
                "configuration.json",
                "generation_config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "original/tokenizer.model",
            ],
        )
    finally:
        os.environ.update(proxy_backup)
    print(f"[model] Student model downloaded to: {model_local_dir}")
    return model_local_dir


def pick_dtype() -> tuple[torch.dtype, str]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"
    return torch.float32, "float32"


def load_student_model_and_tokenizer(args: argparse.Namespace, model_path: str | Path):
    model_path = Path(model_path)
    print(f"[model] Student model path: {model_path}")
    print(f"[model] Teacher model expected path (not used for training): {Path(args.teacher_model_dir)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype, torch_dtype_name = pick_dtype()
    use_4bit = bool(args.use_4bit and torch.cuda.is_available())
    quantization_mode = "lora"
    quantization_config = None

    if use_4bit:
        quantization_mode = "qlora"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
            torch_dtype=None if quantization_config is not None else torch_dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    except Exception as exc:
        if not use_4bit:
            raise
        print(f"[model] 4-bit load failed, falling back to LoRA without quantization: {exc}")
        quantization_mode = "lora_fallback"
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    if quantization_mode.startswith("qlora"):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=resolve_target_modules(args.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    return model, tokenizer, model_path, quantization_mode, torch_dtype_name


def prepare_dataset_paths(args: argparse.Namespace) -> tuple[str, str, dict[str, Any] | None]:
    smoke_metadata = None
    train_file = args.train_file
    eval_file = args.eval_file

    if args.use_smoke_data:
        smoke_train, smoke_eval, smoke_metadata = create_or_sample_smoke_files(
            train_source=train_file,
            eval_source=eval_file,
            output_dir=args.smoke_data_dir,
            seed=args.seed,
            train_size=args.smoke_train_size,
            eval_size=args.smoke_eval_size,
        )
        train_file = str(smoke_train)
        eval_file = str(smoke_eval)

    return train_file, eval_file, smoke_metadata


def build_training_arguments(
    args: argparse.Namespace,
    has_eval: bool,
) -> TrainingArguments:
    torch_dtype, torch_dtype_name = pick_dtype()
    use_bf16 = torch_dtype == torch.bfloat16
    use_fp16 = torch_dtype == torch.float16
    print(f"[runtime] Torch dtype for training: {torch_dtype_name}")

    return TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.eval_steps if has_eval else None,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=True if torch.cuda.is_available() else False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        report_to=args.report_to,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        save_only_model=False,
        do_train=True,
        do_eval=has_eval,
        prediction_loss_only=True,
    )


def save_run_manifest(args: argparse.Namespace, runtime_config: RuntimeConfig) -> None:
    manifest = {"args": vars(args), "runtime": asdict(runtime_config)}
    save_json(Path(args.output_dir) / "run_manifest.json", manifest)


def run_variant(variant: str) -> None:
    args = normalize_args(build_parser(variant).parse_args())
    ensure_dirs(args.output_dir, args.logging_dir, MODEL_ROOT)
    set_seed(args.seed)

    model_path = ensure_student_model(args.model_id, args.model_local_dir, args.model_revision)
    if args.download_only:
        save_json(
            Path(args.output_dir) / "download_only.json",
            {
                "student_model_id": args.model_id,
                "student_model_path": str(model_path),
                "teacher_model_id": args.teacher_model_id,
                "teacher_model_path": args.teacher_model_dir,
            },
        )
        return

    model, tokenizer, model_path, quantization_mode, torch_dtype_name = load_student_model_and_tokenizer(
        args,
        model_path=model_path,
    )

    train_file, eval_file, smoke_metadata = prepare_dataset_paths(args)
    print(f"[data] Train file: {train_file}")
    print(f"[data] Eval file: {eval_file}")

    train_dataset, train_summary = load_variant_dataset(
        path=train_file,
        variant=variant,
        tokenizer=tokenizer,
        max_length=args.max_length,
        refusal_weight=args.refusal_weight,
        benign_weight=args.benign_weight,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    eval_dataset, eval_summary = load_variant_dataset(
        path=eval_file,
        variant=variant,
        tokenizer=tokenizer,
        max_length=args.max_length,
        refusal_weight=args.refusal_weight,
        benign_weight=args.benign_weight,
        max_samples=args.max_eval_samples,
        seed=args.seed + 1,
    )

    print(f"[data] Train summary: {train_summary}")
    print(f"[data] Eval summary: {eval_summary}")

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty after filtering/encoding.")

    training_args = build_training_arguments(args, has_eval=len(eval_dataset) > 0)
    collator = SupervisedDataCollator(tokenizer=tokenizer)
    trainer = SampleWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=collator,
        callbacks=[SaveRuntimeSummaryCallback(Path(args.logging_dir) / "latest_metrics.json")],
    )

    runtime_config = RuntimeConfig(
        variant=variant,
        model_path=str(model_path),
        model_id=args.model_id,
        train_file=train_file,
        eval_file=eval_file,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        quantization_mode=quantization_mode,
        torch_dtype=torch_dtype_name,
        train_summary=train_summary,
        eval_summary=eval_summary,
        smoke_metadata=smoke_metadata,
    )
    save_run_manifest(args, runtime_config)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    eval_metrics = None
    if len(eval_dataset) > 0:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    save_json(
        Path(args.output_dir) / "final_summary.json",
        {
            "runtime": asdict(runtime_config),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
        },
    )
