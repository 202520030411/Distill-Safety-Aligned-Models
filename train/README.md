# Safety Recovery

This folder contains the minimal post-hoc safety recovery scaffold.

## Goal

Patch safety back into an unsafe distilled model using only a small refusal-only dataset.

## Default recovery setup

1. Sample 100 harmful refusal examples from `data/train.jsonl`
2. Start from `outputs/baseline`
3. Merge `baseline` into the base 1B model
4. Train a fresh LoRA adapter on the refusal patch set
5. Save the recovered adapter to `outputs/recovered_baseline`

## Commands

Prepare the patch dataset:

```bash
python data/prepare_recovery_dataset.py --num_examples 100 --output_dir data/recovery_baseline
```

Train the recovery adapter:

```bash
python train/safety_recovery.py \
  --train_file data/recovery_baseline/train.jsonl \
  --val_file data/recovery_baseline/val.jsonl \
  --source_adapter outputs/baseline \
  --output_dir outputs/recovered_baseline
```

Evaluate the recovered model with the existing eval suite:

```bash
python eval/eval_safety.py --model recovered_baseline
python eval/eval_refusal.py --model recovered_baseline
python eval/eval_jailbreak.py --model recovered_baseline
```
