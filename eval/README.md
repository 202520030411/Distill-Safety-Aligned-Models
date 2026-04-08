# Evaluation

This directory fills in the missing evaluation step described in the project README and milestone report.

## What is implemented

- `eval_safety.py`: harmful prompts -> unsafe compliance rate
- `eval_refusal.py`: benign prompts -> false refusal rate
- `eval_quality.py`: benign prompts -> teacher-reference quality proxy
- `eval_jailbreak.py`: harmful prompts with jailbreak templates -> unsafe compliance under attack
- `run_all.py`: runs the whole local suite and writes JSON summaries to `results/eval/`

## Default dataset

All scripts default to `data/val.jsonl`, which is the only held-out split currently in the repo.

That split contains:

- 75 benign prompts
- 25 harmful prompts

This is enough for a reproducible local eval pass, but it is small. If you want stronger final results, create a separate `test.jsonl` from `data/raw_teacher_outputs.jsonl` and point the scripts at it with `--dataset`.

## Important model-loading detail

`outputs/dpo/` is not a standalone final model. DPO training was done on:

1. base `meta-llama/Llama-3.2-1B`
2. plus merged `outputs/baseline/`
3. plus a fresh DPO LoRA adapter

So evaluation must load DPO as:

1. base model
2. merge `baseline`
3. attach `dpo`

`common.py` handles this automatically when you pass `--model dpo`.

## Commands

Run one metric for one model:

```bash
python eval/eval_safety.py --model baseline
python eval/eval_refusal.py --model with_refusals
python eval/eval_quality.py --model weighted
python eval/eval_jailbreak.py --model dpo
```

Run the whole suite:

```bash
python eval/run_all.py --models teacher baseline with_refusals weighted dpo
```

Use a larger but non-held-out dataset for jailbreak stress testing:

```bash
python eval/run_all.py \
  --models baseline with_refusals weighted dpo \
  --dataset data/val.jsonl \
  --jailbreak_dataset data/raw_teacher_outputs.jsonl
```

## Metric caveats

- `safety` and `refusal` use the repo's existing refusal heuristic, not Llama Guard.
- `quality` is a local proxy: token-level F1 against the teacher response saved in the dataset.
- If you need a paper-grade final table, replace the quality proxy with MT-Bench or AlpacaEval and replace the refusal heuristic with a stronger safety judge.
