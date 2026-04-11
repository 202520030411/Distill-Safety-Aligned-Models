# Evaluation

This directory fills in the missing evaluation step described in the project README and milestone report.

## What is implemented

- `eval_safety.py`: harmful prompts -> unsafe compliance rate
- `eval_refusal.py`: benign prompts -> false refusal rate
- `eval_quality.py`: benign prompts -> teacher-reference quality proxy
- `eval_jailbreak.py`: harmful prompts with jailbreak templates -> unsafe compliance under attack
- `run_all.py`: runs the whole local suite and writes JSON summaries to `results/eval/`
- `build_summary.py`: rebuilds `results/eval/summary.json` from existing per-metric JSON files

## Default dataset

All scripts default to `data/val.jsonl`, which is the only held-out split currently in the repo.

That split contains:

- 75 benign prompts
- 25 harmful prompts

This is enough for a reproducible local eval pass, but it is small. If you want stronger final results, create a separate `test.jsonl` from `data/raw_teacher_outputs.jsonl` and point the scripts at it with `--dataset`.

## Important model-loading detail

`outputs/dpo/` and `outputs/on_policy_dpo/` are not standalone final models. Both were trained on:

1. base `meta-llama/Llama-3.2-1B`
2. plus merged `outputs/baseline/`
3. plus a fresh DPO LoRA adapter

So evaluation must load both DPO variants as:

1. base model
2. merge `baseline`
3. attach `dpo` or `on_policy_dpo`

`common.py` handles this automatically when you pass `--model dpo` or `--model on_policy_dpo`.

Likewise, `outputs/dpo_with_refusals/` and `outputs/on_policy_dpo_with_refusals/` are also stacked adapters:

1. base model
2. merge `with_refusals`
3. attach `dpo_with_refusals` or `on_policy_dpo_with_refusals`

`common.py` also handles this automatically.

## Commands

Run one metric for one model:

```bash
python eval/eval_safety.py --model baseline
python eval/eval_refusal.py --model with_refusals
python eval/eval_quality.py --model weighted
python eval/eval_jailbreak.py --model dpo
python eval/eval_safety.py --model on_policy_dpo
python eval/eval_refusal.py --model on_policy_dpo
python eval/eval_quality.py --model on_policy_dpo
python eval/eval_jailbreak.py --model on_policy_dpo
python eval/eval_safety.py --model dpo_with_refusals
python eval/eval_quality.py --model on_policy_dpo_with_refusals
```

Run the whole suite:

```bash
python eval/run_all.py --models teacher baseline with_refusals weighted dpo on_policy_dpo dpo_with_refusals on_policy_dpo_with_refusals
```

Rebuild the summary table without rerunning model inference:

```bash
python eval/build_summary.py
python eval/build_summary.py --models teacher baseline with_refusals weighted dpo on_policy_dpo dpo_with_refusals on_policy_dpo_with_refusals
```

Use a larger but non-held-out dataset for jailbreak stress testing:

```bash
python eval/run_all.py \
  --models baseline with_refusals weighted dpo on_policy_dpo dpo_with_refusals on_policy_dpo_with_refusals \
  --dataset data/val.jsonl \
  --jailbreak_dataset data/raw_teacher_outputs.jsonl
```

## Metric caveats

- `safety` and `refusal` use the repo's existing refusal heuristic, not Llama Guard.
- `quality` is a local proxy: token-level F1 against the teacher response saved in the dataset.
- Before refusal judging, eval applies minimal output normalization: stripping leaked role prefixes and special tokens, normalizing whitespace, and marking "refuse first, then continue helping" outputs as unsafe.
- The SFT refusal-heavy models still show generation degeneration in some outputs (for example repeated refusals). Those artifacts are not aggressively removed from the scored text.
- If you need a paper-grade final table, replace the quality proxy with MT-Bench or AlpacaEval and replace the refusal heuristic with a stronger safety judge.
