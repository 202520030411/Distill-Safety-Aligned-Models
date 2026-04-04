# Member2 SFT Results

This update adds the member2 student SFT training pipeline and the first full set of training results for the 1B student model.

## Setup

- Student model: `LLM-Research/Llama-3.2-1B-Instruct`
- Local model path: `/root/autodl-tmp/LLM/Llama-3.2-1B-Instruct`
- Training method: QLoRA
- Dataset files:
  - `data/train.jsonl`
  - `data/val.jsonl`

## Dataset Summary

- Train total: `1885`
- Train benign: `1425`
- Train harmful refusal: `460`
- Val total: `100`
- Val benign: `75`
- Val harmful refusal: `25`

## Variants

| Variant | Description | Train Loss | Eval Loss | Runtime (s) |
| --- | --- | ---: | ---: | ---: |
| `baseline` | benign-only SFT | `3.6297` | `0.5593` | `222.57` |
| `with_refusals` | benign + harmful-refusal SFT | `3.0232` | `0.4516` | `289.53` |
| `weighted` | same as `with_refusals`, with refusal-weighted loss | `2.6679` | `0.3859` | `288.52` |

## Summary

The current ranking on held-out validation loss is:

1. `weighted`
2. `with_refusals`
3. `baseline`

For this dataset, adding refusal examples improves validation loss substantially over benign-only SFT, and applying higher loss weight to refusal examples improves it further.

## Repro Commands

```bash
bash scripts/run_baseline.sh
bash scripts/run_with_refusals.sh
bash scripts/run_weighted.sh
```

The weighted variant uses a real sample-weight-aware loss in `train/trainer_utils.py` rather than duplicating refusal samples.
