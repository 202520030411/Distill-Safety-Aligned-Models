# Evaluation Results

Local evaluation was run on `data/val.jsonl`, the held-out split currently checked into the repo.

## Eval Setup

- **Benign prompts**: 75
- **Harmful prompts**: 25
- **Models**: `teacher`, `baseline`, `with_refusals`, `weighted`, `dpo`, `on_policy_dpo`
- **Metrics**:
  - unsafe compliance rate on harmful prompts
  - false refusal rate on benign prompts
  - teacher-reference token-level F1 on benign prompts
  - jailbreak unsafe compliance rate (macro average over roleplay / DAN / prefix injection)

## Minimal Output Normalization

The eval code performs minimal normalization before refusal judging:

- strips leaked chat-role prefixes such as `assistant`
- strips leaked special tokens such as `<|eot_id|>`
- normalizes whitespace and removes replacement-character artifacts
- treats "refuse first, then continue helping" outputs as unsafe compliance

This is intended to correct decoding artifacts in generated text, not to rewrite model content.

## Summary Table

| Model | Unsafe Compliance | False Refusal | Quality Proxy (F1) | Jailbreak Unsafe |
|---|---:|---:|---:|---:|
| `teacher` | 0.08 | 0.00 | 0.590 | 0.040 |
| `baseline` | 1.00 | 0.00 | 0.316 | 0.973 |
| `with_refusals` | 0.04 | 0.08 | 0.301 | 0.040 |
| `weighted` | 0.04 | 0.067 | 0.294 | 0.080 |
| `dpo` | 0.96 | 0.027 | 0.311 | 0.827 |
| `on_policy_dpo` | 0.96 | 0.013 | 0.320 | 0.947 |

Source files live under `results/eval/`. Rebuild the aggregate table from existing per-metric JSON files with:

```bash
python eval/build_summary.py
```

## Interpretation

- `baseline` confirms that benign-only distillation does not preserve safety.
- `with_refusals` and `weighted` are the only variants that retain strong refusal behavior under both standard harmful prompts and jailbreak prompts.
- `weighted` slightly reduces false refusals relative to `with_refusals`, but both models show some generation degeneration such as repeated refusals.
- Both DPO variants fail to preserve safety in this setup: they remain close to `baseline` on unsafe compliance and jailbreak vulnerability.
- `on_policy_dpo` improves benign-side helpfulness slightly relative to off-policy `dpo`, but not safety.
