# Evaluation Results

Local evaluation was run on `data/val.jsonl`, the held-out split currently checked into the repo.

## Eval Setup

- **Benign prompts**: 75
- **Harmful prompts**: 25
- **Models**: `teacher`, `baseline`, `with_refusals`, `weighted`, `dpo`, `on_policy_dpo`, `dpo_with_refusals`, `on_policy_dpo_with_refusals`
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

## Summary Table — DPO on baseline (no safety in SFT)

| Model | Unsafe Compliance | False Refusal | Quality Proxy (F1) | Jailbreak Unsafe |
|---|---:|---:|---:|---:|
| `teacher` | 0.08 | 0.00 | 0.590 | 0.040 |
| `baseline` | 1.00 | 0.00 | 0.316 | 0.973 |
| `with_refusals` | 0.04 | 0.08 | 0.301 | 0.040 |
| `weighted` | 0.04 | 0.067 | 0.294 | 0.080 |
| `dpo` | 0.96 | 0.027 | 0.311 | 0.827 |
| `on_policy_dpo` | 0.96 | 0.013 | 0.320 | 0.947 |

## Summary Table — DPO on with_refusals (safety already in SFT)

| Model | Unsafe Compliance | False Refusal | Quality Proxy (F1) | Jailbreak Unsafe |
|---|---:|---:|---:|---:|
| `with_refusals` | 0.04 | 0.080 | 0.304 | 0.040 |
| `dpo_with_refusals` | 0.24 | 0.160 | 0.312 | 0.080 |
| `on_policy_dpo_with_refusals` | 0.04 | 0.053 | 0.332 | 0.080 |

Source files live under `results/eval/`. Rebuild the aggregate table from existing per-metric JSON files with:

```bash
python eval/build_summary.py
```

## Interpretation

- `baseline` confirms that benign-only distillation does not preserve safety.
- `with_refusals` and `weighted` retain strong refusal behavior under both standard harmful prompts and jailbreak prompts.
- Both DPO variants on `baseline` fail to inject safety: they remain close to `baseline` on unsafe compliance and jailbreak vulnerability.
- Off-policy DPO on `with_refusals` partially degrades safety — unsafe compliance rises from `0.04` to `0.24`, and false refusals double.
- **On-policy DPO on `with_refusals` is the best overall model**: it preserves full safety (`0.04`), reduces false refusals (`0.080` → `0.053`), and improves quality (`0.304` → `0.332`).
- The starting point matters more than the alignment method — DPO can refine existing safety but cannot create it from scratch.
