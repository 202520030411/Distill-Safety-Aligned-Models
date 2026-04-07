# Distill Safety-Aligned Models

**Research question:** Does knowledge distillation preserve the safety alignment of large language models, or does it unintentionally remove it?

We distill a safety-aligned teacher model (Llama-3.1-8B-Instruct) into a smaller student model (Llama-3.2-1B base) using synthetic data, then evaluate how well safety is retained across different training strategies.

See `milestone_report.pdf` for the full project plan.

---

## Repository Structure

```
.
├── data/                        # Dataset pipeline
│   ├── generate_data.py         # Run teacher on benign + harmful prompts
│   ├── prepare_dataset.py       # Filter, format, and split into train/val
│   ├── test_logic.py            # Smoke tests (no GPU required)
│   ├── raw_teacher_outputs.jsonl  # 2000 teacher responses (1500 benign, 500 harmful)
│   ├── train.jsonl              # 1885 SFT training examples
│   ├── val.jsonl                # 100 SFT validation examples
│   └── dpo_pairs.jsonl          # 485 DPO preference pairs (chosen=refusal, rejected=base compliance)
│
├── train/                       # Training utilities and DPO scripts
│   ├── common.py                # Shared model loading and path utilities
│   ├── dataset.py               # Dataset loading, filtering, encoding
│   ├── trainer_utils.py         # Custom collator and weighted loss trainer
│   ├── prepare_dpo_data.py      # Generate DPO rejected responses from base model
│   └── dpo.py                   # DPO training on top of SFT baseline
│
├── outputs/                     # Trained LoRA adapters
│   ├── baseline/                # SFT benign-only adapter
│   ├── with_refusals/           # SFT with refusals adapter
│   ├── weighted/                # SFT weighted adapter
│   └── dpo/                     # DPO adapter (built on top of baseline)
│
├── eval/                        # Evaluation scripts (in progress)
├── results/                     # Training summaries and metrics
│   └── sft_results.md           # SFT training setup and variant descriptions
├── kaggle_generate_data.ipynb   # Kaggle notebook: teacher inference (data generation)
├── kaggle_train.ipynb           # Kaggle notebook: all SFT variants + DPO training
└── requirements.txt
```

---

## Pipeline Overview

### Step 1 — Data Generation (Member 1)

Run the teacher model (Llama-3.1-8B-Instruct) on benign and harmful prompts to collect training data.

```
Teacher inference → raw_teacher_outputs.jsonl → train.jsonl / val.jsonl
```

- **Benign prompts**: 1500 examples from the Alpaca dataset
- **Harmful prompts**: 500 examples from AdvBench
- Teacher refuses ~97% of harmful prompts
- Final training set: 1885 examples (benign + teacher refusals, harmful compliances dropped)

Kaggle notebook: `kaggle_generate_data.ipynb`

---

### Step 2 — SFT Training (Member 1)

Three supervised fine-tuning variants on the student model (`meta-llama/Llama-3.2-1B` base) using QLoRA:

| Variant | Data | Description |
|---|---|---|
| `baseline` | Benign only | Teaches instruction-following; no safety signal |
| `with_refusals` | Benign + teacher refusals | Adds teacher refusal responses as training examples |
| `weighted` | Benign + refusals (refusals upweighted 3×) | Extra loss weight on refusal examples |

LoRA config: `r=16`, `alpha=32`, `dropout=0.05`, all attention + MLP projection layers.  
Optimizer: `paged_adamw_8bit` (QLoRA 4-bit). Run on Kaggle T4 via `kaggle_train.ipynb`.

---

### Step 3 — DPO Training (Member 1)

Direct Preference Optimization on top of the SFT baseline to further reinforce safety.

```
Base model (Llama-3.2-1B, no safety) → generates rejected responses
Teacher refusals → chosen responses
→ 485 preference pairs in data/dpo_pairs.jsonl
→ DPO trains outputs/dpo/ adapter
```

**Key design decisions:**
- Rejected responses come from `Llama-3.2-1B` (base, no Instruct) — guaranteed to comply with harmful prompts
- SFT LoRA is merged into base weights before DPO, so reference logits are stable
- Fresh LoRA adapter added on top for DPO updates (β=0.1, lr=5e-5, 2 epochs)
- Train loss: 0.072 after ~55 min on Kaggle T4

Kaggle notebook: `kaggle_train.ipynb` (DPO section runs after SFT)

---

### Step 4 — Evaluation (Members 3 & 4)

Evaluating all four models (baseline, with_refusals, weighted, dpo) on:
- **Unsafe compliance rate (ASR)**: how often the model complies with harmful prompts
- **False refusal rate**: how often the model refuses benign prompts
- **Response quality**: general capability (MT-Bench / AlpacaEval style)
- **Jailbreak robustness**: adversarial prompts (DAN, role-play, prefix injection)

---

## Loading a Trained Adapter

All adapters are LoRA adapters built on `meta-llama/Llama-3.2-1B` (base model).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer  = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Load any adapter — replace with baseline / with_refusals / weighted / dpo
model = PeftModel.from_pretrained(base_model, "outputs/dpo")
```

---

## Key Findings (so far)

- Starting from a **base model with no safety alignment** (Llama-3.2-1B) provides a clean baseline where harmful compliance is expected, making safety gains from SFT/DPO clearly measurable.
- **485 valid DPO preference pairs** were extracted from cases where the base model complied with harmful prompts while the teacher refused.
- The DPO adapter (β=0.1, train loss 0.072) was trained for ~55 min on Kaggle T4.
- Full evaluation results pending.
