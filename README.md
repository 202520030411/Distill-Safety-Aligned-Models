# Distill Safety-Aligned Models

**Research question:** Can safety alignment be efficiently transferred from a large aligned teacher to a small student model by combining output distillation with value-based reward distillation — without requiring human feedback or repeating the teacher's full RLHF pipeline?

We distill a safety-aligned teacher model (Llama-3.1-8B-Instruct) into a smaller student model (Llama-3.2-1B base) using synthetic data, then evaluate how well safety is retained across different training strategies.

See `milestone_report.pdf` for the full project plan.

---

## Research Problem

Large language models encode safety not just as output behavior, but as an internal alignment structure built through RLHF. Standard knowledge distillation transfers only output behavior — the student learns *what* the teacher says, but not *why* it's safe. This leads to the **functional-ethical gap**: students distilled from safety-aligned teachers can comply with up to 86% of adversarial prompts even when trained entirely on benign data ([arXiv:2512.09403](https://arxiv.org/abs/2512.09403)).

---

## Proposed Method

A two-stage pipeline that preserves distillation efficiency while injecting alignment signal:

**Stage 1 — Output Distillation (SFT)**
Train the student on teacher-generated responses via supervised fine-tuning. Three variants:
- `baseline`: benign-only, no safety signal
- `with_refusals`: adds teacher refusal responses
- `weighted`: refusal examples upweighted 3×

**Stage 2 — Value-Based Reward Distillation (TVKD-DPO)**
Apply **Teacher Value-based Knowledge Distillation** (TVKD, NeurIPS 2025) on top of the SFT baseline. TVKD augments the standard DPO loss with a soft reward derived from the teacher's value function:

```
ψ(s, a) = V_φ(s') − V_φ(s)
```

where `V_φ(s) = β · log Σ_a exp(Q_φ(s,a) / β)` is the soft value function of the DPO-trained teacher. This is added to the DPO reward via potential-based reward shaping, which provably preserves the optimal policy while giving the student token-level soft guidance from the teacher's internal alignment signal — not just its output behavior.

**Why TVKD over reward_distill (Zhang et al., 2025)?**
- Reward_distill requires PPO, hundreds of teacher rollouts per query, and extractable ground-truth answers for pseudo-label voting — none of which apply to safety refusal tasks
- TVKD only needs teacher logits precomputed once on the existing DPO pairs, then trains exactly like DPO — computationally feasible on a single Kaggle T4
- TVKD was validated on exactly our model pair: Llama-3.1-8B → Llama-3.2-1B

---

## Repository Structure

```
.
├── data/                          # Dataset pipeline
│   ├── generate_data.py           # Run teacher on benign + harmful prompts
│   ├── prepare_dataset.py         # Filter, format, and split into train/val
│   ├── test_logic.py              # Smoke tests (no GPU required)
│   ├── raw_teacher_outputs.jsonl  # 2000 teacher responses (1500 benign, 500 harmful)
│   ├── train.jsonl                # 1885 SFT training examples
│   ├── val.jsonl                  # 100 SFT validation examples
│   └── dpo_pairs.jsonl            # 485 DPO preference pairs (chosen=refusal, rejected=base compliance)
│
├── outputs/                       # Trained LoRA adapters
│   ├── baseline/                  # SFT benign-only adapter
│   ├── with_refusals/             # SFT with refusals adapter
│   ├── weighted/                  # SFT weighted adapter
│   └── dpo/                       # DPO adapter (on top of SFT baseline)
│
├── eval/                          # Evaluation scripts (in progress)
├── results/                       # Training summaries
│   └── sft_results.md             # SFT setup and variant descriptions
├── kaggle_generate_data.ipynb     # Kaggle notebook: teacher inference (data generation)
├── kaggle_train.ipynb             # Kaggle notebook: all SFT variants + DPO training
├── milestone_report.tex/.pdf      # Project milestone report
└── requirements.txt
```

---

## Pipeline Overview

### Step 1 — Data Generation

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

### Step 2 — SFT Training

Three supervised fine-tuning variants on the student model (`meta-llama/Llama-3.2-1B` base) using QLoRA:

| Variant | Data | Description |
|---|---|---|
| `baseline` | Benign only | Teaches instruction-following; no safety signal |
| `with_refusals` | Benign + teacher refusals | Adds teacher refusal responses as training examples |
| `weighted` | Benign + refusals (refusals upweighted 3×) | Extra loss weight on refusal examples |

LoRA config: `r=16`, `alpha=32`, `dropout=0.05`, all attention + MLP projection layers.  
Optimizer: `paged_adamw_8bit` (QLoRA 4-bit). Run on Kaggle T4 via `kaggle_train.ipynb`.

---

### Step 3 — DPO / TVKD Training

Direct Preference Optimization on top of the SFT baseline to further reinforce safety.

```
Base model (Llama-3.2-1B, no safety) → generates rejected responses
Teacher refusals → chosen responses
→ 485 preference pairs in data/dpo_pairs.jsonl
→ DPO/TVKD trains outputs/dpo/ adapter
```

**Key design decisions:**
- Rejected responses come from `Llama-3.2-1B` (base, no Instruct) — guaranteed to comply with harmful prompts
- SFT LoRA is merged into base weights before DPO, so reference logits are stable
- Current adapter: standard DPO (β=0.1, lr=5e-5, 2 epochs), train loss 0.072, ~55 min on Kaggle T4
- Planned upgrade: **TVKD** — augments DPO with the teacher's soft value function `ψ(s,a) = V_φ(s') − V_φ(s)`, providing token-level alignment signal without additional rollouts

Kaggle notebook: `kaggle_train.ipynb` (DPO section runs after SFT)

---

### Step 4 — Evaluation

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
