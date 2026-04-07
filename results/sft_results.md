# SFT Training Results

All three variants were trained on Kaggle using `kaggle_train.ipynb`.

## Setup

- **Student (base) model**: `meta-llama/Llama-3.2-1B` (base, no instruction tuning)
- **Training method**: QLoRA (4-bit quantization + LoRA)
- **LoRA config**: `r=16`, `alpha=32`, `dropout=0.05`, all attention + MLP projection layers
- **Optimizer**: `paged_adamw_8bit`
- **Hardware**: Kaggle T4 GPU

## Dataset

| Split | Total | Benign | Harmful Refusal |
|---|---|---|---|
| Train | 1885 | 1425 | 460 |
| Val | 100 | 75 | 25 |

## Variants

| Variant | Data | Description |
|---|---|---|
| `baseline` | Benign only | Teaches instruction-following without any safety signal |
| `with_refusals` | Benign + teacher refusals | Includes teacher refusal responses as positive examples |
| `weighted` | Benign + refusals (3× upweight) | Refusal examples receive higher loss weight to emphasize safety |

All adapter checkpoints are saved under `outputs/<variant>/`.
