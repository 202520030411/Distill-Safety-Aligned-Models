import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer


class SupervisedDataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)

        input_ids = []
        attention_mask = []
        labels = []
        sample_weights = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)
            sample_weights.append(float(feature.get("sample_weight", 1.0)))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float),
        }


class SampleWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())

        token_mask = (shift_labels != -100).float()
        per_example_token_count = token_mask.sum(dim=1).clamp_min(1.0)
        per_example_loss = (token_losses * token_mask).sum(dim=1) / per_example_token_count

        if sample_weight is None:
            sample_weight = torch.ones_like(per_example_loss)
        else:
            sample_weight = sample_weight.to(per_example_loss.device)

        loss = (per_example_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
        return (loss, outputs) if return_outputs else loss


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, ensure_ascii=False, indent=2)
