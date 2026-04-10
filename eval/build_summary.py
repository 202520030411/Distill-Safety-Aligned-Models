import argparse
from pathlib import Path

from common import DEFAULT_RESULTS_DIR, available_model_names, save_json


SUMMARY_FIELDS = {
    "safety": "unsafe_compliance_rate",
    "refusal": "false_refusal_rate",
    "quality": "avg_reference_token_f1",
    "jailbreak": "unsafe_compliance_rate_macro",
}

SUMMARY_KEYS = {
    "safety": "unsafe_compliance_rate",
    "refusal": "false_refusal_rate",
    "quality": "reference_quality_proxy",
    "jailbreak": "jailbreak_unsafe_compliance_rate_macro",
}

DEFAULT_MODEL_ORDER = [
    "teacher",
    "baseline",
    "with_refusals",
    "weighted",
    "dpo",
    "on_policy_dpo",
    "dpo_with_refusals",
    "on_policy_dpo_with_refusals",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild results/eval/summary.json from existing per-metric JSON files."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Optional subset of: {', '.join(available_model_names())}",
    )
    parser.add_argument(
        "--input_dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing safety/refusal/quality/jailbreak JSON files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path; defaults to <input_dir>/summary.json",
    )
    return parser.parse_args()


def discover_models(input_dir: Path) -> list[str]:
    models = set()
    for path in input_dir.glob("safety_*.json"):
        models.add(path.stem[len("safety_"):])
    ordered = [model for model in DEFAULT_MODEL_ORDER if model in models]
    extras = sorted(model for model in models if model not in DEFAULT_MODEL_ORDER)
    return ordered + extras


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        import json

        return json.load(handle)


def build_summary(input_dir: Path, models: list[str]) -> dict:
    summary = {}
    for model in models:
        per_model = {}
        for metric_name, field_name in SUMMARY_FIELDS.items():
            metric_path = input_dir / f"{metric_name}_{model}.json"
            if not metric_path.exists():
                raise FileNotFoundError(f"Missing metric file: {metric_path}")
            payload = load_json(metric_path)
            per_model[SUMMARY_KEYS[metric_name]] = payload[field_name]
        summary[model] = per_model
    return summary


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    models = args.models or discover_models(input_dir)
    summary = build_summary(input_dir, models)
    output_path = Path(args.output) if args.output else input_dir / "summary.json"
    save_json(output_path, summary)
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
