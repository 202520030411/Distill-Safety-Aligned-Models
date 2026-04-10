import argparse
from pathlib import Path

import eval_jailbreak
import eval_quality
import eval_refusal
import eval_safety
from common import DEFAULT_DATASET, DEFAULT_RESULTS_DIR, available_model_names, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full local evaluation suite for one or more models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["baseline", "with_refusals", "weighted", "dpo", "on_policy_dpo",
                 "dpo_with_refusals", "on_policy_dpo_with_refusals"],
        help=f"Subset of: {', '.join(available_model_names())}",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Held-out JSONL dataset used for safety/refusal/quality by default",
    )
    parser.add_argument(
        "--jailbreak_dataset",
        default=None,
        help="Optional separate dataset for jailbreak evaluation",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional prompt limit per metric")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for per-metric JSON and summary output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for model_name in args.models:
        print(f"== Evaluating {model_name} ==")
        safety_args = argparse.Namespace(
            model=model_name,
            dataset=args.dataset,
            limit=args.limit,
            seed=args.seed,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output=str(output_dir / f"safety_{model_name}.json"),
        )
        refusal_args = argparse.Namespace(
            model=model_name,
            dataset=args.dataset,
            limit=args.limit,
            seed=args.seed,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output=str(output_dir / f"refusal_{model_name}.json"),
        )
        quality_args = argparse.Namespace(
            model=model_name,
            dataset=args.dataset,
            limit=args.limit,
            seed=args.seed,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output=str(output_dir / f"quality_{model_name}.json"),
        )
        jailbreak_args = argparse.Namespace(
            model=model_name,
            dataset=args.jailbreak_dataset or args.dataset,
            limit=args.limit,
            seed=args.seed,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output=str(output_dir / f"jailbreak_{model_name}.json"),
        )

        safety = eval_safety.evaluate(safety_args)
        refusal = eval_refusal.evaluate(refusal_args)
        quality = eval_quality.evaluate(quality_args)
        jailbreak = eval_jailbreak.evaluate(jailbreak_args)

        summary[model_name] = {
            "unsafe_compliance_rate": safety["unsafe_compliance_rate"],
            "false_refusal_rate": refusal["false_refusal_rate"],
            "reference_quality_proxy": quality["avg_reference_token_f1"],
            "jailbreak_unsafe_compliance_rate_macro": jailbreak["unsafe_compliance_rate_macro"],
        }

    save_json(output_dir / "summary.json", summary)
    print(f"Wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
