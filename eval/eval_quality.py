import argparse
from statistics import mean

from common import (
    DEFAULT_DATASET,
    build_output_path,
    generate_responses,
    load_jsonl,
    load_model_and_tokenizer,
    sample_records,
    save_json,
    token_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate benign response quality against teacher references."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="teacher | baseline | with_refusals | weighted | dpo | on_policy_dpo",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="JSONL dataset with benign references")
    parser.add_argument("--limit", type=int, default=None, help="Optional prompt limit")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed when --limit is used")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default=None, help="Optional explicit output JSON path")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    records = sample_records(
        load_jsonl(args.dataset),
        category="benign",
        limit=args.limit,
        seed=args.seed,
    )
    prompts = [record["prompt"] for record in records]

    model, tokenizer = load_model_and_tokenizer(args.model)
    responses = generate_responses(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    samples = []
    overlap_scores = []
    length_ratios = []
    for record, response in zip(records, responses):
        reference = record["response"]
        overlap = token_f1(response, reference)
        overlap_scores.append(overlap)
        ref_length = max(len(reference.split()), 1)
        length_ratios.append(len(response.split()) / ref_length)
        samples.append(
            {
                "prompt": record["prompt"],
                "response": response,
                "reference": reference,
                "reference_token_f1": overlap,
            }
        )

    result = {
        "metric": "reference_quality_proxy",
        "judge": "teacher_reference_token_f1",
        "model": args.model,
        "dataset": args.dataset,
        "num_examples": len(samples),
        "avg_reference_token_f1": mean(overlap_scores) if overlap_scores else None,
        "avg_length_ratio_vs_reference": mean(length_ratios) if length_ratios else None,
        "samples": samples[:5],
    }
    output_path = args.output or build_output_path("quality", args.model, None)
    save_json(output_path, result)
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print(
        f"[quality] model={result['model']} "
        f"avg_reference_token_f1={result['avg_reference_token_f1']:.4f} "
        f"examples={result['num_examples']}"
    )


if __name__ == "__main__":
    main()
