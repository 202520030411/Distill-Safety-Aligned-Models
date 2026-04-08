import argparse

from common import (
    DEFAULT_DATASET,
    build_output_path,
    generate_responses,
    is_refusal,
    load_jsonl,
    load_model_and_tokenizer,
    sample_records,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate unsafe compliance on harmful prompts."
    )
    parser.add_argument("--model", required=True, help="teacher | baseline | with_refusals | weighted | dpo")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="JSONL dataset with harmful prompts")
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
        category="harmful",
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

    judged = []
    unsafe = 0
    for record, response in zip(records, responses):
        refusal = is_refusal(response)
        unsafe += int(not refusal)
        judged.append(
            {
                "prompt": record["prompt"],
                "response": response,
                "is_refusal": refusal,
                "unsafe_compliance": not refusal,
            }
        )

    total = len(judged)
    result = {
        "metric": "unsafe_compliance_rate",
        "judge": "refusal_heuristic",
        "model": args.model,
        "dataset": args.dataset,
        "num_examples": total,
        "num_unsafe_compliance": unsafe,
        "unsafe_compliance_rate": (unsafe / total) if total else None,
        "samples": judged[:5],
    }
    output_path = args.output or build_output_path("safety", args.model, None)
    save_json(output_path, result)
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print(
        f"[safety] model={result['model']} "
        f"unsafe_compliance_rate={result['unsafe_compliance_rate']:.4f} "
        f"examples={result['num_examples']}"
    )


if __name__ == "__main__":
    main()
