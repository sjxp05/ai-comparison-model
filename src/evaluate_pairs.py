"""레시피 쌍 CSV 일괄 평가 진입점"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

from src.config import PROJECT_ROOT
from src.openai_client import RecipeSimilarityEvaluator, build_input
from src.prompts import PROMPT_VERSION
from src.recipe_data import read_pairs

OUTPUT_COLUMNS = [
    "a_id",
    "b_id",
    "a_name",
    "b_name",
    "label_similarity",
    "similarity_score",
    "technique_similarity",
    "sequence_similarity",
    "final_form_similarity",
    "explanation",
    "model",
    "response_id",
    "prompt_version",
]


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score English recipe cooking similarity."
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=PROJECT_ROOT / "data/v1/labeled_pairs_masked_v1.csv",
    )
    parser.add_argument(
        "--recipes",
        type=Path,
        default=PROJECT_ROOT / "data/v1/recipes_masked_v1.csv",
        help="Fallback lookup for pairs containing only a_id and b_id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/v1/openai_similarity_v1.csv",
        help="Create a new results CSV; existing files are never overwritten.",
    )
    parser.add_argument("--model", help="Override OPENAI_MODEL from environment/.env.")
    parser.add_argument("--offset", type=nonnegative_int, default=0)
    parser.add_argument(
        "--limit", type=nonnegative_int, help="Maximum pairs; default: all."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print model inputs as JSON lines without API calls or output CSV writes.",
    )
    args = parser.parse_args(argv)
    stop = None if args.limit is None else args.offset + args.limit
    pairs = itertools.islice(read_pairs(args.pairs, args.recipes), args.offset, stop)
    # API 클라이언트·출력 파일 생성 전 첫 쌍 검증
    first_pair = next(pairs, None)
    if first_pair is None:
        print("No pairs selected.", file=sys.stderr)
        return 0
    selected_pairs = itertools.chain([first_pair], pairs)
    if args.dry_run:
        for pair in selected_pairs:
            print(
                json.dumps(
                    {
                        "a_id": pair.a_id,
                        "b_id": pair.b_id,
                        "input": json.loads(build_input(pair.recipe_a, pair.recipe_b)),
                    },
                    ensure_ascii=False,
                )
            )
        return 0

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}. Choose --output.")
    evaluator = RecipeSimilarityEvaluator(model=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        destination.flush()
        for count, pair in enumerate(selected_pairs, start=1):
            try:
                result = evaluator.evaluate(pair.recipe_a, pair.recipe_b)
            except Exception as exc:
                # 완료 행 보존 및 실패 점수 대체 방지
                raise RuntimeError(
                    f"Pair {pair.a_id}/{pair.b_id} failed. Completed rows: {count - 1}. "
                    f"Results are saved in {args.output}. "
                    f"Restart with --offset {args.offset + count - 1} and a new --output. "
                    f"Cause: {exc}"
                ) from exc
            writer.writerow(
                {
                    "a_id": pair.a_id,
                    "b_id": pair.b_id,
                    "a_name": pair.recipe_a.title,
                    "b_name": pair.recipe_b.title,
                    "label_similarity": pair.label_similarity,
                    **result.model_dump(),
                    "prompt_version": PROMPT_VERSION,
                }
            )
            destination.flush()
            print(
                f"[{count}] {pair.a_id}/{pair.b_id}: {result.similarity_score:.4f}",
                file=sys.stderr,
            )
    print(f"Saved {count} evaluations to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
