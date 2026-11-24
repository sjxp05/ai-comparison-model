"""레시피별 재료 기여도 일괄 평가 진입점"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

from src.config import PROJECT_ROOT
from src.contribution_data import (
    DEFAULT_SEED,
    DEFAULT_TEST_RATIO,
    read_contribution_recipes,
)
from src.evaluate_pairs import nonnegative_int
from src.openai_client import IngredientContributionEvaluator, build_contribution_input
from src.prompts import CONTRIBUTION_PROMPT_VERSION

OUTPUT_COLUMNS = [
    "recipe_id",
    "recipe_name",
    "ingredient",
    "label_contribution",
    "ai_contribution",
    "is_test",
    "explanation",
    "model",
    "response_id",
    "prompt_version",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score each ingredient's contribution to its recipe."
    )
    parser.add_argument(
        "--contributions",
        type=Path,
        default=PROJECT_ROOT / "data/v3/ingredients_contribution.csv",
    )
    parser.add_argument(
        "--recipes",
        type=Path,
        default=PROJECT_ROOT / "data/v3/recipes_masked_v3.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/v3/contribution_scores.csv",
        help="Create a new results CSV; existing files are never overwritten.",
    )
    parser.add_argument("--model", help="Override OPENAI_MODEL from environment/.env.")
    parser.add_argument(
        "--min-test-pairs",
        type=nonnegative_int,
        default=2,
        help="Only recipes with at least this many stage-2 test ingredients (0: all).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument(
        "--offset",
        type=nonnegative_int,
        default=0,
        help="Recipes to skip after the --min-test-pairs filter.",
    )
    parser.add_argument(
        "--limit", type=nonnegative_int, help="Maximum recipes; default: all."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print model inputs as JSON lines without API calls or output CSV writes.",
    )
    args = parser.parse_args(argv)
    stop = None if args.limit is None else args.offset + args.limit
    recipes = itertools.islice(
        read_contribution_recipes(
            args.contributions,
            args.recipes,
            seed=args.seed,
            test_ratio=args.test_ratio,
            min_test_pairs=args.min_test_pairs,
        ),
        args.offset,
        stop,
    )
    # API 클라이언트·출력 파일 생성 전 첫 레시피 검증
    first_recipe = next(recipes, None)
    if first_recipe is None:
        print("No recipes selected.", file=sys.stderr)
        return 0
    selected_recipes = itertools.chain([first_recipe], recipes)
    if args.dry_run:
        for recipe in selected_recipes:
            print(
                json.dumps(
                    {
                        "recipe_id": recipe.recipe_id,
                        "test_targets": sorted(recipe.test_targets),
                        "input": json.loads(build_contribution_input(recipe)),
                    },
                    ensure_ascii=False,
                )
            )
        return 0

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}. Choose --output.")
    evaluator = IngredientContributionEvaluator(model=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        destination.flush()
        for count, recipe in enumerate(selected_recipes, start=1):
            try:
                result = evaluator.evaluate(recipe)
            except Exception as exc:
                # 완료한 레시피 보존 및 실패 점수 대체 방지
                raise RuntimeError(
                    f"Recipe {recipe.recipe_id} failed. Completed recipes: {count - 1}. "
                    f"Results are saved in {args.output}. "
                    f"Restart with --offset {args.offset + count - 1} and a new --output. "
                    f"Cause: {exc}"
                ) from exc
            # 레시피 한 개의 행을 검증 후 한꺼번에 기록
            writer.writerows(
                {
                    "recipe_id": recipe.recipe_id,
                    "recipe_name": recipe.title,
                    "ingredient": name,
                    "label_contribution": recipe.labels[name],
                    "ai_contribution": score,
                    "is_test": int(name in recipe.test_targets),
                    "explanation": result.explanation,
                    "model": result.model,
                    "response_id": result.response_id,
                    "prompt_version": CONTRIBUTION_PROMPT_VERSION,
                }
                for name, score in result.contributions.items()
            )
            destination.flush()
            print(
                f"[{count}] {recipe.recipe_id}: {len(result.contributions)} ingredients",
                file=sys.stderr,
            )
    print(f"Saved {count} recipes to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
