"""재료 기여도 평가용 레시피 로딩: 마스크 복원, 평가 대상 재료, 2단계 테스트 분할 재현"""

from __future__ import annotations

import ast
import csv
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.recipe_data import parse_steps

# [MASK:I0|grains_starches|dough_structure_formation] 형식. 카테고리·역할은 버린다
MASK_PATTERN = re.compile(r"\[MASK:I(\d+)\|[^\]]*\]")

# recipe-recommendation의 training.data.split_contributions 기본값과 같아야 한다
DEFAULT_SEED = 42
DEFAULT_TEST_RATIO = 0.15

ContributionRow = tuple[str, str, float]  # (레시피 ID, 정규화한 재료 이름, 기여도)


def normalize_name(name: str) -> str:
    return name.strip().lower()


@dataclass(frozen=True)
class ContributionRecipe:
    recipe_id: str
    title: str
    ingredients: list[str]  # 레시피 원재료 목록 (원래 표기)
    steps: list[str]  # 마스크를 재료 이름으로 복원한 조리 단계
    targets: list[str]  # 평가할 재료 (기여도 CSV의 정규화 이름, CSV 순서)
    labels: dict[str, float]  # 재료 -> 정답 기여도 (요청에는 보내지 않음)
    test_targets: frozenset[str]  # 2단계 테스트 분할에 속한 재료

    def to_prompt(self) -> dict[str, object]:
        return {
            "title": self.title,
            "ingredients": self.ingredients,
            "steps": " ".join(self.steps),
            "targets": self.targets,
        }


def read_contributions(path: str | Path) -> list[ContributionRow]:
    """기여도 CSV를 파일 순서대로 읽는다 (training.data.read_contributions와 같은 검사)"""
    rows: list[ContributionRow] = []
    seen: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"ingredient", "recipe_id", "contribution"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "Contributions CSV requires ingredient, recipe_id, and contribution columns."
            )
        for row_number, row in enumerate(reader, start=2):
            recipe_id = (row["recipe_id"] or "").strip()
            name = normalize_name(row["ingredient"] or "")
            if not recipe_id or not name:
                raise ValueError(
                    f"Contributions CSV row {row_number}: missing recipe_id or ingredient."
                )
            try:
                weight = float(row["contribution"])
            except ValueError as exc:
                raise ValueError(
                    f"Contributions CSV row {row_number}: contribution must be a number."
                ) from exc
            if not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError(
                    f"Contributions CSV row {row_number}: contribution must be within 0 and 1."
                )
            if (recipe_id, name) in seen:
                raise ValueError(
                    f"Contributions CSV row {row_number}: duplicate pair {recipe_id}/{name}."
                )
            seen.add((recipe_id, name))
            rows.append((recipe_id, name, weight))
    return rows


def split_test_pairs(
    rows: list[ContributionRow],
    seed: int = DEFAULT_SEED,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> frozenset[tuple[str, str]]:
    """2단계 학습의 테스트 분할 (레시피 ID, 재료) 쌍. split_contributions와 같은 순서로 섞는다"""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_ratio)
    return frozenset((recipe_id, name) for recipe_id, name, _ in shuffled[:n_test])


def parse_ingredients(value: str) -> list[str]:
    """Python 리스트 문자열의 원재료 목록. literal_eval만 사용해 코드를 실행하지 않는다"""
    try:
        parsed = ast.literal_eval(value.strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError("ingredients must be a list of strings") from exc
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) or not item.strip() for item in parsed
    ):
        raise ValueError("ingredients must be a list of non-empty strings")
    return [item.strip() for item in parsed]


def unmask_steps(steps: list[str], ingredients: list[str]) -> list[str]:
    """[MASK:I{k}|카테고리|역할]을 ingredients[k]로 바꾼다. 카테고리·역할 문자열은 남기지 않음"""

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index >= len(ingredients):
            raise ValueError(
                f"mask index I{index} is outside the ingredient list ({len(ingredients)})"
            )
        return ingredients[index]

    restored = [MASK_PATTERN.sub(replace, step) for step in steps]
    if any("[MASK" in step for step in restored):
        raise ValueError("steps contain a malformed mask token")
    return restored


def load_recipe_texts(path: str | Path) -> dict[str, dict[str, str]]:
    csv.field_size_limit(10**9)  # steps 열이 기본 한도보다 길 수 있음
    recipes: dict[str, dict[str, str]] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"id", "name", "ingredients", "steps"}.issubset(reader.fieldnames or []):
            raise ValueError("Recipes CSV requires id, name, ingredients, and steps columns.")
        for row_number, row in enumerate(reader, start=2):
            recipe_id = (row["id"] or "").strip()
            if not recipe_id or recipe_id in recipes:
                raise ValueError(f"Recipes CSV row {row_number}: missing or duplicate id.")
            recipes[recipe_id] = {
                "name": row["name"],
                "ingredients": row["ingredients"],
                "steps": row["steps"],
            }
    return recipes


def read_contribution_recipes(
    contributions_path: str | Path,
    recipes_path: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    test_ratio: float = DEFAULT_TEST_RATIO,
    min_test_pairs: int = 0,
) -> Iterator[ContributionRecipe]:
    """기여도 CSV에 처음 등장한 순서대로 레시피를 반환. 테스트 재료가 min_test_pairs개 미만이면 건너뜀"""
    rows = read_contributions(contributions_path)
    test_pairs = split_test_pairs(rows, seed, test_ratio)
    test_counts = Counter(recipe_id for recipe_id, _ in test_pairs)
    by_recipe: dict[str, list[tuple[str, float]]] = {}
    for recipe_id, name, weight in rows:
        by_recipe.setdefault(recipe_id, []).append((name, weight))

    texts = load_recipe_texts(recipes_path)
    for recipe_id, targets in by_recipe.items():
        if test_counts[recipe_id] < min_test_pairs:
            continue
        text = texts.get(recipe_id)
        if text is None:
            raise ValueError(f"Recipe id {recipe_id!r} was not found in the recipes CSV.")
        try:
            ingredients = parse_ingredients(text["ingredients"])
            steps = unmask_steps(parse_steps(text["steps"]), ingredients)
        except ValueError as exc:
            raise ValueError(f"Recipe {recipe_id}: {exc}") from exc
        title = text["name"].strip()
        if not title:
            raise ValueError(f"Recipe {recipe_id}: missing name.")
        yield ContributionRecipe(
            recipe_id=recipe_id,
            title=title,
            ingredients=ingredients,
            steps=steps,
            targets=[name for name, _ in targets],
            labels=dict(targets),
            test_targets=frozenset(
                name for name, _ in targets if (recipe_id, name) in test_pairs
            ),
        )
