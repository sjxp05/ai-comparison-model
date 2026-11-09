"""마스킹 CSV에서 제목(name), 조리법(steps) 추출"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field, field_validator


def parse_steps(value: list[str] | str) -> list[str]:
    """Python·JSON 리스트 문자열 또는 일반 조리법을 리스트로 변환"""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(
                        "steps must contain a valid list of strings"
                    ) from exc
        else:
            value = value.splitlines()
    if not isinstance(value, list) or not value:
        raise ValueError("steps must be a non-empty list of strings")
    if any(not isinstance(step, str) or not step.strip() for step in value):
        raise ValueError("each step must be a non-empty string")
    return [step.strip() for step in value]


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1)
    steps: list[str]

    @field_validator("steps", mode="before")
    @classmethod
    def normalize_steps(cls, value: list[str] | str) -> list[str]:
        return parse_steps(value)

    def to_prompt(self) -> dict[str, str]:
        return {"title": self.title, "steps": " ".join(self.steps)}


@dataclass(frozen=True)
class RecipePair:
    a_id: str
    b_id: str
    recipe_a: Recipe
    recipe_b: Recipe
    label_similarity: str = ""


def load_recipes(path: str | Path) -> dict[str, Recipe]:
    recipes: dict[str, Recipe] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"id", "name", "steps"}.issubset(reader.fieldnames or []):
            raise ValueError("Recipes CSV requires id, name, and steps columns.")
        for row_number, row in enumerate(reader, start=2):
            recipe_id = (row["id"] or "").strip()
            if not recipe_id or recipe_id in recipes:
                raise ValueError(
                    f"Recipes CSV row {row_number}: missing or duplicate id."
                )
            try:
                recipes[recipe_id] = Recipe(title=row["name"], steps=row["steps"])
            except ValueError as exc:
                raise ValueError(f"Recipes CSV row {row_number}: {exc}") from exc
    return recipes


def read_pairs(
    pairs_path: str | Path, recipes_path: str | Path | None = None
) -> Iterator[RecipePair]:
    """Labeled pairs CSV 로딩 및 누락 필드의 레시피 ID 조회"""
    recipes: dict[str, Recipe] | None = None
    with Path(pairs_path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"a_id", "b_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Pairs CSV requires a_id and b_id columns.")
        for row_number, row in enumerate(reader, start=2):
            pair_recipes = []
            for side in ("a", "b"):
                recipe_id = (row[f"{side}_id"] or "").strip()
                if not recipe_id:
                    raise ValueError(f"Pairs CSV row {row_number}: missing {side}_id.")
                title = row.get(f"{side}_name")
                steps = row.get(f"{side}_steps")
                try:
                    if title and title.strip() and steps and steps.strip():
                        recipe = Recipe(title=title, steps=steps)
                    else:
                        if recipes is None:
                            if recipes_path is None:
                                raise ValueError(
                                    "Missing title/steps; provide a recipes CSV."
                                )
                            recipes = load_recipes(recipes_path)
                        if recipe_id not in recipes:
                            raise ValueError(f"Recipe id {recipe_id!r} was not found.")
                        recipe = recipes[recipe_id]
                except ValueError as exc:
                    raise ValueError(
                        f"Pairs CSV row {row_number}, recipe {side}: {exc}"
                    ) from exc
                pair_recipes.append(recipe)
            yield RecipePair(
                a_id=row["a_id"].strip(),
                b_id=row["b_id"].strip(),
                recipe_a=pair_recipes[0],
                recipe_b=pair_recipes[1],
                label_similarity=row.get("similarity", "") or "",
            )
