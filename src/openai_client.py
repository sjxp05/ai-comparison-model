"""OpenAI 기반 레시피 유사도 평가"""

from __future__ import annotations

import json
from typing import Annotated

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.config import settings
from src.contribution_data import ContributionRecipe, normalize_name
from src.prompts import CONTRIBUTION_SYSTEM_PROMPT, SIMILARITY_SYSTEM_PROMPT
from src.recipe_data import Recipe

UnitScore = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class SimilarityAssessment(BaseModel):
    """모델 응답 형식과 점수 범위(0~1) 검증"""

    model_config = ConfigDict(extra="forbid")

    technique_similarity: UnitScore
    sequence_similarity: UnitScore
    final_form_similarity: UnitScore
    explanation: str = Field(min_length=1)


class SimilarityResult(SimilarityAssessment):
    model: str
    response_id: str

    @computed_field
    @property
    def similarity_score(self) -> float:
        return round(
            0.40 * self.technique_similarity
            + 0.30 * self.sequence_similarity
            + 0.30 * self.final_form_similarity,
            6,
        )


def _parsed_output(response):
    """미완료·거절·파싱 실패 응답을 오류로 처리하고 파싱 결과를 반환 (0점 대체 없음)"""
    if response.status != "completed":
        raise RuntimeError(
            f"OpenAI response {response.id} was not completed "
            f"(status={response.status}, details={response.incomplete_details})."
        )
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "refusal":
                    raise RuntimeError(
                        f"OpenAI refused the evaluation: {content.refusal}"
                    )
    if response.output_parsed is None:
        raise RuntimeError(f"OpenAI response {response.id} has no parsed assessment.")
    return response.output_parsed


def _make_client(client: OpenAI | None) -> OpenAI:
    if client is not None:
        return client
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in the environment or project .env file.")
    return OpenAI(api_key=api_key, timeout=90.0, max_retries=3)


def build_input(recipe_a: Recipe, recipe_b: Recipe) -> str:
    """제목 및 공백으로 결합한 조리법으로 입력 생성"""
    return json.dumps(
        {"recipe_a": recipe_a.to_prompt(), "recipe_b": recipe_b.to_prompt()},
        ensure_ascii=False,
    )


class RecipeSimilarityEvaluator:
    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model or settings.OPENAI_MODEL
        if not self.model:
            raise ValueError("The OpenAI model must not be blank.")
        self.client = _make_client(client)

    def evaluate(self, recipe_a: Recipe, recipe_b: Recipe) -> SimilarityResult:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": SIMILARITY_SYSTEM_PROMPT},
                {"role": "user", "content": build_input(recipe_a, recipe_b)},
            ],
            text_format=SimilarityAssessment,
            store=False,
        )
        assessment = SimilarityAssessment.model_validate(_parsed_output(response))
        return SimilarityResult(
            **assessment.model_dump(), model=response.model, response_id=response.id
        )

    def score(
        self,
        title_a: str,
        steps_a: list[str] | str,
        title_b: str,
        steps_b: list[str] | str,
    ) -> float:
        """단일 쌍의 0~1 유사도 반환"""
        return self.evaluate(
            Recipe(title=title_a, steps=steps_a),
            Recipe(title=title_b, steps=steps_b),
        ).similarity_score


class IngredientScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ingredient: str = Field(min_length=1)
    contribution: UnitScore


class ContributionAssessment(BaseModel):
    """레시피 하나의 재료별 기여도 응답 형식과 점수 범위(0~1) 검증"""

    model_config = ConfigDict(extra="forbid")

    scores: list[IngredientScore]
    explanation: str = Field(min_length=1)


class ContributionResult(BaseModel):
    contributions: dict[str, float]  # 평가 대상 재료 이름(입력 표기) -> 기여도
    explanation: str
    model: str
    response_id: str


def build_contribution_input(recipe: ContributionRecipe) -> str:
    """제목, 원재료 목록, 마스크를 복원한 조리법, 평가 대상 재료로 입력 생성 (정답 기여도 제외)"""
    return json.dumps(recipe.to_prompt(), ensure_ascii=False)


def match_targets(
    assessment: ContributionAssessment, targets: list[str]
) -> dict[str, float]:
    """응답 재료 이름을 대상 재료와 맞춘다. 대소문자·앞뒤 공백만 허용하고 누락·중복·추가는 오류"""
    expected = {normalize_name(name): name for name in targets}
    contributions: dict[str, float] = {}
    for score in assessment.scores:
        key = normalize_name(score.ingredient)
        if key not in expected:
            raise ValueError(f"unexpected ingredient in response: {score.ingredient!r}")
        if expected[key] in contributions:
            raise ValueError(f"duplicate ingredient in response: {score.ingredient!r}")
        contributions[expected[key]] = score.contribution
    missing = [name for name in targets if name not in contributions]
    if missing:
        raise ValueError(f"response is missing ingredients: {missing}")
    return {name: contributions[name] for name in targets}


class IngredientContributionEvaluator:
    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
        attempts: int = 3,
    ) -> None:
        self.model = model or settings.OPENAI_MODEL
        if not self.model:
            raise ValueError("The OpenAI model must not be blank.")
        if attempts < 1:
            raise ValueError("attempts must be positive")
        self.attempts = attempts
        self.client = _make_client(client)

    def evaluate(self, recipe: ContributionRecipe) -> ContributionResult:
        """재료 이름이 대상과 맞지 않는 응답은 attempts번까지 다시 요청"""
        errors = []
        for _ in range(self.attempts):
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": CONTRIBUTION_SYSTEM_PROMPT},
                    {"role": "user", "content": build_contribution_input(recipe)},
                ],
                text_format=ContributionAssessment,
                store=False,
            )
            assessment = ContributionAssessment.model_validate(_parsed_output(response))
            try:
                contributions = match_targets(assessment, recipe.targets)
            except ValueError as exc:
                errors.append(f"{response.id}: {exc}")
                continue
            return ContributionResult(
                contributions=contributions,
                explanation=assessment.explanation,
                model=response.model,
                response_id=response.id,
            )
        raise RuntimeError(
            f"Recipe {recipe.recipe_id}: ingredient names did not match after "
            f"{self.attempts} attempts ({'; '.join(errors)})."
        )
