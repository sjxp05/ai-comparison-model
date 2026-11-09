"""OpenAI 기반 레시피 유사도 평가"""

from __future__ import annotations

import json
from typing import Annotated

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.config import settings
from src.prompts import SIMILARITY_SYSTEM_PROMPT
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
        if client is None:
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError(
                    "Set OPENAI_API_KEY in the environment or project .env file."
                )
            client = OpenAI(api_key=api_key, timeout=90.0, max_retries=3)
        self.client = client

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
            raise RuntimeError(
                f"OpenAI response {response.id} has no parsed assessment."
            )
        assessment = SimilarityAssessment.model_validate(response.output_parsed)
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
