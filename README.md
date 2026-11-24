# Recipe similarity evaluator

Labeled Pairs의 레시피 제목과 조리 과정만 OpenAI API 요청으로 보내 유사도 평가

steps는 CSV의 Python/JSON 리스트 문자열을 리스트로 변환한 뒤 `" ".join(steps)`로 합쳐 전송합니다. 직접 입력한 문자열 리스트도 동일하게 처리합니다.

평가 기준과 설명은 영어입니다. 조리 기법, 의미 있는 조리 순서, 완성 형태를 각각 0~1로 평가하고 다음 가중 평균을 코드에서 계산합니다.

```text
similarity_score = 0.40 * technique_similarity
                 + 0.30 * sequence_similarity
                 + 0.30 * final_form_similarity
```

재료 일치 여부, 재료 종류, 맛, 요리명 단어의 중복은 평가 근거로 사용하지 않습니다. 마스킹 토큰의 번호도 레시피 간 일치 근거로 쓰지 않고 한 레시피 내에서 재료를 구분하는 용도로만 사용됩니다. 제목은 보조 문맥이고 steps를 우선합니다.

가중치는 이 구현의 기본 설정이며 학습된 값은 아닙니다.

점수는 모델의 상대적 판단으로, 통계적으로 보정된 확률이나 검증된 정답이 아닙니다.

## 사용법

프로젝트 루트에서 다음 명령어로 venv 설정 및 의존성 설치

```powershell
python -m venv .venv
.venv/Scripts/Activate
pip install -r requirements.txt
```

Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

루트의 기존 `.env`에 다음 값을 설정

- `OPENAI_API_KEY`
- `OPENAI_MODEL`: 비어 있으면 `gpt-4.1-mini`를 사용, `--model` 옵션으로 변경 가능

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4.1-mini
```

API 호출 없이 입력 형태만 확인하는 명령어

```powershell
python -m src.evaluate_pairs --dry-run --limit 1
```

처음 5쌍 평가 (실제 평가에는 OpenAI API 사용료 발생)

```powershell
python -m src.evaluate_pairs --limit 5
```

전체 평가 (`--limit`을 빼고 실행)

- 입력 기본값: `data/v1/labeled_pairs_masked_v1.csv`
- 출력 기본값: `outputs/v1/openai_similarity_v1.csv`, 같은 이름으로 이미 파일이 존재할 경우 덮어쓰지 않음

```powershell
python -m src.evaluate_pairs --output data/v1/outputs/scores_all.csv
```

테스트 (실제 API 호출하지 않음)

```powershell
python -m unittest discover -s tests -v
```

## 입출력 상세

Labeled Pairs CSV의 `a_name/a_steps/b_name/b_steps`가 있으면 바로 사용합니다.
`a_id/b_id`만 있거나 제목/steps가 빠져 있으면 `--recipes` CSV의 `id/name/steps`에서 찾습니다. 기본 조회 파일은 `data/v1/recipes_masked_v1.csv`입니다.

재료, 태그, description, 정답 라벨은 모델에 전송하지 않습니다.
원본 `similarity` 라벨은 변환 없이 결과의 `label_similarity`에 보존하고, 모델 점수는 별도의 `similarity_score` 열에 저장합니다.\
결과에는 세부 점수, 영어 설명, 실제 응답 모델명, 응답 ID, 프롬프트 버전도 저장합니다.
완료한 행은 매번 파일에 기록합니다.

요청 실패 시 실행을 멈추고 재시작할 `--offset`(원본 CSV에서 건너뛸 쌍의 개수)을 안내합니다.\
재시작할 때는 새 `--output` 경로를 지정합니다.

일시적인 API 오류 발생 시 최대 3회 재시도하며, 거절/미완료/잘못된 응답을 0점으로 대체하지 않습니다.

단일 쌍은 함수로 평가할 수 있습니다. `score()`는 숫자만, `evaluate()`는 상세 결과를 반환합니다.

```python
from src.openai_client import RecipeSimilarityEvaluator
from src.recipe_data import Recipe

evaluator = RecipeSimilarityEvaluator()
score = evaluator.score(
    title_a="Baked vegetable patties",
    steps_a=["Mix the mixture.", "Shape into flat patties.", "Bake until crisp."],
    title_b="Baked bean patties",
    steps_b=["Mix the mixture.", "Form flat patties.", "Bake until crisp."],
)
print(score)  # 0~1 범위 실수

result = evaluator.evaluate(
    Recipe(title="Soup", steps=["Simmer until soft.", "Puree until smooth."]),
    Recipe(title="Fritters", steps=["Shape into pieces.", "Fry until crisp."]),
)
print(result.model_dump())
```

API 형식은 [OpenAI 공식 Structured Outputs 문서](https://developers.openai.com/api/docs/guides/structured-outputs)를 따릅니다.

---

# Ingredient contribution evaluator

레시피 하나의 재료별 기여도(0~1)를 OpenAI API로 평가해, recipe-recommendation 2단계 모델의 예측 기여도와 비교하는 데 사용합니다.

- 기여도 정의: 재료를 빼거나 중립적인 재료로 바꿨을 때 완성된 요리가 얼마나 달라지는가 (요리 정체성 > 구조 > 핵심 맛 순으로 고려)
- 요청 단위: 레시피 1개 = 요청 1개 (같은 레시피 안의 재료를 함께 비교)
- 프롬프트 버전: `ingredient-contribution-v1` (`src/prompts.py`)

## 입력

| 파일 | 내용 |
|---|---|
| `data/v3/ingredients_contribution.csv` | 레시피–재료 정답 기여도 (recipe-recommendation `data/stage2/`에서 복사) |
| `data/v3/recipes_masked_v3.csv` | 레시피 제목·원재료 목록·마스킹된 조리법 (recipe-recommendation `data/stage1/v3/`에서 복사) |

모델에 보내는 값: 제목, 원재료 목록, 조리법, 평가 대상 재료(기여도 CSV의 정규화된 이름)

- 조리법의 `[MASK:I{k}|카테고리|역할]`은 원재료 목록의 k번째 이름으로 복원하고, 카테고리·역할 문자열은 보내지 않습니다
- 정답 기여도와 테스트 분할 여부는 보내지 않습니다

## 평가 대상

recipe-recommendation의 2단계 테스트 분할(`seed=42`, `test_ratio=0.15`)을 같은 방식으로 재현해, 테스트 재료가 `--min-test-pairs`개(기본 2) 이상인 레시피만 평가합니다.

| `--min-test-pairs` | 레시피 | 평가 재료 | 그중 테스트 재료 |
|---|---|---|---|
| 0 (전체) | 2,779 | 19,603 | 2,940 |
| 1 | 1,767 | 14,385 | 2,940 |
| 2 (기본) | 801 | 7,519 | 1,974 |
| 3 | 280 | 2,939 | 932 |

## 실행

```bash
# API 호출 없이 입력 확인
python -m src.evaluate_contributions --dry-run --limit 2

# 파일럿 (사용료 발생)
python -m src.evaluate_contributions --limit 20 --output outputs/v3/contribution_pilot.csv

# 전체
python -m src.evaluate_contributions --output outputs/v3/contribution_scores.csv
```

옵션: `--contributions`, `--recipes`, `--output`, `--model`, `--min-test-pairs`, `--seed`, `--test-ratio`, `--offset`(필터 후 건너뛸 레시피 수), `--limit`, `--dry-run`

- 응답의 재료 이름이 평가 대상과 정확히 맞지 않으면(누락·중복·추가) 최대 3번까지 다시 요청하고, 그래도 맞지 않으면 실패로 멈춥니다. 대소문자·앞뒤 공백 차이는 허용합니다
- 실패 시 완료한 레시피는 보존되고 재시작할 `--offset`을 안내합니다 (유사도 평가와 동일)

## 출력 (`outputs/v3/contribution_scores.csv`, 재료 1개 = 1행)

| 열 | 내용 |
|---|---|
| `recipe_id`, `recipe_name`, `ingredient` | 레시피와 재료 (재료는 정규화된 이름) |
| `label_contribution` | 정답 기여도 |
| `ai_contribution` | 모델 평가 기여도 0~1 |
| `is_test` | 2단계 테스트 분할 재료이면 1 (모델 예측과 비교할 대상) |
| `explanation` | 레시피 단위 영어 설명 (같은 레시피 행에 반복) |
| `model`, `response_id`, `prompt_version` | 재현용 정보 |
