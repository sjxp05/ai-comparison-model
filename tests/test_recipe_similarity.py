"""입력 변환, 평가 요청, CSV 출력 검증"""

import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

from src.evaluate_pairs import main
from src.openai_client import (
    RecipeSimilarityEvaluator,
    SimilarityAssessment,
    SimilarityResult,
    build_input,
)
from src.recipe_data import Recipe, load_recipes, parse_steps, read_pairs


def assessment():
    return SimilarityAssessment(
        technique_similarity=0.8,
        sequence_similarity=0.6,
        final_form_similarity=0.9,
        explanation="Similar baking technique and crisp form; different stage order.",
    )


def response(**overrides):
    fields = dict(
        id="resp_test",
        model="test-model",
        status="completed",
        incomplete_details=None,
        output=[],
        output_parsed=assessment(),
    )
    return SimpleNamespace(**(fields | overrides))


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class RecipeTests(unittest.TestCase):
    def test_serialized_lists_and_join(self):
        for steps in (
            "['Mix the dough.', 'Bake until crisp.']",
            '["Mix the dough.", "Bake until crisp."]',
            ["Mix the dough.", "Bake until crisp."],
        ):
            with self.subTest(steps=steps):
                recipe = Recipe(title="Crisp bake", steps=steps)
                self.assertEqual(recipe.steps, ["Mix the dough.", "Bake until crisp."])
                self.assertEqual(
                    recipe.to_prompt(),
                    {
                        "title": "Crisp bake",
                        "steps": "Mix the dough. Bake until crisp.",
                    },
                )

    def test_plain_text_steps(self):
        self.assertEqual(parse_steps("Mix.\nBake."), ["Mix.", "Bake."])

    def test_invalid_steps(self):
        for steps in (
            "",
            "[]",
            "['Unclosed'",
            "[1]",
            [],
            [""],
            [None],
            {"step": "Bake"},
        ):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                parse_steps(steps)

    def test_no_recipe_code_execution(self):
        with self.assertRaises(ValueError):
            parse_steps("[__import__('os').getcwd()]")

    def test_missing_title(self):
        with self.assertRaises(ValidationError):
            Recipe(title="  ", steps=["Bake."])


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.responses.parse.return_value = response()
        self.evaluator = RecipeSimilarityEvaluator(
            model="test-model", client=self.client
        )
        self.a = Recipe(title="Bake A", steps=["Mix.", "Shape.", "Bake."])
        self.b = Recipe(title="Bake B", steps=["Mix.", "Shape.", "Bake until crisp."])

    def test_request_and_weighted_score(self):
        result = self.evaluator.evaluate(self.a, self.b)
        self.assertAlmostEqual(result.similarity_score, 0.77)
        self.assertEqual(result.model_dump()["similarity_score"], 0.77)
        request = self.client.responses.parse.call_args.kwargs
        self.assertEqual(request["model"], "test-model")
        self.assertIs(request["text_format"], SimilarityAssessment)
        self.assertFalse(request["store"])
        self.assertEqual(
            json.loads(request["input"][1]["content"]),
            {
                "recipe_a": {"title": "Bake A", "steps": "Mix. Shape. Bake."},
                "recipe_b": {
                    "title": "Bake B",
                    "steps": "Mix. Shape. Bake until crisp.",
                },
            },
        )

    def test_score_convenience_method(self):
        self.assertEqual(
            self.evaluator.score("A", "['Mix.', 'Bake.']", "B", ["Bake."]), 0.77
        )

    def test_score_range_and_finiteness(self):
        for value in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                SimilarityAssessment(
                    **(assessment().model_dump() | {"technique_similarity": value})
                )
        for value in (0, 1):
            result = SimilarityResult(
                technique_similarity=value,
                sequence_similarity=value,
                final_form_similarity=value,
                explanation="Equivalent endpoints.",
                model="test-model",
                response_id="resp_test",
            )
            self.assertEqual(result.similarity_score, value)

    def test_refusal_incomplete_and_missing_output(self):
        refusal = SimpleNamespace(type="refusal", refusal="Unavailable")
        outputs = [
            (
                response(output=[SimpleNamespace(type="message", content=[refusal])]),
                "refused",
            ),
            (
                response(status="incomplete", incomplete_details="token limit"),
                "not completed",
            ),
            (response(output_parsed=None), "no parsed assessment"),
        ]
        for output, message in outputs:
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError, message
            ):
                self.client.responses.parse.return_value = output
                self.evaluator.evaluate(self.a, self.b)


class CsvTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pairs_path = self.root / "pairs.csv"
        self.recipes_path = self.root / "recipes.csv"
        self.output_path = self.root / "scores.csv"
        self.rows = [
            {
                "a_id": "001",
                "b_id": "002",
                "a_name": "A",
                "b_name": "B",
                "a_steps": "['Mix.', 'Bake.']",
                "b_steps": "['Shape.', 'Steam.']",
                "similarity": "4",
                "a_ingredients": "SECRET INGREDIENTS",
                "a_tags": "SECRET TAGS",
            }
        ]
        write_csv(self.pairs_path, list(self.rows[0]), self.rows)

    def test_inline_fields_without_label_or_ingredient_leakage(self):
        pair = next(read_pairs(self.pairs_path))
        payload = build_input(pair.recipe_a, pair.recipe_b)
        self.assertEqual(pair.a_id, "001")
        self.assertEqual(pair.label_similarity, "4")
        self.assertNotIn("SECRET", payload)
        self.assertNotIn("similarity", payload)
        self.assertEqual(json.loads(payload)["recipe_a"]["steps"], "Mix. Bake.")

    def test_id_only_recipe_lookup(self):
        write_csv(
            self.recipes_path,
            ["id", "name", "steps"],
            [
                {"id": "001", "name": "First", "steps": "['Mix.', 'Bake.']"},
                {"id": "002", "name": "Second", "steps": "['Steam.']"},
            ],
        )
        write_csv(self.pairs_path, ["a_id", "b_id"], [{"a_id": "001", "b_id": "002"}])
        pair = next(read_pairs(self.pairs_path, self.recipes_path))
        self.assertEqual(pair.recipe_a.title, "First")
        self.assertEqual(pair.recipe_b.to_prompt()["steps"], "Steam.")

    def test_unknown_id_and_missing_columns(self):
        write_csv(self.recipes_path, ["id", "name", "steps"], [])
        write_csv(self.pairs_path, ["a_id", "b_id"], [{"a_id": "001", "b_id": "002"}])
        with self.assertRaisesRegex(ValueError, "not found"):
            list(read_pairs(self.pairs_path, self.recipes_path))
        write_csv(self.pairs_path, ["wrong_column"], [])
        with self.assertRaisesRegex(ValueError, "requires a_id"):
            list(read_pairs(self.pairs_path))

    def test_duplicate_recipe_id(self):
        row = {"id": "1", "name": "A", "steps": "['Bake.']"}
        write_csv(self.recipes_path, list(row), [row, row])
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            load_recipes(self.recipes_path)

    def test_dry_run_without_client_or_output(self):
        stdout = io.StringIO()
        with patch("src.evaluate_pairs.RecipeSimilarityEvaluator") as evaluator:
            with contextlib.redirect_stdout(stdout):
                code = main(
                    ["--pairs", str(self.pairs_path), "--dry-run", "--limit", "1"]
                )
            evaluator.assert_not_called()
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(stdout.getvalue())["input"]["recipe_a"]["steps"], "Mix. Bake."
        )

    def test_csv_output_and_overwrite_protection(self):
        result = SimilarityResult(
            **assessment().model_dump(), model="test-model", response_id="resp_test"
        )
        arguments = [
            "--pairs",
            str(self.pairs_path),
            "--output",
            str(self.output_path),
            "--limit",
            "1",
        ]
        with patch("src.evaluate_pairs.RecipeSimilarityEvaluator") as evaluator:
            evaluator.return_value.evaluate.return_value = result
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            with self.output_path.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["label_similarity"], "4")
            self.assertEqual(rows[0]["similarity_score"], "0.77")
            self.assertEqual(rows[0]["a_id"], "001")
            self.assertNotIn("a_ingredients", rows[0])
            with self.assertRaises(FileExistsError):
                main(arguments)
            self.assertEqual(evaluator.call_count, 1)

    def test_api_failure_preserves_completed_rows_and_restart_offset(self):
        write_csv(self.pairs_path, list(self.rows[0]), self.rows * 3)
        result = SimilarityResult(
            **assessment().model_dump(), model="test-model", response_id="resp_test"
        )
        with patch("src.evaluate_pairs.RecipeSimilarityEvaluator") as evaluator:
            evaluator.return_value.evaluate.side_effect = [
                result,
                RuntimeError("API failure"),
            ]
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "--offset 2"):
                    main(
                        [
                            "--pairs",
                            str(self.pairs_path),
                            "--output",
                            str(self.output_path),
                            "--offset",
                            "1",
                        ]
                    )
        with self.output_path.open(encoding="utf-8-sig", newline="") as source:
            self.assertEqual(len(list(csv.DictReader(source))), 1)


if __name__ == "__main__":
    unittest.main()
