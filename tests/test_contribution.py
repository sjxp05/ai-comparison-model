"""재료 기여도 평가: 데이터 로딩, 마스크 복원, 테스트 분할, 응답 검증, CSV 출력"""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

from src.config import PROJECT_ROOT
from src.contribution_data import (
    ContributionRecipe,
    parse_ingredients,
    read_contribution_recipes,
    read_contributions,
    split_test_pairs,
    unmask_steps,
)
from src.evaluate_contributions import main
from src.openai_client import (
    ContributionAssessment,
    ContributionResult,
    IngredientContributionEvaluator,
    build_contribution_input,
    match_targets,
)

CONTRIBUTION_COLUMNS = ["ingredient", "recipe_id", "contribution", "notes"]
RECIPE_COLUMNS = ["id", "name", "ingredients", "steps"]


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def assessment(*pairs):
    return ContributionAssessment(
        scores=[{"ingredient": name, "contribution": value} for name, value in pairs],
        explanation="Cornmeal defines the muffin; salt is a minor seasoning.",
    )


def response(parsed, response_id="resp_test"):
    return SimpleNamespace(
        id=response_id,
        model="test-model",
        status="completed",
        incomplete_details=None,
        output=[],
        output_parsed=parsed,
    )


def recipe(targets=("cornmeal", "salt")):
    return ContributionRecipe(
        recipe_id="191",
        title="corn muffins",
        ingredients=["cornmeal", "salt", "egg"],
        steps=["whisk cornmeal and salt", "bake"],
        targets=list(targets),
        labels={name: 0.123456 for name in targets},
        test_targets=frozenset({"salt"}),
    )


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contributions = self.root / "contributions.csv"
        self.recipes = self.root / "recipes.csv"
        write_csv(
            self.contributions,
            CONTRIBUTION_COLUMNS,
            [
                {"ingredient": "Cornmeal ", "recipe_id": "191", "contribution": "0.22"},
                {"ingredient": "milk", "recipe_id": "191", "contribution": "0.09"},
                {"ingredient": "rice", "recipe_id": "7", "contribution": "0.5"},
            ],
        )
        write_csv(
            self.recipes,
            RECIPE_COLUMNS,
            [
                {
                    "id": "191",
                    "name": "corn muffins",
                    "ingredients": "['cornmeal', 'skim milk']",
                    "steps": "['whisk [MASK:I0|grains_starches|dough_structure_formation]', "
                    "'add [MASK:I1|dairy|moisture_flavor] and bake']",
                },
                {
                    "id": "7",
                    "name": "rice",
                    "ingredients": "['rice']",
                    "steps": "['boil [MASK:I0|grains_starches|base_component]']",
                },
            ],
        )

    def test_unmask_removes_category_and_role(self):
        steps = unmask_steps(
            ["whisk [MASK:I0|grains_starches|dough_structure_formation] well"],
            ["cornmeal"],
        )
        self.assertEqual(steps, ["whisk cornmeal well"])
        for leaked in ("MASK", "grains_starches", "dough_structure_formation"):
            self.assertNotIn(leaked, steps[0])

    def test_unmask_rejects_bad_tokens(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            unmask_steps(["add [MASK:I3|dairy|moisture]"], ["milk"])
        with self.assertRaisesRegex(ValueError, "malformed"):
            unmask_steps(["add [MASK:milk]"], ["milk"])

    def test_parse_ingredients_without_code_execution(self):
        self.assertEqual(parse_ingredients("[' egg ', 'milk']"), ["egg", "milk"])
        for value in ("[__import__('os').getcwd()]", "'egg'", "[1]", "['']"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_ingredients(value)

    def test_read_contributions_validation(self):
        self.assertEqual(
            read_contributions(self.contributions)[0], ("191", "cornmeal", 0.22)
        )
        bad_rows = [
            {"ingredient": "salt", "recipe_id": "1", "contribution": "1.5"},
            {"ingredient": "salt", "recipe_id": "1", "contribution": "x"},
            {"ingredient": "", "recipe_id": "1", "contribution": "0.1"},
        ]
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                write_csv(self.contributions, CONTRIBUTION_COLUMNS, [row])
                read_contributions(self.contributions)
        write_csv(
            self.contributions,
            CONTRIBUTION_COLUMNS,
            [{"ingredient": "salt", "recipe_id": "1", "contribution": "0.1"}] * 2,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_contributions(self.contributions)

    def test_split_is_deterministic_and_sized(self):
        rows = [(str(i), f"ingredient {i}", 0.1) for i in range(40)]
        first = split_test_pairs(rows, seed=42, test_ratio=0.15)
        self.assertEqual(first, split_test_pairs(rows, seed=42, test_ratio=0.15))
        self.assertEqual(len(first), 6)
        self.assertNotEqual(first, split_test_pairs(rows, seed=7, test_ratio=0.15))
        with self.assertRaises(ValueError):
            split_test_pairs(rows, test_ratio=1)

    def test_recipes_keep_csv_order_targets_and_labels(self):
        recipes = list(read_contribution_recipes(self.contributions, self.recipes))
        self.assertEqual([r.recipe_id for r in recipes], ["191", "7"])
        muffin = recipes[0]
        self.assertEqual(muffin.targets, ["cornmeal", "milk"])
        self.assertEqual(muffin.labels, {"cornmeal": 0.22, "milk": 0.09})
        self.assertEqual(muffin.steps, ["whisk cornmeal", "add skim milk and bake"])
        self.assertEqual(muffin.ingredients, ["cornmeal", "skim milk"])

    def test_min_test_pairs_filter(self):
        rows = read_contributions(self.contributions)
        test_pairs = split_test_pairs(rows, test_ratio=0.5)
        recipes = list(
            read_contribution_recipes(
                self.contributions, self.recipes, test_ratio=0.5, min_test_pairs=1
            )
        )
        self.assertEqual(
            {r.recipe_id for r in recipes}, {recipe_id for recipe_id, _ in test_pairs}
        )
        for item in recipes:
            self.assertEqual(
                item.test_targets,
                {name for rid, name in test_pairs if rid == item.recipe_id},
            )

    def test_unknown_recipe(self):
        write_csv(self.recipes, RECIPE_COLUMNS, [])
        with self.assertRaisesRegex(ValueError, "not found"):
            list(read_contribution_recipes(self.contributions, self.recipes))

    @unittest.skipUnless(
        (PROJECT_ROOT / "data/v3/ingredients_contribution.csv").exists(),
        "v3 data not copied",
    )
    def test_real_split_matches_stage2_size(self):
        rows = read_contributions(PROJECT_ROOT / "data/v3/ingredients_contribution.csv")
        self.assertEqual(len(split_test_pairs(rows)), round(len(rows) * 0.15))


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.evaluator = IngredientContributionEvaluator(
            model="test-model", client=self.client
        )

    def test_request_excludes_labels_and_tags(self):
        self.client.responses.parse.return_value = response(
            assessment(("cornmeal", 0.9), ("salt", 0.1))
        )
        result = self.evaluator.evaluate(recipe())
        self.assertEqual(result.contributions, {"cornmeal": 0.9, "salt": 0.1})
        request = self.client.responses.parse.call_args.kwargs
        self.assertIs(request["text_format"], ContributionAssessment)
        self.assertFalse(request["store"])
        payload = request["input"][1]["content"]
        self.assertNotIn("0.123456", payload)
        self.assertNotIn("test_targets", payload)
        self.assertEqual(
            json.loads(payload),
            {
                "title": "corn muffins",
                "ingredients": ["cornmeal", "salt", "egg"],
                "steps": "whisk cornmeal and salt bake",
                "targets": ["cornmeal", "salt"],
            },
        )

    def test_name_matching_tolerates_case_and_spaces(self):
        matched = match_targets(
            assessment((" Salt", 0.1), ("CORNMEAL ", 0.9)), ["cornmeal", "salt"]
        )
        self.assertEqual(list(matched), ["cornmeal", "salt"])

    def test_name_mismatches_are_errors(self):
        cases = [
            (assessment(("cornmeal", 0.9)), "missing"),
            (assessment(("cornmeal", 0.9), ("salt", 0.1), ("egg", 0.3)), "unexpected"),
            (assessment(("cornmeal", 0.9), ("salt", 0.1), ("salt", 0.2)), "duplicate"),
        ]
        for parsed, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                match_targets(parsed, ["cornmeal", "salt"])

    def test_mismatch_is_retried_then_fails(self):
        bad = response(assessment(("cornmeal", 0.9)), "resp_bad")
        good = response(assessment(("cornmeal", 0.9), ("salt", 0.1)), "resp_good")
        self.client.responses.parse.side_effect = [bad, good]
        self.assertEqual(self.evaluator.evaluate(recipe()).response_id, "resp_good")

        self.client.responses.parse.side_effect = [bad] * 3
        with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
            self.evaluator.evaluate(recipe())

    def test_score_range(self):
        for value in (-0.1, 1.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                assessment(("cornmeal", value))

    def test_refusal_is_not_retried(self):
        refusal = SimpleNamespace(type="refusal", refusal="Unavailable")
        self.client.responses.parse.return_value = SimpleNamespace(
            **(
                vars(response(None))
                | {"output": [SimpleNamespace(type="message", content=[refusal])]}
            )
        )
        with self.assertRaisesRegex(RuntimeError, "refused"):
            self.evaluator.evaluate(recipe())
        self.assertEqual(self.client.responses.parse.call_count, 1)


class CsvTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contributions = self.root / "contributions.csv"
        self.recipes = self.root / "recipes.csv"
        self.output = self.root / "scores.csv"
        write_csv(
            self.contributions,
            CONTRIBUTION_COLUMNS,
            [
                {"ingredient": name, "recipe_id": rid, "contribution": "0.2"}
                for rid in ("1", "2", "3")
                for name in ("rice", "salt")
            ],
        )
        write_csv(
            self.recipes,
            RECIPE_COLUMNS,
            [
                {
                    "id": rid,
                    "name": f"recipe {rid}",
                    "ingredients": "['rice', 'salt']",
                    "steps": "['boil [MASK:I0|grains_starches|base_component]']",
                }
                for rid in ("1", "2", "3")
            ],
        )
        self.base = [
            "--contributions",
            str(self.contributions),
            "--recipes",
            str(self.recipes),
            "--min-test-pairs",
            "0",
        ]

    def result(self):
        return ContributionResult(
            contributions={"rice": 0.8, "salt": 0.1},
            explanation="Rice is the base.",
            model="test-model",
            response_id="resp_test",
        )

    def test_dry_run_without_client_or_output(self):
        stdout = io.StringIO()
        with patch("src.evaluate_contributions.IngredientContributionEvaluator") as ev:
            with contextlib.redirect_stdout(stdout):
                code = main(self.base + ["--dry-run", "--limit", "1"])
            ev.assert_not_called()
        self.assertEqual(code, 0)
        line = json.loads(stdout.getvalue())
        self.assertEqual(line["input"]["steps"], "boil rice")
        self.assertEqual(line["input"]["targets"], ["rice", "salt"])

    def test_csv_output_and_overwrite_protection(self):
        arguments = self.base + ["--output", str(self.output), "--limit", "1"]
        with patch("src.evaluate_contributions.IngredientContributionEvaluator") as ev:
            ev.return_value.evaluate.return_value = self.result()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            with self.output.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([r["ingredient"] for r in rows], ["rice", "salt"])
            self.assertEqual(rows[0]["label_contribution"], "0.2")
            self.assertEqual(rows[0]["ai_contribution"], "0.8")
            self.assertEqual(rows[0]["prompt_version"], "ingredient-contribution-v1")
            self.assertIn(rows[0]["is_test"], {"0", "1"})
            with self.assertRaises(FileExistsError):
                main(arguments)
            self.assertEqual(ev.call_count, 1)

    def test_api_failure_preserves_completed_recipes_and_restart_offset(self):
        with patch("src.evaluate_contributions.IngredientContributionEvaluator") as ev:
            ev.return_value.evaluate.side_effect = [
                self.result(),
                RuntimeError("API failure"),
            ]
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "--offset 2"):
                    main(self.base + ["--output", str(self.output), "--offset", "1"])
        with self.output.open(encoding="utf-8-sig", newline="") as source:
            self.assertEqual(len(list(csv.DictReader(source))), 2)


if __name__ == "__main__":
    unittest.main()
