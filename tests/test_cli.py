from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core.cli import (
    render_comparison_results,
    resolve_interactive_text,
    resolve_requested_text,
    resolve_start_mode_choice,
)
from pln_core.pipeline import AnalysisResult, MatchDetail
from pln_core.samples import ANALYZE_MODE, ANALYZER_STACK_LABEL, COMPARE_MODE, SAMPLE_TEXTS


class CliHelperTests(unittest.TestCase):
    def test_analyze_mode_choice_uses_analyze_mode(self) -> None:
        self.assertEqual(resolve_start_mode_choice("1"), ANALYZE_MODE)

    def test_compare_mode_choice_uses_compare_mode(self) -> None:
        self.assertEqual(resolve_start_mode_choice("2"), COMPARE_MODE)

    def test_positive_menu_choice_uses_positive_sample(self) -> None:
        self.assertEqual(resolve_interactive_text("2"), SAMPLE_TEXTS["positive"])

    def test_negative_menu_choice_uses_negative_sample(self) -> None:
        self.assertEqual(resolve_interactive_text("3"), SAMPLE_TEXTS["negative"])

    def test_neutral_menu_choice_uses_neutral_sample(self) -> None:
        self.assertEqual(resolve_interactive_text("4"), SAMPLE_TEXTS["neutral"])

    def test_command_line_sample_resolves_without_menu(self) -> None:
        args = Namespace(text=None, sample="neutral")
        self.assertEqual(resolve_requested_text(args), SAMPLE_TEXTS["neutral"])

    def test_comparison_renderer_contains_summary_and_details(self) -> None:
        result = AnalysisResult(
            text="teste",
            normalized_text="teste",
            tokens=("muito", "bom"),
            score=1.92,
            label="positive",
            matched_terms=(
                MatchDetail(
                    token="bom",
                    position=1,
                    base_score=1.2,
                    adjusted_score=1.92,
                    applied_rules=("intensifier",),
                ),
            ),
        )
        rendered = render_comparison_results(
            [
                {
                    "example_name": "demo",
                    "text": "muito bom",
                    "result": result,
                }
            ],
            as_json=False,
        )

        self.assertIn(f"stack: {ANALYZER_STACK_LABEL}", rendered)
        self.assertIn("=== example: demo ===", rendered)
        self.assertIn("rules=intensifier", rendered)
        self.assertIn("results table:", rendered)
        self.assertIn("example", rendered)
        self.assertIn("score", rendered)


if __name__ == "__main__":
    unittest.main()
