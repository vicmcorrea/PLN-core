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
    resolve_lexicon_source_choice,
    resolve_requested_text,
    resolve_start_mode_choice,
    resolve_tokenizer_source_choice,
)
from pln_core.lexicon import OPLEXICON_LEXICON_SOURCE, SEED_LEXICON_SOURCE
from pln_core.pipeline import AnalysisResult, MatchDetail
from pln_core.samples import ANALYZE_MODE, COMPARE_MODE, SAMPLE_TEXTS
from pln_core.tokenizers import CUSTOM_TOKENIZER_SOURCE, SPACY_PT_TOKENIZER_SOURCE


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

    def test_seed_source_choice_uses_seed_dictionary(self) -> None:
        self.assertEqual(resolve_lexicon_source_choice("1"), SEED_LEXICON_SOURCE)

    def test_oplexicon_source_choice_uses_oplexicon(self) -> None:
        self.assertEqual(resolve_lexicon_source_choice("2"), OPLEXICON_LEXICON_SOURCE)

    def test_custom_tokenizer_choice_uses_custom_tokenizer(self) -> None:
        self.assertEqual(resolve_tokenizer_source_choice("1"), CUSTOM_TOKENIZER_SOURCE)

    def test_spacy_tokenizer_choice_uses_spacy_tokenizer(self) -> None:
        self.assertEqual(
            resolve_tokenizer_source_choice("2"),
            SPACY_PT_TOKENIZER_SOURCE,
        )

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
                    "variants": [
                        {
                            "lexicon_source": SEED_LEXICON_SOURCE,
                            "tokenizer_source": CUSTOM_TOKENIZER_SOURCE,
                            "result": result,
                        }
                    ],
                }
            ],
            as_json=False,
        )

        self.assertIn("quick summary:", rendered)
        self.assertIn("details:", rendered)
        self.assertIn("configuration 1", rendered)
        self.assertIn("rules=intensifier", rendered)
        self.assertIn("results table:", rendered)
        self.assertIn("example", rendered)
        self.assertIn("score", rendered)


if __name__ == "__main__":
    unittest.main()
