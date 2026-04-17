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
    resolve_interactive_text,
    resolve_lexicon_source_choice,
    resolve_requested_text,
)
from pln_core.lexicon import OPLEXICON_LEXICON_SOURCE, SEED_LEXICON_SOURCE
from pln_core.samples import SAMPLE_TEXTS


class CliHelperTests(unittest.TestCase):
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

    def test_command_line_sample_resolves_without_menu(self) -> None:
        args = Namespace(text=None, sample="neutral")
        self.assertEqual(resolve_requested_text(args), SAMPLE_TEXTS["neutral"])


if __name__ == "__main__":
    unittest.main()
