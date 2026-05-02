from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core.pipeline import SymbolicSentimentAnalyzer
from pln_core.tokenizers import tokenize_spacy_pt_lemmas


class SymbolicSentimentAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = SymbolicSentimentAnalyzer()

    def test_positive_text_is_classified_as_positive(self) -> None:
        result = self.analyzer.analyze("Eu amei o filme, foi muito bom!")
        self.assertEqual(result.label, "positive")
        self.assertGreater(result.score, 0)

    def test_negation_flips_polarity(self) -> None:
        result = self.analyzer.analyze("Nao foi bom.")
        self.assertEqual(result.label, "negative")
        self.assertLess(result.score, 0)

    def test_lemmatized_verbs_can_match_lexicon_entries(self) -> None:
        analyzer = SymbolicSentimentAnalyzer(
            lexicon={"gostar": 1.0},
            tokenizer=tokenize_spacy_pt_lemmas,
        )
        result = analyzer.analyze("Eu gostei do filme.")
        self.assertEqual(result.label, "positive")
        self.assertEqual(result.matched_terms[0].token, "gostar")

    def test_negation_works_with_lemmatized_tokens(self) -> None:
        analyzer = SymbolicSentimentAnalyzer(
            lexicon={"gostar": 1.0},
            tokenizer=tokenize_spacy_pt_lemmas,
        )
        result = analyzer.analyze("Nao gostei do filme.")
        self.assertEqual(result.label, "negative")
        self.assertLess(result.score, 0)

    def test_contrast_gives_more_weight_to_final_clause(self) -> None:
        result = self.analyzer.analyze("O começo foi ruim, mas o final foi otimo.")
        self.assertEqual(result.label, "positive")
        self.assertGreater(result.score, 0)

    def test_neutral_text_is_classified_as_neutral(self) -> None:
        result = self.analyzer.analyze("O arquivo tem quatro paginas e duas tabelas.")
        self.assertEqual(result.label, "neutral")
        self.assertEqual(result.score, 0)


if __name__ == "__main__":
    unittest.main()
