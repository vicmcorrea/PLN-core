from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core.lexicon import load_oplexicon
from pln_core.tokenizers import tokenize_custom, tokenize_spacy_pt, tokenize_spacy_pt_lemmas


class LexiconTests(unittest.TestCase):
    def test_oplexicon_loader_averages_duplicate_terms(self) -> None:
        with TemporaryDirectory() as temp_dir:
            lexicon_path = Path(temp_dir) / "lexico_v3.0.txt"
            lexicon_path.write_text(
                "bom,adj,1,A\n"
                "bom,adj,1,M\n"
                "ruim,adj,-1,A\n"
                ":-),emot,1,A\n",
                encoding="utf-8",
            )

            lexicon = load_oplexicon(lexicon_path)

        self.assertEqual(lexicon["bom"], 1.0)
        self.assertEqual(lexicon["ruim"], -1.0)
        self.assertEqual(lexicon[":-)"], 1.0)

    def test_tokenizer_keeps_basic_emoticons(self) -> None:
        tokens = tokenize_custom("isso foi muito bom :)")
        self.assertIn("bom", tokens)
        self.assertIn(":)", tokens)

    def test_spacy_tokenizer_keeps_portuguese_style_tokens(self) -> None:
        tokens = tokenize_spacy_pt("@ana isso foi muuuuito bom :) #cinema")
        self.assertIn("muuito", tokens)
        self.assertIn("bom", tokens)
        self.assertIn("cinema", tokens)

    def test_spacy_lemmatizer_returns_portuguese_lemmas(self) -> None:
        tokens = tokenize_spacy_pt_lemmas("Eu gostei do filme e amei o final.")
        self.assertIn("gostar", tokens)
        self.assertIn("amar", tokens)

    def test_spacy_lemmatizer_filters_auxiliary_lemmas(self) -> None:
        tokens = tokenize_spacy_pt_lemmas("As telas estão confusas.")
        self.assertIn("confuso", tokens)
        self.assertNotIn("estar", tokens)


if __name__ == "__main__":
    unittest.main()
