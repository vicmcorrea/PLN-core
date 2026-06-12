"""Tests for the evaluation harness (registries, metrics, runner)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core.eval.analyzers import ANALYZER_REGISTRY  # noqa: E402
from pln_core.eval.datasets import (  # noqa: E402
    DATASET_REGISTRY,
    EvalDataset,
    EvalExample,
    create_dataset,
)
from pln_core.eval.metrics import compute_metrics  # noqa: E402
from pln_core.eval.runner import run_evaluation  # noqa: E402


class RegistriesTests(unittest.TestCase):
    def test_dataset_registry_exposes_sample_and_kaggle_tweets(self) -> None:
        names = DATASET_REGISTRY.names()
        self.assertIn("sample", names)
        self.assertIn("kaggle_tweets", names)

    def test_analyzer_registry_exposes_seed_and_oplexicon(self) -> None:
        names = ANALYZER_REGISTRY.names()
        self.assertIn("seed", names)
        self.assertIn("oplexicon", names)

    def test_unknown_dataset_raises(self) -> None:
        with self.assertRaises(KeyError):
            create_dataset("does-not-exist")


class SampleDatasetTests(unittest.TestCase):
    def test_sample_dataset_returns_twenty_examples(self) -> None:
        dataset = create_dataset("sample")
        self.assertIsInstance(dataset, EvalDataset)
        self.assertEqual(len(dataset), 20)
        for example in dataset.examples:
            self.assertIsInstance(example, EvalExample)
            self.assertIn(example.label, {"positive", "negative", "neutral"})


class KaggleTweetsDatasetTests(unittest.TestCase):
    def test_kaggle_loader_normalizes_three_class_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tweets.csv"
            path.write_text(
                "tweet_text,sentiment\n"
                "Amei esse filme,1\n"
                "Odiei esse app,0\n"
                "Recebi o pedido hoje,2\n",
                encoding="utf-8",
            )

            dataset = create_dataset("kaggle_tweets", file_path=str(path))

        self.assertEqual(dataset.name, "kaggle_tweets[test]")
        self.assertEqual(
            [example.label for example in dataset.examples],
            ["positive", "negative", "neutral"],
        )
        self.assertEqual(dataset.examples[0].text, "Amei esse filme")

    def test_kaggle_loader_applies_text_treatment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tweets.csv"
            path.write_text(
                "tweet_text,sentiment\n"
                "\"Amei :) veja http://exemplo.com\",1\n",
                encoding="utf-8",
            )

            dataset = create_dataset(
                "kaggle_tweets",
                file_path=str(path),
                text_treatment="strip_emoticons_urls",
            )

        self.assertEqual(dataset.name, "kaggle_tweets[test|strip_emoticons_urls]")
        self.assertEqual(dataset.examples[0].text, "Amei veja")


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        metrics = compute_metrics(
            expected=["positive", "negative", "neutral"],
            predicted=["positive", "negative", "neutral"],
        )
        self.assertEqual(metrics.total, 3)
        self.assertEqual(metrics.correct, 3)
        self.assertAlmostEqual(metrics.accuracy, 1.0)
        self.assertAlmostEqual(metrics.macro_f1, 1.0)

    def test_confusion_matrix_counts(self) -> None:
        metrics = compute_metrics(
            expected=["positive", "positive", "negative", "neutral"],
            predicted=["positive", "negative", "negative", "positive"],
        )
        self.assertEqual(metrics.confusion[("positive", "positive")], 1)
        self.assertEqual(metrics.confusion[("positive", "negative")], 1)
        self.assertEqual(metrics.confusion[("negative", "negative")], 1)
        self.assertEqual(metrics.confusion[("neutral", "positive")], 1)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics(expected=["positive"], predicted=["positive", "negative"])


class RunnerTests(unittest.TestCase):
    def test_seed_analyzer_on_sample_dataset(self) -> None:
        report = run_evaluation(analyzer_name="seed", dataset_name="sample")
        self.assertEqual(report.analyzer, "seed")
        self.assertEqual(report.metrics.total, 20)
        self.assertGreaterEqual(report.metrics.accuracy, 0.9)
        self.assertEqual(len(report.predictions), 20)


if __name__ == "__main__":
    unittest.main()
