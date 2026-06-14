from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core.app_models import (  # noqa: E402
    HUGGINGFACE_APP_MODEL_NAME,
    AppModelInfo,
    HuggingFaceSentimentAnalyzer,
    choose_default_model_id,
    default_comparison_model_ids,
    discover_app_models,
    predict_sentiment,
)


class FakeSentimentPipeline:
    def __call__(self, texts: list[str]) -> list[list[dict[str, float | str]]]:
        return [
            [
                {"label": "Very Negative", "score": 0.2},
                {"label": "Negative", "score": 0.3},
                {"label": "Neutral", "score": 0.1},
                {"label": "Positive", "score": 0.35},
                {"label": "Very Positive", "score": 0.05},
            ]
            for _ in texts
        ]


class AppModelTests(unittest.TestCase):
    def test_choose_default_prefers_non_symbolic_mode(self) -> None:
        models = (
            AppModelInfo(
                id="symbolic:raw",
                display_name="Symbolic",
                family="symbolic",
                model_name="oplexicon_regex",
                text_treatment="raw",
                description="",
            ),
            AppModelInfo(
                id="external:tabularisai_multilingual_sentiment",
                display_name="Non-symbolic",
                family="external",
                model_name=HUGGINGFACE_APP_MODEL_NAME,
                text_treatment="raw",
                description="",
            ),
        )

        self.assertEqual(
            choose_default_model_id(models),
            "external:tabularisai_multilingual_sentiment",
        )

    def test_default_comparison_uses_two_public_modes(self) -> None:
        models = (
            AppModelInfo(
                id="symbolic:raw",
                display_name="Symbolic",
                family="symbolic",
                model_name="oplexicon_regex",
                text_treatment="raw",
                description="",
            ),
            AppModelInfo(
                id="external:tabularisai_multilingual_sentiment",
                display_name="Non-symbolic",
                family="external",
                model_name=HUGGINGFACE_APP_MODEL_NAME,
                text_treatment="raw",
                description="",
            ),
        )

        self.assertEqual(
            default_comparison_model_ids(models),
            (
                "external:tabularisai_multilingual_sentiment",
                "symbolic:raw",
            ),
        )

    def test_discover_app_models_lists_only_public_modes(self) -> None:
        models = discover_app_models(PROJECT_ROOT)

        self.assertEqual(
            [(model.display_name, model.family) for model in models],
            [
                ("Non-symbolic", "external"),
                ("Symbolic", "symbolic"),
            ],
        )
        self.assertTrue(all(model.can_predict for model in models))

    def test_huggingface_adapter_maps_five_way_sentiment_to_three_labels(self) -> None:
        model_info = AppModelInfo(
            id="external:tabularisai_multilingual_sentiment",
            display_name="Non-symbolic",
            family="external",
            model_name=HUGGINGFACE_APP_MODEL_NAME,
            text_treatment="raw",
            description="",
        )
        analyzer = HuggingFaceSentimentAnalyzer("fake/model", FakeSentimentPipeline())

        prediction = predict_sentiment(model_info, analyzer, "texto de teste")

        self.assertEqual(prediction.label, "negative")
        self.assertAlmostEqual(prediction.class_scores["negative"], 0.5)
        self.assertAlmostEqual(prediction.class_scores["neutral"], 0.1)
        self.assertAlmostEqual(prediction.class_scores["positive"], 0.4)


if __name__ == "__main__":
    unittest.main()
