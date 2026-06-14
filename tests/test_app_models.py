from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core.app_models import (  # noqa: E402
    AppModelInfo,
    choose_default_model_id,
    discover_app_models,
    load_app_model,
    predict_sentiment,
)


class AppModelTests(unittest.TestCase):
    def test_choose_default_prefers_cleaned_tfidf_logreg(self) -> None:
        models = (
            AppModelInfo(
                id="symbolic:strip",
                display_name="OpLexicon",
                family="symbolic",
                model_name="oplexicon_regex",
                text_treatment="strip_emoticons_urls",
                description="",
            ),
            AppModelInfo(
                id="classical:raw:tfidf_logreg",
                display_name="TF-IDF raw",
                family="classical",
                model_name="tfidf_logreg",
                text_treatment="raw",
                description="",
            ),
            AppModelInfo(
                id="classical:clean:tfidf_logreg",
                display_name="TF-IDF clean",
                family="classical",
                model_name="tfidf_logreg",
                text_treatment="strip_emoticons_urls",
                description="",
            ),
        )

        self.assertEqual(choose_default_model_id(models), "classical:clean:tfidf_logreg")

    def test_discover_classical_metadata_and_predicts_with_treatment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            artifact_dir = project_root / "data/models/etapa2_subsymbolic/20260614_app_test"
            artifact_dir.mkdir(parents=True)
            artifact_path = artifact_dir / "tfidf_logreg.joblib"

            pipeline = Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("classifier", DummyClassifier(strategy="constant", constant="positive")),
                ]
            )
            pipeline.fit(
                ["amei o produto", "odiei o produto", "produto entregue"],
                ["positive", "negative", "neutral"],
            )
            joblib.dump(pipeline, artifact_path)

            metadata = {
                "schema_version": 1,
                "family": "classical",
                "run_id": "20260614_app_test",
                "model": "tfidf_logreg",
                "text_treatment": "strip_emoticons_urls",
                "metrics": {
                    "accuracy": 0.8,
                    "macro_f1": 0.79,
                },
            }
            artifact_path.with_suffix(".metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            models = discover_app_models(project_root)
            model_info = next(model for model in models if model.id.endswith(":tfidf_logreg"))

            self.assertEqual(model_info.text_treatment, "strip_emoticons_urls")
            self.assertEqual(model_info.metrics["macro_f1"], 0.79)

            model = load_app_model(model_info)
            prediction = predict_sentiment(
                model_info,
                model,
                "Amei esse produto :) http://exemplo.com",
            )

        self.assertEqual(prediction.label, "positive")
        self.assertEqual(prediction.processed_text, "Amei esse produto")
        self.assertGreaterEqual(prediction.class_scores["positive"], 0.0)


if __name__ == "__main__":
    unittest.main()
