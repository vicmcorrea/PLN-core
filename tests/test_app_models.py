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
    default_comparison_model_ids,
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
                artifact_path=Path("raw.joblib"),
            ),
            AppModelInfo(
                id="classical:clean:tfidf_logreg",
                display_name="TF-IDF clean",
                family="classical",
                model_name="tfidf_logreg",
                text_treatment="strip_emoticons_urls",
                description="",
                artifact_path=Path("clean.joblib"),
            ),
        )

        self.assertEqual(choose_default_model_id(models), "classical:clean:tfidf_logreg")

    def test_default_comparison_uses_cleaned_symbolic_logreg_and_svm(self) -> None:
        models = (
            AppModelInfo(
                id="symbolic:clean",
                display_name="OpLexicon clean",
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
                artifact_path=Path("raw.joblib"),
            ),
            AppModelInfo(
                id="classical:clean:tfidf_logreg",
                display_name="TF-IDF LogReg clean",
                family="classical",
                model_name="tfidf_logreg",
                text_treatment="strip_emoticons_urls",
                description="",
                artifact_path=Path("logreg.joblib"),
            ),
            AppModelInfo(
                id="classical:clean:tfidf_linear_svm",
                display_name="TF-IDF SVM clean",
                family="classical",
                model_name="tfidf_linear_svm",
                text_treatment="strip_emoticons_urls",
                description="",
                artifact_path=Path("svm.joblib"),
            ),
        )

        self.assertEqual(
            default_comparison_model_ids(models),
            (
                "symbolic:clean",
                "classical:clean:tfidf_logreg",
                "classical:clean:tfidf_linear_svm",
            ),
        )

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
            symbolic_info = next(
                model for model in models if model.id == "symbolic:strip_emoticons_urls"
            )

            self.assertEqual(model_info.text_treatment, "strip_emoticons_urls")
            self.assertEqual(model_info.metrics["macro_f1"], 0.79)
            self.assertIn("TF-IDF de palavras + Regressão Logística", model_info.description)
            self.assertIn("Versão do app: 20260614_app_test", model_info.description)
            self.assertIn("OpLexicon v3.0 + regras simbólicas", symbolic_info.description)
            self.assertEqual(
                [model.model_name for model in models],
                [
                    "oplexicon_regex",
                    "tfidf_logreg",
                    "distilbert_multilingual",
                    "xlm_roberta_base",
                    "albertina_ptbr_100m",
                ],
            )

            model = load_app_model(model_info)
            prediction = predict_sentiment(
                model_info,
                model,
                "Amei esse produto :) http://exemplo.com",
            )

        self.assertEqual(prediction.label, "positive")
        self.assertEqual(prediction.processed_text, "Amei esse produto")
        self.assertGreaterEqual(prediction.class_scores["positive"], 0.0)

    def test_discover_prefers_committed_deploy_artifact_over_local_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_id = "20260614_app_test"
            deploy_dir = project_root / f"data/app_models/etapa2_subsymbolic/{run_id}"
            local_dir = project_root / f"data/models/etapa2_subsymbolic/{run_id}"
            deploy_dir.mkdir(parents=True)
            local_dir.mkdir(parents=True)

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
            for artifact_dir, macro_f1 in ((deploy_dir, 0.81), (local_dir, 0.79)):
                artifact_path = artifact_dir / "tfidf_logreg.joblib"
                joblib.dump(pipeline, artifact_path)
                artifact_path.with_suffix(".metadata.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "family": "classical",
                            "run_id": run_id,
                            "model": "tfidf_logreg",
                            "text_treatment": "strip_emoticons_urls",
                            "metrics": {"accuracy": 0.8, "macro_f1": macro_f1},
                        }
                    ),
                    encoding="utf-8",
                )

            matches = [
                model
                for model in discover_app_models(project_root)
                if model.id == f"classical:{run_id}:tfidf_logreg"
            ]

        self.assertEqual(len(matches), 1)
        self.assertIn("data/app_models", str(matches[0].artifact_path))
        self.assertEqual(matches[0].metrics["macro_f1"], 0.81)

    def test_discover_lists_transformer_benchmarks_as_reference_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            models = discover_app_models(project_root)

        transformer_models = [model for model in models if model.family == "transformer"]
        self.assertEqual(
            [model.model_name for model in transformer_models],
            ["distilbert_multilingual", "xlm_roberta_base", "albertina_ptbr_100m"],
        )
        self.assertTrue(all(not model.can_predict for model in transformer_models))
        self.assertIn("fine-tuned", transformer_models[0].description)
        self.assertAlmostEqual(transformer_models[-1].metrics["macro_f1"], 0.7808)


if __name__ == "__main__":
    unittest.main()
