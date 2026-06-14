from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = PROJECT_ROOT / "streamlit_app.py"


def _load_streamlit_app() -> ModuleType:
    spec = importlib.util.spec_from_file_location("streamlit_app_import_test", STREAMLIT_APP)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load streamlit_app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StreamlitAppImportTests(unittest.TestCase):
    def test_streamlit_entrypoint_imports_without_running_ui(self) -> None:
        module = _load_streamlit_app()

        self.assertTrue(callable(module.main))
        self.assertTrue(callable(module.default_comparison_model_ids))

    def test_fallback_comparison_uses_public_modes(self) -> None:
        module = _load_streamlit_app()
        model_cls = module.AppModelInfo
        models = (
            model_cls(
                id="symbolic:raw",
                display_name="Symbolic",
                family="symbolic",
                model_name="oplexicon_regex",
                text_treatment="raw",
                description="",
            ),
            model_cls(
                id="external:tabularisai_multilingual_sentiment",
                display_name="Non-symbolic",
                family="external",
                model_name="tabularisai_multilingual_sentiment",
                text_treatment="raw",
                description="",
            ),
            model_cls(
                id="classical:clean:tfidf_logreg",
                display_name="TF-IDF + SVM linear",
                family="classical",
                model_name="tfidf_logreg",
                text_treatment="strip_emoticons_urls",
                description="",
                artifact_path=Path("logreg.joblib"),
            ),
        )

        self.assertEqual(
            module._fallback_default_comparison_model_ids(models),
            (
                "external:tabularisai_multilingual_sentiment",
                "symbolic:raw",
            ),
        )


if __name__ == "__main__":
    unittest.main()
