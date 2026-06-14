from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = (
    PROJECT_ROOT
    / "etapas"
    / "etapa2_subsimbolica"
    / "pipelines"
    / "run_external_pretrained_benchmark.py"
)


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("external_pretrained_benchmark", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load run_external_pretrained_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExternalPretrainedBenchmarkTests(unittest.TestCase):
    def test_maps_five_way_model_scores_to_project_three_class_schema(self) -> None:
        module = _load_pipeline_module()
        scores = module._scores_from_output(
            [
                {"label": "Very Negative", "score": 0.12},
                {"label": "Negative", "score": 0.18},
                {"label": "Neutral", "score": 0.10},
                {"label": "Positive", "score": 0.34},
                {"label": "Very Positive", "score": 0.26},
            ]
        )

        self.assertAlmostEqual(scores["negative"], 0.30)
        self.assertAlmostEqual(scores["neutral"], 0.10)
        self.assertAlmostEqual(scores["positive"], 0.60)
        self.assertEqual(module._best_label(scores), "positive")

    def test_accepts_generic_label_ids_from_transformers_configs(self) -> None:
        module = _load_pipeline_module()
        scores = module._scores_from_output(
            [
                {"label": "LABEL_0", "score": 0.10},
                {"label": "LABEL_1", "score": 0.20},
                {"label": "LABEL_2", "score": 0.15},
                {"label": "LABEL_3", "score": 0.25},
                {"label": "LABEL_4", "score": 0.30},
            ]
        )

        self.assertAlmostEqual(scores["negative"], 0.30)
        self.assertAlmostEqual(scores["neutral"], 0.15)
        self.assertAlmostEqual(scores["positive"], 0.55)


if __name__ == "__main__":
    unittest.main()
