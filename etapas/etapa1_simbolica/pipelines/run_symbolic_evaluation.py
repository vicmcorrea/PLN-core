"""Etapa 1 symbolic evaluation pipeline.

Examples:
    # Run the production stack on the Kaggle tweets test split
    uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py

    # Run the seed baseline on the hand-curated sample
    uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py \\
        analyzer=seed dataset=sample

    # Run the active analyzer on a 500-tweet subset of the Kaggle corpus
    uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py \\
        analyzer=oplexicon_regex \\
        dataset=kaggle_tweets dataset.kwargs.max_examples=500
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "etapas" / "etapa1_simbolica" / "configs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from pln_core.eval.runner import run_evaluation, save_report_json  # noqa: E402


def _print_report(report) -> None:
    metrics = report.metrics
    print()
    print("=" * 72)
    print(f"analyzer: {report.analyzer}")
    print(f"dataset : {report.dataset}  ({metrics.total} examples)")
    print(f"runtime : {report.elapsed_seconds:.2f}s")
    print("-" * 72)
    print(f"accuracy : {metrics.accuracy:.4f}  ({metrics.correct}/{metrics.total})")
    print(f"macro F1 : {metrics.macro_f1:.4f}")
    print("per-class F1:")
    for label, value in metrics.per_class_f1.items():
        print(f"  {label:8s}  F1={value:.4f}  support={metrics.support.get(label, 0)}")
    print()
    print("classification report:")
    print(metrics.classification_report)
    print("confusion matrix (rows=true, cols=pred):")
    header = " " * 10 + "  ".join(f"{label:>9}" for label in ("positive", "negative", "neutral"))
    print(header)
    for true_label, counts in metrics.confusion_matrix_rows():
        cells = "  ".join(f"{count:>9d}" for count in counts)
        print(f"{true_label:>10}  {cells}")
    print("=" * 72)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="default")
def main(cfg: DictConfig) -> None:
    print("Resolved configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    dataset_kwargs = OmegaConf.to_container(cfg.dataset.kwargs, resolve=True) or {}
    analyzer_kwargs = OmegaConf.to_container(cfg.analyzer.kwargs, resolve=True) or {}

    report = run_evaluation(
        analyzer_name=cfg.analyzer.name,
        dataset_name=cfg.dataset.name,
        dataset_kwargs=dataset_kwargs,
        analyzer_kwargs=analyzer_kwargs,
    )

    _print_report(report)

    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    destination = (
        output_dir
        / "runs"
        / str(cfg.run_id)
        / cfg.dataset.name
        / cfg.analyzer.name
        / "report.json"
    )
    save_report_json(
        report,
        destination=destination,
        include_predictions=bool(cfg.save_predictions),
    )
    print(f"\nSaved JSON report to: {destination.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
