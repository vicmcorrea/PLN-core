"""Etapa 2 classical TF-IDF benchmark suite on the shared Kaggle corpus.

The suite trains classical supervised baselines on the official Kaggle
``Train3Classes.csv`` split and evaluates them on ``Test3classes.csv``. Outputs
are timestamped so Etapa 1 and Etapa 2 artifacts never overwrite each other.

Example:
    uv run python etapas/etapa2_subsimbolica/pipelines/run_classical_benchmark_suite.py
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "etapas" / "etapa2_subsimbolica" / "configs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hydra  # noqa: E402
import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from pln_core.eval.datasets.base import VALID_LABELS, EvalDataset  # noqa: E402
from pln_core.eval.datasets.kaggle_tweets import load_kaggle_tweets  # noqa: E402
from pln_core.eval.metrics import EvaluationMetrics, compute_metrics  # noqa: E402

LABELS = tuple(VALID_LABELS)
METRIC_COLOR = "#0072B2"
BASELINE_COLOR = "#D55E00"
CONFUSION_COLOR = "Blues"


@dataclass(frozen=True, slots=True)
class ClassicalBenchmarkReport:
    """Serializable metadata for one classical Etapa 2 run."""

    model: str
    kind: str
    dataset: str
    train_examples: int
    test_examples: int
    elapsed_seconds: float
    vocabulary_size: int
    model_artifact: str
    metrics: EvaluationMetrics
    model_config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "kind": self.kind,
            "dataset": self.dataset,
            "train_examples": self.train_examples,
            "test_examples": self.test_examples,
            "elapsed_seconds": self.elapsed_seconds,
            "vocabulary_size": self.vocabulary_size,
            "model_artifact": self.model_artifact,
            "metrics": _metrics_as_dict(self.metrics),
            "model_config": self.model_config,
        }


def _project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _display_path(path_like: str | Path) -> str:
    path = Path(path_like)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _metrics_as_dict(metrics: EvaluationMetrics) -> dict[str, Any]:
    return {
        "total": metrics.total,
        "correct": metrics.correct,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "per_class_f1": metrics.per_class_f1,
        "support": metrics.support,
        "confusion": {
            f"{true}->{pred}": count for (true, pred), count in metrics.confusion.items()
        },
        "classification_report": metrics.classification_report,
    }


def _download_kaggle_dataset(slug: str, target_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The Kaggle package is required for automatic dataset download. "
            "Install dependencies with `uv sync` or download the dataset manually."
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset {slug} into {_display_path(target_dir)}")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(target_dir), unzip=True, quiet=False)


def _ensure_dataset(dataset_cfg: DictConfig) -> None:
    root_dir = _project_path(str(dataset_cfg.root_dir))
    expected_files = [
        root_dir / str(dataset_cfg.train_file),
        root_dir / str(dataset_cfg.test_file),
    ]
    missing = [path for path in expected_files if not path.exists()]
    if missing and bool(dataset_cfg.get("download_if_missing", False)):
        _download_kaggle_dataset(str(dataset_cfg.slug), root_dir)
        missing = [path for path in expected_files if not path.exists()]
    if missing:
        missing_paths = ", ".join(_display_path(path) for path in missing)
        raise FileNotFoundError(
            "Missing Kaggle split file(s): "
            f"{missing_paths}. Download augustop/portuguese-tweets-for-sentiment-analysis."
        )


def _load_split(dataset_cfg: DictConfig, split: str) -> EvalDataset:
    return load_kaggle_tweets(
        split=split,
        source_dir=str(_project_path(str(dataset_cfg.root_dir))),
        seed=int(dataset_cfg.seed),
    )


def _load_model_config(model_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / "model" / f"{model_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"model config not found: {_display_path(config_path)}")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"model config must be a mapping: {model_name}")
    if config.get("kind") != "classical":
        raise ValueError(f"model '{model_name}' is not a classical model")
    return config


def _import_class(class_path: str) -> type:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _vectorizer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(config)
    if "ngram_range" in kwargs:
        kwargs["ngram_range"] = tuple(kwargs["ngram_range"])
    return kwargs


def _classifier_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(config)
    kwargs.pop("class_path", None)
    return kwargs


def _build_pipeline(model_config: dict[str, Any]) -> Pipeline:
    vectorizer = TfidfVectorizer(**_vectorizer_kwargs(model_config["vectorizer"]))
    classifier_cls = _import_class(str(model_config["classifier"]["class_path"]))
    classifier = classifier_cls(**_classifier_kwargs(model_config["classifier"]))
    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )


def _save_figure(fig: plt.Figure, destination_base: Path) -> list[str]:
    destination_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix in (".png", ".pdf"):
        destination = destination_base.with_suffix(suffix)
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        outputs.append(_display_path(destination))
    plt.close(fig)
    return outputs


def _plot_confusion(
    model_name: str,
    metrics: EvaluationMetrics,
    destination_base: Path,
) -> list[str]:
    matrix = [
        [metrics.confusion.get((true_label, pred_label), 0) for pred_label in LABELS]
        for true_label in LABELS
    ]

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    image = ax.imshow(matrix, cmap=CONFUSION_COLOR)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion matrix: {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold label")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)

    threshold = max(max(row) for row in matrix) / 2 if matrix else 0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=color)
    return _save_figure(fig, destination_base)


def _plot_metric_bars(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    metric: str,
    destination_base: Path,
) -> list[str]:
    plot_rows = [
        {
            "name": str(baseline["name"]),
            "value": float(baseline[metric]),
            "color": BASELINE_COLOR,
        }
    ]
    for row in rows:
        if row.get("status") == "ok":
            plot_rows.append(
                {
                    "name": str(row["model"]),
                    "value": float(row[metric]),
                    "color": METRIC_COLOR,
                }
            )

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [row["name"] for row in plot_rows]
    values = [row["value"] for row in plot_rows]
    colors = [row["color"] for row in plot_rows]
    ax.bar(names, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Etapa 2 benchmark: {metric.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    for index, value in enumerate(values):
        ax.text(index, min(value + 0.02, 0.98), f"{value:.3f}", ha="center", fontsize=8)
    return _save_figure(fig, destination_base)


def _write_predictions(
    model_name: str,
    texts: list[str],
    expected: list[str],
    predicted: list[str],
    run_dir: Path,
) -> tuple[str, str]:
    prediction_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    paired_predictions = zip(texts, expected, predicted, strict=True)
    for index, (text, gold, pred) in enumerate(paired_predictions, start=1):
        row = {
            "index": index,
            "expected": gold,
            "predicted": pred,
            "correct": gold == pred,
            "text": text,
        }
        prediction_rows.append(row)
        if gold != pred:
            error_rows.append(row)

    fieldnames = ["index", "expected", "predicted", "correct", "text"]
    predictions_path = run_dir / "predictions" / f"{model_name}.csv"
    errors_path = run_dir / "cases" / f"{model_name}_errors.csv"
    _write_csv(predictions_path, prediction_rows, fieldnames)
    _write_csv(errors_path, error_rows, fieldnames)
    return _display_path(predictions_path), _display_path(errors_path)


def _run_model(
    model_config: dict[str, Any],
    train_dataset: EvalDataset,
    test_dataset: EvalDataset,
    run_dir: Path,
    model_dir: Path,
    save_models: bool,
    save_predictions: bool,
    make_figures: bool,
) -> tuple[ClassicalBenchmarkReport, str, str]:
    model_name = str(model_config["name"])
    pipeline = _build_pipeline(model_config)

    train_texts = [example.text for example in train_dataset.examples]
    train_labels = [example.label for example in train_dataset.examples]
    test_texts = [example.text for example in test_dataset.examples]
    test_labels = [example.label for example in test_dataset.examples]

    start = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    predictions = list(pipeline.predict(test_texts))
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(test_labels, predictions)
    vocabulary_size = len(pipeline.named_steps["tfidf"].vocabulary_)

    model_artifact = ""
    if save_models:
        artifact_path = model_dir / f"{model_name}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, artifact_path)
        model_artifact = _display_path(artifact_path)

    predictions_csv = ""
    errors_csv = ""
    if save_predictions:
        predictions_csv, errors_csv = _write_predictions(
            model_name=model_name,
            texts=test_texts,
            expected=test_labels,
            predicted=predictions,
            run_dir=run_dir,
        )

    report = ClassicalBenchmarkReport(
        model=model_name,
        kind=str(model_config["kind"]),
        dataset=test_dataset.name,
        train_examples=len(train_dataset.examples),
        test_examples=len(test_dataset.examples),
        elapsed_seconds=elapsed,
        vocabulary_size=vocabulary_size,
        model_artifact=model_artifact,
        metrics=metrics,
        model_config=model_config,
    )
    report_path = run_dir / "reports" / model_name / "report.json"
    _write_json(report_path, report.as_dict())
    if make_figures:
        _plot_confusion(
            model_name=model_name,
            metrics=metrics,
            destination_base=run_dir / "figures" / f"confusion_{model_name}",
        )
    return report, predictions_csv, errors_csv


def _summary_row(
    report: ClassicalBenchmarkReport,
    predictions_csv: str,
    errors_csv: str,
) -> dict[str, Any]:
    return {
        "model": report.model,
        "status": "ok",
        "dataset": report.dataset,
        "train_examples": report.train_examples,
        "test_examples": report.test_examples,
        "accuracy": report.metrics.accuracy,
        "macro_f1": report.metrics.macro_f1,
        "positive_f1": report.metrics.per_class_f1["positive"],
        "negative_f1": report.metrics.per_class_f1["negative"],
        "neutral_f1": report.metrics.per_class_f1["neutral"],
        "vocabulary_size": report.vocabulary_size,
        "elapsed_seconds": report.elapsed_seconds,
        "model_artifact": report.model_artifact,
        "predictions_csv": predictions_csv,
        "errors_csv": errors_csv,
        "error_file": "",
    }


def _failure_row(model_name: str, error_path: Path) -> dict[str, Any]:
    return {
        "model": model_name,
        "status": "failed",
        "dataset": "",
        "train_examples": "",
        "test_examples": "",
        "accuracy": "",
        "macro_f1": "",
        "positive_f1": "",
        "negative_f1": "",
        "neutral_f1": "",
        "vocabulary_size": "",
        "elapsed_seconds": "",
        "model_artifact": "",
        "predictions_csv": "",
        "errors_csv": "",
        "error_file": _display_path(error_path),
    }


def _write_summary(run_dir: Path, rows: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    successful = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    successful.sort(key=lambda row: float(row["macro_f1"]), reverse=True)
    sorted_rows = successful + failed
    for rank, row in enumerate(successful, start=1):
        row["rank"] = rank
    for row in failed:
        row["rank"] = ""

    fieldnames = [
        "rank",
        "model",
        "status",
        "dataset",
        "train_examples",
        "test_examples",
        "accuracy",
        "macro_f1",
        "positive_f1",
        "negative_f1",
        "neutral_f1",
        "vocabulary_size",
        "elapsed_seconds",
        "model_artifact",
        "predictions_csv",
        "errors_csv",
        "error_file",
    ]
    _write_csv(run_dir / "reports" / "summary_metrics.csv", sorted_rows, fieldnames)

    lines = [
        "# Etapa 2 classical benchmark summary",
        "",
        f"Run directory: `{_display_path(run_dir)}`",
        "",
        "Symbolic baseline from Etapa 1:",
        (
            f"- `{baseline['name']}` run `{baseline['run_id']}`: "
            f"accuracy={float(baseline['accuracy']):.4f}, "
            f"macro-F1={float(baseline['macro_f1']):.4f}"
        ),
        "",
        "| Rank | Model | Accuracy | Macro-F1 | Positive F1 | Negative F1 | "
        "Neutral F1 | Vocab | Time (s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted_rows:
        if row["status"] != "ok":
            lines.append(
                f"|  | {row['model']} | failed | failed | failed | failed | failed |  |  |"
            )
            continue
        lines.append(
            "| {rank} | {model} | {accuracy:.4f} | {macro_f1:.4f} | "
            "{positive_f1:.4f} | {negative_f1:.4f} | {neutral_f1:.4f} | "
            "{vocabulary_size} | {elapsed_seconds:.2f} |".format(
                rank=row["rank"],
                model=row["model"],
                accuracy=float(row["accuracy"]),
                macro_f1=float(row["macro_f1"]),
                positive_f1=float(row["positive_f1"]),
                negative_f1=float(row["negative_f1"]),
                neutral_f1=float(row["neutral_f1"]),
                vocabulary_size=int(row["vocabulary_size"]),
                elapsed_seconds=float(row["elapsed_seconds"]),
            )
        )
    (run_dir / "reports" / "summary_metrics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _prepare_run_dir(output_dir: Path, run_id: str, overwrite: bool) -> Path:
    run_dir = output_dir / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output run directory already exists: {run_dir}. "
            "Use a new run_id or set overwrite=true."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("reports", "predictions", "cases", "figures", "errors"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="benchmark_suite")
def main(cfg: DictConfig) -> None:
    print("Resolved configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    _ensure_dataset(cfg.dataset)
    train_dataset = _load_split(cfg.dataset, "train")
    test_dataset = _load_split(cfg.dataset, "test")

    run_dir = _prepare_run_dir(
        output_dir=_project_path(str(cfg.output_dir)),
        run_id=str(cfg.run_id),
        overwrite=bool(cfg.overwrite),
    )
    model_dir = _project_path(str(cfg.model_output_dir)) / str(cfg.run_id)

    _write_json(
        run_dir / "reports" / "resolved_config.json",
        OmegaConf.to_container(cfg, resolve=True),
    )
    _write_json(
        run_dir / "reports" / "dataset_manifest.json",
        {
            "train_dataset": train_dataset.name,
            "test_dataset": test_dataset.name,
            "train_examples": len(train_dataset.examples),
            "test_examples": len(test_dataset.examples),
            "source_url": str(cfg.dataset.source_url),
            "license": str(cfg.dataset.license),
        },
    )

    rows: list[dict[str, Any]] = []
    for model_name in cfg.models:
        model_name = str(model_name)
        print()
        print(f"Training model: {model_name}")
        try:
            model_config = _load_model_config(model_name)
            report, predictions_csv, errors_csv = _run_model(
                model_config=model_config,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                run_dir=run_dir,
                model_dir=model_dir,
                save_models=bool(cfg.save_models),
                save_predictions=bool(cfg.save_predictions),
                make_figures=bool(cfg.make_figures),
            )
            rows.append(_summary_row(report, predictions_csv, errors_csv))
            print(
                f"  accuracy={report.metrics.accuracy:.4f} "
                f"macro_f1={report.metrics.macro_f1:.4f} "
                f"vocab={report.vocabulary_size} "
                f"time={report.elapsed_seconds:.2f}s"
            )
        except Exception:
            error_path = run_dir / "errors" / f"{model_name}.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            rows.append(_failure_row(model_name, error_path))
            print(f"  failed; traceback saved to {_display_path(error_path)}")
            if bool(cfg.get("fail_fast", False)):
                raise

    baseline = OmegaConf.to_container(cfg.symbolic_baseline, resolve=True)
    if not isinstance(baseline, dict):
        raise ValueError("symbolic_baseline must resolve to a mapping")
    _write_summary(run_dir, rows, baseline)
    if bool(cfg.make_figures):
        _plot_metric_bars(rows, baseline, "accuracy", run_dir / "figures" / "benchmark_accuracy")
        _plot_metric_bars(rows, baseline, "macro_f1", run_dir / "figures" / "benchmark_macro_f1")

    print()
    print(f"Benchmark suite complete: {_display_path(run_dir)}")
    print(f"Summary: {_display_path(run_dir / 'reports' / 'summary_metrics.md')}")


if __name__ == "__main__":
    main()
