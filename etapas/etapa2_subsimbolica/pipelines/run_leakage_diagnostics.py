"""Diagnose shortcut leakage in the shared Kaggle sentiment corpus.

This runner tests whether surface cues used by distant supervision explain the
very high transformer scores observed on the Kaggle split. It trains cue-only
baselines and compares classical TF-IDF models on raw versus cue-stripped text.

Example:
    uv run python etapas/etapa2_subsimbolica/pipelines/run_leakage_diagnostics.py
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "etapas" / "etapa2_subsimbolica" / "configs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hydra  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from pln_core.eval.datasets.base import VALID_LABELS, EvalDataset  # noqa: E402
from pln_core.eval.datasets.kaggle_tweets import load_kaggle_tweets  # noqa: E402
from pln_core.eval.metrics import EvaluationMetrics, compute_metrics  # noqa: E402
from pln_core.eval.text_treatments import (  # noqa: E402
    LEAKAGE_FEATURE_NAMES,
    apply_text_treatment,
    extract_surface_cues,
)

LABELS = tuple(VALID_LABELS)
METRIC_COLORS = {
    "accuracy": "#0072B2",
    "macro_f1": "#D55E00",
}
CONFUSION_COLOR = "Blues"


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    """Serializable metadata for one leakage diagnostic evaluation."""

    model: str
    model_family: str
    train_treatment: str
    test_treatment: str
    feature_set: str
    elapsed_seconds: float
    metrics: EvaluationMetrics

    def as_row(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_family": self.model_family,
            "train_treatment": self.train_treatment,
            "test_treatment": self.test_treatment,
            "feature_set": self.feature_set,
            "accuracy": self.metrics.accuracy,
            "macro_f1": self.metrics.macro_f1,
            "positive_f1": self.metrics.per_class_f1["positive"],
            "negative_f1": self.metrics.per_class_f1["negative"],
            "neutral_f1": self.metrics.per_class_f1["neutral"],
            "elapsed_seconds": self.elapsed_seconds,
        }

    def as_dict(self) -> dict[str, Any]:
        row = self.as_row()
        row["metrics"] = _metrics_as_dict(self.metrics)
        return row


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


def _ensure_dataset(dataset_cfg: DictConfig) -> None:
    root_dir = _project_path(str(dataset_cfg.root_dir))
    expected_files = [
        root_dir / str(dataset_cfg.train_file),
        root_dir / str(dataset_cfg.test_file),
    ]
    missing = [path for path in expected_files if not path.exists()]
    if missing:
        missing_paths = ", ".join(_display_path(path) for path in missing)
        raise FileNotFoundError(
            "Missing Kaggle split file(s): "
            f"{missing_paths}. Run a benchmark pipeline once or download the dataset."
        )


def _load_split(dataset_cfg: DictConfig, split: str) -> EvalDataset:
    return load_kaggle_tweets(
        split=split,
        source_dir=str(_project_path(str(dataset_cfg.root_dir))),
        seed=int(dataset_cfg.seed),
    )


def _prepare_run_dir(output_dir: Path, run_id: str, overwrite: bool) -> Path:
    run_dir = output_dir / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output run directory already exists: {run_dir}. "
            "Use a new run_id or set overwrite=true."
        )
    for child in ("reports", "tables", "figures", "predictions", "cases"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


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


def _build_classical_pipeline(model_config: dict[str, Any]) -> Pipeline:
    vectorizer = TfidfVectorizer(**_vectorizer_kwargs(model_config["vectorizer"]))
    classifier_cls = _import_class(str(model_config["classifier"]["class_path"]))
    classifier = classifier_cls(**_classifier_kwargs(model_config["classifier"]))
    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )


def _labels(dataset: EvalDataset) -> list[str]:
    return [example.label for example in dataset.examples]


def _texts(dataset: EvalDataset, treatment: str) -> list[str]:
    return [apply_text_treatment(example.text, treatment) for example in dataset.examples]


def _cue_matrix(texts: Sequence[str], feature_names: Sequence[str]) -> list[list[int]]:
    rows: list[list[int]] = []
    for text in texts:
        cues = extract_surface_cues(text)
        rows.append([int(cues[feature]) for feature in feature_names])
    return rows


def _majority_label(labels: Sequence[str]) -> str:
    counts = Counter(labels)
    return max(LABELS, key=lambda label: (counts[label], -LABELS.index(label)))


def _predict_cue_rule(texts: Sequence[str], fallback_label: str) -> list[str]:
    predictions: list[str] = []
    for text in texts:
        cues = extract_surface_cues(text)
        if cues["has_negative_emoticon"]:
            predictions.append("negative")
        elif cues["has_positive_emoticon"]:
            predictions.append("positive")
        elif cues["has_url"]:
            predictions.append("neutral")
        else:
            predictions.append(fallback_label)
    return predictions


def _save_figure(fig: plt.Figure, destination_base: Path) -> list[str]:
    destination_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix in (".png", ".pdf"):
        destination = destination_base.with_suffix(suffix)
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        outputs.append(_display_path(destination))
    plt.close(fig)
    return outputs


def _plot_metric_bars(
    rows: list[dict[str, Any]],
    metric: str,
    destination_base: Path,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    names = [str(row["model"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    ax.bar(names, values, color=METRIC_COLORS[metric])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Leakage diagnostics: {metric.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    for index, value in enumerate(values):
        ax.text(index, min(value + 0.02, 0.98), f"{value:.3f}", ha="center", fontsize=8)
    return _save_figure(fig, destination_base)


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


def _write_predictions(
    run_dir: Path,
    model_name: str,
    texts: Sequence[str],
    expected: Sequence[str],
    predicted: Sequence[str],
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


def _run_cue_rule(
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
) -> tuple[DiagnosticResult, list[str]]:
    start = time.perf_counter()
    predictions = _predict_cue_rule(test_texts, fallback_label=_majority_label(train_labels))
    elapsed = time.perf_counter() - start
    metrics = compute_metrics(test_labels, predictions)
    return (
        DiagnosticResult(
            model="cue_rule_emoticon_url",
            model_family="cue_only_rule",
            train_treatment="raw",
            test_treatment="raw",
            feature_set="manual_rule: negative_emoticon > positive_emoticon > url",
            elapsed_seconds=elapsed,
            metrics=metrics,
        ),
        predictions,
    )


def _run_cue_logreg(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
    feature_names: Sequence[str],
    model_name: str,
    test_treatment: str,
) -> tuple[DiagnosticResult, list[str]]:
    start = time.perf_counter()
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(_cue_matrix(train_texts, feature_names), train_labels)
    predictions = list(model.predict(_cue_matrix(test_texts, feature_names)))
    elapsed = time.perf_counter() - start
    metrics = compute_metrics(test_labels, predictions)
    return (
        DiagnosticResult(
            model=model_name,
            model_family="cue_only_logreg",
            train_treatment="raw",
            test_treatment=test_treatment,
            feature_set="+".join(feature_names),
            elapsed_seconds=elapsed,
            metrics=metrics,
        ),
        predictions,
    )


def _run_classical(
    model_name: str,
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
    train_treatment: str,
    test_treatment: str,
) -> tuple[DiagnosticResult, list[str]]:
    model_config = _load_model_config(model_name)
    pipeline = _build_classical_pipeline(model_config)
    start = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    predictions = list(pipeline.predict(test_texts))
    elapsed = time.perf_counter() - start
    metrics = compute_metrics(test_labels, predictions)
    return (
        DiagnosticResult(
            model=f"{model_name}__train_{train_treatment}__test_{test_treatment}",
            model_family=str(model_config["name"]),
            train_treatment=train_treatment,
            test_treatment=test_treatment,
            feature_set="tfidf_word_1_2gram",
            elapsed_seconds=elapsed,
            metrics=metrics,
        ),
        predictions,
    )


def _profile_split(dataset: EvalDataset) -> dict[str, Any]:
    counts = Counter(example.label for example in dataset.examples)
    return {
        "examples": len(dataset.examples),
        "label_counts": {label: counts[label] for label in LABELS},
    }


def _cue_prevalence_rows(dataset: EvalDataset, split: str) -> list[dict[str, Any]]:
    totals = Counter(example.label for example in dataset.examples)
    counts: Counter[tuple[str, str]] = Counter()
    for example in dataset.examples:
        cues = extract_surface_cues(example.text)
        for feature in LEAKAGE_FEATURE_NAMES:
            if cues[feature]:
                counts[(example.label, feature)] += 1

    rows: list[dict[str, Any]] = []
    for label in LABELS:
        for feature in LEAKAGE_FEATURE_NAMES:
            count = counts[(label, feature)]
            total = totals[label]
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "feature": feature,
                    "count": count,
                    "total": total,
                    "share": count / total if total else 0.0,
                }
            )
    return rows


def _write_summary(
    run_dir: Path,
    cfg: DictConfig,
    rows: list[dict[str, Any]],
    train_profile: dict[str, Any],
    test_profile: dict[str, Any],
) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    _write_csv(
        run_dir / "reports" / "summary_metrics.csv",
        sorted_rows,
        [
            "model",
            "model_family",
            "train_treatment",
            "test_treatment",
            "feature_set",
            "accuracy",
            "macro_f1",
            "positive_f1",
            "negative_f1",
            "neutral_f1",
            "elapsed_seconds",
            "predictions_csv",
            "errors_csv",
        ],
    )

    lines = [
        "# Etapa 2 leakage diagnostics",
        "",
        f"Run directory: `{_display_path(run_dir)}`",
        f"Raw treatment: `{cfg.raw_treatment}`",
        f"Stripped treatment: `{cfg.stripped_treatment}`",
        "",
        "## Split Profile",
        "",
        "| Split | Examples | Positive | Negative | Neutral |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Train | {train_profile['examples']} | "
            f"{train_profile['label_counts']['positive']} | "
            f"{train_profile['label_counts']['negative']} | "
            f"{train_profile['label_counts']['neutral']} |"
        ),
        (
            f"| Test | {test_profile['examples']} | "
            f"{test_profile['label_counts']['positive']} | "
            f"{test_profile['label_counts']['negative']} | "
            f"{test_profile['label_counts']['neutral']} |"
        ),
        "",
        "## Diagnostics",
        "",
        "| Model | Train text | Test text | Accuracy | Macro-F1 | Positive F1 | "
        "Negative F1 | Neutral F1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted_rows:
        lines.append(
            "| {model} | {train_treatment} | {test_treatment} | {accuracy:.4f} | "
            "{macro_f1:.4f} | {positive_f1:.4f} | {negative_f1:.4f} | "
            "{neutral_f1:.4f} |".format(
                model=row["model"],
                train_treatment=row["train_treatment"],
                test_treatment=row["test_treatment"],
                accuracy=float(row["accuracy"]),
                macro_f1=float(row["macro_f1"]),
                positive_f1=float(row["positive_f1"]),
                negative_f1=float(row["negative_f1"]),
                neutral_f1=float(row["neutral_f1"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If cue-only baselines score close to the transformer runs, the Kaggle split is "
            "strongly explained by label-associated surface artifacts. In that case, the raw "
            "transformer score is still a valid in-distribution result for this split, but it "
            "is weak evidence that the model learned robust sentiment semantics. The stripped "
            "runs should be reported alongside raw runs in Etapa 2.",
        ]
    )
    (run_dir / "reports" / "summary_metrics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="leakage_diagnostics")
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

    feature_names = [str(feature) for feature in cfg.cue_features]
    if tuple(feature_names) != LEAKAGE_FEATURE_NAMES:
        expected = ", ".join(LEAKAGE_FEATURE_NAMES)
        raise ValueError(f"cue_features must match the leakage feature set: {expected}")

    raw_treatment = str(cfg.raw_treatment)
    stripped_treatment = str(cfg.stripped_treatment)
    train_labels = _labels(train_dataset)
    test_labels = _labels(test_dataset)
    train_raw = _texts(train_dataset, raw_treatment)
    test_raw = _texts(test_dataset, raw_treatment)
    train_stripped = _texts(train_dataset, stripped_treatment)
    test_stripped = _texts(test_dataset, stripped_treatment)

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
            "raw_treatment": raw_treatment,
            "stripped_treatment": stripped_treatment,
        },
    )

    results: list[tuple[DiagnosticResult, list[str], Sequence[str]]] = []
    cue_rule_result, cue_rule_predictions = _run_cue_rule(train_labels, test_raw, test_labels)
    results.append((cue_rule_result, cue_rule_predictions, test_raw))

    cue_logreg_raw, cue_logreg_raw_predictions = _run_cue_logreg(
        train_texts=train_raw,
        train_labels=train_labels,
        test_texts=test_raw,
        test_labels=test_labels,
        feature_names=feature_names,
        model_name="cue_only_logreg_raw_test",
        test_treatment=raw_treatment,
    )
    results.append((cue_logreg_raw, cue_logreg_raw_predictions, test_raw))

    cue_logreg_stripped, cue_logreg_stripped_predictions = _run_cue_logreg(
        train_texts=train_raw,
        train_labels=train_labels,
        test_texts=test_stripped,
        test_labels=test_labels,
        feature_names=feature_names,
        model_name="cue_only_logreg_stripped_test",
        test_treatment=stripped_treatment,
    )
    results.append((cue_logreg_stripped, cue_logreg_stripped_predictions, test_stripped))

    for model_name in cfg.classical_models:
        model_name = str(model_name)
        raw_raw_result, raw_raw_predictions = _run_classical(
            model_name=model_name,
            train_texts=train_raw,
            train_labels=train_labels,
            test_texts=test_raw,
            test_labels=test_labels,
            train_treatment=raw_treatment,
            test_treatment=raw_treatment,
        )
        results.append((raw_raw_result, raw_raw_predictions, test_raw))

        raw_stripped_result, raw_stripped_predictions = _run_classical(
            model_name=model_name,
            train_texts=train_raw,
            train_labels=train_labels,
            test_texts=test_stripped,
            test_labels=test_labels,
            train_treatment=raw_treatment,
            test_treatment=stripped_treatment,
        )
        results.append((raw_stripped_result, raw_stripped_predictions, test_stripped))

        stripped_stripped_result, stripped_stripped_predictions = _run_classical(
            model_name=model_name,
            train_texts=train_stripped,
            train_labels=train_labels,
            test_texts=test_stripped,
            test_labels=test_labels,
            train_treatment=stripped_treatment,
            test_treatment=stripped_treatment,
        )
        results.append((stripped_stripped_result, stripped_stripped_predictions, test_stripped))

    rows: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []
    for result, predictions, texts in results:
        row = result.as_row()
        if bool(cfg.save_predictions):
            predictions_csv, errors_csv = _write_predictions(
                run_dir=run_dir,
                model_name=result.model,
                texts=texts,
                expected=test_labels,
                predicted=predictions,
            )
            row["predictions_csv"] = predictions_csv
            row["errors_csv"] = errors_csv
        else:
            row["predictions_csv"] = ""
            row["errors_csv"] = ""
        rows.append(row)
        detailed_results.append(result.as_dict())

        if bool(cfg.make_figures) and result.model.startswith("cue_"):
            _plot_confusion(
                result.model,
                result.metrics,
                run_dir / "figures" / f"confusion_{result.model}",
            )

    train_profile = _profile_split(train_dataset)
    test_profile = _profile_split(test_dataset)
    _write_csv(
        run_dir / "tables" / "cue_prevalence.csv",
        _cue_prevalence_rows(train_dataset, "train") + _cue_prevalence_rows(test_dataset, "test"),
        ["split", "label", "feature", "count", "total", "share"],
    )
    _write_json(
        run_dir / "reports" / "leakage_diagnostics.json",
        {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "train_profile": train_profile,
            "test_profile": test_profile,
            "results": detailed_results,
        },
    )
    _write_summary(run_dir, cfg, rows, train_profile, test_profile)
    if bool(cfg.make_figures):
        _plot_metric_bars(rows, "accuracy", run_dir / "figures" / "diagnostic_accuracy")
        _plot_metric_bars(rows, "macro_f1", run_dir / "figures" / "diagnostic_macro_f1")

    print()
    print(f"Leakage diagnostics complete: {_display_path(run_dir)}")
    print(f"Summary: {_display_path(run_dir / 'reports' / 'summary_metrics.md')}")


if __name__ == "__main__":
    main()
