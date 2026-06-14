"""Etapa 2 external pretrained benchmark on the shared Kaggle corpus."""

from __future__ import annotations

import csv
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
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from pln_core.eval.datasets.base import VALID_LABELS, EvalDataset  # noqa: E402
from pln_core.eval.datasets.kaggle_tweets import load_kaggle_tweets  # noqa: E402
from pln_core.eval.metrics import EvaluationMetrics, compute_metrics  # noqa: E402
from pln_core.eval.text_treatments import apply_text_treatment  # noqa: E402

LABELS = tuple(VALID_LABELS)
CONFUSION_COLOR = "Blues"
TREATMENT_COLORS = {
    "raw": "#D55E00",
    "strip_emoticons_urls": "#0072B2",
    "strip_social_source_cues": "#009E73",
}
HF_TO_SHARED_LABEL = {
    "label 0": "negative",
    "label_0": "negative",
    "0": "negative",
    "very negative": "negative",
    "negative": "negative",
    "label 1": "negative",
    "label_1": "negative",
    "1": "negative",
    "label 2": "neutral",
    "label_2": "neutral",
    "2": "neutral",
    "neutral": "neutral",
    "label 3": "positive",
    "label_3": "positive",
    "3": "positive",
    "positive": "positive",
    "label 4": "positive",
    "label_4": "positive",
    "4": "positive",
    "very positive": "positive",
}


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkReport:
    model: str
    kind: str
    model_id: str
    base_model: str
    dataset: str
    text_treatment: str
    train_examples: int
    reference_train_examples: int
    test_examples: int
    elapsed_seconds: float
    metrics: EvaluationMetrics
    model_config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "kind": self.kind,
            "model_id": self.model_id,
            "base_model": self.base_model,
            "dataset": self.dataset,
            "text_treatment": self.text_treatment,
            "train_examples": self.train_examples,
            "reference_train_examples": self.reference_train_examples,
            "test_examples": self.test_examples,
            "elapsed_seconds": self.elapsed_seconds,
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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _safe_model_name(model_name: str) -> str:
    return (
        model_name.replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


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


def _load_split(
    dataset_cfg: DictConfig,
    split: str,
    max_examples: int | None = None,
    per_class: int | None = None,
) -> EvalDataset:
    return load_kaggle_tweets(
        split=split,
        source_dir=str(_project_path(str(dataset_cfg.root_dir))),
        max_examples=max_examples,
        per_class=per_class,
        seed=int(dataset_cfg.seed),
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


def _detect_device(configured_device: Any) -> Any:
    if configured_device is None:
        return -1
    device = str(configured_device).strip().lower()
    if device in {"", "none", "cpu"}:
        return -1
    if device not in {"auto", "automatic"}:
        try:
            return int(device)
        except ValueError:
            return str(configured_device)

    try:
        import torch
    except ImportError:
        return -1

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return -1


def _build_pipeline(model_config: dict[str, Any]) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformer dependencies with `uv sync`.") from exc

    inference = model_config.get("inference", {})
    if not isinstance(inference, dict):
        inference = {}
    device = _detect_device(inference.get("device", "auto"))
    print(f"Loading external model {model_config['model_id']} on device={device!r}")
    return pipeline(
        "text-classification",
        model=str(model_config["model_id"]),
        tokenizer=str(model_config["model_id"]),
        top_k=None,
        truncation=True,
        max_length=int(inference.get("max_length", 128)),
        device=device,
    )


def _normalize_hf_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    try:
        return HF_TO_SHARED_LABEL[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown Hugging Face sentiment label: {label!r}") from exc


def _scores_from_output(output: Any) -> dict[str, float]:
    if isinstance(output, dict):
        output = [output]
    scores = {label: 0.0 for label in LABELS}
    for item in output:
        label = _normalize_hf_label(str(item.get("label", "")))
        scores[label] += float(item.get("score", 0.0))
    total = sum(scores.values())
    if total > 0:
        scores = {label: value / total for label, value in scores.items()}
    return scores


def _best_label(scores: dict[str, float]) -> str:
    return max(LABELS, key=lambda label: scores.get(label, 0.0))


def _predict_batches(
    classifier: Any,
    texts: list[str],
    batch_size: int,
    progress_interval: int,
) -> tuple[list[str], list[dict[str, float]]]:
    predictions: list[str] = []
    score_rows: list[dict[str, float]] = []
    total = len(texts)
    next_progress = progress_interval
    for start_index in range(0, total, batch_size):
        batch = texts[start_index : start_index + batch_size]
        outputs = classifier(batch, batch_size=batch_size)
        for output in outputs:
            scores = _scores_from_output(output)
            score_rows.append(scores)
            predictions.append(_best_label(scores))
        completed = min(start_index + batch_size, total)
        if completed >= next_progress or completed == total:
            print(f"  processed {completed}/{total}")
            next_progress += progress_interval
    return predictions, score_rows


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
    text_treatment: str,
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
    ax.set_title(f"Confusion matrix: {model_name} ({text_treatment})")
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
    metric: str,
    destination_base: Path,
) -> list[str]:
    successful = [row for row in rows if row.get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    names = [str(row["text_treatment"]) for row in successful]
    values = [float(row[metric]) for row in successful]
    colors = [TREATMENT_COLORS.get(name, "#0072B2") for name in names]
    ax.bar(names, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"External pretrained benchmark: {metric.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=18)
    for index, value in enumerate(values):
        ax.text(index, min(value + 0.02, 0.98), f"{value:.3f}", ha="center", fontsize=8)
    return _save_figure(fig, destination_base)


def _write_predictions(
    model_name: str,
    text_treatment: str,
    raw_texts: list[str],
    treated_texts: list[str],
    expected: list[str],
    predicted: list[str],
    score_rows: list[dict[str, float]],
    run_dir: Path,
) -> tuple[str, str]:
    prediction_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    paired = zip(raw_texts, treated_texts, expected, predicted, score_rows, strict=True)
    for index, (raw_text, treated_text, gold, pred, scores) in enumerate(paired, start=1):
        confidence = scores.get(pred, 0.0)
        row = {
            "index": index,
            "expected": gold,
            "predicted": pred,
            "correct": gold == pred,
            "confidence": confidence,
            "positive_score": scores.get("positive", 0.0),
            "negative_score": scores.get("negative", 0.0),
            "neutral_score": scores.get("neutral", 0.0),
            "text": treated_text,
            "raw_text": raw_text,
        }
        prediction_rows.append(row)
        if gold != pred:
            error_rows.append(row)

    safe_name = f"{model_name}__{text_treatment}"
    fieldnames = [
        "index",
        "expected",
        "predicted",
        "correct",
        "confidence",
        "positive_score",
        "negative_score",
        "neutral_score",
        "text",
        "raw_text",
    ]
    predictions_path = run_dir / "predictions" / f"{safe_name}.csv"
    errors_path = run_dir / "cases" / f"{safe_name}_errors.csv"
    _write_csv(predictions_path, prediction_rows, fieldnames)
    _write_csv(errors_path, error_rows, fieldnames)
    return _display_path(predictions_path), _display_path(errors_path)


def _summary_row(
    report: ExternalBenchmarkReport,
    predictions_csv: str,
    errors_csv: str,
) -> dict[str, Any]:
    return {
        "model": report.model,
        "status": "ok",
        "model_id": report.model_id,
        "base_model": report.base_model,
        "dataset": report.dataset,
        "text_treatment": report.text_treatment,
        "train_examples": report.train_examples,
        "reference_train_examples": report.reference_train_examples,
        "test_examples": report.test_examples,
        "accuracy": report.metrics.accuracy,
        "macro_f1": report.metrics.macro_f1,
        "positive_f1": report.metrics.per_class_f1["positive"],
        "negative_f1": report.metrics.per_class_f1["negative"],
        "neutral_f1": report.metrics.per_class_f1["neutral"],
        "elapsed_seconds": report.elapsed_seconds,
        "predictions_csv": predictions_csv,
        "errors_csv": errors_csv,
        "error_file": "",
    }


def _failure_row(model_name: str, text_treatment: str, error_path: Path) -> dict[str, Any]:
    return {
        "model": model_name,
        "status": "failed",
        "model_id": "",
        "base_model": "",
        "dataset": "",
        "text_treatment": text_treatment,
        "train_examples": "",
        "reference_train_examples": "",
        "test_examples": "",
        "accuracy": "",
        "macro_f1": "",
        "positive_f1": "",
        "negative_f1": "",
        "neutral_f1": "",
        "elapsed_seconds": "",
        "predictions_csv": "",
        "errors_csv": "",
        "error_file": _display_path(error_path),
    }


def _write_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "status",
        "model_id",
        "base_model",
        "dataset",
        "text_treatment",
        "train_examples",
        "reference_train_examples",
        "test_examples",
        "accuracy",
        "macro_f1",
        "positive_f1",
        "negative_f1",
        "neutral_f1",
        "elapsed_seconds",
        "predictions_csv",
        "errors_csv",
        "error_file",
    ]
    _write_csv(run_dir / "reports" / "summary_metrics.csv", rows, fieldnames)

    lines = [
        "# Etapa 2 external pretrained benchmark summary",
        "",
        f"Run directory: `{_display_path(run_dir)}`",
        "",
        "| Model | Treatment | Accuracy | Macro-F1 | Positive F1 | "
        "Negative F1 | Neutral F1 | Time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] != "ok":
            lines.append(
                f"| {row['model']} | {row['text_treatment']} | failed | failed | "
                "failed | failed | failed |  |"
            )
            continue
        lines.append(
            "| {model} | {text_treatment} | {accuracy:.4f} | {macro_f1:.4f} | "
            "{positive_f1:.4f} | {negative_f1:.4f} | {neutral_f1:.4f} | "
            "{elapsed_seconds:.2f} |".format(
                model=row["model"],
                text_treatment=row["text_treatment"],
                accuracy=float(row["accuracy"]),
                macro_f1=float(row["macro_f1"]),
                positive_f1=float(row["positive_f1"]),
                negative_f1=float(row["negative_f1"]),
                neutral_f1=float(row["neutral_f1"]),
                elapsed_seconds=float(row["elapsed_seconds"]),
            )
        )
    (run_dir / "reports" / "summary_metrics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _run_treatment(
    classifier: Any,
    model_config: dict[str, Any],
    train_dataset: EvalDataset,
    test_dataset: EvalDataset,
    run_dir: Path,
    text_treatment: str,
    save_predictions: bool,
    make_figures: bool,
) -> tuple[ExternalBenchmarkReport, str, str]:
    model_name = str(model_config["name"])
    inference = model_config.get("inference", {})
    batch_size = int(inference.get("batch_size", 32)) if isinstance(inference, dict) else 32
    progress_interval = (
        int(inference.get("progress_interval", 512)) if isinstance(inference, dict) else 512
    )
    raw_texts = [example.text for example in test_dataset.examples]
    treated_texts = [apply_text_treatment(text, text_treatment) for text in raw_texts]
    test_labels = [example.label for example in test_dataset.examples]

    start = time.perf_counter()
    predicted, score_rows = _predict_batches(
        classifier,
        treated_texts,
        batch_size=batch_size,
        progress_interval=progress_interval,
    )
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(test_labels, predicted)
    report = ExternalBenchmarkReport(
        model=model_name,
        kind=str(model_config["kind"]),
        model_id=str(model_config["model_id"]),
        base_model=str(model_config.get("base_model", "")),
        dataset=(
            f"kaggle_tweets[test|{text_treatment}]"
            if text_treatment != "raw"
            else "kaggle_tweets[test]"
        ),
        text_treatment=text_treatment,
        train_examples=0,
        reference_train_examples=len(train_dataset.examples),
        test_examples=len(test_dataset.examples),
        elapsed_seconds=elapsed,
        metrics=metrics,
        model_config=model_config,
    )

    predictions_csv = ""
    errors_csv = ""
    if save_predictions:
        predictions_csv, errors_csv = _write_predictions(
            model_name=model_name,
            text_treatment=text_treatment,
            raw_texts=raw_texts,
            treated_texts=treated_texts,
            expected=test_labels,
            predicted=predicted,
            score_rows=score_rows,
            run_dir=run_dir,
        )

    report_path = run_dir / "reports" / model_name / text_treatment / "report.json"
    _write_json(report_path, report.as_dict())
    if make_figures:
        _plot_confusion(
            model_name=model_name,
            text_treatment=text_treatment,
            metrics=metrics,
            destination_base=run_dir / "figures" / f"confusion_{model_name}__{text_treatment}",
        )
    return report, predictions_csv, errors_csv


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIR),
    config_name="external_pretrained_benchmark",
)
def main(cfg: DictConfig) -> None:
    if str(cfg.model.get("kind")) != "external_pretrained":
        raise ValueError(
            "run_external_pretrained_benchmark.py requires an external_pretrained "
            f"model config, got kind={cfg.model.get('kind')!r}."
        )

    print("Resolved configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    _ensure_dataset(cfg.dataset)
    train_dataset = _load_split(cfg.dataset, "train")
    test_dataset = _load_split(
        cfg.dataset,
        "test",
        max_examples=_optional_int(cfg.test_max_examples),
        per_class=_optional_int(cfg.test_per_class),
    )
    run_dir = _prepare_run_dir(
        output_dir=_project_path(str(cfg.output_dir)),
        run_id=str(cfg.run_id),
        overwrite=bool(cfg.overwrite),
    )

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_config, dict):
        raise ValueError("model config must resolve to a mapping")

    _write_json(
        run_dir / "reports" / "resolved_config.json",
        OmegaConf.to_container(cfg, resolve=True),
    )
    _write_json(
        run_dir / "reports" / "dataset_manifest.json",
        {
            "train_dataset": train_dataset.name,
            "test_dataset": test_dataset.name,
            "reference_train_examples": len(train_dataset.examples),
            "test_examples": len(test_dataset.examples),
            "source_url": str(cfg.dataset.source_url),
            "license": str(cfg.dataset.license),
            "text_treatments": list(cfg.text_treatments),
        },
    )

    classifier = _build_pipeline(model_config)
    model_name = str(model_config["name"])
    rows: list[dict[str, Any]] = []
    for treatment in cfg.text_treatments:
        text_treatment = str(treatment)
        print()
        print(f"Evaluating {model_name} with treatment={text_treatment}")
        try:
            report, predictions_csv, errors_csv = _run_treatment(
                classifier=classifier,
                model_config=model_config,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                run_dir=run_dir,
                text_treatment=text_treatment,
                save_predictions=bool(cfg.save_predictions),
                make_figures=bool(cfg.make_figures),
            )
            rows.append(_summary_row(report, predictions_csv, errors_csv))
            print(
                f"  accuracy={report.metrics.accuracy:.4f} "
                f"macro_f1={report.metrics.macro_f1:.4f} "
                f"time={report.elapsed_seconds:.2f}s"
            )
        except Exception:
            error_path = run_dir / "errors" / f"{model_name}__{text_treatment}.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            rows.append(_failure_row(model_name, text_treatment, error_path))
            print(f"  failed; traceback saved to {_display_path(error_path)}")
            if bool(cfg.get("fail_fast", False)):
                raise

    _write_summary(run_dir, rows)
    if bool(cfg.make_figures):
        _plot_metric_bars(rows, "accuracy", run_dir / "figures" / "benchmark_accuracy")
        _plot_metric_bars(rows, "macro_f1", run_dir / "figures" / "benchmark_macro_f1")

    print()
    print(f"External pretrained benchmark complete: {_display_path(run_dir)}")
    print(f"Summary: {_display_path(run_dir / 'reports' / 'summary_metrics.md')}")


if __name__ == "__main__":
    main()
