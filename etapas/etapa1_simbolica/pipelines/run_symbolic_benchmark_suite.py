"""Etapa 1 symbolic benchmark suite on the shared Kaggle corpus.

The suite does four things in one Hydra-run command:

1. Profiles the official three-class train/test files.
2. Writes normalized split copies for reproducibility.
3. Runs the configured symbolic analyzer through the shared evaluation runner.
4. Saves CSV/JSON reports and static figures in a timestamped output folder.

Example:
    uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_benchmark_suite.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "etapas" / "etapa1_simbolica" / "configs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hydra  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from pln_core.eval.datasets.base import VALID_LABELS  # noqa: E402
from pln_core.eval.datasets.kaggle_tweets import normalize_kaggle_label  # noqa: E402
from pln_core.eval.runner import EvaluationReport, run_evaluation, save_report_json  # noqa: E402

LABELS = tuple(VALID_LABELS)
SPLIT_COLORS = {"train": "#3b82f6", "test": "#ef4444"}
METRIC_COLOR = "#2563eb"
CONFUSION_COLOR = "Blues"


@dataclass(frozen=True, slots=True)
class SplitProfile:
    """Profile data plus length vectors needed for figures."""

    stats: dict[str, Any]
    char_lengths: tuple[int, ...]
    token_lengths: tuple[int, ...]


def _project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _display_path(path: str | Path) -> str:
    path = Path(path)
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        handle.seek(0)
        delimiter = ";" if header.count(";") > header.count(",") else ","
        reader = csv.DictReader(
            handle,
            delimiter=delimiter,
            quotechar='"',
            doublequote=True,
        )
        return list(reader)


def _first_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return value
    return ""


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    lower_weight = upper - position
    upper_weight = position - lower
    return float(ordered[lower] * lower_weight + ordered[upper] * upper_weight)


def _length_summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "min": float(min(values)),
        "p25": _percentile(values, 0.25),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "p75": _percentile(values, 0.75),
        "p95": _percentile(values, 0.95),
        "max": float(max(values)),
    }


def _profile_split(split: str, raw_path: Path, cleaned_path: Path) -> SplitProfile:
    rows = _read_rows(raw_path)
    label_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    char_lengths: list[int] = []
    token_lengths: list[int] = []
    cleaned_rows: list[dict[str, str]] = []
    seen_text_label: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()

    empty_text_rows = 0
    invalid_label_rows = 0
    duplicate_text_label_rows = 0
    duplicate_id_rows = 0

    for row in rows:
        tweet_id = _first_value(row, "id", "tweet_id")
        text = _first_value(row, "tweet_text", "text", "sentence", "tweet").strip()
        raw_label = _first_value(row, "sentiment", "label", "class")
        query = _first_value(row, "query_used", "query", "theme").strip()

        if not text:
            empty_text_rows += 1
            continue

        try:
            label = normalize_kaggle_label(raw_label)
        except ValueError:
            invalid_label_rows += 1
            continue

        text_label = (text, label)
        if text_label in seen_text_label:
            duplicate_text_label_rows += 1
        seen_text_label.add(text_label)

        if tweet_id:
            if tweet_id in seen_ids:
                duplicate_id_rows += 1
            seen_ids.add(tweet_id)

        label_counts[label] += 1
        if query:
            query_counts[query] += 1
        char_lengths.append(len(text))
        token_lengths.append(len(text.split()))
        cleaned_rows.append(
            {
                "id": tweet_id,
                "text": text,
                "label": label,
                "tweet_date": _first_value(row, "tweet_date", "date"),
                "query_used": query,
            }
        )

    _write_csv(
        cleaned_path,
        cleaned_rows,
        ["id", "text", "label", "tweet_date", "query_used"],
    )

    valid_rows = len(cleaned_rows)
    stats = {
        "split": split,
        "raw_file": _display_path(raw_path),
        "cleaned_file": _display_path(cleaned_path),
        "raw_rows": len(rows),
        "valid_rows": valid_rows,
        "empty_text_rows": empty_text_rows,
        "invalid_label_rows": invalid_label_rows,
        "duplicate_text_label_rows": duplicate_text_label_rows,
        "duplicate_id_rows": duplicate_id_rows,
        "label_counts": {label: int(label_counts.get(label, 0)) for label in LABELS},
        "label_proportions": {
            label: (float(label_counts.get(label, 0) / valid_rows) if valid_rows else 0.0)
            for label in LABELS
        },
        "char_length": _length_summary(char_lengths),
        "token_length": _length_summary(token_lengths),
        "top_query_used": [
            {"query_used": query, "count": int(count)}
            for query, count in query_counts.most_common(20)
        ],
    }
    return SplitProfile(
        stats=stats,
        char_lengths=tuple(char_lengths),
        token_lengths=tuple(token_lengths),
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


def _plot_label_distribution(
    profiles: dict[str, SplitProfile],
    destination_base: Path,
) -> list[str]:
    x_positions = range(len(LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, split in enumerate(("train", "test")):
        profile = profiles[split]
        counts = [profile.stats["label_counts"][label] for label in LABELS]
        shifted = [x + (offset - 0.5) * width for x in x_positions]
        ax.bar(
            shifted,
            counts,
            width=width,
            label=split,
            color=SPLIT_COLORS[split],
        )

    ax.set_title("Kaggle Portuguese tweets label distribution")
    ax.set_ylabel("Examples")
    ax.set_xticks(list(x_positions), LABELS)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, destination_base)


def _plot_text_lengths(
    profiles: dict[str, SplitProfile],
    destination_base: Path,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = list(range(0, 301, 20))
    for split in ("train", "test"):
        ax.hist(
            profiles[split].char_lengths,
            bins=bins,
            alpha=0.55,
            label=split,
            color=SPLIT_COLORS[split],
        )

    ax.set_title("Tweet length distribution")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Examples")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, destination_base)


def _plot_metric_bars(
    rows: list[dict[str, Any]],
    metric: str,
    destination_base: Path,
) -> list[str]:
    successful = [row for row in rows if row.get("status") == "ok"]
    successful.sort(key=lambda row: float(row[metric]), reverse=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [str(row["analyzer"]) for row in successful]
    values = [float(row[metric]) for row in successful]
    ax.bar(names, values, color=METRIC_COLOR)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Symbolic benchmark: {metric.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)

    for index, value in enumerate(values):
        ax.text(index, min(value + 0.02, 0.98), f"{value:.3f}", ha="center", fontsize=8)

    return _save_figure(fig, destination_base)


def _plot_confusion(report: EvaluationReport, destination_base: Path) -> list[str]:
    matrix = [
        [report.metrics.confusion.get((true_label, pred_label), 0) for pred_label in LABELS]
        for true_label in LABELS
    ]

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    image = ax.imshow(matrix, cmap=CONFUSION_COLOR)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion matrix: {report.analyzer}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold label")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)

    threshold = max(max(row) for row in matrix) / 2 if matrix else 0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)

    return _save_figure(fig, destination_base)


def _write_dataset_tables(run_dir: Path, profiles: dict[str, SplitProfile]) -> None:
    distribution_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []

    for split, profile in profiles.items():
        valid_rows = int(profile.stats["valid_rows"])
        for label in LABELS:
            count = int(profile.stats["label_counts"][label])
            distribution_rows.append(
                {
                    "split": split,
                    "label": label,
                    "count": count,
                    "proportion": count / valid_rows if valid_rows else 0.0,
                }
            )
        for metric, summary in (
            ("characters", profile.stats["char_length"]),
            ("tokens", profile.stats["token_length"]),
        ):
            length_rows.append({"split": split, "metric": metric, **summary})

    _write_csv(
        run_dir / "dataset" / "label_distribution.csv",
        distribution_rows,
        ["split", "label", "count", "proportion"],
    )
    _write_csv(
        run_dir / "dataset" / "text_length_summary.csv",
        length_rows,
        ["split", "metric", "min", "p25", "median", "mean", "p75", "p95", "max"],
    )


def _write_predictions(report: EvaluationReport, run_dir: Path) -> tuple[str, str]:
    return _write_predictions_with_key(report, run_dir, report.analyzer)


def _write_predictions_with_key(
    report: EvaluationReport,
    run_dir: Path,
    artifact_key: str,
) -> tuple[str, str]:
    prediction_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for index, case in enumerate(report.predictions, start=1):
        row = {
            "index": index,
            "expected": case.expected,
            "predicted": case.predicted,
            "correct": case.expected == case.predicted,
            "score": case.score,
            "text": case.text,
        }
        prediction_rows.append(row)
        if case.expected != case.predicted:
            error_rows.append(row)

    predictions_path = run_dir / "predictions" / f"{artifact_key}.csv"
    errors_path = run_dir / "cases" / f"{artifact_key}_errors.csv"
    fieldnames = ["index", "expected", "predicted", "correct", "score", "text"]
    _write_csv(predictions_path, prediction_rows, fieldnames)
    _write_csv(errors_path, error_rows, fieldnames)
    return _display_path(predictions_path), _display_path(errors_path)


def _metrics_row(
    report: EvaluationReport,
    text_treatment: str,
    artifact_key: str,
    report_path: Path,
    predictions_path: str,
    errors_path: str,
) -> dict[str, Any]:
    return {
        "analyzer": report.analyzer,
        "artifact_key": artifact_key,
        "text_treatment": text_treatment,
        "status": "ok",
        "dataset": report.dataset,
        "total": report.metrics.total,
        "correct": report.metrics.correct,
        "accuracy": report.metrics.accuracy,
        "macro_f1": report.metrics.macro_f1,
        "positive_f1": report.metrics.per_class_f1["positive"],
        "negative_f1": report.metrics.per_class_f1["negative"],
        "neutral_f1": report.metrics.per_class_f1["neutral"],
        "elapsed_seconds": report.elapsed_seconds,
        "report_json": _display_path(report_path),
        "predictions_csv": predictions_path,
        "errors_csv": errors_path,
        "error_file": "",
    }


def _failure_row(analyzer: str, error_path: Path) -> dict[str, Any]:
    return {
        "analyzer": analyzer,
        "artifact_key": analyzer,
        "text_treatment": "",
        "status": "failed",
        "dataset": "",
        "total": "",
        "correct": "",
        "accuracy": "",
        "macro_f1": "",
        "positive_f1": "",
        "negative_f1": "",
        "neutral_f1": "",
        "elapsed_seconds": "",
        "report_json": "",
        "predictions_csv": "",
        "errors_csv": "",
        "error_file": _display_path(error_path),
    }


def _write_summary_tables(run_dir: Path, rows: list[dict[str, Any]]) -> None:
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
        "analyzer",
        "artifact_key",
        "text_treatment",
        "status",
        "dataset",
        "total",
        "correct",
        "accuracy",
        "macro_f1",
        "positive_f1",
        "negative_f1",
        "neutral_f1",
        "elapsed_seconds",
        "report_json",
        "predictions_csv",
        "errors_csv",
        "error_file",
    ]
    _write_csv(run_dir / "reports" / "summary_metrics.csv", sorted_rows, fieldnames)

    lines = [
        "# Etapa 1 symbolic benchmark summary",
        "",
        f"Run directory: `{_display_path(run_dir)}`",
        "",
        "| Rank | Analyzer | Treatment | Accuracy | Macro-F1 | Positive F1 | Negative F1 "
        "| Neutral F1 | Time (s) |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted_rows:
        if row["status"] != "ok":
            lines.append(
                f"|  | {row['analyzer']} | {row['text_treatment']} | failed | failed | "
                "failed | failed | failed |  |"
            )
            continue
        lines.append(
            "| {rank} | {analyzer} | {text_treatment} | {accuracy:.4f} | {macro_f1:.4f} | "
            "{positive_f1:.4f} | {negative_f1:.4f} | {neutral_f1:.4f} | "
            "{elapsed_seconds:.2f} |".format(
                rank=row["rank"],
                analyzer=row["analyzer"],
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


def _prepare_run_dir(output_dir: Path, run_id: str, overwrite: bool) -> Path:
    run_dir = output_dir / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output run directory already exists: {run_dir}. "
            "Use a new run_id or set overwrite=true."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("dataset", "reports", "predictions", "cases", "figures", "errors"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _analyzer_entry(entry: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, str):
        return entry, {}
    container = OmegaConf.to_container(entry, resolve=True)
    if not isinstance(container, dict) or "name" not in container:
        raise ValueError(f"invalid analyzer entry: {entry}")
    kwargs = container.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise ValueError(f"invalid analyzer kwargs for {container['name']}: {kwargs}")
    return str(container["name"]), kwargs


def _configured_text_treatments(cfg: DictConfig) -> list[str]:
    treatments = OmegaConf.to_container(cfg.get("text_treatments", []), resolve=True)
    if isinstance(treatments, str):
        return [treatments]
    if isinstance(treatments, list) and treatments:
        return [str(treatment) for treatment in treatments]
    return [str(cfg.get("text_treatment", "raw"))]


def _artifact_key(analyzer_name: str, text_treatment: str, multiple_treatments: bool) -> str:
    if not multiple_treatments and text_treatment in {"raw", "none"}:
        return analyzer_name
    return f"{analyzer_name}__{text_treatment}"


def _download_kaggle_dataset(slug: str, source_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The Kaggle package is required for automatic dataset download. "
            "Install project dependencies with `uv sync` or run the Kaggle CLI manually."
        ) from exc

    source_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset {slug} into {_display_path(source_dir)}")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(source_dir), unzip=True, quiet=False)


def _profile_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, SplitProfile]:
    source_dir = _project_path(str(cfg.dataset.source_dir))
    split_files = {
        "train": source_dir / str(cfg.dataset.train_file),
        "test": source_dir / str(cfg.dataset.test_file),
    }
    missing = [path for path in split_files.values() if not path.exists()]
    if missing and bool(cfg.dataset.get("download_if_missing", False)):
        _download_kaggle_dataset(str(cfg.dataset.kaggle_slug), source_dir)
        missing = [path for path in split_files.values() if not path.exists()]
    if missing:
        missing_list = ", ".join(_display_path(path) for path in missing)
        raise FileNotFoundError(
            "Missing Kaggle split file(s): "
            f"{missing_list}. Run the Kaggle download command first."
        )

    profiles = {
        split: _profile_split(
            split=split,
            raw_path=path,
            cleaned_path=run_dir / "dataset" / f"{split}_normalized.csv",
        )
        for split, path in split_files.items()
    }

    dataset_profile = {
        "source": {
            "name": "Kaggle Portuguese Tweets for Sentiment Analysis",
            "url": str(cfg.dataset.source_url),
            "license": str(cfg.dataset.license),
            "source_dir": _display_path(source_dir),
        },
        "splits": {split: profile.stats for split, profile in profiles.items()},
    }
    _write_json(run_dir / "dataset" / "dataset_profile.json", dataset_profile)
    _write_dataset_tables(run_dir, profiles)
    return profiles


def _run_benchmarks(cfg: DictConfig, run_dir: Path) -> list[dict[str, Any]]:
    dataset_kwargs = OmegaConf.to_container(cfg.dataset.kwargs, resolve=True) or {}
    if not isinstance(dataset_kwargs, dict):
        raise ValueError("dataset.kwargs must resolve to a mapping")

    source_dir = dataset_kwargs.get("source_dir")
    if source_dir:
        dataset_kwargs["source_dir"] = str(_project_path(str(source_dir)))

    text_treatments = _configured_text_treatments(cfg)
    multiple_treatments = len(text_treatments) > 1
    rows: list[dict[str, Any]] = []
    for text_treatment in text_treatments:
        treatment_dataset_kwargs = dict(dataset_kwargs)
        treatment_dataset_kwargs["text_treatment"] = text_treatment
        for entry in cfg.analyzers:
            analyzer_name, analyzer_kwargs = _analyzer_entry(entry)
            artifact_key = _artifact_key(analyzer_name, text_treatment, multiple_treatments)
            print()
            print(f"Running analyzer: {analyzer_name} (text_treatment={text_treatment})")
            try:
                report = run_evaluation(
                    analyzer_name=analyzer_name,
                    dataset_name=str(cfg.dataset.name),
                    dataset_kwargs=treatment_dataset_kwargs,
                    analyzer_kwargs=analyzer_kwargs,
                )
                report_path = run_dir / "reports" / artifact_key / "report.json"
                save_report_json(
                    report,
                    destination=report_path,
                    include_predictions=bool(cfg.save_predictions),
                )
                predictions_path, errors_path = _write_predictions_with_key(
                    report,
                    run_dir,
                    artifact_key,
                )
                if bool(cfg.make_figures):
                    _plot_confusion(
                        report,
                        run_dir / "figures" / f"confusion_{artifact_key}",
                    )
                rows.append(
                    _metrics_row(
                        report=report,
                        text_treatment=text_treatment,
                        artifact_key=artifact_key,
                        report_path=report_path,
                        predictions_path=predictions_path,
                        errors_path=errors_path,
                    )
                )
                print(
                    f"  accuracy={report.metrics.accuracy:.4f} "
                    f"macro_f1={report.metrics.macro_f1:.4f} "
                    f"n={report.metrics.total} "
                    f"time={report.elapsed_seconds:.2f}s"
                )
            except Exception as exc:  # noqa: BLE001
                error_path = run_dir / "errors" / f"{artifact_key}.txt"
                error_path.write_text(
                    "".join(traceback.format_exception(exc)),
                    encoding="utf-8",
                )
                rows.append(_failure_row(artifact_key, error_path))
                print(f"  failed; see {_display_path(error_path)}")
                if bool(cfg.fail_fast):
                    raise
    return rows


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="benchmark_suite")
def main(cfg: DictConfig) -> None:
    print("Resolved configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    output_dir = _project_path(str(cfg.output_dir))
    run_dir = _prepare_run_dir(
        output_dir=output_dir,
        run_id=str(cfg.run_id),
        overwrite=bool(cfg.overwrite),
    )
    _write_json(
        run_dir / "reports" / "resolved_config.json",
        OmegaConf.to_container(cfg, resolve=True),
    )

    profiles = _profile_dataset(cfg, run_dir)
    if bool(cfg.make_figures):
        _plot_label_distribution(profiles, run_dir / "figures" / "dataset_label_distribution")
        _plot_text_lengths(profiles, run_dir / "figures" / "dataset_text_lengths")

    rows = _run_benchmarks(cfg, run_dir)
    _write_summary_tables(run_dir, rows)
    if bool(cfg.make_figures):
        _plot_metric_bars(rows, "macro_f1", run_dir / "figures" / "benchmark_macro_f1")
        _plot_metric_bars(rows, "accuracy", run_dir / "figures" / "benchmark_accuracy")

    manifest = {
        "run_id": str(cfg.run_id),
        "run_dir": _display_path(run_dir),
        "status": "completed",
        "text_treatments": _configured_text_treatments(cfg),
        "successful_analyzers": [
            row["analyzer"] for row in rows if row.get("status") == "ok"
        ],
        "successful_artifacts": [
            row["artifact_key"] for row in rows if row.get("status") == "ok"
        ],
        "failed_analyzers": [
            row["analyzer"] for row in rows if row.get("status") != "ok"
        ],
        "key_artifacts": {
            "dataset_profile": _display_path(run_dir / "dataset" / "dataset_profile.json"),
            "summary_metrics_csv": _display_path(run_dir / "reports" / "summary_metrics.csv"),
            "summary_metrics_md": _display_path(run_dir / "reports" / "summary_metrics.md"),
            "figures_dir": _display_path(run_dir / "figures"),
        },
    }
    _write_json(run_dir / "reports" / "run_manifest.json", manifest)

    print()
    print(f"Benchmark suite complete: {_display_path(run_dir)}")
    print(f"Summary: {_display_path(run_dir / 'reports' / 'summary_metrics.md')}")


if __name__ == "__main__":
    main()
