"""Etapa 2 transformer fine-tuning benchmark on the shared Kaggle corpus.

This runner uses Hugging Face Transformers for sequence classification while
keeping the same train/test split and metrics used by Etapa 1 and the classical
Etapa 2 suite.

Install the optional dependencies before running:
    uv sync --extra transformers

Example smoke test:
    uv run --extra transformers python \
        etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py \
        model=distilbert_multilingual train_max_examples=120 test_max_examples=60 \
        model.training.epochs=1 trainer.use_cpu=true
"""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Callable
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

import numpy as np  # noqa: E402
from datasets import Dataset  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from pln_core.eval.datasets.base import VALID_LABELS, EvalDataset  # noqa: E402
from pln_core.eval.datasets.kaggle_tweets import load_kaggle_tweets  # noqa: E402
from pln_core.eval.metrics import EvaluationMetrics, compute_metrics  # noqa: E402
from pln_core.eval.text_treatments import apply_text_treatment  # noqa: E402

LABELS = tuple(VALID_LABELS)
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


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
    max_examples: int | None,
    per_class: int | None,
) -> EvalDataset:
    return load_kaggle_tweets(
        split=split,
        source_dir=str(_project_path(str(dataset_cfg.root_dir))),
        max_examples=max_examples,
        per_class=per_class,
        seed=int(dataset_cfg.seed),
    )


def _import_transformers() -> dict[str, Any]:
    try:
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Transformer dependencies are not installed. Run "
            "`uv sync --extra transformers` before using this pipeline."
        ) from exc

    return {
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def _to_hf_dataset(dataset: EvalDataset, text_treatment: str) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [
                apply_text_treatment(example.text, text_treatment)
                for example in dataset.examples
            ],
            "labels": [LABEL2ID[example.label] for example in dataset.examples],
        }
    )


def _make_tokenizer_fn(
    tokenizer: Any,
    max_length: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return tokenize


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


def _build_compute_metrics() -> Callable[[Any], dict[str, float]]:
    def compute_transformer_metrics(eval_pred: Any) -> dict[str, float]:
        logits = (
            eval_pred.predictions[0]
            if isinstance(eval_pred.predictions, tuple)
            else eval_pred.predictions
        )
        pred_ids = np.asarray(logits).argmax(axis=-1)
        expected = [ID2LABEL[int(label_id)] for label_id in eval_pred.label_ids]
        predicted = [ID2LABEL[int(label_id)] for label_id in pred_ids]
        metrics = compute_metrics(expected, predicted)
        return {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "positive_f1": metrics.per_class_f1["positive"],
            "negative_f1": metrics.per_class_f1["negative"],
            "neutral_f1": metrics.per_class_f1["neutral"],
        }

    return compute_transformer_metrics


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
    image = ax.imshow(matrix, cmap="Blues")
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
    texts: list[str],
    expected: list[str],
    predicted: list[str],
    run_dir: Path,
    model_name: str,
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


def _prepare_run_dir(output_dir: Path, run_id: str, overwrite: bool) -> Path:
    run_dir = output_dir / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output run directory already exists: {run_dir}. "
            "Use a new run_id or set overwrite=true."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("reports", "predictions", "cases", "figures"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _training_args(cfg: DictConfig, model_dir: Path) -> dict[str, Any]:
    training_cfg = cfg.model.training
    trainer_cfg = cfg.trainer
    return {
        "output_dir": str(model_dir),
        "learning_rate": float(training_cfg.learning_rate),
        "num_train_epochs": float(training_cfg.epochs),
        "per_device_train_batch_size": int(training_cfg.batch_size),
        "per_device_eval_batch_size": int(
            training_cfg.get("eval_batch_size", training_cfg.batch_size)
        ),
        "gradient_accumulation_steps": int(
            training_cfg.get("gradient_accumulation_steps", 1)
        ),
        "weight_decay": float(training_cfg.weight_decay),
        "eval_strategy": str(trainer_cfg.eval_strategy),
        "save_strategy": str(trainer_cfg.save_strategy),
        "logging_strategy": str(trainer_cfg.logging_strategy),
        "logging_steps": int(trainer_cfg.logging_steps),
        "save_total_limit": int(trainer_cfg.save_total_limit),
        "load_best_model_at_end": bool(trainer_cfg.load_best_model_at_end),
        "metric_for_best_model": str(trainer_cfg.metric_for_best_model),
        "greater_is_better": bool(trainer_cfg.greater_is_better),
        "dataloader_pin_memory": bool(trainer_cfg.dataloader_pin_memory),
        "use_cpu": bool(trainer_cfg.use_cpu),
        "report_to": list(trainer_cfg.report_to),
        "seed": int(training_cfg.seed),
    }


def _write_summary(
    run_dir: Path,
    cfg: DictConfig,
    metrics: EvaluationMetrics,
    model_artifact: str,
    predictions_csv: str,
    errors_csv: str,
) -> None:
    symbolic = OmegaConf.to_container(cfg.symbolic_baseline, resolve=True)
    classical = OmegaConf.to_container(cfg.classical_baseline, resolve=True)
    if not isinstance(symbolic, dict) or not isinstance(classical, dict):
        raise ValueError("baseline configs must resolve to mappings")

    lines = [
        "# Etapa 2 transformer benchmark summary",
        "",
        f"Run directory: `{_display_path(run_dir)}`",
        f"Model: `{cfg.model.name}` (`{cfg.model.model_id}`)",
        f"Text treatment: `{cfg.text_treatment}`",
        "",
        "| System | Accuracy | Macro-F1 | Positive F1 | Negative F1 | Neutral F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {symbolic['name']} | {float(symbolic['accuracy']):.4f} | "
            f"{float(symbolic['macro_f1']):.4f} | {float(symbolic['positive_f1']):.4f} | "
            f"{float(symbolic['negative_f1']):.4f} | {float(symbolic['neutral_f1']):.4f} |"
        ),
        (
            f"| {classical['name']} | {float(classical['accuracy']):.4f} | "
            f"{float(classical['macro_f1']):.4f} | {float(classical['positive_f1']):.4f} | "
            f"{float(classical['negative_f1']):.4f} | {float(classical['neutral_f1']):.4f} |"
        ),
        (
            f"| {cfg.model.name} | {metrics.accuracy:.4f} | {metrics.macro_f1:.4f} | "
            f"{metrics.per_class_f1['positive']:.4f} | {metrics.per_class_f1['negative']:.4f} | "
            f"{metrics.per_class_f1['neutral']:.4f} |"
        ),
        "",
        f"Model artifact: `{model_artifact}`",
        f"Predictions: `{predictions_csv}`",
        f"Errors: `{errors_csv}`",
    ]
    (run_dir / "reports" / "summary_metrics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="transformer_benchmark")
def main(cfg: DictConfig) -> None:
    if str(cfg.model.get("kind")) != "transformer":
        raise ValueError(
            "run_transformer_benchmark.py requires a transformer model config, "
            f"got kind={cfg.model.get('kind')!r}."
        )

    transformers = _import_transformers()
    transformers["set_seed"](int(cfg.seed))

    print("Resolved configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    _ensure_dataset(cfg.dataset)
    train_dataset = _load_split(
        cfg.dataset,
        "train",
        max_examples=_optional_int(cfg.train_max_examples),
        per_class=_optional_int(cfg.train_per_class),
    )
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
    model_dir = _project_path(str(cfg.model_output_dir)) / str(cfg.run_id) / str(cfg.model.name)

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
            "text_treatment": str(cfg.text_treatment),
        },
    )

    tokenizer = transformers["AutoTokenizer"].from_pretrained(str(cfg.model.model_id))
    model = transformers["AutoModelForSequenceClassification"].from_pretrained(
        str(cfg.model.model_id),
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    data_collator = transformers["DataCollatorWithPadding"](tokenizer=tokenizer)

    text_treatment = str(cfg.text_treatment)
    train_hf = _to_hf_dataset(train_dataset, text_treatment)
    test_hf = _to_hf_dataset(test_dataset, text_treatment)
    tokenize = _make_tokenizer_fn(tokenizer, int(cfg.model.training.max_length))
    train_tokenized = train_hf.map(tokenize, batched=True, remove_columns=["text"])
    test_tokenized = test_hf.map(tokenize, batched=True, remove_columns=["text"])

    training_args = transformers["TrainingArguments"](**_training_args(cfg, model_dir))
    trainer = transformers["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_build_compute_metrics(),
    )

    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    predictions_output = trainer.predict(test_tokenized)
    pred_ids = np.asarray(predictions_output.predictions).argmax(axis=-1)
    expected = [ID2LABEL[int(label_id)] for label_id in predictions_output.label_ids]
    predicted = [ID2LABEL[int(label_id)] for label_id in pred_ids]
    metrics = compute_metrics(expected, predicted)

    predictions_csv = ""
    errors_csv = ""
    if bool(cfg.save_predictions):
        predictions_csv, errors_csv = _write_predictions(
            texts=[
                apply_text_treatment(example.text, text_treatment)
                for example in test_dataset.examples
            ],
            expected=expected,
            predicted=predicted,
            run_dir=run_dir,
            model_name=str(cfg.model.name),
        )
    if bool(cfg.make_figures):
        _plot_confusion(
            model_name=str(cfg.model.name),
            metrics=metrics,
            destination_base=run_dir / "figures" / f"confusion_{cfg.model.name}",
        )

    model_artifact = _display_path(model_dir)
    report = {
        "model": str(cfg.model.name),
        "model_id": str(cfg.model.model_id),
        "dataset": test_dataset.name,
        "text_treatment": text_treatment,
        "train_examples": len(train_dataset.examples),
        "test_examples": len(test_dataset.examples),
        "model_artifact": model_artifact,
        "metrics": _metrics_as_dict(metrics),
    }
    _write_json(run_dir / "reports" / str(cfg.model.name) / "report.json", report)
    _write_summary(run_dir, cfg, metrics, model_artifact, predictions_csv, errors_csv)

    print()
    print(
        f"Transformer benchmark complete: accuracy={metrics.accuracy:.4f} "
        f"macro_f1={metrics.macro_f1:.4f}"
    )
    print(f"Summary: {_display_path(run_dir / 'reports' / 'summary_metrics.md')}")


if __name__ == "__main__":
    main()
