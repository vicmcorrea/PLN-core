"""Analyze lexical artifacts in the shared Kaggle sentiment corpus.

The Kaggle corpus uses distant supervision signals from tweets. This analysis
summarizes exact train/test duplicates and simple lexical cues such as
emoticons, URLs, mentions, hashtags, laughter markers, and label-associated
terms. Outputs are timestamped under Etapa 2 so report material never overwrites
model benchmark artifacts.

Example:
    uv run python etapas/etapa2_subsimbolica/pipelines/run_data_artifact_analysis.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
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

from pln_core.eval.datasets.base import VALID_LABELS, EvalDataset, EvalExample  # noqa: E402
from pln_core.eval.datasets.kaggle_tweets import load_kaggle_tweets  # noqa: E402
from pln_core.eval.text_treatments import CUE_PATTERNS, text_for_cue_matching  # noqa: E402

LABELS = tuple(VALID_LABELS)
TOKEN_RE = re.compile(r"(?u)\b[\w@#]+\b|[:;=8xX][-']?[)(/DdpPcC]|<3")
FIGURE_COLOR = "#0072B2"


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
            f"{missing_paths}. Run the benchmark pipeline once or download the dataset."
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
    for child in ("reports", "tables", "figures"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def _split_rows(split: str, dataset: EvalDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in dataset.examples:
        row = {
            "split": split,
            "label": example.label,
            "text": example.text,
            "normalized_text": _normalized_text(example.text),
            "char_count": len(example.text),
            "token_count": len(_tokenize(example.text)),
        }
        rows.append(row)
    return rows


def _profile_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(row["label"]) for row in rows)
    char_counts = [int(row["char_count"]) for row in rows]
    token_counts = [int(row["token_count"]) for row in rows]
    unique_texts = {str(row["normalized_text"]) for row in rows}
    return {
        "rows": len(rows),
        "unique_normalized_texts": len(unique_texts),
        "duplicate_rows": len(rows) - len(unique_texts),
        "label_counts": {label: label_counts[label] for label in LABELS},
        "avg_chars": sum(char_counts) / len(char_counts),
        "avg_tokens": sum(token_counts) / len(token_counts),
    }


def _cue_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], int] = Counter()
    cue_counts: dict[tuple[str, str, str], int] = Counter()
    for row in rows:
        split = str(row["split"])
        label = str(row["label"])
        text = str(row["text"])
        totals[(split, label)] += 1
        for cue_name, pattern in CUE_PATTERNS.items():
            if pattern.search(text_for_cue_matching(text, cue_name)):
                cue_counts[(split, label, cue_name)] += 1

    output_rows: list[dict[str, Any]] = []
    for split, label in sorted(totals):
        total = totals[(split, label)]
        for cue_name in CUE_PATTERNS:
            count = cue_counts[(split, label, cue_name)]
            output_rows.append(
                {
                    "split": split,
                    "label": label,
                    "cue": cue_name,
                    "count": count,
                    "total": total,
                    "share": count / total if total else 0.0,
                }
            )
    return output_rows


def _duplicate_summary(
    train_examples: tuple[EvalExample, ...],
    test_examples: tuple[EvalExample, ...],
) -> dict[str, Any]:
    train_by_text: dict[str, Counter[str]] = defaultdict(Counter)
    test_by_text: dict[str, Counter[str]] = defaultdict(Counter)
    for example in train_examples:
        train_by_text[_normalized_text(example.text)][example.label] += 1
    for example in test_examples:
        test_by_text[_normalized_text(example.text)][example.label] += 1

    overlap = sorted(set(train_by_text) & set(test_by_text))
    inconsistent = 0
    overlap_test_rows = 0
    examples: list[dict[str, Any]] = []
    for text in overlap:
        train_labels = set(train_by_text[text])
        test_labels = set(test_by_text[text])
        overlap_test_rows += sum(test_by_text[text].values())
        if train_labels != test_labels or len(train_labels) != 1:
            inconsistent += 1
        if len(examples) < 20:
            examples.append(
                {
                    "text": text,
                    "train_labels": dict(train_by_text[text]),
                    "test_labels": dict(test_by_text[text]),
                }
            )

    return {
        "train_unique_normalized_texts": len(train_by_text),
        "test_unique_normalized_texts": len(test_by_text),
        "exact_overlap_unique_texts": len(overlap),
        "exact_overlap_test_rows": overlap_test_rows,
        "exact_overlap_inconsistent_unique_texts": inconsistent,
        "examples": examples,
    }


def _term_rows(rows: list[dict[str, Any]], min_count: int, top_n: int) -> list[dict[str, Any]]:
    by_label: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
    totals: Counter[str] = Counter()
    for row in rows:
        label = str(row["label"])
        terms = set(_tokenize(str(row["text"])))
        for term in terms:
            by_label[label][term] += 1
            totals[term] += 1

    output_rows: list[dict[str, Any]] = []
    vocabulary = [term for term, count in totals.items() if count >= min_count]
    for label in LABELS:
        other_total = sum(
            count
            for other, counts in by_label.items()
            if other != label
            for count in counts.values()
        )
        label_total = sum(by_label[label].values())
        scored_terms: list[dict[str, Any]] = []
        for term in vocabulary:
            in_label = by_label[label][term]
            in_other = sum(by_label[other][term] for other in LABELS if other != label)
            label_rate = (in_label + 0.5) / (label_total + 1.0)
            other_rate = (in_other + 0.5) / (other_total + 1.0)
            scored_terms.append(
                {
                    "label": label,
                    "term": term,
                    "count_in_label": in_label,
                    "count_outside_label": in_other,
                    "total_count": totals[term],
                    "log_odds": math.log(label_rate / other_rate),
                }
            )
        scored_terms.sort(key=lambda row: float(row["log_odds"]), reverse=True)
        output_rows.extend(scored_terms[:top_n])
    return output_rows


def _save_figure(fig: plt.Figure, destination_base: Path) -> list[str]:
    destination_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix in (".png", ".pdf"):
        destination = destination_base.with_suffix(suffix)
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        outputs.append(_display_path(destination))
    plt.close(fig)
    return outputs


def _plot_cue_prevalence(cue_rows: list[dict[str, Any]], destination_base: Path) -> list[str]:
    test_rows = [row for row in cue_rows if row["split"] == "test"]
    cues = list(CUE_PATTERNS)
    width = 0.24
    positions = list(range(len(cues)))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    colors = {
        "positive": "#0072B2",
        "negative": "#D55E00",
        "neutral": "#009E73",
    }
    for label_index, label in enumerate(LABELS):
        shares = [
            float(
                next(
                    row["share"]
                    for row in test_rows
                    if row["label"] == label and row["cue"] == cue
                )
            )
            for cue in cues
        ]
        offsets = [position + (label_index - 1) * width for position in positions]
        ax.bar(offsets, shares, width=width, label=label, color=colors[label])

    ax.set_title("Lexical cue prevalence by label on the common test split")
    ax.set_ylabel("Share of tweets")
    ax.set_xticks(positions, cues, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return _save_figure(fig, destination_base)


def _write_markdown(
    path: Path,
    train_profile: dict[str, Any],
    test_profile: dict[str, Any],
    duplicates: dict[str, Any],
    cue_rows: list[dict[str, Any]],
    term_rows: list[dict[str, Any]],
) -> None:
    def cue_share(label: str, cue: str) -> float:
        return float(
            next(
                row["share"]
                for row in cue_rows
                if row["split"] == "test" and row["label"] == label and row["cue"] == cue
            )
        )

    lines = [
        "# Etapa 2 dataset artifact analysis",
        "",
        "This analysis summarizes lexical artifacts in the shared Kaggle corpus.",
        "",
        "## Split Profile",
        "",
        "| Split | Rows | Unique texts | Duplicate rows | Positive | Negative | Neutral |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Train | {train_profile['rows']} | {train_profile['unique_normalized_texts']} | "
            f"{train_profile['duplicate_rows']} | {train_profile['label_counts']['positive']} | "
            f"{train_profile['label_counts']['negative']} | "
            f"{train_profile['label_counts']['neutral']} |"
        ),
        (
            f"| Test | {test_profile['rows']} | {test_profile['unique_normalized_texts']} | "
            f"{test_profile['duplicate_rows']} | {test_profile['label_counts']['positive']} | "
            f"{test_profile['label_counts']['negative']} | "
            f"{test_profile['label_counts']['neutral']} |"
        ),
        "",
        "## Exact Train/Test Overlap",
        "",
        f"- Unique normalized texts in both splits: {duplicates['exact_overlap_unique_texts']}",
        f"- Test rows covered by those overlaps: {duplicates['exact_overlap_test_rows']}",
        (
            "- Unique overlapping texts with label conflict: "
            f"{duplicates['exact_overlap_inconsistent_unique_texts']}"
        ),
        "",
        "## Cue Prevalence on Test Split",
        "",
        "| Cue | Positive | Negative | Neutral |",
        "| --- | ---: | ---: | ---: |",
    ]
    for cue in CUE_PATTERNS:
        lines.append(
            f"| {cue} | {cue_share('positive', cue):.4f} | "
            f"{cue_share('negative', cue):.4f} | {cue_share('neutral', cue):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Top Label-Associated Terms",
            "",
            "| Label | Term | Log-odds | Count in label | Count outside label |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for label in LABELS:
        label_terms = [row for row in term_rows if row["label"] == label][:10]
        for row in label_terms:
            lines.append(
                f"| {row['label']} | `{row['term']}` | {float(row['log_odds']):.3f} | "
                f"{row['count_in_label']} | {row['count_outside_label']} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The corpus contains explicit surface cues such as emoticons, mentions, "
            "hashtags, laughter markers, punctuation, and repeated characters. "
            "These cues should be discussed as distant-supervision artifacts when "
            "interpreting very high neural scores.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="data_artifact_analysis")
def main(cfg: DictConfig) -> None:
    _ensure_dataset(cfg.dataset)
    train_dataset = _load_split(cfg.dataset, "train")
    test_dataset = _load_split(cfg.dataset, "test")
    run_dir = _prepare_run_dir(
        output_dir=_project_path(str(cfg.output_dir)),
        run_id=str(cfg.run_id),
        overwrite=bool(cfg.overwrite),
    )

    train_rows = _split_rows("train", train_dataset)
    test_rows = _split_rows("test", test_dataset)
    all_rows = train_rows + test_rows
    train_profile = _profile_dataset(train_rows)
    test_profile = _profile_dataset(test_rows)
    duplicates = _duplicate_summary(train_dataset.examples, test_dataset.examples)
    cue_rows = _cue_rows(all_rows)
    terms = _term_rows(
        all_rows,
        min_count=int(cfg.min_term_count),
        top_n=int(cfg.top_terms_per_label),
    )

    payload = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "train_profile": train_profile,
        "test_profile": test_profile,
        "duplicates": duplicates,
    }
    _write_json(run_dir / "reports" / "artifact_analysis.json", payload)
    _write_csv(
        run_dir / "tables" / "cue_prevalence.csv",
        cue_rows,
        ["split", "label", "cue", "count", "total", "share"],
    )
    _write_csv(
        run_dir / "tables" / "label_associated_terms.csv",
        terms,
        ["label", "term", "count_in_label", "count_outside_label", "total_count", "log_odds"],
    )
    if bool(cfg.make_figures):
        _plot_cue_prevalence(cue_rows, run_dir / "figures" / "cue_prevalence_test")
    _write_markdown(
        run_dir / "reports" / "artifact_analysis.md",
        train_profile=train_profile,
        test_profile=test_profile,
        duplicates=duplicates,
        cue_rows=cue_rows,
        term_rows=terms,
    )

    print(f"Dataset artifact analysis complete: {_display_path(run_dir)}")
    print(f"Summary: {_display_path(run_dir / 'reports' / 'artifact_analysis.md')}")


if __name__ == "__main__":
    main()
