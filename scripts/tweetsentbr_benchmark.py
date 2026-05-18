"""TweetSentBR benchmark: in-house analyzers vs external symbolic tool (LeIA).

Runs every analyzer registered in ``ANALYZER_REGISTRY`` plus the LeIA port of
VADER (the de-facto Portuguese lexicon-based baseline) on a balanced subset of
the few-shot test split of TweetSentBR (1,703 tweets after stratified sampling
of up to 600 per class with seed 42).

Outputs:
    * Markdown comparison table on stdout.
    * ``data/eval_reports/tweetsentbr_benchmark.md`` (same table).
    * Per-analyzer JSON report under ``data/eval_reports/tweetsentbr/``.

Run as::

    uv run python -m scripts.tweetsentbr_benchmark
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pln_core.eval.datasets import create_dataset  # noqa: E402
from pln_core.eval.metrics import EvaluationMetrics, compute_metrics  # noqa: E402
from pln_core.eval.runner import (  # noqa: E402
    EvaluationReport,
    run_evaluation,
    save_report_json,
)

REPORTS_DIR = REPO_ROOT / "data" / "eval_reports" / "tweetsentbr"


@dataclass(frozen=True, slots=True)
class Entry:
    """One row in the benchmark table."""

    name: str
    metrics: EvaluationMetrics
    elapsed: float
    notes: str


def _run_local(name: str, label: str, notes: str, kwargs: dict) -> Entry:
    """Evaluate an in-house analyzer (registered in ANALYZER_REGISTRY)."""

    report: EvaluationReport = run_evaluation(
        analyzer_name=name,
        dataset_name="tweetsentbr",
        dataset_kwargs={"split": "test", "balanced": True, "per_class": 600, "seed": 42},
        analyzer_kwargs=kwargs,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_report_json(report, REPORTS_DIR / f"{label.replace(' ', '_').replace('+', 'plus')}.json")
    return Entry(label, report.metrics, report.elapsed_seconds, notes)


def _run_leia() -> Entry:
    """Evaluate LeIA (Léxico para Inferência Adaptada, the PT-BR VADER port)."""

    import time

    from LeIA import SentimentIntensityAnalyzer

    dataset = create_dataset(
        "tweetsentbr", split="test", balanced=True, per_class=600, seed=42
    )
    sia = SentimentIntensityAnalyzer()
    expected: list[str] = []
    predicted: list[str] = []
    start = time.perf_counter()
    for example in dataset.examples:
        compound = sia.polarity_scores(example.text)["compound"]
        if compound >= 0.05:
            predicted.append("positive")
        elif compound <= -0.05:
            predicted.append("negative")
        else:
            predicted.append("neutral")
        expected.append(example.label)
    elapsed = time.perf_counter() - start
    return Entry(
        "LeIA (Almeida 2018, PT-BR VADER)",
        compute_metrics(expected=expected, predicted=predicted),
        elapsed,
        "external lexicon-based tool, default thresholds (±0.05)",
    )


def _format_rows(entries: list[Entry]) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in entries:
        pc = entry.metrics.per_class_f1
        rows.append([
            entry.name,
            str(entry.metrics.total),
            f"{entry.metrics.accuracy:.3f}",
            f"{entry.metrics.macro_f1:.3f}",
            f"{pc.get('positive', 0.0):.3f}",
            f"{pc.get('negative', 0.0):.3f}",
            f"{pc.get('neutral', 0.0):.3f}",
            f"{entry.elapsed:.1f}s",
            entry.notes,
        ])
    return rows


def _to_markdown(rows: list[list[str]]) -> str:
    header = ["analyzer", "n", "acc", "F1m", "F1 pos", "F1 neg", "F1 neu", "time", "notes"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    entries: list[Entry] = []

    entries.append(_run_local("majority", "majority baseline",
                              "always predicts the majority class", {"label": "positive"}))
    entries.append(_run_local("seed", "seed lexicon",
                              "30-word didactic seed lexicon + regex tokenizer", {}))
    entries.append(_run_local("oplexicon_regex", "OpLexicon (regex)",
                              "OpLexicon v3.0 + regex tokenizer", {}))
    entries.append(_run_local("oplexicon", "OpLexicon (spaCy lemmas)",
                              "OpLexicon v3.0 + spaCy pt_core_news_sm", {}))
    entries.append(_run_leia())
    entries.append(_run_local("oplexicon_tweet", "OpLexicon + tweet rules",
                              "OpLexicon v3.0 + slang/emoji + tweet normalization",
                              {"positive_threshold": 0.5, "negative_threshold": -0.5}))
    entries.append(_run_local("oplexicon_tweet_plus", "OpLexicon + SentiLex + tweet (best)",
                              "Multi-lexicon fusion (OpLexicon + SentiLex-PT 02 + slang/emoji)",
                              {"positive_threshold": 0.3, "negative_threshold": -0.3,
                               "use_negation": False, "use_intensifier": True,
                               "use_caps": False, "use_contrast": False, "sentilex_weight": 0.5}))

    rows = _format_rows(entries)
    table = _to_markdown(rows)
    print("\n" + table + "\n")

    REPORTS_DIR.parent.mkdir(parents=True, exist_ok=True)
    summary = REPORTS_DIR.parent / "tweetsentbr_benchmark.md"
    summary.write_text(
        "# TweetSentBR symbolic benchmark\n\n"
        "Balanced 3-class test set: up to 600 tweets per class (seed=42), "
        "1,703 examples in total.\n\n"
        + table
        + "\n",
        encoding="utf-8",
    )
    (REPORTS_DIR.parent / "tweetsentbr_benchmark.json").write_text(
        json.dumps(
            [
                {
                    "analyzer": e.name,
                    "accuracy": e.metrics.accuracy,
                    "macro_f1": e.metrics.macro_f1,
                    "per_class_f1": e.metrics.per_class_f1,
                    "confusion": {
                        f"{t}->{p}": c for (t, p), c in e.metrics.confusion.items()
                    },
                    "elapsed_seconds": e.elapsed,
                    "notes": e.notes,
                }
                for e in entries
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
