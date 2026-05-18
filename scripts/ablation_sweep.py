"""Ablation sweep on TweetSentBR.

Runs the symbolic pipeline under many configurations (threshold sweep, rule
ablation, tokenizer swap, majority baseline) and prints a single comparison
table. The goal is to find which knobs actually move the needle on a real
corpus, so we can prioritize improvements.

Usage:
    uv run python scripts/ablation_sweep.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core.eval.runner import run_evaluation  # noqa: E402


@dataclass(frozen=True, slots=True)
class Config:
    label: str
    analyzer: str
    analyzer_kwargs: dict[str, object]


def _all_rules_off() -> dict[str, object]:
    return dict(
        use_negation=False,
        use_intensifier=False,
        use_diminisher=False,
        use_contrast=False,
        use_exclamation=False,
    )


CONFIGS: list[Config] = [
    Config(
        label="majority_positive",
        analyzer="majority",
        analyzer_kwargs={"label": "positive"},
    ),
    Config(
        label="oplexicon_baseline_T=0.75 (current)",
        analyzer="oplexicon",
        analyzer_kwargs={},
    ),
    Config(
        label="oplexicon_T=0.50",
        analyzer="oplexicon",
        analyzer_kwargs={"positive_threshold": 0.5, "negative_threshold": -0.5},
    ),
    Config(
        label="oplexicon_T=0.25",
        analyzer="oplexicon",
        analyzer_kwargs={"positive_threshold": 0.25, "negative_threshold": -0.25},
    ),
    Config(
        label="oplexicon_T=0.01 (any signal)",
        analyzer="oplexicon",
        analyzer_kwargs={"positive_threshold": 0.01, "negative_threshold": -0.01},
    ),
    Config(
        label="oplexicon_no_rules_T=0.75",
        analyzer="oplexicon",
        analyzer_kwargs=_all_rules_off(),
    ),
    Config(
        label="oplexicon_no_rules_T=0.01",
        analyzer="oplexicon",
        analyzer_kwargs={
            **_all_rules_off(),
            "positive_threshold": 0.01,
            "negative_threshold": -0.01,
        },
    ),
    Config(
        label="oplexicon_no_negation",
        analyzer="oplexicon",
        analyzer_kwargs={"use_negation": False},
    ),
    Config(
        label="oplexicon_no_intensifier",
        analyzer="oplexicon",
        analyzer_kwargs={"use_intensifier": False},
    ),
    Config(
        label="oplexicon_no_contrast",
        analyzer="oplexicon",
        analyzer_kwargs={"use_contrast": False},
    ),
    Config(
        label="oplexicon_no_exclamation",
        analyzer="oplexicon",
        analyzer_kwargs={"use_exclamation": False},
    ),
    Config(
        label="oplexicon_no_diminisher",
        analyzer="oplexicon",
        analyzer_kwargs={"use_diminisher": False},
    ),
    Config(
        label="oplexicon_regex_T=0.75",
        analyzer="oplexicon_regex",
        analyzer_kwargs={},
    ),
    Config(
        label="oplexicon_regex_T=0.01",
        analyzer="oplexicon_regex",
        analyzer_kwargs={"positive_threshold": 0.01, "negative_threshold": -0.01},
    ),
]


def main() -> None:
    print(f"running {len(CONFIGS)} configurations on TweetSentBR (test, 2010 tweets)\n")
    header = (
        f"{'configuration':45s} | {'acc':>6s} | "
        f"{'F1 mac':>7s} | {'F1 pos':>7s} | {'F1 neg':>7s} | {'F1 neu':>7s} | "
        f"{'pred dist (P/N/Neu)':>22s}"
    )
    print(header)
    print("-" * len(header))

    rows: list[tuple[str, float, float]] = []
    for cfg in CONFIGS:
        report = run_evaluation(
            analyzer_name=cfg.analyzer,
            dataset_name="tweetsentbr",
            analyzer_kwargs=cfg.analyzer_kwargs,
        )
        metrics = report.metrics
        per_class = metrics.per_class_f1
        pred_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for case in report.predictions:
            pred_counts[case.predicted] += 1
        dist = (
            f"{pred_counts['positive']}/"
            f"{pred_counts['negative']}/"
            f"{pred_counts['neutral']}"
        )
        print(
            f"{cfg.label:45s} | "
            f"{metrics.accuracy:6.3f} | "
            f"{metrics.macro_f1:7.3f} | "
            f"{per_class['positive']:7.3f} | "
            f"{per_class['negative']:7.3f} | "
            f"{per_class['neutral']:7.3f} | "
            f"{dist:>22s}"
        )
        rows.append((cfg.label, metrics.accuracy, metrics.macro_f1))

    print("\nsorted by accuracy:")
    for label, acc, mf1 in sorted(rows, key=lambda r: r[1], reverse=True):
        print(f"  {acc:6.3f}  F1m={mf1:.3f}  {label}")


if __name__ == "__main__":
    main()
