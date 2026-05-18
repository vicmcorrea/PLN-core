"""Legacy entry point that reproduces the original 20-sentence smoke test.

The actual logic lives in :mod:`pln_core.eval`; this script is kept so callers
who relied on ``python scripts/evaluate.py`` keep working. For new experiments
prefer the Hydra entry point at ``run/pipeline/analysis/run_evaluation.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core.eval.runner import run_evaluation  # noqa: E402


def _print_summary(name: str, accuracy: float, correct: int, total: int) -> None:
    print(f"\n  {name}: {correct}/{total} = {accuracy:.0%}")


def main() -> None:
    for analyzer_name, label in (
        ("seed", "SEED (didactic baseline)"),
        ("oplexicon", "OpLexicon v3.0 + spaCy pt_core_news_sm (production)"),
    ):
        report = run_evaluation(
            analyzer_name=analyzer_name,
            dataset_name="sample",
        )
        print(f"\n=== {label} ===")
        for case in report.predictions:
            marker = "OK " if case.predicted == case.expected else "X  "
            print(
                f"  {marker} pred={case.predicted:8s} exp={case.expected:8s} "
                f"score={case.score:+.3f} | {case.text}"
            )
        _print_summary(
            analyzer_name,
            report.metrics.accuracy,
            report.metrics.correct,
            report.metrics.total,
        )


if __name__ == "__main__":
    main()
