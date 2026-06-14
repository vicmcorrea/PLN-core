"""Download and organize the shared Kaggle corpus used by both project stages."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "augustop/portuguese-tweets-for-sentiment-analysis"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "portuguese-tweets-for-sentiment-analysis"
EXPECTED_FILES = (
    Path("TrainingDatasets") / "Train3Classes.csv",
    Path("TestDatasets") / "Test3classes.csv",
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _missing_files(target_dir: Path) -> list[Path]:
    return [
        relative_path
        for relative_path in EXPECTED_FILES
        if not (target_dir / relative_path).exists()
    ]


def _download_dataset(slug: str, target_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "Install project dependencies with `uv sync` before downloading from Kaggle."
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(target_dir), unzip=True, quiet=False)


def prepare_dataset(slug: str, target_dir: Path, force: bool) -> None:
    target_dir = target_dir.expanduser().resolve()
    missing_before = _missing_files(target_dir)
    if not missing_before and not force:
        print(f"Dataset already prepared in {_display_path(target_dir)}")
        return

    if force and target_dir.exists():
        shutil.rmtree(target_dir)

    print(f"Downloading {slug}")
    print(f"Destination: {_display_path(target_dir)}")
    _download_dataset(slug, target_dir)

    missing_after = _missing_files(target_dir)
    if missing_after:
        missing = ", ".join(str(path) for path in missing_after)
        raise FileNotFoundError(
            f"Kaggle download finished, but expected split file(s) were not found: {missing}"
        )

    print("Dataset ready.")
    for relative_path in EXPECTED_FILES:
        print(f"- {_display_path(target_dir / relative_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Kaggle Portuguese tweets sentiment corpus into data/raw."
    )
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle dataset slug.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory where the Kaggle files should be extracted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the existing target directory before downloading again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(slug=args.slug, target_dir=args.target_dir, force=bool(args.force))


if __name__ == "__main__":
    main()
