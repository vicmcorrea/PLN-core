"""SentiLex-PT 02 loader (Silva, Carvalho and Sarmento, 2012; PROPOR 2010).

SentiLex-PT 02 is a Portuguese sentiment lexicon with 7,014 lemmas and 82,347
inflected forms, manually or semi-automatically annotated with polarity in
``{-1, 0, +1}``. We load the inflected file because the project tokenizes
without lemmatization by default; the inflected file already covers most
surface forms encountered in tweets.

File format (latin-1, one entry per line)::

    inflected,lemma.PoS=Adj;FLEX=fs;TG=HUM:N0;POL:N0=-1;ANOT=JALC

We extract ``inflected`` and ``POL:N0`` (or ``POL:N1``) and discard the rest.
Entries with polarity ``0`` are kept out so they do not dilute the score.
"""

from __future__ import annotations

import re
from pathlib import Path

from pln_core.text_utils import fold_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENTILEX_FLEX_PATH = PROJECT_ROOT / "data" / "external" / "SentiLex-flex-PT02.txt"

POL_PATTERN = re.compile(r"POL:N[01]=(-?\d+)")


def load_sentilex_flex(path: str | Path | None = None) -> dict[str, float]:
    """Parse SentiLex-flex-PT02 into a ``{token: score}`` mapping.

    Tokens are accent-folded so they match the rest of the project's pipeline.
    When a single surface form appears more than once with conflicting
    polarities (rare), the polarities are averaged.
    """

    target = Path(path) if path else SENTILEX_FLEX_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"SentiLex flex file not found at {target}. Download it from "
            "https://github.com/sillasgonzaga/lexiconPT (data-raw/)."
        )

    raw: dict[str, list[float]] = {}
    with target.open("r", encoding="latin-1") as handle:
        for line in handle:
            if "," not in line:
                continue
            surface, rest = line.split(",", 1)
            match = POL_PATTERN.search(rest)
            if not match:
                continue
            polarity = float(match.group(1))
            if polarity == 0:
                continue
            key = fold_text(surface.strip())
            if not key:
                continue
            raw.setdefault(key, []).append(polarity)

    return {token: sum(scores) / len(scores) for token, scores in raw.items()}
