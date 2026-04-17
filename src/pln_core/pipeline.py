from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pln_core.lexicon import load_lexicon
from pln_core.text_utils import normalize_text, tokenize

NEGATIONS = {"nao", "nem", "nunca", "jamais", "sem"}
INTENSIFIERS = {
    "muito": 1.6,
    "super": 1.5,
    "bem": 1.2,
    "demais": 1.4,
    "realmente": 1.2,
    "bastante": 1.3,
}
DIMINISHERS = {
    "pouco": 0.6,
    "meio": 0.75,
    "quase": 0.8,
}
CONTRAST_MARKERS = {"mas", "porem", "contudo", "entretanto"}


@dataclass(frozen=True, slots=True)
class MatchDetail:
    token: str
    position: int
    base_score: float
    adjusted_score: float
    applied_rules: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    text: str
    normalized_text: str
    tokens: tuple[str, ...]
    score: float
    label: str
    matched_terms: tuple[MatchDetail, ...]


class SymbolicSentimentAnalyzer:
    """Lexicon and rule based sentiment analyzer for short Portuguese texts."""

    def __init__(
        self,
        lexicon: Mapping[str, float] | None = None,
        positive_threshold: float = 0.75,
        negative_threshold: float = -0.75,
    ) -> None:
        self.lexicon = dict(lexicon or load_lexicon())
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def analyze(self, text: str) -> AnalysisResult:
        """Analyze text and return token matches, score, and final label."""

        normalized_text = normalize_text(text)
        tokens = tokenize(text)
        matches: list[MatchDetail] = []
        contrast_index = self._find_contrast_index(tokens)
        exclamation_count = min(text.count("!"), 3)
        exclamation_multiplier = 1 + (0.05 * exclamation_count) if exclamation_count else 1.0

        for index, token in enumerate(tokens):
            base_score = self.lexicon.get(token)
            if base_score is None:
                continue

            adjusted_score = base_score
            applied_rules: list[str] = []
            previous_tokens = tokens[max(0, index - 3) : index]
            previous_token = tokens[index - 1] if index > 0 else None

            if any(previous in NEGATIONS for previous in previous_tokens):
                adjusted_score *= -1
                applied_rules.append("negation")

            if previous_token in INTENSIFIERS:
                adjusted_score *= INTENSIFIERS[previous_token]
                applied_rules.append("intensifier")

            if previous_token in DIMINISHERS:
                adjusted_score *= DIMINISHERS[previous_token]
                applied_rules.append("diminisher")

            if contrast_index is not None and index < contrast_index:
                adjusted_score *= 0.7
                applied_rules.append("pre-contrast")
            elif contrast_index is not None and index > contrast_index:
                adjusted_score *= 1.3
                applied_rules.append("post-contrast")

            if exclamation_count:
                adjusted_score *= exclamation_multiplier
                applied_rules.append("exclamation")

            matches.append(
                MatchDetail(
                    token=token,
                    position=index,
                    base_score=base_score,
                    adjusted_score=round(adjusted_score, 3),
                    applied_rules=tuple(applied_rules),
                )
            )

        score = round(sum(match.adjusted_score for match in matches), 3)
        label = self._label_for(score, matches)

        return AnalysisResult(
            text=text,
            normalized_text=normalized_text,
            tokens=tuple(tokens),
            score=score,
            label=label,
            matched_terms=tuple(matches),
        )

    def _find_contrast_index(self, tokens: list[str]) -> int | None:
        contrast_positions = [
            index for index, token in enumerate(tokens) if token in CONTRAST_MARKERS
        ]
        if not contrast_positions:
            return None
        return contrast_positions[-1]

    def _label_for(self, score: float, matches: list[MatchDetail]) -> str:
        if not matches:
            return "neutral"
        if score >= self.positive_threshold:
            return "positive"
        if score <= self.negative_threshold:
            return "negative"
        return "neutral"
