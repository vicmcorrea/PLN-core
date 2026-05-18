from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pln_core.lexicon import load_lexicon
from pln_core.text_utils import normalize_text
from pln_core.tokenizers import tokenize_custom

NEGATIONS = {"nao", "nem", "nunca", "jamais", "sem"}
NEGATION_SCOPE_TERMINATORS = {".", ",", ";", ":", "!", "?", "mas", "porem", "porém"}
NEGATION_SCOPE_TOKENS = 4
CAPS_BOOST = 1.4
CAPS_MIN_LENGTH = 3
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
        tokenizer: Callable[[str], list[str]] | None = None,
        positive_threshold: float = 0.75,
        negative_threshold: float = -0.75,
        use_negation: bool = True,
        use_intensifier: bool = True,
        use_diminisher: bool = True,
        use_contrast: bool = True,
        use_exclamation: bool = True,
        use_caps: bool = False,
    ) -> None:
        self.lexicon = dict(lexicon or load_lexicon())
        self.tokenizer = tokenizer or tokenize_custom
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.use_negation = use_negation
        self.use_intensifier = use_intensifier
        self.use_diminisher = use_diminisher
        self.use_contrast = use_contrast
        self.use_exclamation = use_exclamation
        self.use_caps = use_caps

    def analyze(self, text: str) -> AnalysisResult:
        """Analyze text and return token matches, score, and final label."""

        normalized_text = normalize_text(text)
        tokens = self.tokenizer(text)
        caps_tokens = self._caps_token_set(text) if self.use_caps else set()
        matches: list[MatchDetail] = []
        contrast_index = self._find_contrast_index(tokens) if self.use_contrast else None
        if self.use_exclamation:
            exclamation_count = min(text.count("!"), 3)
            exclamation_multiplier = (
                1 + (0.05 * exclamation_count) if exclamation_count else 1.0
            )
        else:
            exclamation_count = 0
            exclamation_multiplier = 1.0

        negation_active_until = -1

        for index, token in enumerate(tokens):
            if self.use_negation and token in NEGATIONS:
                negation_active_until = index + NEGATION_SCOPE_TOKENS
                continue

            if (
                self.use_negation
                and index <= negation_active_until
                and token in NEGATION_SCOPE_TERMINATORS
            ):
                negation_active_until = -1

            base_score = self.lexicon.get(token)
            if base_score is None:
                continue

            adjusted_score = base_score
            applied_rules: list[str] = []
            previous_token = tokens[index - 1] if index > 0 else None

            if self.use_negation and index <= negation_active_until:
                adjusted_score *= -1
                applied_rules.append("negation")

            if self.use_intensifier and previous_token in INTENSIFIERS:
                adjusted_score *= INTENSIFIERS[previous_token]
                applied_rules.append("intensifier")

            if self.use_diminisher and previous_token in DIMINISHERS:
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

            if self.use_caps and token in caps_tokens:
                adjusted_score *= CAPS_BOOST
                applied_rules.append("caps")

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

    def _caps_token_set(self, text: str) -> set[str]:
        """Return the set of accent-folded tokens that appear in ALL CAPS.

        A token counts as "all caps" only when it is at least
        :data:`CAPS_MIN_LENGTH` letters long and has no lowercase letter in
        the original text, to avoid mistaking acronyms and 2-letter words.
        """

        from pln_core.text_utils import fold_text

        caps: set[str] = set()
        for raw in text.split():
            stripped = "".join(ch for ch in raw if ch.isalpha())
            if (
                len(stripped) >= CAPS_MIN_LENGTH
                and stripped.isupper()
            ):
                caps.add(fold_text(stripped))
        return caps

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
        strong = [m for m in matches if abs(m.base_score) >= 0.8]
        if strong and -0.2 < score < self.positive_threshold:
            avg = sum(m.adjusted_score for m in strong) / len(strong)
            if avg >= 0.5:
                return "positive"
            if avg <= -0.5:
                return "negative"
        return "neutral"
