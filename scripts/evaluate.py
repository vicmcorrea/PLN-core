"""Qualitative benchmark of the symbolic sentiment pipeline.

Runs a fixed set of short Brazilian Portuguese sentences through both the
seed lexicon and the OpLexicon + spaCy production stack, prints per-sentence
results, and reports accuracy for each configuration. The same sentences and
expected labels are reported in the project report under section "Resultados".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pln_core.lexicon import OPLEXICON_LEXICON_SOURCE, load_lexicon
from pln_core.pipeline import SymbolicSentimentAnalyzer
from pln_core.tokenizers import tokenize_spacy_pt_lemmas

BENCHMARK_CASES: tuple[tuple[str, str], ...] = (
    ("Eu amei o filme, foi muito bom!", "positive"),
    ("Nao gostei do app, esta bem confuso e bugado.", "negative"),
    ("O arquivo tem quatro paginas e duas tabelas.", "neutral"),
    ("O comeco foi ruim, mas o final foi otimo.", "positive"),
    ("Nao foi bom.", "negative"),
    ("Adorei o atendimento, super rapido!", "positive"),
    ("Pessimo produto, nao recomendo.", "negative"),
    ("Maravilhoso, amei demais!", "positive"),
    ("O servico esta horrivel, horrivel mesmo.", "negative"),
    ("Recebi o pedido hoje.", "neutral"),
    ("Top demais, recomendo!", "positive"),
    ("Que dia chato.", "negative"),
    ("Filme muito bom, recomendo!", "positive"),
    ("App lento e travando.", "negative"),
    ("A entrega foi tranquila.", "positive"),
    ("Estou triste com o resultado.", "negative"),
    ("O produto chegou na data.", "neutral"),
    ("Nao gostei nem um pouco.", "negative"),
    ("Maravilha, ficou perfeito!", "positive"),
    ("Confuso, mal feito e caro.", "negative"),
)


@dataclass(frozen=True, slots=True)
class CaseResult:
    text: str
    expected: str
    predicted: str
    score: float


def _run(name: str, analyzer: SymbolicSentimentAnalyzer) -> list[CaseResult]:
    print(f"\n=== {name} ===")
    results: list[CaseResult] = []
    for text, expected in BENCHMARK_CASES:
        prediction = analyzer.analyze(text)
        hit = prediction.label == expected
        marker = "OK " if hit else "X  "
        print(
            f"  {marker} pred={prediction.label:8s} exp={expected:8s} "
            f"score={prediction.score:+.3f} | {text}"
        )
        results.append(
            CaseResult(
                text=text,
                expected=expected,
                predicted=prediction.label,
                score=prediction.score,
            )
        )
    return results


def _summary(name: str, results: list[CaseResult]) -> None:
    total = len(results)
    correct = sum(1 for case in results if case.predicted == case.expected)
    print(f"\n  {name}: {correct}/{total} = {correct / total:.0%}")
    errors = [case for case in results if case.predicted != case.expected]
    if not errors:
        return
    print("  errors by (expected -> predicted):")
    counter = Counter((case.expected, case.predicted) for case in errors)
    for (expected, predicted), count in counter.most_common():
        print(f"    {expected:8s} -> {predicted:8s} : {count}")


def main() -> None:
    seed_analyzer = SymbolicSentimentAnalyzer()
    seed_results = _run("SEED (didactic baseline)", seed_analyzer)

    op_lexicon = load_lexicon(source=OPLEXICON_LEXICON_SOURCE)
    production_analyzer = SymbolicSentimentAnalyzer(
        lexicon=op_lexicon,
        tokenizer=tokenize_spacy_pt_lemmas,
    )
    production_results = _run(
        "OpLexicon v3.0 + spaCy pt_core_news_sm (production)",
        production_analyzer,
    )

    _summary("SEED", seed_results)
    _summary("OpLexicon + spaCy", production_results)


if __name__ == "__main__":
    main()
