"""Hand-curated benchmark used as a quick smoke test for the pipeline.

The same twenty sentences originally lived in ``scripts/evaluate.py`` and are
referenced in the project report as the didactic baseline.
"""

from __future__ import annotations

from pln_core.eval.datasets.base import EvalDataset, EvalExample
from pln_core.eval.datasets.registry import DATASET_REGISTRY

SAMPLE_CASES: tuple[tuple[str, str], ...] = (
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


@DATASET_REGISTRY.register("sample")
def load_sample_dataset() -> EvalDataset:
    """Return the 20 hand-written PT-BR sentences used as a sanity benchmark."""

    examples = tuple(EvalExample(text=text, label=label) for text, label in SAMPLE_CASES)
    return EvalDataset(
        name="sample",
        description="20 hand-written PT-BR sentences (didactic baseline)",
        examples=examples,
    )
