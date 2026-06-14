"""Small free-text probe used to choose the public non-symbolic app model."""

from __future__ import annotations

import csv
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib  # noqa: E402
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: E402

from pln_core.app_models import HuggingFaceSentimentAnalyzer  # noqa: E402
from pln_core.eval.datasets.base import VALID_LABELS  # noqa: E402
from pln_core.eval.text_treatments import apply_text_treatment  # noqa: E402
from pln_core.factory import build_production_analyzer  # noqa: E402

APP_CLASSICAL_RUN_ID = "20260614_113447_389024"
TEXT_TREATMENT = "strip_emoticons_urls"
TABULARISAI_MODEL_ID = "tabularisai/multilingual-sentiment-analysis"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "etapa2_subsymbolic" / "free_text_probe"


@dataclass(frozen=True, slots=True)
class FreeTextSample:
    index: int
    text: str
    label: str


@dataclass(frozen=True, slots=True)
class ProbeSystem:
    name: str
    predict: Callable[[list[str]], list[str]]


FREE_TEXT_SAMPLES = (
    FreeTextSample(1, "O app ficou maravilhoso, simples e divertido.", "positive"),
    FreeTextSample(2, "Gostei muito do atendimento, foi rápido e educado.", "positive"),
    FreeTextSample(3, "A apresentação ficou linda e fácil de entender.", "positive"),
    FreeTextSample(4, "A música me deixou animado e com vontade de sorrir.", "positive"),
    FreeTextSample(5, "Esse jogo está muito legal, parabéns pelo trabalho.", "positive"),
    FreeTextSample(6, "A interface ficou bonita e funciona bem.", "positive"),
    FreeTextSample(7, "Estou feliz com o resultado final.", "positive"),
    FreeTextSample(8, "Eu odiei esse app, ficou péssimo e muito confuso.", "negative"),
    FreeTextSample(9, "O sistema travou várias vezes e perdi minha resposta.", "negative"),
    FreeTextSample(10, "Não gostei da experiência, achei lenta e irritante.", "negative"),
    FreeTextSample(11, "A tela está bagunçada e difícil de usar.", "negative"),
    FreeTextSample(12, "Fiquei frustrado porque nada funcionou direito.", "negative"),
    FreeTextSample(13, "Esse resultado me deixou triste e preocupado.", "negative"),
    FreeTextSample(14, "A recomendação não combina comigo e ficou ruim.", "negative"),
    FreeTextSample(15, "O aplicativo mostra uma tela com três botões.", "neutral"),
    FreeTextSample(16, "A reunião começa às nove horas na sala dois.", "neutral"),
    FreeTextSample(17, "O arquivo tem quatro páginas e duas tabelas.", "neutral"),
    FreeTextSample(18, "A frase foi digitada no campo de texto.", "neutral"),
    FreeTextSample(19, "Hoje é domingo e a feira acontece à tarde.", "neutral"),
    FreeTextSample(20, "O relatório contém uma seção de resultados.", "neutral"),
)


def _project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _classical_model_path(model_name: str) -> Path:
    return _project_path(
        Path("data")
        / "app_models"
        / "etapa2_subsymbolic"
        / APP_CLASSICAL_RUN_ID
        / f"{model_name}.joblib"
    )


def _transformer_model_path(run_id: str, model_name: str) -> Path:
    return _project_path(
        Path("data")
        / "models"
        / "etapa2_subsymbolic"
        / "transformers"
        / run_id
        / model_name
    )


def _predict_symbolic(texts: list[str]) -> list[str]:
    analyzer = build_production_analyzer()
    return [analyzer.analyze(text).label for text in texts]


def _load_classical_system(display_name: str, model_name: str) -> ProbeSystem:
    model = joblib.load(_classical_model_path(model_name))

    def predict(texts: list[str]) -> list[str]:
        processed = [apply_text_treatment(text, TEXT_TREATMENT) for text in texts]
        return [str(label) for label in model.predict(processed)]

    return ProbeSystem(display_name, predict)


def _normalize_hf_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    if "negative" in normalized or normalized in {"negativo", "neg"}:
        return "negative"
    if "neutral" in normalized or normalized in {"neutro", "neu"}:
        return "neutral"
    if "positive" in normalized or normalized in {"positivo", "pos"}:
        return "positive"
    return normalized


def _best_transformer_labels(outputs: Any) -> list[str]:
    labels: list[str] = []
    for output in outputs:
        rows = output if isinstance(output, list) else [output]
        best = max(rows, key=lambda row: float(row.get("score", 0.0)))
        labels.append(_normalize_hf_label(str(best.get("label", ""))))
    return labels


def _load_transformer_system(display_name: str, run_id: str, model_name: str) -> ProbeSystem:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Install transformer dependencies with `uv sync --extra transformers`."
        ) from exc

    model_path = _transformer_model_path(run_id, model_name)
    classifier = pipeline(
        "text-classification",
        model=str(model_path),
        tokenizer=str(model_path),
        top_k=None,
        truncation=True,
        device=-1,
    )

    def predict(texts: list[str]) -> list[str]:
        processed = [apply_text_treatment(text, TEXT_TREATMENT) for text in texts]
        return _best_transformer_labels(classifier(processed))

    return ProbeSystem(display_name, predict)


def _load_tabularisai_system() -> ProbeSystem:
    analyzer = HuggingFaceSentimentAnalyzer(TABULARISAI_MODEL_ID)

    def predict(texts: list[str]) -> list[str]:
        return [str(label) for label in analyzer.predict(texts)]

    return ProbeSystem("TabularisAI pronto", predict)


def _systems() -> tuple[ProbeSystem, ...]:
    return (
        ProbeSystem("OpLexicon com regras", _predict_symbolic),
        _load_classical_system("TF-IDF + Reg. Logística", "tfidf_logreg"),
        _load_classical_system("TF-IDF + SVM linear", "tfidf_linear_svm"),
        _load_transformer_system("XLM-R fine-tuned", "20260612_132734_004665", "xlm_roberta_base"),
        _load_transformer_system(
            "Albertina fine-tuned",
            "20260612_134601_518530",
            "albertina_ptbr_100m",
        ),
        _load_tabularisai_system(),
    )


def _metric_row(system: str, expected: list[str], predicted: list[str]) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected,
        predicted,
        labels=list(VALID_LABELS),
        average="macro",
        zero_division=0,
    )
    errors = sum(exp != pred for exp, pred in zip(expected, predicted, strict=True))
    return {
        "system": system,
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "errors": errors,
        "total": len(expected),
    }


def run_probe() -> Path:
    run_dir = OUTPUT_DIR / _timestamp()
    texts = [sample.text for sample in FREE_TEXT_SAMPLES]
    expected = [sample.label for sample in FREE_TEXT_SAMPLES]
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for system in _systems():
        predicted = system.predict(texts)
        metric_rows.append(_metric_row(system.name, expected, predicted))
        for sample, label in zip(FREE_TEXT_SAMPLES, predicted, strict=True):
            prediction_rows.append(
                {
                    "system": system.name,
                    "index": sample.index,
                    "text": sample.text,
                    "expected": sample.label,
                    "predicted": label,
                    "correct": sample.label == label,
                }
            )

    _write_json(
        run_dir / "reports" / "metrics.json",
        {
            "sample_size": len(FREE_TEXT_SAMPLES),
            "sample_source": (
                "Free short sentences written by the project team, about five per "
                "participant, with no fixed topic and no attempt to imitate tweets."
            ),
            "labels": list(VALID_LABELS),
            "class_counts": {
                label: expected.count(label)
                for label in VALID_LABELS
            },
            "metrics": metric_rows,
        },
    )
    _write_csv(
        run_dir / "reports" / "metrics.csv",
        metric_rows,
        ["system", "accuracy", "macro_precision", "macro_recall", "macro_f1", "errors", "total"],
    )
    _write_csv(
        run_dir / "predictions" / "predictions.csv",
        prediction_rows,
        ["system", "index", "text", "expected", "predicted", "correct"],
    )
    return run_dir


def main() -> None:
    run_dir = run_probe()
    print(f"Free-text probe saved to {run_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
