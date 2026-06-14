"""Model discovery and prediction helpers for the Streamlit app."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import joblib

from pln_core.eval.text_treatments import apply_text_treatment
from pln_core.factory import (
    PRODUCTION_ANALYZER_NAME,
    build_production_analyzer,
)
from pln_core.pipeline import AnalysisResult, SymbolicSentimentAnalyzer

DEPLOY_CLASSICAL_MODELS_DIR = Path("data/app_models/etapa2_subsymbolic")
LOCAL_CLASSICAL_MODELS_DIR = Path("data/models/etapa2_subsymbolic")
DEFAULT_CLASSICAL_REPORTS_DIR = Path("outputs/etapa2_subsymbolic/benchmark_suite")
DEFAULT_APP_TEXT_TREATMENT = "strip_emoticons_urls"
CLASSICAL_MODEL_SEARCH_DIRS = (
    DEPLOY_CLASSICAL_MODELS_DIR,
    LOCAL_CLASSICAL_MODELS_DIR,
)

MODEL_DISPLAY_NAMES = {
    "oplexicon_regex": "OpLexicon regex",
    "tfidf_logreg": "TF-IDF + Reg. Logística",
    "tfidf_linear_svm": "TF-IDF + SVM linear",
    "distilbert_multilingual": "DistilBERT multilingual",
    "xlm_roberta_base": "XLM-R base",
    "albertina_ptbr_100m": "Albertina 100M pt-BR",
}

MODEL_TECHNICAL_NAMES = {
    "oplexicon_regex": "OpLexicon v3.0 + regras simbólicas",
    "tfidf_logreg": "TF-IDF de palavras + Regressão Logística",
    "tfidf_linear_svm": "TF-IDF de palavras + SVM linear",
    "distilbert_multilingual": "distilbert/distilbert-base-multilingual-cased fine-tuned",
    "xlm_roberta_base": "FacebookAI/xlm-roberta-base fine-tuned",
    "albertina_ptbr_100m": "PORTULAN/albertina-100m-portuguese-ptbr-encoder fine-tuned",
}

TEXT_TREATMENT_DISPLAY_NAMES = {
    "raw": "texto bruto",
    "none": "texto bruto",
    "strip_emoticons_urls": "sem emoticons/URLs",
    "strip_social_source_cues": "sem pistas sociais",
    "strip_emoticons": "sem emoticons",
    "strip_urls": "sem URLs",
}

SYMBOLIC_BEST_METRICS = {
    "accuracy": 0.3697,
    "macro_f1": 0.3668,
}

BEST_TRANSFORMER_MODELS = (
    {
        "model_name": "distilbert_multilingual",
        "run_id": "20260612_132142_936441",
        "accuracy": 0.7465,
        "macro_f1": 0.7385,
    },
    {
        "model_name": "xlm_roberta_base",
        "run_id": "20260612_132734_004665",
        "accuracy": 0.7586,
        "macro_f1": 0.7494,
    },
    {
        "model_name": "albertina_ptbr_100m",
        "run_id": "20260612_134601_518530",
        "accuracy": 0.7822,
        "macro_f1": 0.7808,
    },
)

APP_MODEL_ORDER = (
    "oplexicon_regex",
    "tfidf_logreg",
    "tfidf_linear_svm",
    "distilbert_multilingual",
    "xlm_roberta_base",
    "albertina_ptbr_100m",
)


class PredictivePipeline(Protocol):
    """Small protocol for sklearn-like pipelines used by the app."""

    classes_: Any

    def predict(self, texts: list[str]) -> Any:
        """Predict labels for a batch of texts."""

    def predict_proba(self, texts: list[str]) -> Any:
        """Optionally return class probabilities."""

    def decision_function(self, texts: list[str]) -> Any:
        """Optionally return class margins."""


@dataclass(frozen=True, slots=True)
class AppModelInfo:
    """Metadata needed to show and load one app model."""

    id: str
    display_name: str
    family: str
    model_name: str
    text_treatment: str
    description: str
    run_id: str = ""
    artifact_path: Path | None = None
    metadata_path: Path | None = None
    report_path: Path | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_symbolic(self) -> bool:
        return self.family == "symbolic"

    @property
    def is_classical(self) -> bool:
        return self.family == "classical"

    @property
    def can_predict(self) -> bool:
        return self.is_symbolic or self.artifact_path is not None


@dataclass(frozen=True, slots=True)
class AppPrediction:
    """Uniform app prediction for symbolic and subsymbolic models."""

    model: AppModelInfo
    raw_text: str
    processed_text: str
    label: str
    score: float
    score_name: str
    class_scores: Mapping[str, float] = field(default_factory=dict)
    symbolic_result: AnalysisResult | None = None

    @property
    def confidence(self) -> float | None:
        return self.class_scores.get(self.label)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _project_path(project_root: Path, path_like: str | Path | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root / path


def _metric_summary(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return {}
    names = ("accuracy", "macro_f1")
    summary: dict[str, float] = {}
    for name in names:
        value = metrics.get(name)
        if value is not None:
            summary[name] = float(value)
    return summary


def _metric_text(metrics: Mapping[str, float]) -> str:
    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    parts: list[str] = []
    if macro_f1 is not None:
        parts.append(f"macro F1 {macro_f1:.3f}")
    if accuracy is not None:
        parts.append(f"accuracy {accuracy:.3f}")
    if not parts:
        return ""
    return " Resultado validado: " + ", ".join(parts) + "."


def text_treatment_label(text_treatment: str) -> str:
    """Return a human-friendly treatment label."""

    return TEXT_TREATMENT_DISPLAY_NAMES.get(text_treatment, text_treatment)


def model_label(model_name: str) -> str:
    """Return a human-friendly model label."""

    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def model_technical_name(model_name: str) -> str:
    """Return the concrete model/version used behind a friendly label."""

    return MODEL_TECHNICAL_NAMES.get(model_name, model_name)


def _display_name_for_model(model_name: str, text_treatment: str) -> str:
    return model_label(model_name)


def _symbolic_models() -> tuple[AppModelInfo, ...]:
    technical_name = model_technical_name(PRODUCTION_ANALYZER_NAME)
    return (
        AppModelInfo(
            id=f"symbolic:{DEFAULT_APP_TEXT_TREATMENT}",
            display_name=MODEL_DISPLAY_NAMES[PRODUCTION_ANALYZER_NAME],
            family="symbolic",
            model_name=PRODUCTION_ANALYZER_NAME,
            text_treatment=DEFAULT_APP_TEXT_TREATMENT,
            description=(
                f"Modelo usado: {technical_name}. Melhor versão válida com "
                f"{text_treatment_label(DEFAULT_APP_TEXT_TREATMENT)}. Procura "
                "palavras positivas e negativas e combina regras simples para "
                f"decidir o sentimento.{_metric_text(SYMBOLIC_BEST_METRICS)}"
            ),
            metrics=SYMBOLIC_BEST_METRICS,
        ),
    )


def _report_payload_for_artifact(
    project_root: Path,
    run_id: str,
    model_name: str,
) -> dict[str, Any]:
    report_path = (
        project_root
        / DEFAULT_CLASSICAL_REPORTS_DIR
        / run_id
        / "reports"
        / model_name
        / "report.json"
    )
    if not report_path.exists():
        return {}
    payload = _read_json(report_path)
    payload.setdefault("report_path", str(report_path.relative_to(project_root)))
    payload.setdefault("run_id", run_id)
    return payload


def _classical_info_from_artifact(project_root: Path, artifact_path: Path) -> AppModelInfo:
    run_id = artifact_path.parent.name
    model_name = artifact_path.stem
    metadata_path = artifact_path.with_suffix(".metadata.json")

    payload: dict[str, Any]
    if metadata_path.exists():
        payload = _read_json(metadata_path)
    else:
        payload = _report_payload_for_artifact(project_root, run_id, model_name)

    text_treatment = str(payload.get("text_treatment") or "raw")
    display_name = _display_name_for_model(model_name, text_treatment)
    report_path = _project_path(project_root, payload.get("report_path"))
    technical_name = model_technical_name(model_name)
    treatment_label = text_treatment_label(text_treatment)
    description = (
        f"Modelo usado: {technical_name}. Melhor versão válida com "
        f"{treatment_label}. Versão do app: {run_id}. Aprende padrões de "
        f"palavras em frases curtas e responde rapidamente.{_metric_text(_metric_summary(payload))}"
    )

    return AppModelInfo(
        id=f"classical:{run_id}:{model_name}",
        display_name=display_name,
        family="classical",
        model_name=model_name,
        text_treatment=text_treatment,
        description=description,
        run_id=run_id,
        artifact_path=artifact_path,
        metadata_path=metadata_path if metadata_path.exists() else None,
        report_path=report_path if report_path and report_path.exists() else None,
        metrics=_metric_summary(payload),
    )


def _benchmark_transformer_models() -> tuple[AppModelInfo, ...]:
    models: list[AppModelInfo] = []
    treatment_label = text_treatment_label(DEFAULT_APP_TEXT_TREATMENT)
    for payload in BEST_TRANSFORMER_MODELS:
        model_name = str(payload["model_name"])
        metrics = {
            "accuracy": float(payload["accuracy"]),
            "macro_f1": float(payload["macro_f1"]),
        }
        technical_name = model_technical_name(model_name)
        run_id = str(payload["run_id"])
        models.append(
            AppModelInfo(
                id=f"transformer:{run_id}:{model_name}",
                display_name=model_label(model_name),
                family="transformer",
                model_name=model_name,
                text_treatment=DEFAULT_APP_TEXT_TREATMENT,
                description=(
                    f"Modelo usado: {technical_name}. Melhor versão válida com "
                    f"{treatment_label}. Esta rodada fine-tuned fica cadastrada "
                    "como referência da Etapa 2. Para classificar frases com "
                    "ela, é preciso carregar o modelo fine-tuned completo."
                    f"{_metric_text(metrics)}"
                ),
                run_id=run_id,
                metrics=metrics,
            )
        )
    return tuple(models)


def _model_order_key(model: AppModelInfo) -> tuple[int, str]:
    try:
        index = APP_MODEL_ORDER.index(model.model_name)
    except ValueError:
        index = len(APP_MODEL_ORDER)
    return index, model.display_name


def discover_app_models(project_root: Path) -> tuple[AppModelInfo, ...]:
    """Discover the best valid app and benchmark models."""

    models = list(_symbolic_models())
    candidates: dict[str, list[AppModelInfo]] = {
        "tfidf_logreg": [],
        "tfidf_linear_svm": [],
    }
    for relative_root in CLASSICAL_MODEL_SEARCH_DIRS:
        classical_root = project_root / relative_root
        if not classical_root.exists():
            continue

        artifacts = sorted(classical_root.glob("*/*.joblib"), reverse=True)
        for artifact_path in artifacts:
            model_name = artifact_path.stem
            if model_name not in candidates:
                continue
            info = _classical_info_from_artifact(project_root, artifact_path)
            if info.text_treatment == DEFAULT_APP_TEXT_TREATMENT:
                candidates[model_name].append(info)

    for model_name in ("tfidf_logreg", "tfidf_linear_svm"):
        if candidates[model_name]:
            models.append(candidates[model_name][0])

    models.extend(_benchmark_transformer_models())
    return tuple(sorted(models, key=_model_order_key))


def default_comparison_model_ids(models: tuple[AppModelInfo, ...]) -> tuple[str, ...]:
    """Return the preferred cleaned models for side-by-side comparison."""

    preferred_names = (
        PRODUCTION_ANALYZER_NAME,
        "tfidf_logreg",
        "tfidf_linear_svm",
    )
    selected: list[str] = []
    for model_name in preferred_names:
        for model in models:
            if (
                model.model_name == model_name
                and model.text_treatment == DEFAULT_APP_TEXT_TREATMENT
                and model.can_predict
            ):
                selected.append(model.id)
                break
    return tuple(selected)


def choose_default_model_id(models: tuple[AppModelInfo, ...]) -> str:
    """Prefer the cleaned TF-IDF LogReg artifact when it exists."""

    preferences = (
        ("classical", "tfidf_logreg", DEFAULT_APP_TEXT_TREATMENT),
        ("classical", "tfidf_linear_svm", DEFAULT_APP_TEXT_TREATMENT),
        ("classical", "tfidf_logreg", "strip_social_source_cues"),
        ("classical", "tfidf_logreg", "raw"),
        ("symbolic", PRODUCTION_ANALYZER_NAME, DEFAULT_APP_TEXT_TREATMENT),
    )
    for family, model_name, text_treatment in preferences:
        for model in models:
            if (
                model.family == family
                and model.model_name == model_name
                and model.text_treatment == text_treatment
                and model.can_predict
            ):
                return model.id
    for model in models:
        if model.can_predict:
            return model.id
    if not models:
        raise ValueError("no app models available")
    return models[0].id


def load_app_model(info: AppModelInfo) -> SymbolicSentimentAnalyzer | PredictivePipeline:
    """Load one model resource for prediction."""

    if info.is_symbolic:
        return build_production_analyzer()
    if info.artifact_path is None:
        raise FileNotFoundError(f"model artifact is not configured for {info.id}")
    return joblib.load(info.artifact_path)


def _as_float_list(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list) and values and isinstance(values[0], list):
        values = values[0]
    elif isinstance(values, tuple) and values and isinstance(values[0], tuple):
        values = values[0]
    elif not isinstance(values, (list, tuple)):
        values = [values]
    return [float(value) for value in values]


def _class_names(model: PredictivePipeline) -> list[str]:
    classes = getattr(model, "classes_", ())
    if hasattr(classes, "tolist"):
        classes = classes.tolist()
    return [str(label) for label in classes]


def _signed_sentiment_score(label: str, strength: float) -> float:
    bounded = max(0.0, min(1.0, abs(float(strength))))
    if label == "negative":
        return -bounded
    if label == "neutral":
        return 0.0
    return bounded


def _squash_margin(margin: float) -> float:
    return margin / (1.0 + abs(margin))


def _class_scores(
    model: PredictivePipeline,
    text: str,
    label: str,
) -> tuple[str, dict[str, float], float]:
    classes = _class_names(model)
    if hasattr(model, "predict_proba"):
        values = _as_float_list(model.predict_proba([text]))
        scores = {class_name: value for class_name, value in zip(classes, values, strict=False)}
        confidence = scores.get(label, max(scores.values()) if scores else 0.0)
        return "confianca", scores, _signed_sentiment_score(label, confidence)

    if hasattr(model, "decision_function"):
        values = _as_float_list(model.decision_function([text]))
        scores = {class_name: value for class_name, value in zip(classes, values, strict=False)}
        margin = scores.get(label, max(values, key=abs) if values else 0.0)
        return "margem", scores, _signed_sentiment_score(label, _squash_margin(margin))

    return "saida", {}, 0.0


def predict_sentiment(
    info: AppModelInfo,
    model: SymbolicSentimentAnalyzer | PredictivePipeline,
    text: str,
) -> AppPrediction:
    """Predict sentiment with a symbolic or app-loadable classical model."""

    processed_text = apply_text_treatment(text, info.text_treatment)
    if info.is_symbolic:
        if not isinstance(model, SymbolicSentimentAnalyzer):
            raise TypeError("symbolic model info requires a SymbolicSentimentAnalyzer")
        result = model.analyze(processed_text)
        return AppPrediction(
            model=info,
            raw_text=text,
            processed_text=processed_text,
            label=result.label,
            score=result.score,
            score_name="escore simbolico",
            symbolic_result=result,
        )

    label = str(model.predict([processed_text])[0])
    score_name, class_scores, score = _class_scores(model, processed_text, label)
    return AppPrediction(
        model=info,
        raw_text=text,
        processed_text=processed_text,
        label=label,
        score=score,
        score_name=score_name,
        class_scores=class_scores,
    )
