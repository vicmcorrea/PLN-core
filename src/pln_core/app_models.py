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
    PRODUCTION_ANALYZER_LABEL,
    PRODUCTION_ANALYZER_NAME,
    build_production_analyzer,
)
from pln_core.pipeline import AnalysisResult, SymbolicSentimentAnalyzer

DEFAULT_CLASSICAL_MODELS_DIR = Path("data/models/etapa2_subsymbolic")
DEFAULT_CLASSICAL_REPORTS_DIR = Path("outputs/etapa2_subsymbolic/benchmark_suite")
DEFAULT_APP_TEXT_TREATMENT = "strip_emoticons_urls"

MODEL_DISPLAY_NAMES = {
    "oplexicon_regex": "OpLexicon regex",
    "tfidf_logreg": "TF-IDF + Regressao Logistica",
    "tfidf_linear_svm": "TF-IDF + Linear SVM",
}

TEXT_TREATMENT_DISPLAY_NAMES = {
    "raw": "texto bruto",
    "none": "texto bruto",
    "strip_emoticons_urls": "sem emoticons/URLs",
    "strip_social_source_cues": "sem pistas sociais",
    "strip_emoticons": "sem emoticons",
    "strip_urls": "sem URLs",
}


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


def text_treatment_label(text_treatment: str) -> str:
    """Return a human-friendly treatment label."""

    return TEXT_TREATMENT_DISPLAY_NAMES.get(text_treatment, text_treatment)


def model_label(model_name: str) -> str:
    """Return a human-friendly model label."""

    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def _symbolic_models() -> tuple[AppModelInfo, ...]:
    return (
        AppModelInfo(
            id=f"symbolic:{DEFAULT_APP_TEXT_TREATMENT}",
            display_name=(
                f"{MODEL_DISPLAY_NAMES[PRODUCTION_ANALYZER_NAME]} "
                f"({text_treatment_label(DEFAULT_APP_TEXT_TREATMENT)})"
            ),
            family="symbolic",
            model_name=PRODUCTION_ANALYZER_NAME,
            text_treatment=DEFAULT_APP_TEXT_TREATMENT,
            description=(
                f"{PRODUCTION_ANALYZER_LABEL}; aplica o mesmo tratamento textual "
                "preferido para comparacao justa com a Etapa 2."
            ),
        ),
        AppModelInfo(
            id="symbolic:raw",
            display_name=f"{MODEL_DISPLAY_NAMES[PRODUCTION_ANALYZER_NAME]} (texto bruto)",
            family="symbolic",
            model_name=PRODUCTION_ANALYZER_NAME,
            text_treatment="raw",
            description=f"{PRODUCTION_ANALYZER_LABEL}; versao simbolica no texto original.",
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
    display_name = (
        f"{model_label(model_name)} ({text_treatment_label(text_treatment)}, run {run_id})"
    )
    report_path = _project_path(project_root, payload.get("report_path"))
    description = (
        "Modelo classico treinado no split Kaggle comum. "
        f"Tratamento de texto: {text_treatment_label(text_treatment)}."
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


def discover_app_models(project_root: Path) -> tuple[AppModelInfo, ...]:
    """Discover symbolic and saved app-loadable classical models."""

    models = list(_symbolic_models())
    classical_root = project_root / DEFAULT_CLASSICAL_MODELS_DIR
    if classical_root.exists():
        artifacts = sorted(classical_root.glob("*/*.joblib"), reverse=True)
        models.extend(_classical_info_from_artifact(project_root, path) for path in artifacts)
    return tuple(models)


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
            ):
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
