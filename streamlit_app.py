from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core.app_models import (  # noqa: E402
    AppModelInfo,
    AppPrediction,
    choose_default_model_id,
    discover_app_models,
    load_app_model,
    model_label,
    predict_sentiment,
    text_treatment_label,
)
from pln_core.lexicon import LexiconDownloadError  # noqa: E402
from pln_core.pipeline import AnalysisResult  # noqa: E402
from pln_core.recommender import Song, recommend_ranked  # noqa: E402
from pln_core.samples import SAMPLE_TEXTS  # noqa: E402

SESSION_KEYS_TO_CLEAR = (
    "text_input",
    "sample_choice",
    "last_prediction",
    "recommendation_index",
)

LABEL_COLORS: dict[str, str] = {
    "positive": "green",
    "negative": "red",
    "neutral": "gray",
}

LABEL_TRANSLATIONS: dict[str, str] = {
    "positive": "positivo",
    "negative": "negativo",
    "neutral": "neutro",
}

SAMPLE_LABELS: dict[str, str] = {
    "positive": "Exemplo positivo",
    "negative": "Exemplo negativo",
    "neutral": "Exemplo neutro",
}

RULE_TRANSLATIONS: dict[str, str] = {
    "negation": "negação",
    "intensifier": "intensificador",
    "diminisher": "atenuador",
    "pre-contrast": "pré-contraste",
    "post-contrast": "pós-contraste",
    "exclamation": "exclamação",
}

LABEL_ORDER = ("positive", "negative", "neutral")

st.set_page_config(
    page_title="PLN Core",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def initialize_session_state(models: tuple[AppModelInfo, ...]) -> None:
    st.session_state.setdefault("text_input", "")
    st.session_state.setdefault("sample_choice", None)
    st.session_state.setdefault("last_prediction", None)
    st.session_state.setdefault("recommendation_index", 0)

    model_ids = {model.id for model in models}
    if st.session_state.get("model_choice") not in model_ids:
        st.session_state.model_choice = choose_default_model_id(models)


@st.cache_resource(show_spinner="Carregando modelo...")
def get_model_resource(model_id: str, artifact_path: str) -> Any:
    """Load and cache one model resource by id and artifact path."""

    models = {model.id: model for model in discover_app_models(PROJECT_ROOT)}
    return load_app_model(models[model_id])


def model_resource_key(model: AppModelInfo) -> str:
    if model.artifact_path is None:
        return ""
    return str(model.artifact_path)


def on_sample_change() -> None:
    sample = st.session_state.sample_choice
    if sample:
        st.session_state.text_input = SAMPLE_TEXTS[sample]
        st.session_state.last_prediction = None
        st.session_state.recommendation_index = 0


def on_model_change() -> None:
    st.session_state.last_prediction = None
    st.session_state.recommendation_index = 0


def reset_analysis_state() -> None:
    for key in SESSION_KEYS_TO_CLEAR:
        st.session_state.pop(key, None)
    st.session_state.last_prediction = None
    st.session_state.recommendation_index = 0
    st.session_state.text_input = ""
    st.session_state.sample_choice = None
    st.rerun()


def analyze_current_text(model: AppModelInfo) -> None:
    text = st.session_state.text_input.strip()
    if not text:
        st.warning("Escreva algum texto antes de classificar.")
        return

    try:
        resource = get_model_resource(model.id, model_resource_key(model))
        st.session_state.last_prediction = predict_sentiment(model, resource, text)
        st.session_state.recommendation_index = 0
    except LexiconDownloadError:
        st.error("Não foi possível carregar o OpLexicon. Verifique a conexão e tente novamente.")
        st.session_state.last_prediction = None
    except FileNotFoundError as exc:
        st.error(f"Artefato do modelo não encontrado: {exc}")
        st.session_state.last_prediction = None


def translate_label(label: str) -> str:
    return LABEL_TRANSLATIONS.get(label, label)


def translate_rules(rules: tuple[str, ...]) -> str:
    if not rules:
        return "base"
    return ", ".join(RULE_TRANSLATIONS.get(rule, rule) for rule in rules)


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def format_prediction_score(prediction: AppPrediction) -> str:
    if prediction.score_name == "confianca" and prediction.confidence is not None:
        return f"{prediction.confidence:.1%}"
    return f"{prediction.score:.3f}"


def render_model_selector(models: tuple[AppModelInfo, ...]) -> AppModelInfo:
    model_by_id = {model.id: model for model in models}
    options = list(model_by_id)

    def _label(model_id: str) -> str:
        return model_by_id[model_id].display_name

    if len(options) <= 5:
        st.segmented_control(
            "Modelo",
            options=options,
            format_func=_label,
            key="model_choice",
            on_change=on_model_change,
            selection_mode="single",
        )
    else:
        st.selectbox(
            "Modelo",
            options=options,
            format_func=_label,
            key="model_choice",
            on_change=on_model_change,
        )

    return model_by_id[str(st.session_state.model_choice)]


def render_model_card(model: AppModelInfo) -> None:
    with st.container(border=True):
        st.caption("Modelo selecionado")
        st.markdown(f"### {model_label(model.model_name)}")
        st.write(model.description)
        st.caption(f"Tratamento textual: `{model.text_treatment}`")

        if model.metrics:
            cols = st.columns(2)
            cols[0].metric("Acurácia no teste", format_metric(model.metrics.get("accuracy")))
            cols[1].metric("Macro-F1 no teste", format_metric(model.metrics.get("macro_f1")))
        elif model.is_classical:
            st.caption("Sem métricas locais encontradas para este artefato.")

        if model.artifact_path is not None:
            st.caption(f"Artefato: `{model.artifact_path.relative_to(PROJECT_ROOT)}`")
        if model.run_id:
            st.caption(f"Run: `{model.run_id}`")
        if model.text_treatment == "raw":
            st.warning(
                "Este modelo usa o texto bruto. No corpus Kaggle, emoticons e URLs vazam "
                "pistas fortes do rótulo; use estes resultados só como diagnóstico.",
                icon=":material/warning:",
            )


def render_label_card(prediction: AppPrediction) -> None:
    color = LABEL_COLORS.get(prediction.label, "gray")
    cols = st.columns(3)

    with cols[0].container(border=True):
        st.caption("Rótulo")
        st.markdown(f"### :{color}[{translate_label(prediction.label)}]")

    with cols[1].container(border=True):
        st.caption(prediction.score_name)
        st.markdown(f"### `{format_prediction_score(prediction)}`")

    with cols[2].container(border=True):
        st.caption("Tratamento")
        st.markdown(f"### {text_treatment_label(prediction.model.text_treatment)}")


def render_text_card(prediction: AppPrediction) -> None:
    with st.container(border=True):
        st.caption("Texto original")
        st.write(prediction.raw_text or "(vazio)")

        if prediction.processed_text != prediction.raw_text:
            st.caption("Texto entregue ao modelo")
            st.write(prediction.processed_text or "(vazio após limpeza)")


def render_class_scores(prediction: AppPrediction) -> None:
    if not prediction.class_scores:
        return

    rows = [
        {
            "rótulo": translate_label(label),
            "valor": prediction.class_scores[label],
        }
        for label in LABEL_ORDER
        if label in prediction.class_scores
    ]
    with st.container(border=True):
        st.caption("Saídas por classe")
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            column_config={
                "rótulo": st.column_config.TextColumn("rótulo", pinned=True),
                "valor": st.column_config.NumberColumn("valor", format="%.4f"),
            },
        )


def render_symbolic_text_card(result: AnalysisResult) -> None:
    with st.container(border=True):
        st.caption("Texto normalizado")
        st.write(result.normalized_text or "(vazio)")

        st.caption("Lemas usados na busca")
        if result.tokens:
            st.markdown(" ".join(f"`{token}`" for token in result.tokens))
        else:
            st.write("(nenhum)")


def render_symbolic_matches(result: AnalysisResult) -> None:
    if not result.matched_terms:
        st.info("Nenhum lema encontrado no léxico, o escore é zero.")
        return

    rows = [
        {
            "lema": match.token,
            "posição": match.position,
            "escore base": match.base_score,
            "escore ajustado": match.adjusted_score,
            "regras": translate_rules(match.applied_rules),
        }
        for match in result.matched_terms
    ]

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "lema": st.column_config.TextColumn("lema", pinned=True),
            "posição": st.column_config.NumberColumn("posição", format="%d"),
            "escore base": st.column_config.NumberColumn("escore base", format="%.3f"),
            "escore ajustado": st.column_config.NumberColumn("escore ajustado", format="%.3f"),
            "regras": st.column_config.TextColumn("regras"),
        },
    )


def _recommendation_prev() -> None:
    st.session_state.recommendation_index = max(int(st.session_state.recommendation_index) - 1, 0)


def _recommendation_next(max_idx: int) -> None:
    st.session_state.recommendation_index = min(
        int(st.session_state.recommendation_index) + 1,
        max_idx,
    )


def render_recommendations(prediction: AppPrediction) -> None:
    songs: tuple[Song, ...] = recommend_ranked(prediction.label, prediction.score)
    with st.container(border=True):
        st.subheader("música recomendada")
        if not songs:
            st.caption("nenhuma música disponível para esse rótulo.")
            return

        idx = int(st.session_state.recommendation_index)
        if idx >= len(songs):
            idx = 0
            st.session_state.recommendation_index = 0
        song = songs[idx]
        st.caption(
            f"{translate_label(prediction.label)} · "
            f"{prediction.score_name} {format_prediction_score(prediction)}"
        )
        if len(songs) > 1:
            st.caption(f"opção {idx + 1} de {len(songs)}")
        st.markdown(f"### {song.title}")
        st.caption(song.artist)
        if len(songs) >= 2:
            last = len(songs) - 1
            with st.container(horizontal=True, horizontal_alignment="distribute"):
                st.button(
                    "voltar",
                    on_click=_recommendation_prev,
                    disabled=idx == 0,
                    key=f"rec_prev_{prediction.model.id}_{idx}",
                    use_container_width=True,
                )
                st.button(
                    "outra sugestão",
                    on_click=_recommendation_next,
                    kwargs={"max_idx": last},
                    disabled=idx >= last,
                    key=f"rec_next_{prediction.model.id}_{idx}",
                    use_container_width=True,
                )
        st.video(song.youtube_url)
        st.link_button("abrir no youtube", song.search_url, width="stretch")


def render_prediction(prediction: AppPrediction) -> None:
    render_label_card(prediction)
    st.space("medium")
    render_text_card(prediction)
    st.space("medium")
    render_class_scores(prediction)

    if prediction.symbolic_result is not None:
        st.space("medium")
        render_symbolic_text_card(prediction.symbolic_result)
        st.space("medium")
        render_symbolic_matches(prediction.symbolic_result)

    if st.toggle("Mostrar recomendação musical", value=False):
        st.space("medium")
        render_recommendations(prediction)


def main() -> None:
    models = discover_app_models(PROJECT_ROOT)
    initialize_session_state(models)

    _, page, _ = st.columns([1, 6, 1])

    with page:
        st.title("PLN Core", text_alignment="center")
        st.caption(
            "Classificação de sentimentos em português brasileiro com modelos simbólicos e TF-IDF.",
            text_alignment="center",
        )

        st.space("medium")
        selected_model = render_model_selector(models)

        if not any(model.is_classical for model in models):
            st.info(
                "Nenhum artefato TF-IDF foi encontrado em `data/models/etapa2_subsymbolic/`. "
                "Rode a suíte clássica da Etapa 2 para habilitar o modelo padrão do app.",
                icon=":material/info:",
            )

        st.space("medium")

        control_col, result_col = st.columns([1, 2], gap="large", vertical_alignment="top")

        with control_col:
            render_model_card(selected_model)

        with result_col:
            st.pills(
                "Exemplos",
                options=list(SAMPLE_TEXTS.keys()),
                format_func=lambda key: SAMPLE_LABELS[key],
                key="sample_choice",
                on_change=on_sample_change,
                label_visibility="collapsed",
                selection_mode="single",
            )

            with st.form("analysis_form", border=False):
                st.text_area(
                    "Texto",
                    key="text_input",
                    height=150,
                    placeholder="Digite uma frase curta em português brasileiro...",
                    label_visibility="collapsed",
                )

                with st.container(horizontal=True, horizontal_alignment="distribute"):
                    clear_clicked = st.form_submit_button("Limpar")
                    analyze_clicked = st.form_submit_button("Classificar", type="primary")

        if clear_clicked:
            reset_analysis_state()

        if analyze_clicked:
            analyze_current_text(selected_model)

        prediction = st.session_state.last_prediction
        if prediction is not None:
            with result_col:
                st.space("medium")
                render_prediction(prediction)


main()
