from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core import app_models as app_model_registry  # noqa: E402
from pln_core.lexicon import LexiconDownloadError  # noqa: E402
from pln_core.recommender import Song, recommend_ranked  # noqa: E402
from pln_core.samples import SAMPLE_TEXTS  # noqa: E402

AppModelInfo = app_model_registry.AppModelInfo
AppPrediction = app_model_registry.AppPrediction
choose_default_model_id = app_model_registry.choose_default_model_id
discover_app_models = app_model_registry.discover_app_models
load_app_model = app_model_registry.load_app_model
predict_sentiment = app_model_registry.predict_sentiment

SESSION_KEYS_TO_CLEAR = (
    "text_input",
    "sample_choice",
    "last_prediction",
    "last_comparison_predictions",
    "recommendation_index",
)

MODE_SINGLE = "Classificar"
MODE_COMPARE = "Comparar"

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

SCORE_TRANSLATIONS: dict[str, str] = {
    "confianca": "Confiança",
    "margem": "Força",
    "escore simbolico": "Força",
    "saida": "Resposta",
}


def _fallback_default_comparison_model_ids(models: tuple[AppModelInfo, ...]) -> tuple[str, ...]:
    """Choose cleaned symbolic and TF-IDF modes if the helper is unavailable."""

    preferred_names = ("oplexicon_regex", "tfidf_logreg", "tfidf_linear_svm")
    selected: list[str] = []
    for model_name in preferred_names:
        for model in models:
            if model.model_name == model_name and model.text_treatment == "strip_emoticons_urls":
                selected.append(model.id)
                break
    return tuple(selected)


default_comparison_model_ids = getattr(
    app_model_registry,
    "default_comparison_model_ids",
    _fallback_default_comparison_model_ids,
)

st.set_page_config(
    page_title="PLN Core",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def initialize_session_state(models: tuple[AppModelInfo, ...]) -> None:
    st.session_state.setdefault("text_input", "")
    st.session_state.setdefault("sample_choice", None)
    st.session_state.setdefault("last_prediction", None)
    st.session_state.setdefault("last_comparison_predictions", ())
    st.session_state.setdefault("recommendation_index", 0)
    st.session_state.setdefault("app_mode", MODE_SINGLE)

    model_ids = {model.id for model in models}
    if st.session_state.get("model_choice") not in model_ids:
        st.session_state.model_choice = choose_default_model_id(models)

    comparison_choices = tuple(st.session_state.get("comparison_model_choices", ()))
    if not comparison_choices or not set(comparison_choices).issubset(model_ids):
        st.session_state.comparison_model_choices = default_comparison_model_ids(models)


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
        st.session_state.last_comparison_predictions = ()
        st.session_state.recommendation_index = 0


def on_model_change() -> None:
    st.session_state.last_prediction = None
    st.session_state.last_comparison_predictions = ()
    st.session_state.recommendation_index = 0


def on_mode_change() -> None:
    st.session_state.last_prediction = None
    st.session_state.last_comparison_predictions = ()
    st.session_state.recommendation_index = 0


def reset_analysis_state() -> None:
    for key in SESSION_KEYS_TO_CLEAR:
        st.session_state.pop(key, None)
    st.session_state.last_prediction = None
    st.session_state.last_comparison_predictions = ()
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
    except FileNotFoundError:
        st.error("Esse modo de análise não está disponível agora. Tente outro.")
        st.session_state.last_prediction = None


def compare_current_text(models: tuple[AppModelInfo, ...]) -> None:
    text = st.session_state.text_input.strip()
    if not text:
        st.warning("Escreva algum texto antes de comparar.")
        return
    if not models:
        st.warning("Selecione pelo menos um modelo para comparar.")
        return

    predictions: list[AppPrediction] = []
    failed: list[str] = []
    for model in models:
        try:
            resource = get_model_resource(model.id, model_resource_key(model))
            predictions.append(predict_sentiment(model, resource, text))
        except LexiconDownloadError:
            failed.append(f"{model.display_name}: indisponível agora")
        except FileNotFoundError:
            failed.append(f"{model.display_name}: indisponível agora")

    st.session_state.last_comparison_predictions = tuple(predictions)
    st.session_state.last_prediction = None
    st.session_state.recommendation_index = 0

    if failed:
        st.warning("Alguns modos não responderam agora: " + "; ".join(failed))


def translate_label(label: str) -> str:
    return LABEL_TRANSLATIONS.get(label, label)


def format_prediction_score(prediction: AppPrediction) -> str:
    if prediction.score_name == "confianca" and prediction.confidence is not None:
        return f"{prediction.confidence:.1%}"
    return f"{prediction.score:.3f}"


def translate_score_name(score_name: str) -> str:
    return SCORE_TRANSLATIONS.get(score_name, score_name)


def render_mode_selector() -> str:
    return str(
        st.segmented_control(
            "Ação",
            options=[MODE_SINGLE, MODE_COMPARE],
            key="app_mode",
            on_change=on_mode_change,
            selection_mode="single",
        )
    )


def render_model_selector(models: tuple[AppModelInfo, ...]) -> AppModelInfo:
    model_by_id = {model.id: model for model in models}
    options = list(model_by_id)

    def _label(model_id: str) -> str:
        return model_by_id[model_id].display_name

    if len(options) <= 5:
        st.segmented_control(
            "Como analisar",
            options=options,
            format_func=_label,
            key="model_choice",
            on_change=on_model_change,
            selection_mode="single",
        )
    else:
        st.selectbox(
            "Como analisar",
            options=options,
            format_func=_label,
            key="model_choice",
            on_change=on_model_change,
        )

    return model_by_id[str(st.session_state.model_choice)]


def render_comparison_selector(models: tuple[AppModelInfo, ...]) -> tuple[AppModelInfo, ...]:
    model_by_id = {model.id: model for model in models}
    cleaned_ids = [
        model.id for model in models if model.text_treatment == "strip_emoticons_urls"
    ]
    options = cleaned_ids or list(model_by_id)
    current_ids = tuple(st.session_state.get("comparison_model_choices", ()))
    if not current_ids or not set(current_ids).issubset(set(options)):
        st.session_state.comparison_model_choices = default_comparison_model_ids(models)

    st.pills(
        "Como comparar",
        options=options,
        format_func=lambda model_id: model_by_id[model_id].display_name,
        key="comparison_model_choices",
        on_change=on_model_change,
        selection_mode="multi",
    )

    selected_ids = tuple(st.session_state.get("comparison_model_choices", ()))
    return tuple(model_by_id[model_id] for model_id in selected_ids if model_id in model_by_id)


def render_model_card(model: AppModelInfo) -> None:
    with st.container(border=True):
        st.caption("Modo escolhido")
        st.markdown(f"### {model.display_name}")
        st.write(model.description)


def render_comparison_model_card(models: tuple[AppModelInfo, ...]) -> None:
    with st.container(border=True):
        st.caption("Modelos comparados")
        if not models:
            st.write("Nenhum modelo selecionado.")
            return

        rows = [
            {
                "modo": model.display_name,
                "ideia": model.description,
            }
            for model in models
        ]
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            column_config={
                "modo": st.column_config.TextColumn("modo", pinned=True),
                "ideia": st.column_config.TextColumn("ideia"),
            },
        )


def render_label_card(prediction: AppPrediction) -> None:
    color = LABEL_COLORS.get(prediction.label, "gray")
    cols = st.columns(3)

    with cols[0].container(border=True):
        st.caption("Sentimento")
        st.markdown(f"### :{color}[{translate_label(prediction.label)}]")

    with cols[1].container(border=True):
        st.caption(translate_score_name(prediction.score_name))
        st.markdown(f"### `{format_prediction_score(prediction)}`")

    with cols[2].container(border=True):
        st.caption("Modo")
        st.markdown(f"### {prediction.model.display_name}")


def render_text_card(prediction: AppPrediction) -> None:
    with st.container(border=True):
        st.caption("Sua frase")
        st.write(prediction.raw_text or "(vazio)")


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
            f"{translate_score_name(prediction.score_name).lower()} "
            f"{format_prediction_score(prediction)}"
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

    if st.toggle("Mostrar recomendação musical", value=False):
        st.space("medium")
        render_recommendations(prediction)


def render_comparison(predictions: tuple[AppPrediction, ...]) -> None:
    if not predictions:
        return

    cols = st.columns(len(predictions))
    for col, prediction in zip(cols, predictions, strict=True):
        color = LABEL_COLORS.get(prediction.label, "gray")
        with col.container(border=True):
            st.caption(prediction.model.display_name)
            st.markdown(f"### :{color}[{translate_label(prediction.label)}]")
            st.write(f"`{format_prediction_score(prediction)}`")

    rows = [
        {
            "modo": prediction.model.display_name,
            "sentimento": translate_label(prediction.label),
            "resposta": format_prediction_score(prediction),
        }
        for prediction in predictions
    ]

    st.space("medium")
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "modo": st.column_config.TextColumn("modo", pinned=True),
            "sentimento": st.column_config.TextColumn("sentimento"),
            "resposta": st.column_config.TextColumn("resposta"),
        },
    )


def main() -> None:
    models = discover_app_models(PROJECT_ROOT)
    initialize_session_state(models)

    _, page, _ = st.columns([1, 6, 1])

    with page:
        st.title("PLN Core", text_alignment="center")
        st.caption(
            "Digite uma frase e veja se ela parece positiva, negativa ou neutra.",
            text_alignment="center",
        )

        st.space("medium")
        app_mode = render_mode_selector()
        if app_mode == MODE_COMPARE:
            selected_models = render_comparison_selector(models)
            selected_model = None
        else:
            selected_model = render_model_selector(models)
            selected_models = ()

        if not any(model.is_classical for model in models):
            st.info(
                "Alguns modos de análise não estão disponíveis nesta versão do app.",
                icon=":material/info:",
            )

        st.space("medium")

        control_col, result_col = st.columns([1, 2], gap="large", vertical_alignment="top")

        with control_col:
            if selected_model is not None:
                render_model_card(selected_model)
            else:
                render_comparison_model_card(selected_models)

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
                    action_label = "Comparar" if app_mode == MODE_COMPARE else "Classificar"
                    analyze_clicked = st.form_submit_button(action_label, type="primary")

        if clear_clicked:
            reset_analysis_state()

        if analyze_clicked:
            if app_mode == MODE_COMPARE:
                compare_current_text(selected_models)
            elif selected_model is not None:
                analyze_current_text(selected_model)

        with result_col:
            if app_mode == MODE_COMPARE:
                comparison_predictions = tuple(st.session_state.last_comparison_predictions)
                if comparison_predictions:
                    st.space("medium")
                    render_comparison(comparison_predictions)
            else:
                prediction = st.session_state.last_prediction
                if prediction is not None:
                    st.space("medium")
                    render_prediction(prediction)


if __name__ == "__main__":
    main()
