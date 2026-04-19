from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pln_core.lexicon import (
    OPLEXICON_LEXICON_SOURCE,
    LexiconDownloadError,
    load_lexicon,
)
from pln_core.pipeline import AnalysisResult, SymbolicSentimentAnalyzer
from pln_core.recommender import Song, recommend_ranked
from pln_core.samples import ANALYZER_STACK_LABEL, SAMPLE_TEXTS
from pln_core.tokenizers import SPACY_PT_TOKENIZER_SOURCE, get_tokenizer

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

st.set_page_config(
    page_title="PLN Core",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.session_state.setdefault("text_input", "")
st.session_state.setdefault("sample_choice", None)
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("recommendation_index", 0)


@st.cache_resource(show_spinner="Carregando OpLexicon v3.0...")
def get_analyzer() -> SymbolicSentimentAnalyzer:
    """Build and cache the project analyzer with the fixed stack."""

    lexicon = load_lexicon(source=OPLEXICON_LEXICON_SOURCE)
    tokenizer = get_tokenizer(SPACY_PT_TOKENIZER_SOURCE)
    return SymbolicSentimentAnalyzer(lexicon=lexicon, tokenizer=tokenizer)


def on_sample_change() -> None:
    sample = st.session_state.sample_choice
    if sample:
        st.session_state.text_input = SAMPLE_TEXTS[sample]


def translate_label(label: str) -> str:
    return LABEL_TRANSLATIONS.get(label, label)


def translate_rules(rules: tuple[str, ...]) -> str:
    if not rules:
        return "base"
    return ", ".join(RULE_TRANSLATIONS.get(rule, rule) for rule in rules)


def render_label_card(result: AnalysisResult) -> None:
    color = LABEL_COLORS.get(result.label, "gray")
    cols = st.columns(2)

    with cols[0].container(border=True):
        st.caption("Rótulo")
        st.markdown(f"### :{color}[{translate_label(result.label)}]")

    with cols[1].container(border=True):
        st.caption("Escore")
        st.markdown(f"### `{result.score}`")


def render_text_card(result: AnalysisResult) -> None:
    with st.container(border=True):
        st.caption("Texto original")
        st.write(result.text or "(vazio)")

        st.caption("Texto normalizado")
        st.write(result.normalized_text or "(vazio)")

        st.caption("Tokens")
        if result.tokens:
            st.markdown(" ".join(f"`{token}`" for token in result.tokens))
        else:
            st.write("(nenhum)")


def render_matches(result: AnalysisResult) -> None:
    if not result.matched_terms:
        st.info("Nenhum token encontrado no léxico, o escore é zero.")
        return

    rows = [
        {
            "token": match.token,
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
            "token": st.column_config.TextColumn("token", pinned=True),
            "posição": st.column_config.NumberColumn("posição", format="%d"),
            "escore base": st.column_config.NumberColumn("escore base", format="%.3f"),
            "escore ajustado": st.column_config.NumberColumn(
                "escore ajustado", format="%.3f"
            ),
            "regras": st.column_config.TextColumn("regras"),
        },
    )


def _recommendation_prev() -> None:
    st.session_state.recommendation_index = max(
        int(st.session_state.recommendation_index) - 1, 0
    )


def _recommendation_next(max_idx: int) -> None:
    st.session_state.recommendation_index = min(
        int(st.session_state.recommendation_index) + 1, max_idx
    )


def render_recommendations(result: AnalysisResult) -> None:
    songs: tuple[Song, ...] = recommend_ranked(result.label, result.score)
    rec_key = str(id(result))
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
        st.caption(f"{translate_label(result.label)} · escore {result.score}")
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
                    key=f"rec_prev_{rec_key}",
                    use_container_width=True,
                )
                st.button(
                    "outra sugestão",
                    on_click=_recommendation_next,
                    kwargs={"max_idx": last},
                    disabled=idx >= last,
                    key=f"rec_next_{rec_key}",
                    use_container_width=True,
                )
        st.video(song.youtube_url)
        st.link_button("abrir no youtube", song.search_url, width="stretch")


def render_result(result: AnalysisResult) -> None:
    render_label_card(result)
    st.space("medium")
    render_text_card(result)
    st.space("medium")
    render_matches(result)


def render_recommendation_panel(result: AnalysisResult | None) -> None:
    if result is None:
        with st.container(border=True):
            st.subheader("música recomendada")
            st.caption("aparece após você analisar um texto.")
        return

    render_recommendations(result)


def main() -> None:
    _, page, _ = st.columns([1, 6, 1])

    with page:
        st.title("PLN Core", text_alignment="center")
        st.caption(
            "Análise simbólica de sentimentos para português brasileiro.",
            text_alignment="center",
        )
        st.markdown(
            f"**Pipeline:** {ANALYZER_STACK_LABEL}",
            text_alignment="center",
        )

        st.space("medium")

        recommendation_col, analyzer_col = st.columns(
            [1, 2], gap="large", vertical_alignment="top"
        )

        with analyzer_col:
            st.pills(
                "Exemplos",
                options=list(SAMPLE_TEXTS.keys()),
                format_func=lambda key: SAMPLE_LABELS[key],
                key="sample_choice",
                on_change=on_sample_change,
                label_visibility="collapsed",
                selection_mode="single",
            )

            st.text_area(
                "Texto",
                key="text_input",
                height=140,
                placeholder="Digite uma frase curta em português brasileiro...",
                label_visibility="collapsed",
            )

            with st.container(horizontal=True, horizontal_alignment="distribute"):
                clear_clicked = st.button("Limpar")
                analyze_clicked = st.button("Analisar", type="primary")

        if clear_clicked:
            for key in (
                "text_input",
                "sample_choice",
                "last_result",
                "recommendation_index",
            ):
                st.session_state.pop(key, None)
            st.session_state.setdefault("recommendation_index", 0)
            st.rerun()

        if analyze_clicked:
            text = st.session_state.text_input.strip()
            if not text:
                st.warning("Escreva algum texto antes de rodar o analisador.")
            else:
                try:
                    analyzer = get_analyzer()
                    st.session_state.last_result = analyzer.analyze(text)
                    st.session_state.recommendation_index = 0
                except LexiconDownloadError:
                    st.error(
                        "Não foi possível carregar o OpLexicon. "
                        "Verifique a conexão e tente novamente."
                    )
                    st.session_state.last_result = None

        with recommendation_col:
            render_recommendation_panel(st.session_state.last_result)

        if st.session_state.last_result is not None:
            with analyzer_col:
                st.space("medium")
                render_result(st.session_state.last_result)


main()
