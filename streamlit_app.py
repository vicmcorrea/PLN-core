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
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.session_state.setdefault("text_input", "")
st.session_state.setdefault("sample_choice", None)
st.session_state.setdefault("last_result", None)


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


def render_result(result: AnalysisResult) -> None:
    render_label_card(result)
    st.space("medium")
    render_text_card(result)
    st.space("medium")
    render_matches(result)


def main() -> None:
    st.title("PLN Core")
    st.caption("Análise simbólica de sentimentos para português brasileiro.")
    st.markdown(f"**Pipeline:** {ANALYZER_STACK_LABEL}")

    st.divider()

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
        st.session_state.text_input = ""
        st.session_state.sample_choice = None
        st.session_state.last_result = None
        st.rerun()

    if analyze_clicked:
        text = st.session_state.text_input.strip()
        if not text:
            st.warning("Escreva algum texto antes de rodar o analisador.")
        else:
            try:
                analyzer = get_analyzer()
                st.session_state.last_result = analyzer.analyze(text)
            except LexiconDownloadError:
                st.error(
                    "Não foi possível carregar o OpLexicon. "
                    "Verifique a conexão e tente novamente."
                )
                st.session_state.last_result = None

    if st.session_state.last_result is not None:
        st.divider()
        render_result(st.session_state.last_result)


main()
