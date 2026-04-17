from __future__ import annotations

import argparse
import json

from pln_core.lexicon import (
    OPLEXICON_LEXICON_SOURCE,
    SEED_LEXICON_SOURCE,
    LexiconDownloadError,
    load_lexicon,
)
from pln_core.pipeline import AnalysisResult, SymbolicSentimentAnalyzer
from pln_core.samples import (
    LEXICON_SOURCE_LABELS,
    LEXICON_SOURCE_OPTIONS,
    MENU_OPTIONS,
    SAMPLE_TEXTS,
)

MENU_TO_SAMPLE_KEY = {
    "2": "positive",
    "3": "negative",
    "4": "neutral",
}
MENU_TO_LEXICON_SOURCE = {
    option: source
    for option, source, _label in LEXICON_SOURCE_OPTIONS
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for the analyzer."""

    parser = argparse.ArgumentParser(
        prog="pln-sentiment",
        description="Symbolic sentiment analysis for short Brazilian Portuguese texts.",
    )
    parser.add_argument("text", nargs="?", help="Text to analyze.")
    parser.add_argument(
        "--lexicon",
        help="Optional path to a custom .json, .csv, .tsv, or OpLexicon .txt lexicon.",
    )
    parser.add_argument(
        "--lexicon-source",
        choices=(SEED_LEXICON_SOURCE, OPLEXICON_LEXICON_SOURCE),
        default=SEED_LEXICON_SOURCE,
        help="Choose the built-in seed dictionary or OpLexicon.",
    )
    parser.add_argument(
        "--sample",
        choices=tuple(SAMPLE_TEXTS),
        help="Analyze one of the built-in sample texts without using the menu.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open the interactive menu even if other modes are available.",
    )
    parser.add_argument("--json", action="store_true", help="Print the analysis as JSON.")
    return parser


def render_text_result(result: AnalysisResult) -> str:
    """Render a human-readable analysis result."""

    lines = [
        f"text: {result.text}",
        f"normalized: {result.normalized_text}",
        f"tokens: {', '.join(result.tokens) if result.tokens else '(none)'}",
        f"label: {result.label}",
        f"score: {result.score}",
    ]
    if result.matched_terms:
        lines.append("matches:")
        for match in result.matched_terms:
            rules = ", ".join(match.applied_rules) if match.applied_rules else "base"
            lines.append(
                f"  - token={match.token} position={match.position} "
                f"base={match.base_score} adjusted={match.adjusted_score} rules={rules}"
            )
    else:
        lines.append("matches: none")
    return "\n".join(lines)


def result_to_dict(result: AnalysisResult) -> dict[str, object]:
    """Convert an analysis result into a JSON-friendly dictionary."""

    return {
        "text": result.text,
        "normalized_text": result.normalized_text,
        "tokens": list(result.tokens),
        "score": result.score,
        "label": result.label,
        "matched_terms": [
            {
                "token": match.token,
                "position": match.position,
                "base_score": match.base_score,
                "adjusted_score": match.adjusted_score,
                "applied_rules": list(match.applied_rules),
            }
            for match in result.matched_terms
        ],
    }


def prompt_menu_choice() -> str:
    """Ask the user to choose one of the four interactive menu options."""

    print("pln-core")
    print("choose one option:")
    for option, label in MENU_OPTIONS:
        print(f"{option}. {label}")

    while True:
        choice = input("Option: ").strip()
        if choice in {"1", "2", "3", "4"}:
            return choice
        print("Please choose 1, 2, 3, or 4.")


def prompt_lexicon_source_choice() -> str:
    """Ask the user which lexicon source should be used."""

    print("choose the lexicon source:")
    for option, _source, label in LEXICON_SOURCE_OPTIONS:
        print(f"{option}. {label}")

    while True:
        choice = input("source: ").strip()
        if choice in MENU_TO_LEXICON_SOURCE:
            return resolve_lexicon_source_choice(choice)
        print("please choose 1 or 2.")


def prompt_custom_text() -> str:
    """Prompt for a non-empty custom text."""

    while True:
        text = input("Write the text to analyze: ").strip()
        if text:
            return text
        print("The text cannot be empty.")


def prompt_continue() -> bool:
    """Ask whether the user wants to analyze another text."""

    while True:
        choice = input("analyze another text? [y/n]: ").strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("please answer y or n.")


def resolve_interactive_text(choice: str) -> str:
    """Map an interactive menu choice to the selected text."""

    if choice == "1":
        return prompt_custom_text()
    return SAMPLE_TEXTS[MENU_TO_SAMPLE_KEY[choice]]


def resolve_lexicon_source_choice(choice: str) -> str:
    """Map an interactive source choice to the selected lexicon source."""

    return MENU_TO_LEXICON_SOURCE[choice]


def resolve_requested_text(args: argparse.Namespace) -> str | None:
    """Resolve the text requested by command line arguments."""

    if args.text:
        return args.text
    if args.sample:
        return SAMPLE_TEXTS[args.sample]
    return None


def build_analyzer(
    lexicon_source: str,
    lexicon_path: str | None,
) -> SymbolicSentimentAnalyzer:
    """Build an analyzer for the requested lexicon source."""

    if lexicon_path is None and lexicon_source == OPLEXICON_LEXICON_SOURCE:
        print("loading oplexicon v3.0...")

    lexicon = load_lexicon(path=lexicon_path, source=lexicon_source)
    return SymbolicSentimentAnalyzer(lexicon=lexicon)


def render_output(result: AnalysisResult, as_json: bool) -> str:
    """Render the final output in text or JSON format."""

    if as_json:
        return json.dumps(result_to_dict(result), ensure_ascii=False, indent=2)
    return render_text_result(result)


def main() -> None:
    """Run the CLI in interactive or direct mode."""

    parser = build_parser()
    args = parser.parse_args()
    requested_text = resolve_requested_text(args)
    lexicon_source = args.lexicon_source

    if args.json and requested_text is None and not args.interactive:
        parser.error("JSON output requires a direct text or a sample choice.")

    if args.interactive or requested_text is None:
        try:
            if args.lexicon is None:
                lexicon_source = prompt_lexicon_source_choice()

            analyzer = build_analyzer(lexicon_source=lexicon_source, lexicon_path=args.lexicon)

            while True:
                requested_text = resolve_interactive_text(prompt_menu_choice())
                result = analyzer.analyze(requested_text)
                print()
                print(f"lexicon source: {LEXICON_SOURCE_LABELS[lexicon_source]}")
                print(render_output(result, as_json=False))
                print()
                if not prompt_continue():
                    return
                print()
        except LexiconDownloadError:
            print("could not load oplexicon. check your connection or use the built-in dictionary.")
            return
        except KeyboardInterrupt:
            print()
            return

    try:
        analyzer = build_analyzer(lexicon_source=lexicon_source, lexicon_path=args.lexicon)
    except LexiconDownloadError:
        parser.error(
            "could not load oplexicon. check your connection or use the built-in dictionary."
        )

    result = analyzer.analyze(requested_text)
    print(render_output(result, as_json=args.json))
