from __future__ import annotations

import argparse
import json

from pln_core.lexicon import (
    OPLEXICON_LEXICON_SOURCE,
    LexiconDownloadError,
    load_lexicon,
)
from pln_core.pipeline import AnalysisResult, SymbolicSentimentAnalyzer
from pln_core.samples import (
    ANALYZE_MODE,
    ANALYZER_STACK_LABEL,
    COMPARE_MODE,
    COMPARISON_EXAMPLES,
    MENU_OPTIONS,
    SAMPLE_TEXTS,
    START_MODE_OPTIONS,
)
from pln_core.tokenizers import (
    SPACY_PT_TOKENIZER_SOURCE,
    get_tokenizer,
)

MENU_TO_SAMPLE_KEY = {
    "2": "positive",
    "3": "negative",
    "4": "neutral",
}
MENU_TO_START_MODE = {
    option: mode
    for option, mode, _label in START_MODE_OPTIONS
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for the analyzer."""

    parser = argparse.ArgumentParser(
        prog="pln-sentiment",
        description=(
            "Symbolic sentiment analysis for short Brazilian Portuguese texts "
            "using OpLexicon v3.0 and spaCy."
        ),
    )
    parser.add_argument("text", nargs="?", help="Text to analyze.")
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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the built-in comparison examples instead of a single analysis.",
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


def prompt_start_mode_choice() -> str:
    """Ask the user whether they want analysis mode or comparison mode."""

    print("pln-core")
    print("choose the cli mode:")
    for option, _mode, label in START_MODE_OPTIONS:
        print(f"{option}. {label}")

    while True:
        choice = input("mode: ").strip()
        if choice in MENU_TO_START_MODE:
            return resolve_start_mode_choice(choice)
        print("please choose 1 or 2.")


def prompt_menu_choice() -> str:
    """Ask the user to choose one of the four interactive menu options."""

    print("choose one option:")
    for option, label in MENU_OPTIONS:
        print(f"{option}. {label}")

    while True:
        choice = input("Option: ").strip()
        if choice in {"1", "2", "3", "4"}:
            return choice
        print("Please choose 1, 2, 3, or 4.")


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


def resolve_start_mode_choice(choice: str) -> str:
    """Map an interactive mode choice to the selected CLI mode."""

    return MENU_TO_START_MODE[choice]


def resolve_requested_text(args: argparse.Namespace) -> str | None:
    """Resolve the text requested by command line arguments."""

    if args.text:
        return args.text
    if args.sample:
        return SAMPLE_TEXTS[args.sample]
    return None


def build_analyzer(announce_loading: bool = True) -> SymbolicSentimentAnalyzer:
    """Build the analyzer with the fixed project stack."""

    if announce_loading:
        print("loading oplexicon v3.0...")

    lexicon = load_lexicon(source=OPLEXICON_LEXICON_SOURCE)
    tokenizer = get_tokenizer(SPACY_PT_TOKENIZER_SOURCE)
    return SymbolicSentimentAnalyzer(lexicon=lexicon, tokenizer=tokenizer)


def build_comparison_results() -> list[dict[str, object]]:
    """Run the built-in example set with the fixed analyzer stack."""

    analyzer = build_analyzer()
    example_results: list[dict[str, object]] = []
    for example_name, text in COMPARISON_EXAMPLES:
        example_results.append(
            {
                "example_name": example_name,
                "text": text,
                "result": analyzer.analyze(text),
            }
        )

    return example_results


def render_comparison_results(
    comparison_results: list[dict[str, object]],
    as_json: bool,
) -> str:
    """Render the built-in example set in text or JSON format."""

    if as_json:
        return json.dumps(
            [
                {
                    "example_name": item["example_name"],
                    "text": item["text"],
                    "stack": ANALYZER_STACK_LABEL,
                    "analysis": result_to_dict(item["result"]),
                }
                for item in comparison_results
            ],
            ensure_ascii=False,
            indent=2,
        )

    sections = [f"stack: {ANALYZER_STACK_LABEL}"]
    table_rows: list[tuple[str, str, str]] = []
    for item in comparison_results:
        result = item["result"]
        table_rows.append((str(item["example_name"]), result.label, str(result.score)))

        lines = [
            f"=== example: {item['example_name']} ===",
            f"input text: {item['text']}",
            f"label: {result.label}",
            f"score: {result.score}",
            f"tokens: {', '.join(result.tokens) if result.tokens else '(none)'}",
        ]

        if result.matched_terms:
            lines.append("matches:")
            for match in result.matched_terms:
                rules = ", ".join(match.applied_rules) if match.applied_rules else "base"
                lines.append(
                    "  - "
                    f"{match.token} -> adjusted={match.adjusted_score} "
                    f"(base={match.base_score}; rules={rules})"
                )
        else:
            lines.append("matches: none")

        sections.append("\n".join(lines))

    headers = ("example", "label", "score")
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]

    def format_row(row: tuple[str, str, str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    table_lines = [
        "results table:",
        format_row(headers),
        separator,
        *(format_row(row) for row in table_rows),
    ]

    sections.append("\n".join(table_lines))
    return "\n\n".join(sections)


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
    start_mode = COMPARE_MODE if args.compare else ANALYZE_MODE

    if args.json and requested_text is None and not args.interactive and not args.compare:
        parser.error("JSON output requires a direct text or a sample choice.")

    if (args.interactive or requested_text is None) and not args.compare:
        try:
            start_mode = prompt_start_mode_choice()
            print()

            if start_mode == COMPARE_MODE:
                comparison_results = build_comparison_results()
                print(render_comparison_results(comparison_results, as_json=False))
                return

            print(f"stack: {ANALYZER_STACK_LABEL}")
            print()
            analyzer = build_analyzer()

            while True:
                requested_text = resolve_interactive_text(prompt_menu_choice())
                result = analyzer.analyze(requested_text)
                print()
                print(render_output(result, as_json=False))
                print()
                if not prompt_continue():
                    return
                print()
        except LexiconDownloadError:
            print("could not load oplexicon. check your connection and try again.")
            return
        except KeyboardInterrupt:
            print()
            return

    if start_mode == COMPARE_MODE:
        try:
            comparison_results = build_comparison_results()
        except LexiconDownloadError:
            parser.error("could not load oplexicon. check your connection and try again.")
        print(render_comparison_results(comparison_results, as_json=args.json))
        return

    try:
        analyzer = build_analyzer()
    except LexiconDownloadError:
        parser.error("could not load oplexicon. check your connection and try again.")

    result = analyzer.analyze(requested_text)
    print(render_output(result, as_json=args.json))
