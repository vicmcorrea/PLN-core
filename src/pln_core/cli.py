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
    ANALYZE_MODE,
    COMPARE_MODE,
    COMPARISON_EXAMPLES,
    LEXICON_SOURCE_LABELS,
    LEXICON_SOURCE_OPTIONS,
    MENU_OPTIONS,
    SAMPLE_TEXTS,
    START_MODE_OPTIONS,
    TOKENIZER_SOURCE_LABELS,
    TOKENIZER_SOURCE_OPTIONS,
)
from pln_core.tokenizers import (
    CUSTOM_TOKENIZER_SOURCE,
    SPACY_PT_TOKENIZER_SOURCE,
    get_tokenizer,
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
MENU_TO_TOKENIZER_SOURCE = {
    option: source
    for option, source, _label in TOKENIZER_SOURCE_OPTIONS
}
MENU_TO_START_MODE = {
    option: mode
    for option, mode, _label in START_MODE_OPTIONS
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
        "--tokenizer-source",
        choices=(CUSTOM_TOKENIZER_SOURCE, SPACY_PT_TOKENIZER_SOURCE),
        default=CUSTOM_TOKENIZER_SOURCE,
        help="Choose the built-in regex tokenizer or the spaCy Portuguese tokenizer.",
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


def prompt_tokenizer_source_choice() -> str:
    """Ask the user which tokenizer source should be used."""

    print("choose the tokenizer source:")
    for option, _source, label in TOKENIZER_SOURCE_OPTIONS:
        print(f"{option}. {label}")

    while True:
        choice = input("tokenizer: ").strip()
        if choice in MENU_TO_TOKENIZER_SOURCE:
            return resolve_tokenizer_source_choice(choice)
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


def resolve_tokenizer_source_choice(choice: str) -> str:
    """Map an interactive source choice to the selected tokenizer source."""

    return MENU_TO_TOKENIZER_SOURCE[choice]


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


def build_analyzer(
    lexicon_source: str,
    lexicon_path: str | None,
    tokenizer_source: str,
    announce_loading: bool = True,
) -> SymbolicSentimentAnalyzer:
    """Build an analyzer for the requested lexicon source."""

    if announce_loading and lexicon_path is None and lexicon_source == OPLEXICON_LEXICON_SOURCE:
        print("loading oplexicon v3.0...")

    lexicon = load_lexicon(path=lexicon_path, source=lexicon_source)
    tokenizer = get_tokenizer(tokenizer_source)
    return SymbolicSentimentAnalyzer(lexicon=lexicon, tokenizer=tokenizer)


def build_comparison_results() -> list[dict[str, object]]:
    """Run built-in examples across lexicon and tokenizer combinations."""

    lexicon_cache = {
        SEED_LEXICON_SOURCE: load_lexicon(source=SEED_LEXICON_SOURCE),
    }
    print("loading oplexicon v3.0...")
    lexicon_cache[OPLEXICON_LEXICON_SOURCE] = load_lexicon(source=OPLEXICON_LEXICON_SOURCE)

    analyzers: dict[tuple[str, str], SymbolicSentimentAnalyzer] = {}
    for lexicon_source in (SEED_LEXICON_SOURCE, OPLEXICON_LEXICON_SOURCE):
        for tokenizer_source in (CUSTOM_TOKENIZER_SOURCE, SPACY_PT_TOKENIZER_SOURCE):
            analyzers[(lexicon_source, tokenizer_source)] = SymbolicSentimentAnalyzer(
                lexicon=lexicon_cache[lexicon_source],
                tokenizer=get_tokenizer(tokenizer_source),
            )

    comparison_results: list[dict[str, object]] = []
    for example_name, text in COMPARISON_EXAMPLES:
        variants: list[dict[str, object]] = []
        for lexicon_source in (SEED_LEXICON_SOURCE, OPLEXICON_LEXICON_SOURCE):
            for tokenizer_source in (CUSTOM_TOKENIZER_SOURCE, SPACY_PT_TOKENIZER_SOURCE):
                result = analyzers[(lexicon_source, tokenizer_source)].analyze(text)
                variants.append(
                    {
                        "lexicon_source": lexicon_source,
                        "tokenizer_source": tokenizer_source,
                        "result": result,
                    }
                )

        comparison_results.append(
            {
                "example_name": example_name,
                "text": text,
                "variants": variants,
            }
        )

    return comparison_results


def render_comparison_results(comparison_results: list[dict[str, object]], as_json: bool) -> str:
    """Render built-in comparison examples in text or JSON format."""

    if as_json:
        return json.dumps(
            [
                {
                    "example_name": item["example_name"],
                    "text": item["text"],
                    "variants": [
                        {
                            "lexicon_source": variant["lexicon_source"],
                            "tokenizer_source": variant["tokenizer_source"],
                            "analysis": result_to_dict(variant["result"]),
                        }
                        for variant in item["variants"]
                    ],
                }
                for item in comparison_results
            ],
            ensure_ascii=False,
            indent=2,
        )

    sections: list[str] = []
    table_rows: list[tuple[str, str, str, str, str]] = []
    for item in comparison_results:
        summary_lines = []
        detail_lines = []

        for index, variant in enumerate(item["variants"], start=1):
            result = variant["result"]
            lexicon_label = LEXICON_SOURCE_LABELS[variant["lexicon_source"]]
            tokenizer_label = TOKENIZER_SOURCE_LABELS[variant["tokenizer_source"]]

            summary_lines.append(
                f"  {index}. {lexicon_label} + {tokenizer_label} -> "
                f"{result.label} ({result.score})"
            )
            table_rows.append(
                (
                    str(item["example_name"]),
                    lexicon_label.replace("use ", ""),
                    tokenizer_label.replace("use ", ""),
                    result.label,
                    str(result.score),
                )
            )

            detail_lines.extend(
                [
                    f"configuration {index}",
                    f"  lexicon: {lexicon_label}",
                    f"  tokenizer: {tokenizer_label}",
                    f"  label: {result.label}",
                    f"  score: {result.score}",
                    f"  tokens: {', '.join(result.tokens) if result.tokens else '(none)'}",
                ]
            )

            if result.matched_terms:
                detail_lines.append("  matches:")
                for match in result.matched_terms:
                    rules = ", ".join(match.applied_rules) if match.applied_rules else "base"
                    detail_lines.append(
                        "    - "
                        f"{match.token} -> adjusted={match.adjusted_score} "
                        f"(base={match.base_score}; rules={rules})"
                    )
            else:
                detail_lines.append("  matches: none")

            if index < len(item["variants"]):
                detail_lines.append("")

        lines = [
            f"=== comparison: {item['example_name']} ===",
            f"input text: {item['text']}",
            "",
            "quick summary:",
            *summary_lines,
            "",
            "details:",
        ]
        lines.extend(detail_lines)
        sections.append("\n".join(lines))

    headers = ("example", "lexicon", "tokenizer", "label", "score")
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]

    def format_row(row: tuple[str, str, str, str, str]) -> str:
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
    lexicon_source = args.lexicon_source
    tokenizer_source = args.tokenizer_source
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

            if args.lexicon is None:
                lexicon_source = prompt_lexicon_source_choice()
            tokenizer_source = prompt_tokenizer_source_choice()

            analyzer = build_analyzer(
                lexicon_source=lexicon_source,
                lexicon_path=args.lexicon,
                tokenizer_source=tokenizer_source,
            )

            while True:
                requested_text = resolve_interactive_text(prompt_menu_choice())
                result = analyzer.analyze(requested_text)
                print()
                print(f"lexicon source: {LEXICON_SOURCE_LABELS[lexicon_source]}")
                print(f"tokenizer source: {TOKENIZER_SOURCE_LABELS[tokenizer_source]}")
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

    if start_mode == COMPARE_MODE:
        try:
            comparison_results = build_comparison_results()
        except LexiconDownloadError:
            parser.error(
                "could not load oplexicon. check your connection or use the built-in dictionary."
            )
        print(render_comparison_results(comparison_results, as_json=args.json))
        return

    try:
        analyzer = build_analyzer(
            lexicon_source=lexicon_source,
            lexicon_path=args.lexicon,
            tokenizer_source=tokenizer_source,
        )
    except LexiconDownloadError:
        parser.error(
            "could not load oplexicon. check your connection or use the built-in dictionary."
        )

    result = analyzer.analyze(requested_text)
    print(render_output(result, as_json=args.json))
