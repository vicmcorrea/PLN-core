"""Build publication-style figures for the Etapa 2 report draft."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPORT_DIR / "figures"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
}


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _to_linear_srgb(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _contrast_text_color(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _alpha = rgba
    linear_red = _to_linear_srgb(red)
    linear_green = _to_linear_srgb(green)
    linear_blue = _to_linear_srgb(blue)
    luminance = 0.2126 * linear_red + 0.7152 * linear_green + 0.0722 * linear_blue
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "black" if black_contrast >= white_contrast else "white"


def plot_macro_f1_tracks() -> None:
    systems = [
        "Baseline\nde pistas",
        "OpLexicon\n+ regras",
        "TF-IDF\nReg. Log.",
        "TF-IDF\nSVM linear",
        "XLM-R",
        "Albertina",
        "TabularisAI",
    ]
    raw = np.array([0.9970, 0.5960, 0.8164, 0.8084, 0.9968, 0.9972, 0.5248])
    stripped = np.array([0.1666, 0.3668, 0.8094, 0.8030, 0.7494, 0.7808, 0.5557])
    strict = np.array([0.1666, 0.3665, 0.8030, 0.7962, np.nan, np.nan, 0.5723])

    x = np.arange(len(systems))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    raw_bars = ax.bar(
        x - width,
        raw,
        width,
        label="Kaggle bruto (com vazamento)",
        color="#B8B8B8",
        edgecolor=OKABE_ITO["black"],
        linewidth=0.5,
        hatch="///",
    )
    ax.bar(x, stripped, width, label="sem emoticons/URLs", color=OKABE_ITO["orange"])
    ax.bar(
        x + width,
        strict,
        width,
        label="sem pistas sociais/fontes",
        color=OKABE_ITO["green"],
    )

    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.19)
    ax.set_xticks(x, systems)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(
        "baseline de pistas = 0,997\nresolve o teste bruto\nsem ler sentimento",
        xy=(raw_bars[0].get_x() + raw_bars[0].get_width() / 2, raw[0]),
        xytext=(0.58, 1.04),
        textcoords=("data", "data"),
        ha="left",
        va="bottom",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": OKABE_ITO["red"]},
        color=OKABE_ITO["red"],
    )
    ax.annotate(
        "OpLexicon também cai\n0,596 -> 0,367",
        xy=(x[1], stripped[1]),
        xytext=(0.78, 0.86),
        textcoords=("data", "data"),
        ha="left",
        va="bottom",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": OKABE_ITO["red"]},
        color=OKABE_ITO["red"],
    )

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.015,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    _save(fig, "macro_f1_tracks")


def plot_cue_prevalence() -> None:
    labels = ["Positivo", "Negativo", "Neutro"]
    cues = ["Emoticon +", "Emoticon -", "URL", "Menção", "Hashtag"]
    values = np.array(
        [
            [0.9928, 0.0066, 0.2346, 0.5135, 0.0312],
            [0.0000, 0.9988, 0.1849, 0.4724, 0.0132],
            [0.0006, 0.0000, 0.9970, 0.0222, 0.0900],
        ]
    )

    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    image = ax.imshow(values, cmap="cividis", vmin=0, vmax=1)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proporção no teste")
    ax.set_xticks(np.arange(len(cues)), cues, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Vazamento de rótulo por pistas superficiais por classe")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color = "white" if value > 0.55 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "cue_prevalence_heatmap")


def plot_transformer_drop() -> None:
    systems = ["XLM-R fine-tuned", "Albertina fine-tuned", "TabularisAI pronto"]
    raw = np.array([0.9968, 0.9972, 0.5248])
    stripped = np.array([0.7494, 0.7808, 0.5557])
    colors = [OKABE_ITO["red"], OKABE_ITO["purple"], OKABE_ITO["sky"]]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for index, system in enumerate(systems):
        ax.plot([0, 1], [raw[index], stripped[index]], marker="o", color=colors[index])
        ax.text(
            1.03,
            stripped[index],
            f"{system}  {stripped[index]:.3f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(-0.2, 1.65)
    ax.set_ylim(0.48, 1.03)
    ax.set_xticks([0, 1], ["bruto\n(com pistas)", "sem emoticons/URLs"])
    ax.set_ylabel("Macro F1")
    ax.set_title("O efeito da limpeza muda conforme o tipo de modelo neural")
    ax.grid(axis="y", alpha=0.25)
    ax.annotate(
        "bruto quase perfeito,\nmas contaminado",
        xy=(0, 0.9968),
        xytext=(0.18, 0.982),
        ha="left",
        va="top",
        fontsize=8,
        color=OKABE_ITO["red"],
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["red"], "lw": 1.0},
    )
    ax.text(
        1.0,
        0.505,
        "modelo pronto melhora um pouco\nquando retiramos pistas",
        ha="center",
        va="bottom",
        fontsize=8,
        color=OKABE_ITO["green"],
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "transformer_drop")


def plot_clean_confusion_matrices() -> None:
    labels = ["positivo", "negativo", "neutro"]
    cmap = plt.get_cmap("cividis")
    systems = [
        (
            "OpLexicon\nMacro F1 0,367",
            np.array(
                [
                    [596, 329, 742],
                    [300, 456, 910],
                    [448, 422, 796],
                ]
            ),
        ),
        (
            "TF-IDF Reg. Log.\nMacro F1 0,809",
            np.array(
                [
                    [1256, 379, 32],
                    [428, 1213, 25],
                    [80, 13, 1573],
                ]
            ),
        ),
        (
            "TF-IDF SVM\nMacro F1 0,803",
            np.array(
                [
                    [1202, 433, 32],
                    [413, 1236, 17],
                    [77, 18, 1571],
                ]
            ),
        ),
        (
            "XLM-R\nMacro F1 0,749",
            np.array(
                [
                    [809, 783, 75],
                    [271, 1353, 42],
                    [36, 0, 1630],
                ]
            ),
        ),
        (
            "Albertina\nMacro F1 0,781",
            np.array(
                [
                    [1113, 490, 64],
                    [468, 1173, 25],
                    [22, 20, 1624],
                ]
            ),
        ),
        (
            "TabularisAI\nMacro F1 0,556",
            np.array(
                [
                    [660, 524, 483],
                    [495, 868, 303],
                    [99, 267, 1300],
                ]
            ),
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), constrained_layout=True)
    for ax, (title, counts) in zip(axes.flat, systems, strict=True):
        row_totals = counts.sum(axis=1, keepdims=True)
        percentages = counts / row_totals
        image = ax.imshow(percentages, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)), [f"Prev. {label}" for label in labels])
        ax.set_yticks(np.arange(len(labels)), [f"Real {label}" for label in labels])
        ax.tick_params(axis="x", rotation=30)

        for row in range(counts.shape[0]):
            for col in range(counts.shape[1]):
                value = percentages[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.0%}\n({counts[row, col]})",
                    ha="center",
                    va="center",
                    color=_contrast_text_color(cmap(value)),
                    fontsize=7,
                )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
    cbar.set_label("Proporção dentro da classe real")
    _save(fig, "confusion_clean_condition")


def main() -> None:
    _configure()
    plot_macro_f1_tracks()
    plot_cue_prevalence()
    plot_transformer_drop()
    plot_clean_confusion_matrices()
    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
