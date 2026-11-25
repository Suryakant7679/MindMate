"""E5 figures (static PNG for the report; each figure also writes its data as a CSV table view).

Colour follows the condition everywhere (validated categorical slots 1-3 of the reference palette, light surface):
E5 blue, E4 orange, E5-old aqua; E1 / E2 are context marks in neutral gray.  Every point is direct-labelled (the aqua
slot is below 3:1 contrast on the light surface, so labels carry identity, not colour alone).

  coverage.png   per model: share of replies where steering was attempted (outline) and retained (fill)
  tradeoff.png   per model: change in quality (x) and tone (y) against E3, 95% CIs, with the quality margin
  alpha.png      per model: distribution of applied steering strengths, E4 vs E5
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .e5_common import MARGIN_QUALITY, TAGS

SURFACE, INK, INK_2, GRID, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
COLOR = {"E5": "#2a78d6", "E4": "#eb6834", "E5old": "#1baf7a", "E2": CONTEXT, "E1": CONTEXT}
LABEL = {"E1": "E1 baseline", "E2": "E2 steer only", "E4": "E4 KG+steer", "E5old": "E5-old policy", "E5": "E5 selective"}
# Fixed label offsets (points) per condition, so neighbouring points do not stack their labels.
OFFSET = {"E1": (7, -12), "E2": (7, 6), "E4": (-30, 8), "E5": (7, 8), "E5old": (7, -4)}
TITLE = {"llama31_8b": "Llama 3.1-8B", "emollama_chat_7b": "EmoLLaMA-7B", "llama2_13b_chat": "Llama-2-13B",
         "sarvam_30b": "Sarvam-30B"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _figure(n: int, width: float = 3.3, height: float = 3.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n, figsize=(max(width * n, 6.5), height), squeeze=False, facecolor=SURFACE)
    return plt, fig, axes[0]


def _title(fig, text: str) -> None:
    import textwrap
    width = int(fig.get_figwidth() * 13)                  # about 13 characters per inch at 9 pt
    fig.suptitle("\n".join(textwrap.wrap(text, width)), fontsize=9, color=INK, x=0.01, ha="left")


def _models(base: Path) -> list[str]:
    return [tag for tag in TAGS if (base / "heldout" / tag / "summary.json").exists()]


def coverage(base: Path) -> pd.DataFrame:
    import json
    rows = []
    for tag in _models(base):
        coverage = json.loads((base / "heldout" / tag / "summary.json").read_text())["coverage"]
        for condition in ("E2", "E4", "E5old", "E5"):
            rows.append({"model": tag, "condition": condition, "attempted": coverage[condition]["attempt_rate"],
                         "retained": coverage[condition]["retained_rate"]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    plt, fig, axes = _figure(len(_models(base)))
    for ax, (tag, part) in zip(axes, frame.groupby("model", sort=False)):
        _style(ax)
        y = np.arange(len(part))
        for i, row in enumerate(part.itertuples()):
            ax.barh(i, row.attempted, height=0.6, color="none", edgecolor=COLOR[row.condition], linewidth=1.2, zorder=2)
            ax.barh(i, row.retained, height=0.6, color=COLOR[row.condition], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(min(row.attempted, 1.0) + 0.03, i, f"{row.retained:.0%} sent\n{row.attempted:.0%} tried",
                    va="center", fontsize=7, color=INK_2, linespacing=1.1)
        ax.set_yticks(y, [LABEL[c] for c in part.condition], fontsize=8, color=INK)
        ax.set_xlim(0, 1.45)
        ax.set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
        ax.invert_yaxis()
        ax.set_title(TITLE.get(tag, tag), fontsize=9, color=INK, loc="left")
    _title(fig, "Steering coverage on held-out messages (fill: steered reply sent; outline: candidate generated)")
    fig.tight_layout()
    fig.savefig(base / "plots" / "coverage.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return frame


def tradeoff(base: Path) -> pd.DataFrame:
    from .e5_report import difference, estimate, load, wide_of
    from .e5_common import CONDITIONS
    rows = []
    for tag in _models(base):
        table, _ = load(base / "heldout" / tag)
        wide = wide_of(table[table.condition.isin(CONDITIONS)])
        index = wide["E3"].index
        for condition in ("E1", "E2", "E4", "E5old", "E5"):
            tone = estimate(difference(wide, condition, "E3", index, "judge_emotion_both"))
            quality = estimate(difference(wide, condition, "E3", index, "quality_both"))
            safety = estimate(difference(wide, condition, "E3", index, "safety_both"))
            rows.append({"model": tag, "condition": condition, "d_tone": tone["mean"], "tone_lo": tone["ci_low"],
                         "tone_hi": tone["ci_high"], "d_quality": quality["mean"], "quality_lo": quality["ci_low"],
                         "quality_hi": quality["ci_high"], "d_safety": safety["mean"], "safety_lo": safety["ci_low"],
                         "safety_hi": safety["ci_high"]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    plt, fig, axes = _figure(len(_models(base)), 3.4, 3.2)
    for ax, (tag, part) in zip(axes, frame.groupby("model", sort=False)):
        _style(ax)
        ax.axvline(-MARGIN_QUALITY, color=INK_2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.axvline(0, color=GRID, linewidth=1.2, zorder=1)
        ax.axhline(0, color=GRID, linewidth=1.2, zorder=1)
        for row in part.itertuples():
            color = COLOR[row.condition]
            ax.errorbar(row.d_quality, row.d_tone, xerr=[[row.d_quality - row.quality_lo], [row.quality_hi - row.d_quality]],
                        yerr=[[row.d_tone - row.tone_lo], [row.tone_hi - row.d_tone]], fmt="o", color=color, ms=8,
                        mec=SURFACE, mew=2, elinewidth=1.5, capsize=0, zorder=3)
            ax.annotate(LABEL[row.condition].split(" ")[0], (row.d_quality, row.d_tone), xytext=OFFSET[row.condition],
                        textcoords="offset points", fontsize=8, color=INK)
        ax.set_xlabel("change in quality vs E3 (1-5)", fontsize=8, color=INK_2)
        ax.set_ylabel("change in tone vs E3 (1-5)", fontsize=8, color=INK_2)
        ax.set_title(TITLE.get(tag, tag), fontsize=9, color=INK, loc="left")
        ax.text(-MARGIN_QUALITY, ax.get_ylim()[1], " quality margin", fontsize=7, color=INK_2, va="top")
    _title(fig, "Tone gained vs quality kept, against KG-only replies (95% bootstrap CIs)")
    fig.tight_layout()
    fig.savefig(base / "plots" / "tradeoff.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return frame


def alpha(base: Path) -> pd.DataFrame:
    import json
    rows = []
    for tag in _models(base):
        coverage = json.loads((base / "heldout" / tag / "summary.json").read_text())["coverage"]
        for condition in ("E4", "E5"):
            rows += [{"model": tag, "condition": condition, "alpha": value} for value in coverage[condition]["alpha_values"]]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    models = _models(base)
    plt, fig, axes = _figure(len(models))
    for ax, tag in zip(axes, models):
        _style(ax)
        part = frame[frame.model == tag]
        values = part.alpha.to_numpy()
        bins = np.linspace(0, max(values.max(), 0.8) * 1.05, 16) if len(values) else 10
        for condition in ("E4", "E5"):
            data = part.loc[part.condition == condition, "alpha"]
            if len(data):
                ax.hist(data, bins=bins, histtype="step", linewidth=2, color=COLOR[condition], zorder=3,
                        label=f"{LABEL[condition]} (n={len(data)})")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK)
        ax.set_xlabel("applied alpha", fontsize=8, color=INK_2)
        ax.set_ylabel("steered replies", fontsize=8, color=INK_2)
        ax.set_title(TITLE.get(tag, tag), fontsize=9, color=INK, loc="left")
    _title(fig, "Strength of the steering actually sent")
    fig.tight_layout()
    fig.savefig(base / "plots" / "alpha.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return frame


def draw_all(base: Path) -> None:
    (base / "plots").mkdir(parents=True, exist_ok=True)
    for name, draw in (("coverage", coverage), ("tradeoff", tradeoff), ("alpha", alpha)):
        frame = draw(base)
        if frame.empty:
            print(f"{name}: no held-out summaries yet")
            continue
        frame.to_csv(base / "plots" / f"{name}.csv", index=False)
        print(f"{name}: {base / 'plots' / (name + '.png')} (+ table view {name}.csv)")


__all__ = ["draw_all"]
