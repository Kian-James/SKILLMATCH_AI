"""
charts.py
=========
Chart generation companion for evaluate_pipeline.py.

Called automatically when --charts is passed to the evaluator.
Produces a single multi-page PDF (eval_charts.pdf by default) and,
optionally, individual PNG files alongside it.

Charts produced (per available data)
-------------------------------------
OCR stage
  1. OCR Quality % per sample                (horizontal bar)
  2. Character count per sample              (horizontal bar)
  3. Text similarity vs ground truth         (horizontal bar, if expected_text present)

NLP stage
  4. Precision / Recall / F1 per sample      (grouped horizontal bar)
  5. TP / FP / FN per sample                 (stacked horizontal bar)
  6. Aggregate P / R / F1 radar              (radar / spider)
  7. Skill breakdown per sample              (matched / missed / extra pills)
  8. Skill frequency analysis                (most commonly missed vs extra)

FAISS stage
  9. Hybrid score per sample                 (horizontal bar + min_score threshold line)
 10. Cosine score per sample                 (horizontal bar)
 11. Skill match ratio per sample            (horizontal bar)
 12. FAISS rank of expected title per sample (dot / lollipop)

Pipeline overview
 13. Per-sample total elapsed time           (horizontal bar)
 14. Stage timing breakdown                  (stacked horizontal bar)
 15. Overall pass-rate summary               (bar chart)

Usage (standalone — run directly against a saved report JSON)
-------------------------------------------------------------
  python charts.py report.json
  python charts.py report.json --output-dir ./charts_out --prefix run1 --png

Usage (as a library)
---------------------
  from charts import generate_charts
  generate_charts(report, output_dir="eval_charts/", prefix="run1")
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger("evaluator.charts")

# ── Colour palette (accessible, works on white background) ──────────────────
_GREEN   = "#2ecc71"
_YELLOW  = "#f39c12"
_RED     = "#e74c3c"
_BLUE    = "#3498db"
_PURPLE  = "#9b59b6"
_TEAL    = "#1abc9c"
_ORANGE  = "#e67e22"
_GREY    = "#95a5a6"
_DARK    = "#2c3e50"

_STAGE_COLOURS = {
    "ocr":   _BLUE,
    "nlp":   _PURPLE,
    "faiss": _TEAL,
}

# ── Inline colour helper (mirrors evaluate_pipeline._c; needed when running
#    standalone so _chart_skill_breakdown doesn't blow up) ──────────────────
try:
    import colorama
    colorama.init()
    _ANSI = dict(
        RESET="\033[0m", BOLD="\033[1m",
        GREEN="\033[92m", RED="\033[91m", YELLOW="\033[93m",
        CYAN="\033[96m",  BLUE="\033[94m", MAGENTA="\033[95m",
        DIM="\033[2m",
    )
except ImportError:
    _ANSI = {k: "" for k in ("RESET","BOLD","GREEN","RED","YELLOW",
                             "CYAN","BLUE","MAGENTA","DIM")}


def _c(colour: str, text: str) -> str:
    """ANSI colour wrapper — safe fallback when colorama is absent."""
    return f"{_ANSI.get(colour, '')}{text}{_ANSI['RESET']}"


# ── Score → colour helpers ───────────────────────────────────────────────────
def _score_colour(val: float, lo: float = 0.5, hi: float = 0.75) -> str:
    if val >= hi:
        return _GREEN
    if val >= lo:
        return _YELLOW
    return _RED


def _bar_colours(values: list[float], lo: float = 0.5, hi: float = 0.75) -> list[str]:
    return [_score_colour(v, lo, hi) for v in values]


def _short_name(path: str, max_len: int = 28) -> str:
    name = Path(path).name
    if len(name) > max_len:
        name = name[:max_len - 1] + "…"
    return name


# ── Shared figure helpers ────────────────────────────────────────────────────
def _hbar(
        ax,
        labels: list[str],
        values: list[float],
        colours: list[str],
        title: str,
        xlabel: str,
        xlim: tuple = (0, 1),
        vline: Optional[float] = None,
        vline_label: Optional[str] = None,
        fmt: str = ".1%",
        fontsize: int = 8,
) -> None:
    """Generic horizontal bar chart."""
    y = range(len(labels))
    bars = ax.barh(list(y), values, color=colours, edgecolor="white", height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    for bar, val in zip(bars, values):
        label_text = format(val, fmt)
        ax.text(
            min(val + 0.01, xlim[1] - 0.01),
            bar.get_y() + bar.get_height() / 2,
            label_text, va="center", ha="left", fontsize=7, color=_DARK,
            )

    if vline is not None:
        ax.axvline(vline, color=_RED, linestyle="--", linewidth=1.2,
                   label=vline_label or f"threshold {vline:{fmt}}")
        ax.legend(fontsize=7, loc="lower right")


# ── Chart builders ───────────────────────────────────────────────────────────
def _chart_ocr_quality(results: list[dict], ax) -> None:
    samples = [r for r in results if r.get("ocr") and not r["ocr"].get("error")]
    if not samples:
        ax.text(0.5, 0.5, "No OCR data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("OCR Quality %", fontsize=11, fontweight="bold")
        return
    labels = [_short_name(r["file"]) for r in samples]
    vals   = [r["ocr"]["quality"] / 100 for r in samples]
    _hbar(ax, labels, vals, _bar_colours(vals, 0.4, 0.7),
          "OCR Quality (alnum%)", "Alnum ratio", xlim=(0, 1.05),
          vline=0.70, vline_label="quality ≥ 70%")


def _chart_char_count(results: list[dict], ax) -> None:
    samples = [r for r in results if r.get("ocr") and not r["ocr"].get("error")]
    if not samples:
        ax.text(0.5, 0.5, "No OCR data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("OCR Character Count", fontsize=11, fontweight="bold")
        return
    labels = [_short_name(r["file"]) for r in samples]
    vals   = [r["ocr"]["char_count"] for r in samples]
    max_v  = max(vals) if vals else 1
    y = range(len(labels))
    bars = ax.barh(list(y), vals, color=_BLUE, edgecolor="white", height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Characters", fontsize=9)
    ax.set_title("OCR Character Count", fontsize=11, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(val + max_v * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=7, color=_DARK)


def _chart_text_similarity(results: list[dict], ax) -> None:
    samples = [r for r in results
               if r.get("ocr") and r["ocr"].get("text_similarity") is not None]
    if not samples:
        ax.set_facecolor("#f8f9fa")
        ax.text(0.5, 0.62, "⏳  Coming Soon",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color=_GREY, transform=ax.transAxes)
        ax.text(0.5, 0.42,
                "Add  expected_text  to your labels.json entries\n"
                "to enable character-level OCR accuracy scoring.",
                ha="center", va="center", fontsize=9, color=_GREY,
                transform=ax.transAxes, linespacing=1.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax.set_title("OCR Text Similarity vs Ground Truth", fontsize=11,
                     fontweight="bold", pad=8)
        return
    labels = [_short_name(r["file"]) for r in samples]
    vals   = [r["ocr"]["text_similarity"] for r in samples]
    _hbar(ax, labels, vals, _bar_colours(vals, 0.65, 0.85),
          "OCR Text Similarity vs Ground Truth", "Character similarity (0–1)",
          vline=0.85, vline_label="target ≥ 85%")


def _chart_nlp_prf(results: list[dict], ax) -> None:
    import numpy as np
    samples = [r for r in results
               if r.get("nlp") and r["nlp"].get("precision") is not None]
    if not samples:
        ax.text(0.5, 0.5, "No NLP label data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("NLP Precision / Recall / F1", fontsize=11, fontweight="bold")
        return

    labels = [_short_name(r["file"]) for r in samples]
    prec   = [r["nlp"]["precision"] for r in samples]
    rec    = [r["nlp"]["recall"]    for r in samples]
    f1     = [r["nlp"]["f1"]        for r in samples]

    n   = len(labels)
    y   = np.arange(n)
    h   = 0.22
    ax.barh(y + h, prec, height=h, color=_BLUE,   label="Precision", edgecolor="white")
    ax.barh(y,     rec,  height=h, color=_PURPLE,  label="Recall",    edgecolor="white")
    ax.barh(y - h, f1,   height=h, color=_TEAL,   label="F1",        edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Score", fontsize=9)
    ax.set_title("NLP Precision / Recall / F1 per Sample", fontsize=11,
                 fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0.75, color=_RED, linestyle="--", linewidth=1, label="target 75%")
    ax.legend(fontsize=7, loc="lower right")

    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        for val, off in ((p, h), (r, 0), (f, -h)):
            ax.text(val + 0.01, y[i] + off, f"{val:.0%}",
                    va="center", fontsize=6, color=_DARK)


def _chart_nlp_counts(results: list[dict], ax) -> None:
    import numpy as np
    samples = [r for r in results
               if r.get("nlp") and r["nlp"].get("precision") is not None]
    if not samples:
        ax.text(0.5, 0.5, "No NLP label data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("NLP TP / FP / FN Breakdown", fontsize=11, fontweight="bold")
        return

    labels = [_short_name(r["file"]) for r in samples]
    tp = np.array([r["nlp"]["tp"] for r in samples])
    fp = np.array([r["nlp"]["fp"] for r in samples])
    fn = np.array([r["nlp"]["fn"] for r in samples])
    y  = np.arange(len(labels))

    ax.barh(y, tp,            color=_GREEN,  edgecolor="white", height=0.55, label="TP (correct)")
    ax.barh(y, fp, left=tp,   color=_YELLOW, edgecolor="white", height=0.55, label="FP (extra)")
    ax.barh(y, fn, left=tp+fp, color=_RED,   edgecolor="white", height=0.55, label="FN (missed)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Skill count", fontsize=9)
    ax.set_title("NLP TP / FP / FN per Sample", fontsize=11, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="lower right")


def _chart_nlp_radar(results: list[dict], ax) -> None:
    """Aggregate P/R/F1 radar chart — one spoke per metric, one polygon per sample."""
    import numpy as np

    samples = [r for r in results
               if r.get("nlp") and r["nlp"].get("precision") is not None]
    if not samples:
        ax.text(0.5, 0.5, "No NLP label data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("NLP Aggregate Radar", fontsize=11, fontweight="bold")
        return

    categories = ["Precision", "Recall", "F1"]
    N = len(categories)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    colours_cycle = [_BLUE, _PURPLE, _TEAL, _ORANGE, _GREEN, _RED]
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6, color=_GREY)
    ax.spines["polar"].set_color(_GREY)

    for i, r in enumerate(samples):
        vals = [r["nlp"]["precision"], r["nlp"]["recall"], r["nlp"]["f1"]]
        vals += vals[:1]
        col  = colours_cycle[i % len(colours_cycle)]
        ax.plot(angles, vals, color=col, linewidth=1.5, linestyle="solid")
        ax.fill(angles, vals, color=col, alpha=0.08)

    ax.set_title("NLP Aggregate Radar (P/R/F1)", fontsize=11, fontweight="bold", pad=14)


def _chart_skill_breakdown(results: list[dict], ax) -> None:
    """
    Per-sample skill detail table rendered as a figure.
    Each row = one sample. Columns show matched (✔), missed (✘), extra (+) skills
    as coloured pill-text so the reader can see exactly which skills landed where.
    """
    import matplotlib.patches as mpatches

    samples = [r for r in results
               if r.get("nlp") and r["nlp"].get("precision") is not None]
    if not samples:
        ax.text(0.5, 0.5, "No NLP label data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color=_GREY)
        ax.set_title("Skill Breakdown per Sample (Matched / Missed / Extra)",
                     fontsize=11, fontweight="bold")
        return

    ax.axis("off")
    ax.set_title("Skill Breakdown per Sample", fontsize=11, fontweight="bold", pad=10)

    n     = len(samples)
    row_h = 1.0 / (n + 1)
    col_x = [0.0, 0.02, 0.36, 0.64]

    # Header
    for text, x in zip(["Sample", "✔  Matched", "✘  Missed", "+  Extra"], col_x):
        colour = (_DARK    if text == "Sample"
                  else _GREEN  if "Matched" in text
        else _RED    if "Missed"  in text
        else _YELLOW)
        ax.text(x, 1.0 - row_h * 0.3, text,
                transform=ax.transAxes, fontsize=8, fontweight="bold", color=colour,
                va="top", ha="left", clip_on=True)

    ax.plot([0, 1], [1.0 - row_h * 0.7, 1.0 - row_h * 0.7],
            color=_DARK, linewidth=0.6, transform=ax.transAxes,
            clip_on=False, solid_capstyle="butt")

    def _pills(sub_ax, items, x, y, colour, transform):
        """Draw skill names as small coloured text badges on one row."""
        if not items:
            sub_ax.text(x, y, "—", transform=transform,
                        fontsize=6.5, color=_GREY, va="top", ha="left", clip_on=True)
            return
        lines, line = [], []
        for item in items:
            line.append(item)
            if len(line) == 4:
                lines.append(line)
                line = []
        if line:
            lines.append(line)
        for li, pill_line in enumerate(lines[:3]):
            text = "  ".join(pill_line)
            sub_ax.text(x, y - li * 0.012, text,
                        transform=transform, fontsize=6, color=colour,
                        va="top", ha="left", clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc=colour + "22", ec=colour + "66", lw=0.5))

    for i, r in enumerate(samples):
        nlp  = r["nlp"]
        name = _short_name(r["file"], max_len=30)
        y    = 1.0 - row_h * (i + 1.1)

        # Alternating row background
        if i % 2 == 0:
            rect = mpatches.FancyBboxPatch(
                (0, y - row_h * 0.05), 1, row_h,
                boxstyle="square,pad=0", transform=ax.transAxes,
                fc="#f4f6f8", ec="none", zorder=0, clip_on=False,
            )
            ax.add_patch(rect)

        ax.text(col_x[0], y, name,
                transform=ax.transAxes, fontsize=7, color=_DARK,
                va="top", ha="left", clip_on=True, fontweight="bold")

        _pills(ax, nlp.get("matched", []), col_x[1], y, _GREEN,  ax.transAxes)
        _pills(ax, nlp.get("missed",  []), col_x[2], y, _RED,    ax.transAxes)
        _pills(ax, nlp.get("extra",   []), col_x[3], y, _ORANGE, ax.transAxes)

        ax.plot([0, 1], [y - row_h * 0.05, y - row_h * 0.05],
                color="#dee2e6", linewidth=0.4, transform=ax.transAxes,
                clip_on=False, solid_capstyle="butt")

    legend_items = [
        mpatches.Patch(fc=_GREEN  + "22", ec=_GREEN  + "66", label="Matched (TP)"),
        mpatches.Patch(fc=_RED    + "22", ec=_RED    + "66", label="Missed  (FN)"),
        mpatches.Patch(fc=_ORANGE + "22", ec=_ORANGE + "66", label="Extra   (FP)"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              fontsize=7, framealpha=0.9, ncol=3)


def _chart_skill_frequency(results: list[dict], ax) -> None:
    """
    Horizontal bar chart of the most commonly missed and most commonly
    predicted-extra skills across all samples — shows systemic gaps.
    """
    import numpy as np
    from collections import Counter

    samples = [r for r in results
               if r.get("nlp") and r["nlp"].get("precision") is not None]
    if not samples:
        ax.text(0.5, 0.5, "No NLP label data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color=_GREY)
        ax.set_title("Skill Frequency Analysis (Missed vs Extra)",
                     fontsize=11, fontweight="bold")
        return

    missed_ctr = Counter()
    extra_ctr  = Counter()
    for r in samples:
        missed_ctr.update(r["nlp"].get("missed", []))
        extra_ctr.update(r["nlp"].get("extra",  []))

    top_n      = 12
    top_missed = missed_ctr.most_common(top_n)
    top_extra  = extra_ctr.most_common(top_n)

    if not top_missed and not top_extra:
        ax.text(0.5, 0.5,
                "No missed or extra skills across all samples!\n🎉  Perfect extraction",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=_GREEN)
        ax.set_title("Skill Frequency Analysis", fontsize=11, fontweight="bold")
        return

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax.axis("off")
    ax.set_title("Skill Frequency Analysis  —  Most Commonly Missed vs Extra",
                 fontsize=11, fontweight="bold", pad=10)

    ax_miss = inset_axes(ax, width="47%", height="85%", loc="center left",
                         bbox_to_anchor=(0.01, 0, 1, 1), bbox_transform=ax.transAxes)
    ax_ext  = inset_axes(ax, width="47%", height="85%", loc="center right",
                         bbox_to_anchor=(0.01, 0, 1, 1), bbox_transform=ax.transAxes)

    def _freq_bar(sub_ax, items, colour, title):
        if not items:
            sub_ax.text(0.5, 0.5, "None", ha="center", va="center",
                        transform=sub_ax.transAxes, color=_GREY, fontsize=9)
            sub_ax.set_title(title, fontsize=9, fontweight="bold", color=colour)
            sub_ax.axis("off")
            return
        labels = [i[0] for i in reversed(items)]
        counts = [i[1] for i in reversed(items)]
        alpha_scale = [0.45 + 0.55 * (c / counts[-1]) for c in counts]
        y    = np.arange(len(labels))
        bars = sub_ax.barh(y, counts, color=colour, edgecolor="white", height=0.65)
        for bar, a in zip(bars, alpha_scale):
            bar.set_alpha(a)
        sub_ax.set_yticks(y)
        sub_ax.set_yticklabels(labels, fontsize=7)
        sub_ax.set_xlabel("# samples", fontsize=7)
        sub_ax.set_title(title, fontsize=9, fontweight="bold", color=colour, pad=6)
        sub_ax.spines[["top", "right"]].set_visible(False)
        sub_ax.tick_params(axis="both", labelsize=7)
        max_c = max(counts)
        for bar, val in zip(bars, counts):
            sub_ax.text(val + max_c * 0.03, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=6.5, color=_DARK)
        sub_ax.set_xlim(0, max_c * 1.25)
        import matplotlib
        sub_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    _freq_bar(ax_miss, top_missed, _RED,    "Top Missed Skills (FN)")
    _freq_bar(ax_ext,  top_extra,  _ORANGE, "Top Extra Skills  (FP)")


def _chart_faiss_scores(results: list[dict], ax) -> None:
    samples = [r for r in results if r.get("faiss") and not r["faiss"].get("error")]
    if not samples:
        ax.text(0.5, 0.5, "No FAISS data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("FAISS Hybrid Score", fontsize=11, fontweight="bold")
        return
    labels     = [_short_name(r["file"]) for r in samples]
    vals       = [r["faiss"]["best_score"] for r in samples]
    min_scores = [r["faiss"]["min_score"] / 100 for r in samples]
    avg_min    = sum(min_scores) / len(min_scores) if min_scores else 0.5
    _hbar(ax, labels, vals, _bar_colours(vals, avg_min, 0.75),
          "FAISS Hybrid Score per Sample", "Hybrid score (0–1)",
          vline=avg_min, vline_label=f"avg min_score ({avg_min:.0%})")


def _chart_faiss_cosine(results: list[dict], ax) -> None:
    samples = [r for r in results if r.get("faiss") and not r["faiss"].get("error")]
    if not samples:
        ax.text(0.5, 0.5, "No FAISS data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("FAISS Cosine Score", fontsize=11, fontweight="bold")
        return
    labels = [_short_name(r["file"]) for r in samples]
    vals   = [r["faiss"]["best_cosine"] for r in samples]
    _hbar(ax, labels, vals, _bar_colours(vals, 0.55, 0.72),
          "FAISS Cosine Score (raw SBERT)", "Cosine similarity",
          xlim=(0, 1.05), vline=0.72, vline_label="strong ≥ 0.72", fmt=".3f")


def _chart_faiss_match_ratio(results: list[dict], ax) -> None:
    samples = [r for r in results if r.get("faiss") and not r["faiss"].get("error")]
    if not samples:
        ax.text(0.5, 0.5, "No FAISS data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("FAISS Skill Match Ratio", fontsize=11, fontweight="bold")
        return
    labels = [_short_name(r["file"]) for r in samples]
    vals   = [r["faiss"]["best_match_ratio"] for r in samples]
    _hbar(ax, labels, vals, _bar_colours(vals, 0.45, 0.75),
          "Skill Match Ratio (best job)", "Fraction of required skills matched",
          vline=0.75, vline_label="strong ≥ 75%")


def _chart_faiss_rank(results: list[dict], ax, top_k: int = 5) -> None:
    import numpy as np
    samples = [r for r in results if r.get("faiss") and not r["faiss"].get("error")]
    if not samples:
        ax.text(0.5, 0.5, "No FAISS data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("FAISS Expected Title Rank", fontsize=11, fontweight="bold")
        return

    labels = [_short_name(r["file"]) for r in samples]
    ranks  = [
        r["faiss"]["expected_title_rank"] if r["faiss"]["expected_title_rank"]
        else top_k + 1
        for r in samples
    ]
    y = np.arange(len(labels))

    colours = [
        _GREEN  if rk == 1
        else _YELLOW if rk <= 3
        else _ORANGE if rk <= top_k
        else _RED
        for rk in ranks
    ]
    ax.hlines(y, 0, ranks, color=_GREY, linewidth=1, linestyle=":")
    ax.scatter(ranks, y, color=colours, zorder=5, s=70)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Rank (lower = better)", fontsize=9)
    ax.set_title(f"Expected Title Rank in Top-{top_k} Results",
                 fontsize=11, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.set_xlim(0, top_k + 2)
    ax.set_xticks(range(1, top_k + 2))
    ax.set_xticklabels(
        [str(i) for i in range(1, top_k + 1)] + [f">{top_k}"],
        fontsize=8,
        )
    ax.axvline(top_k + 0.5, color=_RED, linestyle="--", linewidth=1,
               label=f"Top-{top_k} cutoff")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7)

    for rk, yi in zip(ranks, y):
        label = str(rk) if rk <= top_k else f">{top_k}"
        ax.text(rk + 0.15, yi, label, va="center", fontsize=7, color=_DARK)


def _chart_timing_total(results: list[dict], ax) -> None:
    labels = [_short_name(r["file"]) for r in results]
    vals   = [r.get("elapsed_total_s", 0.0) for r in results]
    max_v  = max(vals) if vals else 1
    y = range(len(labels))
    bars = ax.barh(list(y), vals, color=_BLUE, edgecolor="white", height=0.55)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Seconds", fontsize=9)
    ax.set_title("Total Elapsed Time per Sample", fontsize=11, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(val + max_v * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=7, color=_DARK)


def _chart_timing_breakdown(results: list[dict], ax) -> None:
    import numpy as np
    labels = [_short_name(r["file"]) for r in results]
    ocr_t  = np.array([r.get("ocr",   {}).get("elapsed_s", 0) or 0 for r in results])
    nlp_t  = np.array([r.get("nlp",   {}).get("elapsed_s", 0) or 0 for r in results])
    fai_t  = np.array([r.get("faiss", {}).get("elapsed_s", 0) or 0 for r in results])
    y      = np.arange(len(labels))

    ax.barh(y, ocr_t,                      color=_BLUE,   edgecolor="white", height=0.55, label="OCR")
    ax.barh(y, nlp_t,  left=ocr_t,         color=_PURPLE, edgecolor="white", height=0.55, label="NLP")
    ax.barh(y, fai_t,  left=ocr_t + nlp_t, color=_TEAL,   edgecolor="white", height=0.55, label="FAISS")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Seconds", fontsize=9)
    ax.set_title("Stage Timing Breakdown per Sample", fontsize=11, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="lower right")


def _chart_pass_rate_donut(results: list[dict], ax, mode: str = "pipeline", top_k: int = 5) -> None:
    """Overall pass-rate bar chart — one bar per stage metric present in this run."""
    slices: list[tuple[str, float, str]] = []

    ocrs = [r for r in results if r.get("ocr") and not r["ocr"].get("error")]
    if ocrs and mode in ("ocr", "pipeline"):
        rate = sum(1 for r in ocrs if r["ocr"]["quality"] >= 70) / len(ocrs)
        slices.append((f"OCR quality ≥ 70%\n({rate:.0%})", rate, _BLUE))

    nlps = [r for r in results if r.get("nlp") and r["nlp"].get("f1") is not None]
    if nlps and mode in ("nlp", "pipeline"):
        rate = sum(1 for r in nlps if r["nlp"]["f1"] >= 0.75) / len(nlps)
        slices.append((f"NLP F1 ≥ 75%\n({rate:.0%})", rate, _PURPLE))

    fais = [r for r in results if r.get("faiss") and not r["faiss"].get("error")]
    if fais and mode in ("faiss", "pipeline"):
        rate_score = sum(1 for r in fais if r["faiss"]["score_pass"]) / len(fais)
        rate_found = sum(1 for r in fais if r["faiss"]["expected_title_found"]) / len(fais)
        slices.append((f"Score pass\n({rate_score:.0%})", rate_score, _TEAL))
        slices.append((f"Title in top-{top_k}\n({rate_found:.0%})", rate_found, _ORANGE))

    if not slices:
        ax.text(0.5, 0.5, "No pass-rate data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Pass Rate Summary", fontsize=11, fontweight="bold")
        return

    names  = [s[0] for s in slices]
    rates  = [s[1] for s in slices]
    colors = [s[2] for s in slices]

    y    = range(len(names))
    bars = ax.barh(list(y), rates, color=colors, edgecolor="white", height=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Pass rate", fontsize=9)
    ax.set_title("Overall Pass-Rate Summary", fontsize=11, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(1.0, color=_GREY, linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, rates):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=8,
                color=_GREEN if val >= 0.75 else (_YELLOW if val >= 0.5 else _RED),
                fontweight="bold")


# ── Layout planner ───────────────────────────────────────────────────────────
def _plan_charts(mode: str, results: list[dict]) -> list[tuple]:
    """
    Return an ordered list of (builder_fn, kwargs) to render based on
    which stages are present and which data is actually available.
    """
    has_nlp_labels = any(
        r.get("nlp", {}).get("precision") is not None for r in results
    )

    charts = []

    if mode in ("ocr", "pipeline"):
        charts.append((_chart_ocr_quality,     {}))
        charts.append((_chart_char_count,      {}))
        charts.append((_chart_text_similarity, {}))

    if mode in ("nlp", "pipeline") and has_nlp_labels:
        charts.append((_chart_nlp_prf,         {}))
        charts.append((_chart_nlp_counts,      {}))
        charts.append((_chart_nlp_radar,       {}))
        charts.append((_chart_skill_breakdown, {}))
        charts.append((_chart_skill_frequency, {}))

    if mode in ("faiss", "pipeline"):
        charts.append((_chart_faiss_scores,      {}))
        charts.append((_chart_faiss_cosine,      {}))
        charts.append((_chart_faiss_match_ratio, {}))

    # Rank chart uses top_k — injected by generate_charts
    charts.append(("__rank__", {}))

    if mode == "pipeline":
        charts.append((_chart_timing_total,     {}))
        charts.append((_chart_timing_breakdown, {}))

    charts.append(("__pass_rate__", {}))

    return charts


# ── Public API ───────────────────────────────────────────────────────────────
def generate_charts(
        report: dict,
        output_dir: str = ".",
        prefix: str = "eval",
        save_png: bool = False,
) -> list[str]:
    """
    Generate all charts from an evaluate() report dict and save them.

    Parameters
    ----------
    report     : dict returned by evaluate_pipeline.evaluate()
    output_dir : directory to write files into (created if absent)
    prefix     : filename prefix, e.g. "run1" → "run1_charts.pdf"
    save_png   : if True also write individual .png files per chart page

    Returns
    -------
    List of paths written (PDF first, then any PNGs).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        log.error("matplotlib is required for chart generation. "
                  "Install it with: pip install matplotlib")
        return []

    mode    = report.get("mode", "pipeline")
    top_k   = report.get("top_k", 5)
    results = report.get("results", [])

    if not results:
        log.warning("No results in report — nothing to chart.")
        return []

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{prefix}_charts.pdf"
    written: list[str] = []

    chart_specs = _plan_charts(mode, results)
    # Swap sentinel strings for real functions now that we have top_k
    chart_specs = [
        (_chart_faiss_rank,      {"top_k": top_k}) if fn == "__rank__"
        else (_chart_pass_rate_donut, {"mode": mode, "top_k": top_k}) if fn == "__pass_rate__"
        else (fn, kwargs)
        for fn, kwargs in chart_specs
    ]

    n_charts  = len(chart_specs)
    n_samples = len(results)
    row_h     = max(4.0, 0.45 * n_samples + 1.5)

    log.info(f"Generating {n_charts} charts → {pdf_path}")

    with PdfPages(str(pdf_path)) as pdf:
        # ── Cover page ────────────────────────────────────────────────────
        fig_cover, ax_cover = plt.subplots(figsize=(10, 4))
        ax_cover.axis("off")
        ax_cover.text(
            0.5, 0.65,
            "Pipeline Evaluation Report",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=_DARK, transform=ax_cover.transAxes,
        )
        agg   = report.get("aggregates", {})
        lines = [
            f"Mode: {mode.upper()}    Samples: {report.get('samples', '?')}    Top-K: {top_k}",
        ]
        if agg.get("ocr"):
            lines.append(
                f"OCR avg quality: {agg['ocr']['avg_quality']:.1f}%    "
                f"avg chars: {agg['ocr']['avg_chars']:.0f}"
            )
        if agg.get("nlp"):
            lines.append(
                f"NLP avg P: {agg['nlp']['avg_precision']:.1%}    "
                f"R: {agg['nlp']['avg_recall']:.1%}    "
                f"F1: {agg['nlp']['avg_f1']:.1%}"
            )
        if agg.get("faiss"):
            lines.append(
                f"FAISS title-found: {agg['faiss']['title_found_rate']:.0%}    "
                f"score-pass: {agg['faiss']['score_pass_rate']:.0%}    "
                f"avg score: {agg['faiss']['avg_hybrid_score']:.1f}/100"
            )
        ax_cover.text(
            0.5, 0.35, "\n".join(lines),
            ha="center", va="center", fontsize=11, color=_DARK,
            transform=ax_cover.transAxes, linespacing=1.8,
        )
        fig_cover.tight_layout()
        pdf.savefig(fig_cover, bbox_inches="tight")
        plt.close(fig_cover)

        # ── One chart per page ─────────────────────────────────────────────
        # Charts that turn the axis off or use inset_axes are incompatible
        # with tight_layout; skip it for those and rely on bbox_inches="tight".
        _no_tight = {_chart_skill_breakdown, _chart_skill_frequency,
                     _chart_nlp_radar, _chart_pass_rate_donut}

        for i, (fn, kwargs) in enumerate(chart_specs, start=1):
            if fn is _chart_nlp_radar:
                fig     = plt.figure(figsize=(7, 7))
                ax_main = fig.add_subplot(111, polar=True)
            else:
                fig, ax_main = plt.subplots(figsize=(10, row_h))

            fn(results, ax_main, **kwargs)

            if fn not in _no_tight:
                fig.tight_layout(pad=1.5)

            pdf.savefig(fig, bbox_inches="tight")

            if save_png:
                fn_name  = fn.__name__.replace("_chart_", "")
                png_path = out_dir / f"{prefix}_{i:02d}_{fn_name}.png"
                fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
                written.append(str(png_path))
                log.info(f"  PNG → {png_path}")

            plt.close(fig)

    written.insert(0, str(pdf_path))
    log.info(f"Charts saved: {pdf_path}  ({n_charts} charts + cover)")
    return written


# ── Standalone CLI ────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate charts PDF from a saved evaluate_pipeline report JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Basic — reads report.json, writes eval_charts.pdf in the current directory
  python charts.py report.json

  # Custom output directory and filename prefix
  python charts.py report.json --output-dir ./charts_out --prefix run1

  # Also save individual PNG files alongside the PDF
  python charts.py report.json --output-dir ./charts_out --prefix run1 --png

  # Quiet mode (suppress INFO logging)
  python charts.py report.json --quiet
""",
    )
    p.add_argument("report", help="Path to the report JSON file produced by evaluate_pipeline.py")
    p.add_argument("--output-dir", default=".",
                   help="Directory to write chart files into (default: current directory)")
    p.add_argument("--prefix", default="eval",
                   help="Filename prefix for chart files, e.g. 'run1' → 'run1_charts.pdf' (default: eval)")
    p.add_argument("--png", action="store_true",
                   help="Also save individual PNG files per chart page alongside the PDF")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress INFO log messages")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"ERROR: report file not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    with report_path.open() as f:
        report = json.load(f)

    written = generate_charts(
        report,
        output_dir=args.output_dir,
        prefix=args.prefix,
        save_png=args.png,
    )

    if written:
        print(f"\nFiles written ({len(written)}):")
        for path in written:
            print(f"  {path}")
    else:
        print("No charts were generated. Check that the report contains results.",
              file=sys.stderr)
        sys.exit(1)