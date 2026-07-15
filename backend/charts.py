"""
charts.py
=========
Chart generation companion for evaluate_pipeline.py.

REWRITTEN FOR READABILITY: instead of 22 charts, this produces 6 focused
pages — one for each thing you actually need to know when judging pipeline
quality. Every chart is sorted (best -> worst) so problem samples jump out
immediately, uses larger fonts, and drops metrics that are either redundant
with something else on the page or too noisy to be useful (raw vs aligned
CER, char-count bar, cosine-only bar, skill-pill tables, radar chart,
per-sample timing breakdown, confusion heatmap with an unmeasured TN cell).

Pages produced (per available data)
------------------------------------
  0. Cover page                — run summary in one glance
  1. OCR accuracy              — character & word accuracy (aligned), per sample
  2. NLP extraction quality    — precision / recall / F1 per sample, TP/FP/FN annotated
  3. NLP summary               — macro vs weighted average, skills + certs
  4. FAISS retrieval & scoring — hybrid score bars + rank dot plot, side by side
  5. Overall pass-rate summary — one bar per stage's pass criterion
  6. Aggregate statistics table — copy-ready numbers for a report/thesis

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
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger("evaluator.charts")

# ── Colour palette ────────────────────────────────────────────────────────
_GREEN  = "#2ecc71"
_YELLOW = "#f39c12"
_RED    = "#e74c3c"
_BLUE   = "#3498db"
_PURPLE = "#9b59b6"
_TEAL   = "#1abc9c"
_ORANGE = "#e67e22"
_GREY   = "#95a5a6"
_DARK   = "#2c3e50"


def _short_name(path: str, max_len: int = 30) -> str:
    name = Path(path).name
    if len(name) > max_len:
        name = name[:max_len - 1] + "…"
    return name


def _score_colour(val: float, lo: float, hi: float) -> str:
    if val >= hi:
        return _GREEN
    if val >= lo:
        return _YELLOW
    return _RED


def _no_data_msg(ax, title: str, detail: str = "") -> None:
    ax.set_facecolor("#f8f9fa")
    ax.text(0.5, 0.55, "No data for this stage",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=_GREY, transform=ax.transAxes)
    if detail:
        ax.text(0.5, 0.4, detail, ha="center", va="center", fontsize=9,
                color=_GREY, transform=ax.transAxes, linespacing=1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)


def _hbar(ax, labels, values, colours, title, xlabel, xlim=(0, 1),
          vlines=None, fmt=".0%", fontsize=10) -> None:
    """Generic, larger-font horizontal bar chart with value labels."""
    y = range(len(labels))
    bars = ax.barh(list(y), values, color=colours, edgecolor="white", height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)

    for bar, val in zip(bars, values):
        ax.text(min(val + (xlim[1] - xlim[0]) * 0.015, xlim[1] - 0.01),
                bar.get_y() + bar.get_height() / 2,
                format(val, fmt), va="center", ha="left", fontsize=9, color=_DARK)

    if vlines:
        for v, label, colour in vlines:
            ax.axvline(v, color=colour, linestyle="--", linewidth=1.3, label=label)
        ax.legend(fontsize=9, loc="lower right")


# ── 1. OCR accuracy ──────────────────────────────────────────────────────
def _chart_ocr_accuracy(results: list[dict], ax) -> None:
    """
    Character & Word Accuracy (order-tolerant / aligned) per sample — the
    single number that matters for OCR quality. Sorted worst-to-best so
    problem files are immediately visible at the top.
    """
    import numpy as np
    samples = [r for r in results if r.get("ocr") and r["ocr"].get("benchmark")]
    if not samples:
        _no_data_msg(ax, "OCR Accuracy per Sample",
                     "Add expected_text to labels.json to enable OCR scoring.")
        return

    samples.sort(key=lambda r: r["ocr"]["benchmark"]["char_accuracy_aligned"])
    labels    = [_short_name(r["file"]) for r in samples]
    char_acc  = np.array([r["ocr"]["benchmark"]["char_accuracy_aligned"] for r in samples])
    word_acc  = np.array([r["ocr"]["benchmark"]["word_accuracy_aligned"] for r in samples])

    y = np.arange(len(labels))
    h = 0.32
    ax.barh(y + h / 2, char_acc, height=h, color=_BLUE,  label="Character accuracy", edgecolor="white")
    ax.barh(y - h / 2, word_acc, height=h, color=_TEAL,  label="Word accuracy",      edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Accuracy (1 - error rate; higher = better)", fontsize=11)
    ax.set_title("OCR Accuracy per Sample (order-tolerant)", fontsize=14, fontweight="bold", pad=10)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0.85, color=_GREEN, linestyle="--", linewidth=1, label="Good ≥ 85%")
    ax.legend(fontsize=9, loc="lower right")

    for yi, c, w in zip(y, char_acc, word_acc):
        ax.text(c + 0.015, yi + h / 2, f"{c:.0%}", va="center", fontsize=8, color=_DARK)
        ax.text(w + 0.015, yi - h / 2, f"{w:.0%}", va="center", fontsize=8, color=_DARK)


# ── 2. NLP extraction quality ───────────────────────────────────────────
def _chart_nlp_prf(results: list[dict], ax) -> None:
    """
    Precision / Recall / F1 per sample for skill extraction, sorted
    worst-to-best F1. TP/FP/FN counts are annotated on each row instead of
    a separate stacked-bar chart, so this one view answers both "how good"
    and "what kind of error."
    """
    import numpy as np
    samples = [r for r in results if r.get("nlp") and r["nlp"].get("precision") is not None]
    if not samples:
        _no_data_msg(ax, "NLP Extraction Quality per Sample",
                     "Add expected_skills to labels.json to enable this chart.")
        return

    samples.sort(key=lambda r: r["nlp"]["f1"])
    labels = [_short_name(r["file"]) for r in samples]
    prec   = [r["nlp"]["precision"] for r in samples]
    rec    = [r["nlp"]["recall"]    for r in samples]
    f1     = [r["nlp"]["f1"]        for r in samples]

    n = len(labels)
    y = np.arange(n)
    h = 0.24
    ax.barh(y + h, prec, height=h, color=_BLUE,   label="Precision", edgecolor="white")
    ax.barh(y,     rec,  height=h, color=_PURPLE, label="Recall",    edgecolor="white")
    ax.barh(y - h, f1,   height=h, color=_TEAL,   label="F1",        edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.35)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title("NLP Skill Extraction: Precision / Recall / F1 per Sample",
                 fontsize=14, fontweight="bold", pad=10)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0.75, color=_RED, linestyle="--", linewidth=1, label="target 75%")
    ax.legend(fontsize=9, loc="lower right")

    for i, r in enumerate(samples):
        n_ = r["nlp"]
        for val, off in ((prec[i], h), (rec[i], 0), (f1[i], -h)):
            ax.text(val + 0.015, y[i] + off, f"{val:.0%}", va="center", fontsize=8, color=_DARK)
        ax.text(1.14, y[i], f"TP {n_['tp']}  FP {n_['fp']}  FN {n_['fn']}",
                va="center", fontsize=8, color=_GREY)


# ── 3. NLP summary (macro vs weighted, skills + certs) ──────────────────
def _chart_nlp_summary(results: list[dict], ax) -> None:
    """
    One bar group per metric (Precision/Recall/F1), macro vs weighted
    average, shown for skills and — if labelled — certifications. This is
    the "so what's the overall number" chart, complementing the per-sample
    detail chart above it.
    """
    import numpy as np

    skill_samples = [r for r in results if r.get("nlp") and r["nlp"].get("precision") is not None]
    cert_samples  = [r for r in results if r.get("nlp")
                     and r["nlp"].get("cert_metrics", {}).get("precision") is not None]

    if not skill_samples and not cert_samples:
        _no_data_msg(ax, "NLP Summary: Macro vs Weighted Average")
        return

    def _macro_weighted(samples, key_fn):
        p   = np.array([key_fn(r)["precision"] for r in samples])
        r_  = np.array([key_fn(r)["recall"]    for r in samples])
        f1  = np.array([key_fn(r)["f1"]        for r in samples])
        sup = np.array([key_fn(r)["tp"] + key_fn(r)["fn"] for r in samples])
        w   = sup.sum() or 1
        macro    = np.array([p.mean(), r_.mean(), f1.mean()])
        weighted = np.array([(p * sup).sum() / w, (r_ * sup).sum() / w, (f1 * sup).sum() / w])
        return macro, weighted, int(sup.sum())

    metrics = ["Precision", "Recall", "F1-score"]
    x = np.arange(len(metrics))
    groups = []  # (label, macro, weighted, colour_macro, colour_weighted, support, n)
    if skill_samples:
        macro, weighted, sup = _macro_weighted(skill_samples, lambda r: r["nlp"])
        groups.append(("Skills", macro, weighted, _BLUE, "#9fd3f0", sup, len(skill_samples)))
    if cert_samples:
        macro, weighted, sup = _macro_weighted(cert_samples, lambda r: r["nlp"]["cert_metrics"])
        groups.append(("Certs", macro, weighted, _PURPLE, "#d7b8ea", sup, len(cert_samples)))

    n_groups = len(groups)
    total_w  = 0.34
    bar_w    = total_w / (2 * n_groups)
    offset0  = -total_w / 2

    for gi, (label, macro, weighted, cm, cw, sup, n) in enumerate(groups):
        pos_m = x + offset0 + (2 * gi) * bar_w + bar_w / 2
        pos_w = x + offset0 + (2 * gi + 1) * bar_w + bar_w / 2
        bars_m = ax.bar(pos_m, macro, width=bar_w, color=cm,
                        label=f"{label} — macro avg", edgecolor="white")
        bars_w = ax.bar(pos_w, weighted, width=bar_w, color=cw,
                        label=f"{label} — weighted avg", edgecolor="white")
        for bar in list(bars_m) + list(bars_w):
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}",
                    ha="center", fontsize=8, color=_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("NLP Summary: Macro vs Weighted Average", fontsize=14, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.75, color=_RED, linestyle="--", linewidth=1)
    ax.legend(fontsize=8, ncol=n_groups, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    caption = "   ".join(f"{label}: support={sup}, n={n} samples" for label, *_r, sup, n in groups)
    ax.text(0.5, 1.08, caption, transform=ax.transAxes, ha="center", fontsize=9, color=_GREY)


# ── 4. FAISS retrieval & scoring (combined) ─────────────────────────────
def _chart_faiss_combined(results: list[dict], fig, top_k: int = 5) -> None:
    """
    Two panels on one page: hybrid score per sample (left) and the rank at
    which the correct job was actually found (right). Together they answer
    "how confident was the match" and "did it even find the right job."
    """
    import numpy as np
    samples = [r for r in results if r.get("faiss") and not r["faiss"].get("error")]

    ax_score = fig.add_subplot(1, 2, 1)
    ax_rank  = fig.add_subplot(1, 2, 2)

    if not samples:
        _no_data_msg(ax_score, "FAISS Hybrid Score")
        _no_data_msg(ax_rank, "FAISS Rank of Correct Job")
        return

    # -- left: hybrid score, sorted worst-to-best --
    s_sorted = sorted(samples, key=lambda r: r["faiss"]["best_score"])
    labels   = [_short_name(r["file"]) for r in s_sorted]
    scores   = [r["faiss"]["best_score"] / 100 for r in s_sorted]
    min_sc   = [r["faiss"]["min_score"] / 100 for r in s_sorted]
    avg_min  = sum(min_sc) / len(min_sc) if min_sc else 0.5
    colours  = [_score_colour(v, avg_min, 0.75) for v in scores]
    _hbar(ax_score, labels, scores, colours, "Hybrid Match Score per Sample",
          "Score (0-1)", vlines=[(avg_min, f"avg min_score ({avg_min:.0%})", _RED)])

    # -- right: rank of expected title, sorted worst-to-best --
    r_sorted = sorted(samples, key=lambda r: r["faiss"]["expected_title_rank"] or top_k + 1)
    labels2  = [_short_name(r["file"]) for r in r_sorted]
    ranks    = [r["faiss"]["expected_title_rank"] or top_k + 1 for r in r_sorted]
    y2 = np.arange(len(labels2))
    colours2 = [_GREEN if rk == 1 else _YELLOW if rk <= 3 else _ORANGE if rk <= top_k else _RED
                for rk in ranks]
    ax_rank.hlines(y2, 0, ranks, color=_GREY, linewidth=1, linestyle=":")
    ax_rank.scatter(ranks, y2, color=colours2, zorder=5, s=90)
    ax_rank.set_yticks(y2)
    ax_rank.set_yticklabels(labels2, fontsize=10)
    ax_rank.set_xlabel("Rank of correct job (lower = better)", fontsize=11)
    ax_rank.set_title(f"Where the Correct Job Landed (Top-{top_k})", fontsize=14, fontweight="bold", pad=10)
    ax_rank.invert_yaxis()
    ax_rank.set_xlim(0, top_k + 2)
    ax_rank.set_xticks(range(1, top_k + 2))
    ax_rank.set_xticklabels([str(i) for i in range(1, top_k + 1)] + [f">{top_k}"], fontsize=9)
    ax_rank.spines[["top", "right"]].set_visible(False)
    for rk, yi in zip(ranks, y2):
        label = str(rk) if rk <= top_k else f">{top_k}"
        ax_rank.text(rk + 0.2, yi, label, va="center", fontsize=9, color=_DARK)


# ── 5. Overall pass-rate summary ────────────────────────────────────────
def _chart_pass_rate(results: list[dict], ax, mode: str = "pipeline", top_k: int = 5) -> None:
    """One bar per pass/fail criterion actually evaluated in this run."""
    slices: list[tuple[str, float, str]] = []

    ocrs = [r for r in results if r.get("ocr") and r["ocr"].get("benchmark")]
    if ocrs and mode in ("ocr", "pipeline"):
        rate = sum(1 for r in ocrs if r["ocr"]["benchmark"]["char_accuracy_aligned"] >= 0.85) / len(ocrs)
        slices.append((f"OCR char accuracy ≥ 85%", rate, _BLUE))

    nlps = [r for r in results if r.get("nlp") and r["nlp"].get("f1") is not None]
    if nlps and mode in ("nlp", "pipeline"):
        rate = sum(1 for r in nlps if r["nlp"]["f1"] >= 0.75) / len(nlps)
        slices.append(("NLP skill F1 ≥ 75%", rate, _PURPLE))

    fais = [r for r in results if r.get("faiss") and not r["faiss"].get("error")]
    if fais and mode in ("faiss", "pipeline"):
        rate_score = sum(1 for r in fais if r["faiss"]["score_pass"]) / len(fais)
        rate_found = sum(1 for r in fais if r["faiss"]["expected_title_found"]) / len(fais)
        slices.append(("FAISS score passes threshold", rate_score, _TEAL))
        slices.append((f"Correct job in top-{top_k}", rate_found, _ORANGE))

    if not slices:
        _no_data_msg(ax, "Overall Pass-Rate Summary")
        return

    names  = [f"{s[0]}\n({s[1]:.0%})" for s in slices]
    rates  = [s[1] for s in slices]
    colors = [s[2] for s in slices]

    y = range(len(names))
    bars = ax.barh(list(y), rates, color=colors, edgecolor="white", height=0.55)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Pass rate", fontsize=11)
    ax.set_title("Overall Pass-Rate Summary", fontsize=14, fontweight="bold", pad=10)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(1.0, color=_GREY, linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, rates):
        ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.0%}",
                va="center", fontsize=11, fontweight="bold",
                color=_GREEN if val >= 0.75 else (_YELLOW if val >= 0.5 else _RED))


# ── 6. Aggregate statistics table ───────────────────────────────────────
def _chart_aggregate_table(results: list[dict], ax, mode: str = "pipeline") -> None:
    """Copy-ready numeric summary — the only page meant to be read as text."""
    import numpy as np
    ax.axis("off")
    ax.set_title("Aggregate Statistics — Copy-Ready", fontsize=14, fontweight="bold", pad=14)

    rows = []

    ocrs = [r for r in results if r.get("ocr") and r["ocr"].get("benchmark")]
    if ocrs and mode in ("ocr", "pipeline"):
        rows.append(("OCR", "", ""))
        ca = np.array([r["ocr"]["benchmark"]["char_accuracy_aligned"] for r in ocrs])
        wa = np.array([r["ocr"]["benchmark"]["word_accuracy_aligned"] for r in ocrs])
        rows.append(("  Character accuracy (aligned)", f"{ca.mean():.1%}", f"±{ca.std():.1%}"))
        rows.append(("  Word accuracy (aligned)",       f"{wa.mean():.1%}", f"±{wa.std():.1%}"))
        rows.append(("  n samples", str(len(ocrs)), ""))

    nlps = [r for r in results if r.get("nlp") and r["nlp"].get("precision") is not None]
    if nlps and mode in ("nlp", "pipeline"):
        rows.append(("NLP — Skills", "", ""))
        p = np.array([r["nlp"]["precision"] for r in nlps])
        rc = np.array([r["nlp"]["recall"]    for r in nlps])
        f1 = np.array([r["nlp"]["f1"]        for r in nlps])
        sup = np.array([r["nlp"]["tp"] + r["nlp"]["fn"] for r in nlps])
        w = sup.sum() or 1
        rows.append(("  Macro-avg P / R / F1", f"{p.mean():.1%} / {rc.mean():.1%} / {f1.mean():.1%}", ""))
        rows.append(("  Weighted-avg P / R / F1",
                     f"{(p*sup).sum()/w:.1%} / {(rc*sup).sum()/w:.1%} / {(f1*sup).sum()/w:.1%}", ""))
        rows.append(("  Total support / TP / FP / FN",
                     f"{int(sup.sum())} / {sum(r['nlp']['tp'] for r in nlps)} / "
                     f"{sum(r['nlp']['fp'] for r in nlps)} / {sum(r['nlp']['fn'] for r in nlps)}", ""))
        rows.append(("  n samples", str(len(nlps)), ""))

    certs = [r for r in results if r.get("nlp")
             and r["nlp"].get("cert_metrics", {}).get("precision") is not None]
    if certs and mode in ("nlp", "pipeline"):
        rows.append(("NLP — Certifications", "", ""))
        cm = [r["nlp"]["cert_metrics"] for r in certs]
        p = np.array([c["precision"] for c in cm])
        rc = np.array([c["recall"]    for c in cm])
        f1 = np.array([c["f1"]        for c in cm])
        rows.append(("  Macro-avg P / R / F1", f"{p.mean():.1%} / {rc.mean():.1%} / {f1.mean():.1%}", ""))
        rows.append(("  n samples", str(len(certs)), ""))

    fais = [r for r in results if r.get("faiss") and not r["faiss"].get("error")]
    if fais and mode in ("faiss", "pipeline"):
        rows.append(("FAISS — Job Retrieval", "", ""))
        rr   = np.array([r["faiss"].get("reciprocal_rank", 0.0) for r in fais])
        ndcg = np.array([r["faiss"].get("ndcg_at_k",       0.0) for r in fais])
        h1   = np.array([r["faiss"].get("hit_at_1",        0)   for r in fais], dtype=float)
        rows.append(("  MRR", f"{rr.mean():.3f}", f"±{rr.std():.3f}"))
        rows.append(("  Mean NDCG@K", f"{ndcg.mean():.3f}", f"±{ndcg.std():.3f}"))
        rows.append(("  Mean Hit@1", f"{h1.mean():.1%}", ""))
        fnd   = sum(1 for r in fais if r["faiss"]["expected_title_found"])
        spass = sum(1 for r in fais if r["faiss"]["score_pass"])
        hyb   = np.array([r["faiss"]["best_score"] for r in fais])
        rows.append(("  Title found in top-K", f"{fnd/len(fais):.1%}", ""))
        rows.append(("  Score-pass rate", f"{spass/len(fais):.1%}", ""))
        rows.append(("  Avg hybrid score", f"{hyb.mean():.1f}/100", f"±{hyb.std():.1f}"))
        rows.append(("  n samples", str(len(fais)), ""))

    if not rows:
        ax.text(0.5, 0.5, "No aggregate data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color=_GREY)
        return

    col_x   = [0.02, 0.62, 0.85]
    y_start = 0.90
    row_h   = min(0.85 / max(len(rows), 1), 0.06)

    for cx, cl in zip(col_x, ["Metric", "Value", "Std Dev"]):
        ax.text(cx, y_start, cl, transform=ax.transAxes, fontsize=10, fontweight="bold", color=_DARK)
    ax.plot([0, 1], [y_start - 0.02, y_start - 0.02], color=_GREY, linewidth=0.8,
            transform=ax.transAxes, clip_on=False)

    for i, (metric, val, std) in enumerate(rows):
        y = y_start - 0.045 - i * row_h
        if y < 0.02:
            break
        is_section = (val == "" and std == "")
        colour, weight = (_DARK, "bold") if is_section else ("#333333", "normal")
        if not is_section and i % 2 == 0:
            ax.fill_between([0, 1], [y - row_h * 0.35] * 2, [y + row_h * 0.55] * 2,
                            color="#f0f4f8", zorder=0, transform=ax.transAxes, clip_on=False)
        ax.text(col_x[0], y, metric, transform=ax.transAxes, fontsize=9, color=colour, fontweight=weight)
        ax.text(col_x[1], y, val, transform=ax.transAxes, fontsize=9, color=colour)
        ax.text(col_x[2], y, std, transform=ax.transAxes, fontsize=9, color=_GREY)


# ── Public API ───────────────────────────────────────────────────────────
def generate_charts(
        report: dict,
        output_dir: str = ".",
        prefix: str = "eval",
        save_png: bool = False,
) -> list[str]:
    """
    Generate the 6 essential charts (+ cover page) from a report dict.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["font.size"] = 10
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

    n_samples = len(results)
    row_h     = max(4.0, 0.5 * n_samples + 1.5)

    log.info(f"Generating charts → {pdf_path}")

    with PdfPages(str(pdf_path)) as pdf:
        # ── Cover page ────────────────────────────────────────────────
        fig_cover, ax_cover = plt.subplots(figsize=(10, 4.5))
        ax_cover.axis("off")
        ax_cover.text(0.5, 0.68, "Pipeline Evaluation Report",
                      ha="center", va="center", fontsize=24, fontweight="bold",
                      color=_DARK, transform=ax_cover.transAxes)
        agg = report.get("aggregates", {})
        lines = [f"Mode: {mode.upper()}    Samples: {report.get('samples', '?')}    Top-K: {top_k}"]
        if agg.get("ocr") and agg["ocr"].get("benchmark_samples"):
            ca = agg["ocr"].get("avg_char_accuracy_aligned")
            wa = agg["ocr"].get("avg_word_accuracy_aligned")
            if ca is not None:
                lines.append(f"OCR — char accuracy: {ca:.1%}    word accuracy: {wa:.1%}"
                             f"  (n={agg['ocr']['benchmark_samples']})")
        if agg.get("nlp"):
            n = agg["nlp"]
            lines.append(f"NLP skills — macro P: {n.get('macro_avg_precision', 0):.1%}  "
                         f"R: {n.get('macro_avg_recall', 0):.1%}  F1: {n.get('macro_avg_f1', 0):.1%}  "
                         f"(support: {n.get('total_support', 0)})")
        if agg.get("faiss"):
            fa = agg["faiss"]
            line = (f"FAISS — title-found: {fa['title_found_rate']:.0%}    "
                    f"score-pass: {fa['score_pass_rate']:.0%}    "
                    f"avg score: {fa['avg_hybrid_score']:.1f}/100")
            if fa.get("mrr") is not None:
                line += f"    MRR: {fa['mrr']:.3f}"
            lines.append(line)
        ax_cover.text(0.5, 0.35, "\n".join(lines), ha="center", va="center",
                      fontsize=12, color=_DARK, transform=ax_cover.transAxes, linespacing=2.0)
        fig_cover.tight_layout()
        pdf.savefig(fig_cover, bbox_inches="tight")
        plt.close(fig_cover)

        page_i = 0

        def _save(fig, name):
            nonlocal page_i
            page_i += 1
            pdf.savefig(fig, bbox_inches="tight")
            if save_png:
                png_path = out_dir / f"{prefix}_{page_i:02d}_{name}.png"
                fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
                written.append(str(png_path))
            plt.close(fig)

        # 1. OCR accuracy
        if mode in ("ocr", "pipeline"):
            fig, ax = plt.subplots(figsize=(10, row_h))
            _chart_ocr_accuracy(results, ax)
            fig.tight_layout(pad=1.5)
            _save(fig, "ocr_accuracy")

        # 2 & 3. NLP
        has_nlp_labels = any(r.get("nlp", {}).get("precision") is not None for r in results)
        if mode in ("nlp", "pipeline") and has_nlp_labels:
            fig, ax = plt.subplots(figsize=(11, row_h))
            _chart_nlp_prf(results, ax)
            fig.tight_layout(pad=1.5)
            _save(fig, "nlp_extraction")

            fig, ax = plt.subplots(figsize=(9, 5.5))
            _chart_nlp_summary(results, ax)
            _save(fig, "nlp_summary")

        # 4. FAISS combined
        if mode in ("faiss", "pipeline"):
            fig = plt.figure(figsize=(13, row_h))
            _chart_faiss_combined(results, fig, top_k=top_k)
            fig.tight_layout(pad=1.5)
            _save(fig, "faiss_retrieval")

        # 5. Pass-rate summary
        fig, ax = plt.subplots(figsize=(9, 4))
        _chart_pass_rate(results, ax, mode=mode, top_k=top_k)
        fig.tight_layout(pad=1.5)
        _save(fig, "pass_rate")

        # 6. Aggregate table (always last)
        fig, ax = plt.subplots(figsize=(9, 6))
        _chart_aggregate_table(results, ax, mode=mode)
        _save(fig, "aggregate_table")

    written.insert(0, str(pdf_path))
    log.info(f"Charts saved: {pdf_path}  ({page_i} charts + cover)")
    return written


# ── Standalone CLI ────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a compact, readable charts PDF from a saved "
                    "evaluate_pipeline report JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python charts.py report.json
  python charts.py report.json --output-dir ./charts_out --prefix run1
  python charts.py report.json --output-dir ./charts_out --prefix run1 --png
  python charts.py report.json --quiet
""",
    )
    p.add_argument("report", help="Path to the report JSON file produced by evaluate_pipeline.py")
    p.add_argument("--output-dir", default=".", help="Directory to write chart files into")
    p.add_argument("--prefix", default="eval", help="Filename prefix for chart files")
    p.add_argument("--png", action="store_true", help="Also save individual PNG files")
    p.add_argument("--quiet", action="store_true", help="Suppress INFO log messages")
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

    written = generate_charts(report, output_dir=args.output_dir,
                              prefix=args.prefix, save_png=args.png)

    if written:
        print(f"\nFiles written ({len(written)}):")
        for path in written:
            print(f"  {path}")
    else:
        print("No charts were generated. Check that the report contains results.",
              file=sys.stderr)
        sys.exit(1)