"""
generate_charts.py
==================
Diagnostic chart generation for evaluate_pipeline.py.

Call generate_charts(report, out_dir) after evaluate() to produce a
folder of diagnostic PNGs.  Works with the JSON report dict returned by
evaluate() or loaded from --out report.json.

Every chart reads only fields that evaluate_pipeline.py actually emits.
Field reference (per_resume dict keys):
    filename, skill_recall, skill_precision, skill_f1,
    matched_skills, missing_skills, extra_skills,
    nlp_pipeline_accuracy,       # pipeline's own fuzzy NLP_ACCURACY (0-1 float)
    job_title_match, expected_title, best_title, title_similarity,
    faiss_top1_match, faiss_top5_match, faiss_best_rank,
    faiss_cosine_score, faiss_match_ratio,
    best_hybrid_score,           # 0-100 scale
    min_match_score, score_pass,
    ocr_quality_pct,             # alnum-ratio 0-100
    text_source,                 # 'native_pdf' | 'ocr' | 'camera_scan_mlkit'
    text_source_correct,         # 1/0/None (only when label has expected_text_source)
    cert_recall

Charts produced
---------------
OCR stage  (what text came in and how clean it was)
  01_ocr_quality_by_resume.png       — bar: ocr_quality_pct per resume, colored by text_source
  02_ocr_quality_distribution.png    — histogram: ocr_quality_pct split by text_source
  03_ocr_source_breakdown.png        — bar: count of native_pdf / ocr / camera_scan_mlkit
  04_ocr_f1_by_source.png            — box: skill F1 grouped by text_source (shows OCR impact on NLP)

NLP extraction stage  (how well skills were pulled from the text)
  05_nlp_skill_f1_per_resume.png         — horizontal bar: skill F1 per resume sorted
  06_nlp_recall_vs_precision.png         — scatter: recall vs precision, colored by pass/fail
  07_nlp_self_score_vs_gt_f1.png         — scatter: pipeline NLP self-score vs ground-truth F1
                                            (calibration check — are they tracking each other?)
  08_nlp_missing_skills.png              — top-N most frequently missed skills
  09_nlp_extra_skills.png                — top-N most frequently hallucinated skills
  10_nlp_recall_precision_gap.png        — bar: recall − precision per resume

FAISS / scoring stage  (how well the ranker placed the right job)
  11_faiss_rank_distribution.png         — histogram: faiss_best_rank (1-6) across all resumes
  12_faiss_cosine_vs_ratio.png           — scatter: cosine_score vs match_ratio, colored by top1 hit
  13_faiss_score_distribution.png        — histogram: best_hybrid_score with threshold line
  14_faiss_score_vs_f1.png               — scatter: hybrid score vs skill F1 (calibration)
  15_faiss_title_by_category.png         — stacked bar: top1/top5/miss by job category

Combined pipeline
  16_pipeline_failure_attribution.png    — bar: resumes attributed to each failure stage
  17_pipeline_stage_summary.png          — grouped bar: OCR / NLP / FAISS means by text_source
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Palette ───────────────────────────────────────────────────────────────────
PASS_COLOR   = "#1D9E75"   # teal   — pass / native_pdf
FAIL_COLOR   = "#D85A30"   # coral  — fail / image
MID_COLOR    = "#378ADD"   # blue   — neutral / ocr
WARN_COLOR   = "#BA7517"   # amber  — warning / camera
GRAY_COLOR   = "#888780"   # gray   — both-fail / structural
PURPLE_COLOR = "#7B5EA7"   # purple — top5 hit (not top1)
GRID_COLOR   = "#E8E8E8"
BG_COLOR     = "#FAFAFA"
TEXT_COLOR   = "#2C2C2A"

# One colour per text_source value — consistent across all charts
SOURCE_COLORS = {
    "native_pdf":        PASS_COLOR,
    "ocr":               MID_COLOR,
    "camera_scan_mlkit": WARN_COLOR,
    "unknown":           GRAY_COLOR,
}

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "figure.dpi":        140,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _display_name(filename: str) -> str:
    name = Path(filename).stem
    return name.split(" - ", 1)[1].strip() if " - " in name else name.strip()


def _source_color(r: dict) -> str:
    return SOURCE_COLORS.get(r.get("text_source", "unknown"), GRAY_COLOR)


def _save(fig, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [chart] {out}")


def _patch(color: str, label: str) -> mpatches.Patch:
    return mpatches.Patch(color=color, label=label)


def _source_legend_patches(sources_present: set[str]) -> list[mpatches.Patch]:
    order = ["native_pdf", "ocr", "camera_scan_mlkit", "unknown"]
    labels = {
        "native_pdf":        "native PDF",
        "ocr":               "EasyOCR fallback",
        "camera_scan_mlkit": "Camera / ML Kit",
        "unknown":           "unknown",
    }
    return [_patch(SOURCE_COLORS[s], labels[s])
            for s in order if s in sources_present]


# ══════════════════════════════════════════════════════════════════════════════
#  OCR STAGE
# ══════════════════════════════════════════════════════════════════════════════

def chart_ocr_quality_by_resume(per_resume: list[dict], out: Path) -> None:
    """
    Bar: ocr_quality_pct per resume sorted ascending, bars colored by text_source.

    ocr_quality_pct is the pipeline's alnum-ratio (0-100).  It is NOT a
    character-error-rate — it measures how clean the extracted text is, not
    whether every word is correct.  Bars grouped by source show whether the
    EasyOCR/camera paths produce noisier output than native PDF.
    """
    sorted_r = sorted(per_resume, key=lambda r: r.get("ocr_quality_pct", 0))
    names    = [_display_name(r["filename"]) for r in sorted_r]
    vals     = [r.get("ocr_quality_pct", 0) for r in sorted_r]
    colors   = [_source_color(r) for r in sorted_r]
    sources  = {r.get("text_source", "unknown") for r in sorted_r}

    n = len(names)
    h = max(4, n * 0.28)
    fig, ax = plt.subplots(figsize=(9, h))
    ax.barh(range(n), vals, color=colors, height=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("OCR alnum-ratio quality (%)")
    ax.set_title("OCR text quality per resume — by extraction source\n"
                 "(alnum-ratio: how clean the extracted text is, not CER)")
    ax.axvline(70, color=WARN_COLOR, linewidth=1.2, linestyle="--",
               alpha=0.8, label="70% quality floor")
    ax.set_xlim(0, 108)
    ax.grid(axis="x")
    ax.legend(handles=_source_legend_patches(sources) + [
        _patch(WARN_COLOR, "70% floor")
    ], frameon=False, fontsize=9)
    _save(fig, out, "01_ocr_quality_by_resume.png")


def chart_ocr_quality_distribution(per_resume: list[dict], out: Path) -> None:
    """
    Overlaid histograms of ocr_quality_pct split by text_source.

    A left-shifted image/OCR distribution vs a right-shifted native_pdf
    distribution confirms that EasyOCR or camera paths are noisier.
    """
    groups: dict[str, list[float]] = {}
    for r in per_resume:
        src = r.get("text_source", "unknown")
        groups.setdefault(src, []).append(r.get("ocr_quality_pct", 0))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 100, 26)
    for src, vals in groups.items():
        color = SOURCE_COLORS.get(src, GRAY_COLOR)
        ax.hist(vals, bins=bins, color=color, alpha=0.65,
                label=f"{src} (n={len(vals)})", edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(vals), color=color, linewidth=1.4, linestyle="--", alpha=0.9)

    ax.axvline(70, color=WARN_COLOR, linewidth=1.2, linestyle=":",
               alpha=0.8, label="70% floor")
    ax.set_xlabel("OCR alnum-ratio quality (%)")
    ax.set_ylabel("Number of resumes")
    ax.set_title("OCR quality distribution by text source\n"
                 "(dashed vertical = group mean)")
    ax.set_xlim(0, 105)
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out, "02_ocr_quality_distribution.png")


def chart_ocr_source_breakdown(per_resume: list[dict], out: Path) -> None:
    """
    Bar: count of each text_source value, with mean OCR quality annotated.

    Shows what proportion of your test set hit each extraction path and
    how clean that path's output was on average.
    """
    groups: dict[str, list[float]] = {}
    for r in per_resume:
        src = r.get("text_source", "unknown")
        groups.setdefault(src, []).append(r.get("ocr_quality_pct", 0))

    order  = ["native_pdf", "ocr", "camera_scan_mlkit", "unknown"]
    labels = [s for s in order if s in groups]
    counts = [len(groups[s]) for s in labels]
    means  = [np.mean(groups[s]) for s in labels]
    colors = [SOURCE_COLORS.get(s, GRAY_COLOR) for s in labels]
    display = {
        "native_pdf":        "Native PDF\n(PyMuPDF)",
        "ocr":               "EasyOCR\nfallback",
        "camera_scan_mlkit": "Camera\n(ML Kit)",
        "unknown":           "Unknown",
    }

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.8), 5))
    bars = ax.bar([display.get(l, l) for l in labels], counts,
                  color=colors, width=0.5, edgecolor="white", linewidth=0.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f"avg quality\n{mean:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of resumes")
    ax.set_title("OCR — extraction path breakdown\n"
                 "(count per path + mean quality score)")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.grid(axis="y")
    _save(fig, out, "03_ocr_source_breakdown.png")


def chart_ocr_f1_by_source(per_resume: list[dict], out: Path) -> None:
    """
    Box plot: skill F1 grouped by text_source.

    This is the key downstream-impact chart: if native_pdf F1 >> ocr F1,
    the OCR path is garbling text enough to hurt skill extraction.
    """
    groups: dict[str, list[float]] = {}
    for r in per_resume:
        src = r.get("text_source", "unknown")
        groups.setdefault(src, []).append(r["skill_f1"])

    if len(groups) < 2:
        # Still draw — single group is informative as a single box
        pass

    order  = ["native_pdf", "ocr", "camera_scan_mlkit", "unknown"]
    labels = [s for s in order if s in groups]
    data   = [groups[s] for s in labels]
    colors = [SOURCE_COLORS.get(s, GRAY_COLOR) for s in labels]
    disp   = {
        "native_pdf":        f"Native PDF\n(n={len(groups.get('native_pdf',[]))})",
        "ocr":               f"EasyOCR\n(n={len(groups.get('ocr',[]))})",
        "camera_scan_mlkit": f"Camera ML Kit\n(n={len(groups.get('camera_scan_mlkit',[]))})",
        "unknown":           f"Unknown\n(n={len(groups.get('unknown',[]))})",
    }
    xlabels = [disp[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 2), 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops=dict(color=TEXT_COLOR, linewidth=1.8))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Skill F1  (vs ground-truth labels)")
    ax.set_title("OCR downstream impact — skill F1 by extraction path\n"
                 "(low F1 on OCR/camera paths = OCR errors hurting skill extraction)")
    ax.set_ylim(-0.05, 1.12)
    ax.axhline(0.8, color=PASS_COLOR, linewidth=1, linestyle="--",
               alpha=0.6, label="F1 = 0.80 target")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out, "04_ocr_f1_by_source.png")


# ══════════════════════════════════════════════════════════════════════════════
#  NLP EXTRACTION STAGE
# ══════════════════════════════════════════════════════════════════════════════

def chart_nlp_f1_per_resume(per_resume: list[dict], out: Path) -> None:
    """Horizontal bar: skill F1 per resume sorted, colored by F1 band."""
    sorted_r = sorted(per_resume, key=lambda r: r["skill_f1"])
    names    = [_display_name(r["filename"]) for r in sorted_r]
    vals     = [r["skill_f1"] for r in sorted_r]
    colors   = [
        PASS_COLOR if v >= 0.8 else (WARN_COLOR if v >= 0.5 else FAIL_COLOR)
        for v in vals
    ]

    n = len(names)
    h = max(4, n * 0.28)
    fig, ax = plt.subplots(figsize=(9, h))
    ax.barh(range(n), vals, color=colors, height=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Skill F1  (vs your ground-truth labels)")
    ax.set_title("NLP skill extraction — F1 per resume (sorted)\n"
                 "(measures: did the LLM find the right skills from the text?)")
    ax.axvline(0.8, color=PASS_COLOR, linewidth=1.2, linestyle="--",
               alpha=0.7, label="F1 = 0.80 target")
    ax.axvline(0.5, color=FAIL_COLOR, linewidth=1.2, linestyle="--",
               alpha=0.7, label="F1 = 0.50 floor")
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x")
    ax.legend(handles=[
        _patch(PASS_COLOR, "F1 ≥ 0.80"),
        _patch(WARN_COLOR, "0.50 ≤ F1 < 0.80"),
        _patch(FAIL_COLOR, "F1 < 0.50"),
    ], frameon=False, fontsize=9)
    _save(fig, out, "05_nlp_skill_f1_per_resume.png")


def chart_nlp_recall_vs_precision(per_resume: list[dict], out: Path) -> None:
    """
    Scatter: recall (x) vs precision (y).

    Points above the diagonal → the pipeline is conservative (misses skills).
    Points below the diagonal → the pipeline is hallucinating extra skills.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for r in per_resume:
        passed = r["job_title_match"] and r["score_pass"]
        ax.scatter(r["skill_recall"], r["skill_precision"],
                   color=PASS_COLOR if passed else FAIL_COLOR,
                   alpha=0.75, s=50, linewidths=0)
    ax.plot([0, 1], [0, 1], color=GRID_COLOR, linewidth=1.2,
            linestyle="--", label="recall = precision")
    ax.set_xlabel("Skill recall  (found ÷ expected)")
    ax.set_ylabel("Skill precision  (correct ÷ extracted)")
    ax.set_title("NLP extraction — recall vs precision\n"
                 "(above diagonal = conservative/missing, below = hallucinating)")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True)
    ax.legend(handles=[
        _patch(PASS_COLOR, "Pass (title + score)"),
        _patch(FAIL_COLOR, "Fail"),
    ], frameon=False)
    _save(fig, out, "06_nlp_recall_vs_precision.png")


def chart_nlp_self_score_vs_gt_f1(per_resume: list[dict], out: Path) -> None:
    """
    Scatter: pipeline's self-reported NLP accuracy vs ground-truth skill F1.

    nlp_pipeline_accuracy  — fuzzy_skill_verify() result vs best job's required_skills.
    skill_f1               — your ground-truth labels vs pipeline output.

    If both track closely (tight diagonal cluster): the corpus job definitions
    are a good proxy for your ground truth labels.
    If they diverge: either the job corpus has different skill names to your labels
    (canonicalisation drift) or the pipeline's self-assessment is miscalibrated.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    sources = {r.get("text_source", "unknown") for r in per_resume}
    for r in per_resume:
        ax.scatter(
            r["nlp_pipeline_accuracy"] * 100,   # x: self-score (already 0-1)
            r["skill_f1"] * 100,                 # y: ground-truth F1
            color=_source_color(r),
            alpha=0.75, s=50, linewidths=0,
            )

    # Perfect calibration line
    ax.plot([0, 100], [0, 100], color=GRID_COLOR, linewidth=1.2,
            linestyle="--", label="perfect calibration")

    # Trend line
    xs = np.array([r["nlp_pipeline_accuracy"] * 100 for r in per_resume])
    ys = np.array([r["skill_f1"] * 100 for r in per_resume])
    if len(xs) > 2:
        m, b = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xline, m * xline + b, color=MID_COLOR, linewidth=1.4,
                linestyle="-", alpha=0.7, label=f"trend  (slope {m:.2f})")

    ax.set_xlabel("Pipeline NLP self-score (%)\n(fuzzy match vs best-job required skills)")
    ax.set_ylabel("Ground-truth skill F1 (%)\n(your labels vs pipeline output)")
    ax.set_title("NLP calibration — self-score vs ground-truth F1\n"
                 "(diagonal = well-calibrated; flat = self-assessment meaningless)")
    ax.set_xlim(-5, 108)
    ax.set_ylim(-5, 108)
    ax.grid(True)

    handles = _source_legend_patches(sources) + [
        mpatches.Patch(color=GRID_COLOR, label="perfect calibration"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9)
    _save(fig, out, "07_nlp_self_score_vs_gt_f1.png")


def chart_nlp_missing_skills(per_resume: list[dict], out: Path, top_n: int = 20) -> None:
    """Top-N skills most frequently absent from pipeline output (false negatives)."""
    counter: Counter = Counter()
    for r in per_resume:
        counter.update(r.get("missing_skills", []))
    if not counter:
        return
    items  = counter.most_common(top_n)
    labels = [i[0] for i in items]
    counts = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.38)))
    ax.barh(range(len(labels)), counts, color=FAIL_COLOR, height=0.65)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of resumes where this skill was missed")
    ax.set_title(f"NLP — top {top_n} most frequently missed skills\n"
                 "(fix: check allowlist canonicalisation or LLM prompt for these terms)")
    ax.grid(axis="x")
    _save(fig, out, "08_nlp_missing_skills.png")


def chart_nlp_extra_skills(per_resume: list[dict], out: Path, top_n: int = 20) -> None:
    """Top-N skills most frequently hallucinated (false positives)."""
    counter: Counter = Counter()
    for r in per_resume:
        counter.update(r.get("extra_skills", []))
    if not counter:
        return
    items  = counter.most_common(top_n)
    labels = [i[0] for i in items]
    counts = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.38)))
    ax.barh(range(len(labels)), counts, color=WARN_COLOR, height=0.65)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of resumes where this skill was hallucinated")
    ax.set_title(f"NLP — top {top_n} most frequently hallucinated skills\n"
                 "(fix: tighten allowlist drop-guard or LLM system prompt)")
    ax.grid(axis="x")
    _save(fig, out, "09_nlp_extra_skills.png")


def chart_nlp_recall_precision_gap(per_resume: list[dict], out: Path) -> None:
    """
    Bar: recall − precision per resume, sorted by absolute gap.

    Positive (amber) = pipeline misses real skills (recall > precision).
    Negative (coral) = pipeline hallucinates extra skills (precision > recall).
    """
    sorted_r = sorted(per_resume,
                      key=lambda r: abs(r["skill_recall"] - r["skill_precision"]),
                      reverse=True)
    names  = [_display_name(r["filename"]) for r in sorted_r]
    gaps   = [r["skill_recall"] - r["skill_precision"] for r in sorted_r]
    colors = [WARN_COLOR if g > 0 else FAIL_COLOR for g in gaps]

    n = len(names)
    h = max(4, n * 0.28)
    fig, ax = plt.subplots(figsize=(9, h))
    ax.barh(range(n), gaps, color=colors, height=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color=TEXT_COLOR, linewidth=0.8)
    ax.set_xlabel("Recall − Precision  (positive = misses skills,  negative = hallucinates)")
    ax.set_title("NLP extraction gap — recall vs precision per resume\n"
                 "(sorted by absolute gap — worst calibration at top)")
    ax.grid(axis="x")
    ax.legend(handles=[
        _patch(WARN_COLOR, "Recall > precision (misses real skills)"),
        _patch(FAIL_COLOR, "Precision > recall (hallucinates extra skills)"),
    ], frameon=False, fontsize=9)
    _save(fig, out, "10_nlp_recall_precision_gap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FAISS / SCORING STAGE
# ══════════════════════════════════════════════════════════════════════════════

def chart_faiss_rank_distribution(per_resume: list[dict], out: Path) -> None:
    """
    Histogram: faiss_best_rank (1-6) across all resumes.

    Rank 1 = expected job was #1.  Rank 6 = not found in top-5 at all.
    A healthy system should have a large spike at rank 1.

    Note: this system uses IndexFlatIP (exact search) so rank errors come from
    the hybrid scoring formula (70% SBERT cosine + 30% skill ratio), not from
    FAISS approximation.
    """
    ranks  = [r["faiss_best_rank"] for r in per_resume]
    top1   = ranks.count(1)
    top5   = sum(1 for rv in ranks if rv <= 5)
    missed = ranks.count(6)
    total  = len(ranks)

    colors = [
        PASS_COLOR if rv == 1
        else (PURPLE_COLOR if rv <= 5
              else FAIL_COLOR)
        for rv in sorted(ranks)
    ]

    rank_counts = [ranks.count(i) for i in range(1, 7)]
    rank_labels = ["#1", "#2", "#3", "#4", "#5", "Not\nfound"]
    bar_colors  = [PASS_COLOR, PURPLE_COLOR, PURPLE_COLOR,
                   PURPLE_COLOR, PURPLE_COLOR, FAIL_COLOR]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(rank_labels, rank_counts, color=bar_colors,
                  width=0.55, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, rank_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{count}\n({count/total*100:.0f}%)",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of resumes")
    ax.set_title(
        f"FAISS ranking — where did the expected job land?\n"
        f"Top-1: {top1}/{total} ({top1/total*100:.0f}%)   "
        f"Top-5: {top5}/{total} ({top5/total*100:.0f}%)   "
        f"Missed: {missed}/{total}"
    )
    ax.set_ylim(0, max(rank_counts) * 1.25)
    ax.grid(axis="y")
    ax.legend(handles=[
        _patch(PASS_COLOR,   "Rank #1 (top-1 hit)"),
        _patch(PURPLE_COLOR, "Rank #2–5 (top-5 hit)"),
        _patch(FAIL_COLOR,   "Not in top-5"),
    ], frameon=False, fontsize=9)
    _save(fig, out, "11_faiss_rank_distribution.png")


def chart_faiss_cosine_vs_ratio(per_resume: list[dict], out: Path) -> None:
    """
    Scatter: faiss_cosine_score (x) vs faiss_match_ratio (y).

    Colored by whether the expected job ranked #1 (green), top-5 (purple),
    or was missed (coral).

    Hybrid = 0.70 * cosine + 0.30 * ratio.
    Points in the top-right are well-matched on both dimensions.
    Points along the left edge have high skill overlap but poor semantic fit.
    Points along the bottom have good semantic fit but skill gap.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for r in per_resume:
        rank = r["faiss_best_rank"]
        if rank == 1:
            color, marker = PASS_COLOR, "o"
        elif rank <= 5:
            color, marker = PURPLE_COLOR, "s"
        else:
            color, marker = FAIL_COLOR, "^"
        ax.scatter(r["faiss_cosine_score"], r["faiss_match_ratio"],
                   color=color, marker=marker, alpha=0.75, s=55, linewidths=0)

    # Iso-hybrid lines (0.70*cos + 0.30*ratio = constant)
    for level, alpha in [(0.5, 0.3), (0.7, 0.5), (0.85, 0.7)]:
        x_vals = np.linspace(0, 1, 100)
        y_vals = (level - 0.70 * x_vals) / 0.30
        mask   = (y_vals >= 0) & (y_vals <= 1)
        ax.plot(x_vals[mask], y_vals[mask], color=GRAY_COLOR,
                linewidth=0.8, linestyle="--", alpha=alpha)
        if mask.any():
            xi = x_vals[mask][-1]
            yi = y_vals[mask][-1]
            ax.text(xi + 0.01, yi, f"h={level:.2f}",
                    fontsize=7, color=GRAY_COLOR, alpha=alpha + 0.1)

    ax.set_xlabel("SBERT cosine score  (semantic similarity)")
    ax.set_ylabel("Skill match ratio  (skill overlap)")
    ax.set_title("FAISS scoring decomposition — cosine vs skill ratio\n"
                 "(dashed = iso-hybrid lines;  hybrid = 0.70·cos + 0.30·ratio)")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 1.12)
    ax.grid(True, alpha=0.4)
    ax.legend(handles=[
        _patch(PASS_COLOR,   "Rank #1 (top-1 hit)"),
        _patch(PURPLE_COLOR, "Rank #2–5 (top-5 hit)"),
        _patch(FAIL_COLOR,   "Not in top-5"),
    ], frameon=False, fontsize=9)
    _save(fig, out, "12_faiss_cosine_vs_ratio.png")


def chart_faiss_score_distribution(per_resume: list[dict], out: Path) -> None:
    """
    Histogram: best_hybrid_score (0-100) with threshold line.

    best_hybrid_score is already on a 0-100 scale (evaluate_pipeline multiplies
    the pipeline's 0-1 hybrid float by 100 for readability).
    """
    vals      = [r["best_hybrid_score"] for r in per_resume]
    # min_match_score is already on 0-100 scale in labels.json
    min_score = per_resume[0].get("min_match_score", 50) if per_resume else 50

    fig, ax = plt.subplots(figsize=(7, 4))
    n_bins = min(20, max(5, len(vals) // 2))
    n, bins, patches = ax.hist(vals, bins=n_bins, edgecolor="white", linewidth=0.5)
    for patch, left in zip(patches, bins):
        patch.set_facecolor(PASS_COLOR if left >= min_score else FAIL_COLOR)

    ax.axvline(min_score, color=FAIL_COLOR, linewidth=1.5, linestyle="--",
               label=f"pass threshold ({min_score})")
    ax.axvline(np.mean(vals), color=TEXT_COLOR, linewidth=1.2, linestyle=":",
               label=f"mean {np.mean(vals):.1f}")

    ax.set_xlabel("Best hybrid score  (0–100)")
    ax.set_ylabel("Number of resumes")
    ax.set_title("FAISS — best hybrid score distribution\n"
                 "(green = above threshold, coral = below)")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    _save(fig, out, "13_faiss_score_distribution.png")


def chart_faiss_score_vs_f1(per_resume: list[dict], out: Path) -> None:
    """
    Scatter: best_hybrid_score (x) vs skill_f1 (y).

    A strong upward trend means the corpus's job definitions are aligned with
    your ground-truth skill labels — resumes that scored high also had good
    skill extraction.  A flat/random trend means scoring and extraction are
    decoupled (a corpus or embedding quality problem).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in per_resume:
        passed = r["job_title_match"] and r["score_pass"]
        ax.scatter(r["best_hybrid_score"], r["skill_f1"] * 100,
                   color=PASS_COLOR if passed else FAIL_COLOR,
                   alpha=0.75, s=50, linewidths=0)

    min_score = per_resume[0].get("min_match_score", 50) if per_resume else 50
    ax.axvline(min_score, color=FAIL_COLOR, linewidth=1.2, linestyle="--",
               alpha=0.6, label=f"score threshold ({min_score})")

    xs = np.array([r["best_hybrid_score"] for r in per_resume])
    ys = np.array([r["skill_f1"] * 100 for r in per_resume])
    if len(xs) > 2:
        m, b = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xline, m * xline + b, color=MID_COLOR, linewidth=1.4,
                linestyle="-", alpha=0.65, label=f"trend  (slope {m:.2f})")

    ax.set_xlabel("Best hybrid score  (0–100)")
    ax.set_ylabel("Ground-truth skill F1 (%)")
    ax.set_title("FAISS calibration — hybrid score vs ground-truth skill F1\n"
                 "(steep upward trend = corpus well-aligned with your labels)")
    ax.set_ylim(-5, 110)
    ax.grid(True)
    ax.legend(handles=[
        _patch(PASS_COLOR, "Pass (title + score)"),
        _patch(FAIL_COLOR, "Fail"),
    ], frameon=False, fontsize=9)
    _save(fig, out, "14_faiss_score_vs_f1.png")


def chart_faiss_title_by_category(per_resume: list[dict], out: Path, top_n: int = 25) -> None:
    """
    Stacked bar: top-1 hit / top-5 hit / not found by job category.

    Groups resumes by the first two words of expected_title so related
    roles cluster together.  Shows which job categories the ranker
    consistently gets right vs where it struggles.
    """
    def _cat(title: str) -> str:
        words = title.split()
        return " ".join(words[:2]) if len(words) >= 2 else (title or "Unknown")

    cat_top1: Counter = Counter()
    cat_top5: Counter = Counter()   # top-5 but NOT top-1
    cat_miss: Counter = Counter()
    for r in per_resume:
        cat  = _cat(r.get("expected_title", "Unknown"))
        rank = r["faiss_best_rank"]
        if rank == 1:
            cat_top1[cat] += 1
        elif rank <= 5:
            cat_top5[cat] += 1
        else:
            cat_miss[cat] += 1

    all_cats = sorted(
        set(cat_top1) | set(cat_top5) | set(cat_miss),
        key=lambda c: -(cat_top1[c] + cat_top5[c] + cat_miss[c]),
        )[:top_n]

    top1s = [cat_top1[c] for c in all_cats]
    top5s = [cat_top5[c] for c in all_cats]
    misss = [cat_miss[c]  for c in all_cats]

    n = len(all_cats)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.38)))
    y = range(n)
    ax.barh(y, top1s, color=PASS_COLOR,   height=0.6, label="Rank #1 (top-1)")
    ax.barh(y, top5s, left=top1s,          color=PURPLE_COLOR, height=0.6, label="Rank #2–5 (top-5)")
    ax.barh(y, misss, left=[a + b for a, b in zip(top1s, top5s)],
            color=FAIL_COLOR, height=0.6, label="Not in top-5")
    ax.set_yticks(y)
    ax.set_yticklabels(all_cats, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of resumes")
    ax.set_title("FAISS ranking by job category\n"
                 "(green = ranked correctly at #1, purple = top-5, coral = missed)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x")
    _save(fig, out, "15_faiss_title_by_category.png")


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def chart_pipeline_failure_attribution(per_resume: list[dict], out: Path) -> None:
    """
    Bar: attribute each resume to the first stage that failed.

    Stage thresholds (in order):
      OCR    : ocr_quality_pct < 70   (text quality floor)
      NLP    : skill_f1 < 0.70        (skill extraction floor)
      FAISS  : faiss_top5_match == 0  (expected job not retrieved at all)
      Score  : score_pass == 0        (hybrid score below label threshold)
      Title  : job_title_match == 0   (best job title doesn't match expected)
      Pass   : everything ok

    Note: OCR is measured by alnum-ratio, NOT character-error-rate.
    A resume flagged here may still have OCR transcription errors that
    aren't visible in the quality score.
    """
    buckets: Counter = Counter()
    for r in per_resume:
        ocr_ok   = r.get("ocr_quality_pct", 100) >= 70
        nlp_ok   = r["skill_f1"] >= 0.70
        faiss_ok = bool(r.get("faiss_top5_match", 1))
        score_ok = bool(r["score_pass"])
        title_ok = bool(r["job_title_match"])

        if not ocr_ok:
            buckets["OCR quality\n< 70%"] += 1
        elif not nlp_ok:
            buckets["NLP skill F1\n< 0.70"] += 1
        elif not faiss_ok:
            buckets["FAISS: job not\nin top-5"] += 1
        elif not score_ok:
            buckets["Score below\nthreshold"] += 1
        elif not title_ok:
            buckets["Title\nmismatch"] += 1
        else:
            buckets["Pass ✓"] += 1

    order  = ["Pass ✓", "OCR quality\n< 70%", "NLP skill F1\n< 0.70",
              "FAISS: job not\nin top-5", "Score below\nthreshold", "Title\nmismatch"]
    colors = [PASS_COLOR, FAIL_COLOR, WARN_COLOR, PURPLE_COLOR, MID_COLOR, GRAY_COLOR]
    labels = [o for o in order if o in buckets]
    vals   = [buckets[o] for o in labels]
    cols   = [colors[order.index(o)] for o in labels]
    total  = len(per_resume)

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 5))
    bars = ax.bar(labels, vals, color=cols, width=0.55,
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{v}\n({v/total*100:.0f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of resumes")
    ax.set_title("Pipeline failure attribution — first failing stage per resume\n"
                 "(tallest non-green bar = biggest bottleneck to fix first)")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="y")
    _save(fig, out, "16_pipeline_failure_attribution.png")


def chart_pipeline_stage_summary(per_resume: list[dict], out: Path) -> None:
    """
    Grouped bar: mean OCR quality / NLP skill F1 / FAISS top-1 rate
    broken down by text_source — one-glance overview of where each
    extraction path sits on each metric.
    """
    sources_present = sorted({r.get("text_source", "unknown") for r in per_resume})
    groups: dict[str, list[dict]] = {s: [] for s in sources_present}
    for r in per_resume:
        groups[r.get("text_source", "unknown")].append(r)

    metrics      = ["OCR quality\n(alnum %)", "NLP skill F1", "FAISS top-1\nrank rate"]
    x            = np.arange(len(metrics))
    width        = 0.7 / max(len(sources_present), 1)
    source_order = ["native_pdf", "ocr", "camera_scan_mlkit", "unknown"]
    ordered      = [s for s in source_order if s in groups] + \
                   [s for s in sources_present if s not in source_order]

    fig, ax = plt.subplots(figsize=(max(7, len(ordered) * 2.5), 5))
    display_labels = {
        "native_pdf":        "Native PDF",
        "ocr":               "EasyOCR",
        "camera_scan_mlkit": "Camera ML Kit",
        "unknown":           "Unknown",
    }

    for i, src in enumerate(ordered):
        rs = groups[src]
        if not rs:
            continue
        ocr_mean   = float(np.mean([r.get("ocr_quality_pct", 0) for r in rs]))
        f1_mean    = float(np.mean([r["skill_f1"] for r in rs])) * 100
        faiss_mean = float(np.mean([r.get("faiss_top1_match", 0) for r in rs])) * 100
        vals       = [ocr_mean, f1_mean, faiss_mean]
        offset     = (i - len(ordered) / 2 + 0.5) * width
        color      = SOURCE_COLORS.get(src, GRAY_COLOR)
        bars = ax.bar(x + offset, vals, width * 0.9,
                      color=color, label=f"{display_labels.get(src, src)} (n={len(rs)})",
                      edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score / rate (%)")
    ax.set_title("Pipeline stage summary by extraction path\n"
                 "(gap per metric shows where a text_source hurts most)")
    ax.set_ylim(0, 118)
    ax.axhline(70, color=WARN_COLOR, linewidth=1, linestyle="--",
               alpha=0.5, label="70% floor")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out, "17_pipeline_stage_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_charts(
        report:  dict,
        out_dir: str | Path = "eval_charts",
) -> Path:
    """
    Generate all diagnostic charts from an evaluate() report dict.

    Parameters
    ----------
    report  : dict returned by evaluate() or loaded from --out report.json
    out_dir : directory to write PNGs into (created if missing)

    Returns
    -------
    Path to the output directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_resume: list[dict] = report.get("per_resume", [])
    if not per_resume:
        print("  [charts] No per_resume data — nothing to plot.")
        return out

    print(f"\n  Generating charts -> {out}/")

    print("  — OCR stage —")
    chart_ocr_quality_by_resume(per_resume, out)
    chart_ocr_quality_distribution(per_resume, out)
    chart_ocr_source_breakdown(per_resume, out)
    chart_ocr_f1_by_source(per_resume, out)

    print("  — NLP extraction stage —")
    chart_nlp_f1_per_resume(per_resume, out)
    chart_nlp_recall_vs_precision(per_resume, out)
    chart_nlp_self_score_vs_gt_f1(per_resume, out)
    chart_nlp_missing_skills(per_resume, out)
    chart_nlp_extra_skills(per_resume, out)
    chart_nlp_recall_precision_gap(per_resume, out)

    print("  — FAISS / scoring stage —")
    chart_faiss_rank_distribution(per_resume, out)
    chart_faiss_cosine_vs_ratio(per_resume, out)
    chart_faiss_score_distribution(per_resume, out)
    chart_faiss_score_vs_f1(per_resume, out)
    chart_faiss_title_by_category(per_resume, out)

    print("  — Combined pipeline —")
    chart_pipeline_failure_attribution(per_resume, out)
    chart_pipeline_stage_summary(per_resume, out)

    n = len(list(out.glob("*.png")))
    print(f"\n  Done. {n} charts saved to {out}/\n")
    return out

if __name__ == "__main__":
    import json
    import sys

    report_path = sys.argv[1] if len(sys.argv) > 1 else "report.json"
    out_dir     = sys.argv[2] if len(sys.argv) > 2 else "analysis"

    with open(report_path) as f:
        report = json.load(f)

    generate_charts(report, out_dir=out_dir)