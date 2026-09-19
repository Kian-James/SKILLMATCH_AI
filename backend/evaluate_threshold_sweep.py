"""
evaluate_threshold_sweep.py
============================
Empirical justification for the hardcoded 0.80 SBERT cosine thresholds.

The live pipeline has THREE separate 0.80 constants, and this script sweeps
the stages they belong to, each against the real pipeline output:

    core/extractor_base.py  _SBERT_COSINE_THRESHOLD  (default arg of _sbert_match)
        Stage 2 "discovery": each candidate phrase from the resume's SKILLS
        section is matched to the job-corpus allowlist. This decides WHICH
        skills get discovered at all.

    core/scoring.py         ITEM_SEMANTIC_THRESHOLD  (default arg of
    core/validation.py      _SEMANTIC_THRESHOLD       _semantic_item_match /
                                                      semantic_verify)
        Stage 3 matching: discovered skills are compared to a job's
        required skills. Here it is applied to discovered-vs-expected_skills.

WHAT IT RUNS
------------
For every labeled resume:

    Stage 1  core.extraction.extract_text(<pdf>)         -- the real OCR path
    Stage 2  core.skills.discover_skills_dynamically()   -- the real discovery,
             with the same job corpus / Firestore allowlist the app uses.
    Stage 3  core.scoring.calculate_cosine_similarity()  -- the real job
             ranking (sweep 3), fed the skills Stage 2 really discovered.

No LLM is used for skill extraction or scoring. The only LLM in the pipeline
is generate_fit_reason() (Llama via the HF router) in Stage 3, which writes
descriptive text for each top-K job and cannot change any score or rank. It is
switched OFF during sweep 3 (it would otherwise make thousands of slow, paid
API calls when HF_TOKEN is set).

Stage 1 is threshold-independent, so it runs ONCE per resume and is cached
(--text-cache). Nothing in core/ is edited: the discovery threshold is
injected at runtime by wrapping extractor_base._sbert_match, and two pure
speed-ups (memoised SBERT encodings, memoised OcrCorrector) are installed.
Both are output-identical to the un-cached code.

THREE SWEEPS
------------
1. DISCOVERY sweep  (--thresholds)
   Varies extractor_base's threshold. The discovered skills are scored
   against labels.json `expected_skills` with the matcher held at its
   PRODUCTION threshold, so every change is attributable to discovery only.

2. MATCHING sweep   (--match-thresholds, defaults to --thresholds)
   Discovery is held at its production threshold (one real run per resume);
   the threshold used to compare discovered skills to expected_skills is
   varied for both metrics below.

3. RANKING sweep   (--match-thresholds)
   The real Stage 3 is run for every resume at each matching threshold,
   scored against labels.json `expected_job_title` exactly as
   evaluate_pipeline.py does (fuzzy title match >= 70 within the top-K):
   MRR, Hit@1, Hit@K, NDCG@K, average best hybrid score and the score-pass
   rate (best score >= min_match_score). It also reports WHERE the correct
   job landed at each threshold (how many resumes had it at rank 1, 2, ...,
   K, or missing). Every threshold is judged on its own against the ground
   truth; no other threshold is used as a reference. The threshold cannot
   alter WHICH K jobs are retrieved (that is pure cosine similarity), only
   how they are re-scored and ordered, so the correct job can only move
   around inside the top-K.

Metrics, side by side in sweeps 1 and 2:
  * Precision / Recall / F1 via core.scoring._semantic_item_match() (the same
    matcher evaluate_pipeline.py uses; checks both directions, so an
    over-eager threshold shows up as a precision drop).
  * NLP_ACCURACY via core.validation.semantic_verify() -- the recall-only
    sliding-window figure the product shows to users.

NOT COVERED: certificates (cert discovery is not run, so Stage 3 receives no
discovered certs; certs are reported but never enter the hybrid score).

REQUIREMENTS
------------
Run it where the backend runs: sentence-transformers (all-MiniLM-L6-v2),
faiss, rapidfuzz, the OCR microservice reachable at OCR_SERVICE_URL, and
Firestore credentials for the job corpus. To avoid re-doing those on every
run, use the two cache files below; after the first run the sweep needs only
SBERT.

USAGE
-----
    python evaluate_threshold_sweep.py --labels labels.json --resumes resumes
    python evaluate_threshold_sweep.py --labels labels.json --resumes resumes \\
        --thresholds 0.60:0.95:0.05 --output sweep_report.json --charts

Output
------
Three tables on stdout, an optional JSON report (--output) and, with --charts,
three PNGs: <chart-path>_discovery.png, _matching.png and _ranking.png.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Threshold grid parsing
# ---------------------------------------------------------------------------

def _parse_thresholds(spec: str) -> list[float]:
    """
    Accepts either a comma-separated list ("0.6,0.7,0.75,0.8,0.85,0.9") or a
    start:stop:step range ("0.60:0.95:0.05", inclusive of stop).
    """
    if ":" in spec:
        start_s, stop_s, step_s = spec.split(":")
        start, stop, step = float(start_s), float(stop_s), float(step_s)
        n_steps = round((stop - start) / step)
        return [round(start + i * step, 4) for i in range(n_steps + 1)]
    return [round(float(t.strip()), 4) for t in spec.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Job corpus (Firestore) with an on-disk cache, so the sweep is repeatable
# ---------------------------------------------------------------------------

def _load_corpus(cache_path: Path | None) -> tuple[list, list, list]:
    """(firestore_skills, jobs_corpus, firestore_certs), cached if requested."""
    if cache_path and cache_path.exists():
        d = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Job corpus loaded from cache {cache_path} "
              f"({len(d['jobs_corpus'])} jobs).")
        return d["firestore_skills"], d["jobs_corpus"], d["firestore_certs"]

    from db import load_analyzer_data
    print("Loading job corpus from Firestore ...")
    firestore_skills, jobs_corpus, firestore_certs = load_analyzer_data()
    print(f"  {len(jobs_corpus)} jobs, {len(firestore_skills)} allowlist skills.")
    if cache_path:
        cache_path.write_text(json.dumps(dict(
            firestore_skills=firestore_skills,
            jobs_corpus=jobs_corpus,
            firestore_certs=firestore_certs,
        ), default=str), encoding="utf-8")
        print(f"  cached to {cache_path} (delete it to refresh).")
    return firestore_skills, jobs_corpus, firestore_certs


# ---------------------------------------------------------------------------
# Stage 1 -- text acquisition, run once per resume and cached
# ---------------------------------------------------------------------------

def _acquire_texts(names: list[str], resumes_dir: Path,
                   cache_path: Path) -> tuple[dict, list[str], list[str]]:
    """
    Returns (texts, missing_pdf, ocr_failed) where texts maps
    name -> (resume_text, is_native_pdf). A cached text is used as-is (no
    PDF needed); otherwise the real core.extraction.extract_text() runs.
    Empty OCR results are never cached, so they are retried next run.
    """
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    texts, missing, failed = {}, [], []
    extract_text = None
    dirty = 0
    for i, name in enumerate(names, 1):
        c = cache.get(name)
        if c and (c.get("text") or "").strip():
            texts[name] = (c["text"], bool(c.get("is_native", False)))
            continue

        pdf = resumes_dir / name
        if not pdf.exists():
            missing.append(name)
            continue
        if extract_text is None:
            from core.extraction import extract_text as _et
            extract_text = _et

        print(f"  [OCR {i}/{len(names)}] {name}")
        text, _quality, is_native = extract_text(str(pdf))
        if not (text or "").strip():
            failed.append(name)
            continue
        cache[name] = dict(text=text, is_native=bool(is_native))
        texts[name] = (text, bool(is_native))
        dirty += 1
        if dirty % 10 == 0:   # crash-safe: OCR is the slow part
            cache_path.write_text(json.dumps(cache), encoding="utf-8")

    if dirty:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return texts, missing, failed


# ---------------------------------------------------------------------------
# Runtime patches: threshold injection + two output-identical speed-ups
# ---------------------------------------------------------------------------

class _Patches:
    """
    Nothing in core/ is edited on disk.

      * discovery_threshold(t): wraps extractor_base._sbert_match so it always
        runs at threshold t. discover_items_dynamically() looks the function
        up in its own module globals at call time, so the wrapper is seen.
      * encode memoisation: SentenceTransformer.encode is wrapped to memoise
        per (text, normalize_embeddings). Embeddings are deterministic, so
        this only avoids re-encoding the same allowlist thousands of times.
      * build_ocr_corrector memoisation: the corrector depends only on the
        allowlist, which is identical for every resume.
    """

    def __init__(self) -> None:
        from core import extractor_base as eb
        from core.ocr_correction import get_sbert

        self.eb = eb
        self._orig_match = eb._sbert_match
        self._orig_build = eb.build_ocr_corrector
        self._corr_cache: dict = {}

        sbert = get_sbert()
        self._install_encode_cache(sbert)
        eb.build_ocr_corrector = self._cached_build

    @staticmethod
    def _install_encode_cache(sbert) -> None:
        orig = sbert.encode
        cache: dict = {}

        def cached(sentences, *args, **kwargs):
            single = isinstance(sentences, str)
            texts = [sentences] if single else list(sentences)
            if not texts:
                return orig(sentences, *args, **kwargs)
            norm = bool(kwargs.get("normalize_embeddings", False))
            missing = [t for t in dict.fromkeys(texts) if (t, norm) not in cache]
            if missing:
                vecs = orig(missing, *args, **kwargs)
                for t, v in zip(missing, vecs):
                    cache[(t, norm)] = v
            out = np.stack([cache[(t, norm)] for t in texts])
            return out[0] if single else out

        sbert.encode = cached

    def _cached_build(self, allowlist):
        key = tuple(allowlist)
        if key not in self._corr_cache:
            self._corr_cache[key] = self._orig_build(allowlist)
        return self._corr_cache[key]

    @property
    def production_discovery_threshold(self) -> float:
        return float(self.eb._SBERT_COSINE_THRESHOLD)

    @contextmanager
    def ranking_threshold(self, item_t: float, verify_t: float):
        """
        Run core.scoring with its matching thresholds forced. Stage 3 reaches
        them through scoring._partition_items -> _semantic_item_match (default
        arg = ITEM_SEMANTIC_THRESHOLD) and scoring.semantic_verify (default
        arg = validation._SEMANTIC_THRESHOLD); both are looked up in scoring's
        module globals at call time, so wrapping them here is seen.

        generate_fit_reason is replaced with a no-op for the duration: it only
        produces display text (an LLM call when HF_TOKEN is set) and has no
        effect on any score or ranking.
        """
        from core import scoring as sc
        o_item, o_verify, o_fit = (sc._semantic_item_match,
                                   sc.semantic_verify,
                                   sc.generate_fit_reason)

        def item(target_item, candidate_norms, candidate_index, threshold=None):
            return o_item(target_item, candidate_norms, candidate_index,
                          threshold=item_t)

        def verify(discovered_list, ground_truth_list, threshold=None,
                   resume_text=""):
            return o_verify(discovered_list=discovered_list,
                            ground_truth_list=ground_truth_list,
                            threshold=verify_t, resume_text=resume_text)

        sc._semantic_item_match = item
        sc.semantic_verify      = verify
        sc.generate_fit_reason  = lambda *a, **k: ""
        try:
            yield
        finally:
            sc._semantic_item_match = o_item
            sc.semantic_verify      = o_verify
            sc.generate_fit_reason  = o_fit

    @contextmanager
    def discovery_threshold(self, t: float | None):
        """t=None -> leave discovery completely untouched (production)."""
        if t is None:
            yield
            return
        orig = self._orig_match

        def patched(entries, allowlist, already_found,
                    threshold=None, label="SBERT"):
            return orig(entries, allowlist, already_found,
                        threshold=t, label=label)

        self.eb._sbert_match = patched
        try:
            yield
        finally:
            self.eb._sbert_match = orig


# ---------------------------------------------------------------------------
# Stage 2 -- the real discovery
# ---------------------------------------------------------------------------

def _discover(text: str, is_native: bool, corpus: tuple) -> list[str]:
    from core.skills import discover_skills_dynamically
    firestore_skills, jobs_corpus, _ = corpus
    return discover_skills_dynamically(
        resume_text=text,
        jobs_corpus=jobs_corpus,
        extra_skills=firestore_skills,
        is_native_pdf=is_native,
    )


def _discover_all(samples: dict, corpus: tuple, patches: _Patches,
                  t: float | None) -> dict[str, list[str]]:
    with patches.discovery_threshold(t):
        return {name: _discover(text, native, corpus)
                for name, (text, native, _exp) in samples.items()}


# ---------------------------------------------------------------------------
# Per-sample, per-threshold metrics
# ---------------------------------------------------------------------------

def _item_metrics_at_threshold(predicted: list[str], expected: list[str],
                               threshold: float) -> dict:
    """
    Same shape as evaluate_pipeline.py's _item_metrics(), but with the matcher
    threshold exposed as a parameter (that function hardcodes the default).
    """
    from core.scoring import _semantic_item_match, _encode_norm, _build_index
    from core.utils   import normalize

    if not expected:
        return dict(tp=0, fp=0, fn=0)

    pred_norms = [normalize(p) for p in predicted]
    pred_index = _build_index(_encode_norm(pred_norms)) if pred_norms else None
    exp_norms  = [normalize(e) for e in expected]
    exp_index  = _build_index(_encode_norm(exp_norms))

    tp = sum(1 for e in expected
             if _semantic_item_match(e, pred_norms, pred_index, threshold=threshold))
    fn = len(expected) - tp
    fp = sum(1 for p in predicted
             if not _semantic_item_match(p, exp_norms, exp_index, threshold=threshold))
    return dict(tp=tp, fp=fp, fn=fn)


def _semantic_verify_recall_at_threshold(predicted: list[str],
                                         expected: list[str],
                                         threshold: float) -> float | None:
    """The production metric (core.validation.semantic_verify)."""
    from core.validation import semantic_verify

    if not expected:
        return None
    pct = semantic_verify(discovered_list=predicted,
                          ground_truth_list=expected,
                          threshold=threshold)["NLP_ACCURACY"]
    return None if pct == "N/A" else float(pct.rstrip("%")) / 100.0


def _aggregate(preds: dict[str, list[str]], samples: dict,
               item_thr: float, verify_thr: float) -> dict:
    """Micro-averaged P/R/F1 (+ mean NLP_ACCURACY) over all samples."""
    tp = fp = fn = 0
    verify, n_disc, zero = [], 0, 0
    for name, predicted in preds.items():
        expected = samples[name][2]
        m = _item_metrics_at_threshold(predicted, expected, item_thr)
        tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        r = _semantic_verify_recall_at_threshold(predicted, expected, verify_thr)
        if r is not None:
            verify.append(r)
        n_disc += len(predicted)
        zero += (len(predicted) == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    nlp = sum(verify) / len(verify) if verify else None
    return dict(
        precision=round(precision, 4), recall=round(recall, 4),
        f1=round(f1, 4),
        nlp_accuracy=round(nlp, 4) if nlp is not None else None,
        tp=tp, fp=fp, fn=fn,
        avg_discovered=round(n_disc / max(len(preds), 1), 2),
        zero_discovered=zero,
    )


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def _log_row(kind: str, t: float, r: dict, secs: float) -> None:
    nlp = "n/a" if r["nlp_accuracy"] is None else f"{r['nlp_accuracy']:.4f}"
    print(f"  {kind}={t:.2f}  precision={r['precision']:.4f}  "
          f"recall={r['recall']:.4f}  f1={r['f1']:.4f}  nlp_accuracy={nlp}  "
          f"(tp={r['tp']} fp={r['fp']} fn={r['fn']})  [{secs:.0f}s]")


def run_discovery_sweep(samples: dict, corpus: tuple, patches: _Patches,
                        thresholds: list[float],
                        item_thr: float, verify_thr: float) -> list[dict]:
    rows = []
    for t in thresholds:
        t0 = time.perf_counter()
        preds = _discover_all(samples, corpus, patches, t)
        r = _aggregate(preds, samples, item_thr, verify_thr)
        rows.append(dict(threshold=t, **r))
        _log_row("discovery_threshold", t, r, time.perf_counter() - t0)
    return rows


def run_matching_sweep(samples: dict, prod_preds: dict,
                       thresholds: list[float]) -> list[dict]:
    rows = []
    for t in thresholds:
        t0 = time.perf_counter()
        r = _aggregate(prod_preds, samples, t, t)
        rows.append(dict(threshold=t, **r))
        _log_row("match_threshold    ", t, r, time.perf_counter() - t0)
    return rows


# ---------------------------------------------------------------------------
# Sweep 3 -- Stage 3 job ranking (the real calculate_cosine_similarity)
# ---------------------------------------------------------------------------

def _title_rank(expected_title: str, top_matches: list[dict],
                top_k: int) -> int | None:
    """1-based rank of expected_title in the top-K, or None. Mirrors
    evaluate_pipeline._title_in_topk (token_sort_ratio >= 70 on normalised
    titles)."""
    from rapidfuzz import fuzz
    from core.utils import normalize

    exp = normalize(expected_title)
    for rank, job in enumerate(top_matches[:top_k], start=1):
        if fuzz.token_sort_ratio(exp, normalize(job.get("title", ""))) >= 70:
            return rank
    return None


def _rank_all(resume_texts: dict, skills: dict, corpus: tuple, patches: _Patches,
              item_t: float, verify_t: float, top_k: int) -> dict[str, list[dict]]:
    """Run the real Stage 3 for every resume at the given matching thresholds."""
    from core.scoring import calculate_cosine_similarity
    out = {}
    with patches.ranking_threshold(item_t, verify_t):
        for name, text in resume_texts.items():
            out[name] = calculate_cosine_similarity(
                resume_text=text,
                resume_skills=skills[name],
                jobs_corpus=corpus[1],
                top_k=top_k,
                resume_certs=[],          # cert discovery is not run here
            )
    return out


def _rank_metrics(results: dict, meta: dict, top_k: int) -> dict:
    """
    IR metrics (same definitions as evaluate_pipeline.py) plus the full
    distribution of where the correct job landed. Each threshold is judged
    on its own against the ground-truth job title.
    """
    import math
    rr, ndcg, scores, nlps, found_ranks = [], [], [], [], []
    hit1 = hit_k = passed = 0
    rank_counts = {str(k): 0 for k in range(1, top_k + 1)}
    rank_counts["miss"] = 0
    for name, top in results.items():
        title, min_score = meta[name]
        rank = _title_rank(title, top, top_k)
        rr.append(1.0 / rank if rank else 0.0)
        ndcg.append(1.0 / math.log2(rank + 1) if rank else 0.0)
        hit1  += (rank == 1)
        hit_k += (rank is not None)
        if rank:
            rank_counts[str(rank)] += 1
            found_ranks.append(rank)
        else:
            rank_counts["miss"] += 1
        best = top[0] if top else {}
        scores.append(best.get("score", 0.0))
        passed += best.get("score", 0.0) >= min_score
        pct = best.get("nlp_validation", {}).get("NLP_ACCURACY", "N/A")
        if pct != "N/A":
            nlps.append(float(pct.rstrip("%")) / 100.0)
    n = max(len(results), 1)
    return dict(
        mrr=round(sum(rr) / n, 4),
        hit1=round(hit1 / n, 4),
        hit_k=round(hit_k / n, 4),
        ndcg=round(sum(ndcg) / n, 4),
        avg_best_score=round(sum(scores) / n, 2),
        score_pass=round(passed / n, 4),
        best_nlp=round(sum(nlps) / len(nlps), 4) if nlps else None,
        mean_rank=(round(sum(found_ranks) / len(found_ranks), 2)
                   if found_ranks else None),
        rank_counts=rank_counts,
    )


def run_ranking_sweep(resume_texts: dict, meta: dict, skills: dict,
                      corpus: tuple, patches: _Patches, thresholds: list[float],
                      top_k: int) -> list[dict]:
    rows = []
    for t in thresholds:
        t0 = time.perf_counter()
        results = _rank_all(resume_texts, skills, corpus, patches, t, t, top_k)
        r = _rank_metrics(results, meta, top_k)
        rows.append(dict(threshold=t, **r))
        landed = " ".join(f"#{k}={v}" if k != "miss" else f"miss={v}"
                          for k, v in r["rank_counts"].items())
        print(f"  match_threshold={t:.2f}  MRR={r['mrr']:.4f}  Hit@1={r['hit1']:.4f}  "
              f"Hit@{top_k}={r['hit_k']:.4f}  avg_score={r['avg_best_score']:.2f}  "
              f"score_pass={r['score_pass']:.4f}  correct job landed: {landed}  "
              f"[{time.perf_counter() - t0:.0f}s]")
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_ranking_table(rows: list[dict], production: float, top_k: int) -> None:
    def mark(r):
        return "  <- production" if abs(r["threshold"] - production) < 1e-9 else ""

    w = 96
    print("\n" + "=" * w)
    print("Sweep 3: Stage 3 job ranking vs matching threshold "
          "(scoring.ITEM_SEMANTIC_THRESHOLD / validation._SEMANTIC_THRESHOLD)")
    print("=" * w)
    print(f"{'Threshold':>10} | {'MRR':>7} | {'Hit@1':>7} | {f'Hit@{top_k}':>7} | "
          f"{'NDCG':>7} | {'AvgScore':>8} | {'ScorePass':>9} | {'BestNLP':>8}")
    print("-" * w)
    for r in rows:
        nlp = "n/a" if r["best_nlp"] is None else f"{r['best_nlp']:.4f}"
        print(f"{r['threshold']:>10.2f} | {r['mrr']:>7.4f} | {r['hit1']:>7.4f} | "
              f"{r['hit_k']:>7.4f} | {r['ndcg']:>7.4f} | {r['avg_best_score']:>8.2f} | "
              f"{r['score_pass']:>9.4f} | {nlp:>8}{mark(r)}")
    print("=" * w)

    print("\nWhere the correct job landed (number of resumes per rank):")
    print("=" * w)
    ranks = [str(k) for k in range(1, top_k + 1)]
    print(f"{'Threshold':>10} | " + " | ".join(f"{'#' + k:>5}" for k in ranks)
          + f" | {'Miss':>5} | {'MeanRank':>8}")
    print("-" * w)
    for r in rows:
        mr = "n/a" if r["mean_rank"] is None else f"{r['mean_rank']:.2f}"
        print(f"{r['threshold']:>10.2f} | "
              + " | ".join(f"{r['rank_counts'][k]:>5}" for k in ranks)
              + f" | {r['rank_counts']['miss']:>5} | {mr:>8}{mark(r)}")
    print("=" * w)
    print("The threshold cannot change WHICH jobs are retrieved (pure cosine "
          "similarity), only how they are re-scored and ordered.")


def _save_ranking_chart(rows: list[dict], path: str, top_k: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["threshold"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [r["mrr"] for r in rows], marker="o", label="MRR")
    plt.plot(xs, [r["hit1"] for r in rows], marker="o", label="Hit@1")
    plt.plot(xs, [r["ndcg"] for r in rows], marker="o", label=f"NDCG@{top_k}")
    plt.plot(xs, [r["score_pass"] for r in rows], marker="o", label="Score-pass rate")
    plt.plot(xs, [r["avg_best_score"] / 100.0 for r in rows], marker="o",
             linestyle="--", label="Avg best hybrid score (/100)")
    plt.xlabel("Matching cosine similarity threshold")
    plt.ylabel("Score")
    plt.title("Stage 3 job ranking: SBERT matching-threshold sweep")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {path}")


def _print_table(rows: list[dict], title: str, production: float) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    print(f"{'Threshold':>10} | {'Precision':>9} | {'Recall':>9} | {'F1':>9} | "
          f"{'NLP_ACC':>9}")
    print("-" * 76)
    for r in rows:
        nlp = "n/a" if r["nlp_accuracy"] is None else f"{r['nlp_accuracy']:.4f}"
        mark = "  <- production" if abs(r["threshold"] - production) < 1e-9 else ""
        print(f"{r['threshold']:>10.2f} | {r['precision']:>9.4f} | "
              f"{r['recall']:>9.4f} | {r['f1']:>9.4f} | {nlp:>9}{mark}")
    print("=" * 76)

    best = max(rows, key=lambda r: r["f1"])
    print(f"\nBest threshold by F1: {best['threshold']:.2f} "
          f"(precision={best['precision']:.4f}, recall={best['recall']:.4f}, "
          f"f1={best['f1']:.4f})")
    prod_row = next((r for r in rows if abs(r["threshold"] - production) < 1e-9), None)
    if prod_row is None:
        print(f"Production threshold ({production:.2f}) is not in this grid.")
    elif abs(best["f1"] - prod_row["f1"]) < 1e-9:
        print(f"Production threshold ({production:.2f}) ties for the best F1 "
              f"(f1={prod_row['f1']:.4f}).")
    else:
        print(f"Production threshold ({production:.2f}): f1={prod_row['f1']:.4f}, "
              f"lower than the swept best by {best['f1'] - prod_row['f1']:.4f}.")


def _save_chart(rows: list[dict], path: str, title: str, xlabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["threshold"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [r["precision"] for r in rows], marker="o", label="Precision")
    plt.plot(xs, [r["recall"] for r in rows], marker="o", label="Recall")
    plt.plot(xs, [r["f1"] for r in rows], marker="o", label="F1")
    nlp_xs = [r["threshold"] for r in rows if r["nlp_accuracy"] is not None]
    nlp_ys = [r["nlp_accuracy"] for r in rows if r["nlp_accuracy"] is not None]
    if nlp_ys:
        plt.plot(nlp_xs, nlp_ys, marker="o", linestyle="--",
                 label="NLP_ACCURACY (semantic_verify)")

    best = max(rows, key=lambda r: r["f1"])
    plt.axvline(best["threshold"], color="gray", linestyle=":", linewidth=1)
    plt.annotate(f"best F1 @ {best['threshold']:.2f}",
                 (best["threshold"], best["f1"]),
                 textcoords="offset points", xytext=(6, 6))

    plt.xlabel(xlabel)
    plt.ylabel("Score")
    plt.title(title)
    # Legend below the axes (outside the data area) so it never covers a line.
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default="labels.json", help="Path to labels.json")
    p.add_argument("--resumes", default="resumes",
                   help="Folder holding the PDFs named as the labels.json keys")
    p.add_argument("--backend-dir", default=str(Path(__file__).resolve().parent),
                   help="Folder containing core/, db/, config.py (default: this file's folder)")
    p.add_argument("--thresholds", default="0.60:0.95:0.05",
                   help="DISCOVERY grid: comma list or start:stop:step")
    p.add_argument("--match-thresholds", default=None,
                   help="MATCHING grid (defaults to --thresholds)")
    p.add_argument("--sweep", default="all",
                   choices=["all", "discovery", "matching", "ranking", "both"],
                   help="which sweep(s) to run; 'both' = discovery + matching")
    p.add_argument("--top-k", type=int, default=None,
                   help="Stage 3 top-K (default: config.TOP_K)")
    p.add_argument("--text-cache", default="sweep_text_cache.json",
                   help="Stage-1 OCR text cache (delete to force re-OCR)")
    p.add_argument("--corpus-cache", default="sweep_corpus_cache.json",
                   help="Job-corpus snapshot (delete to re-read Firestore)")
    p.add_argument("--output", default=None, help="Optional path to write the report as JSON")
    p.add_argument("--charts", action="store_true", help="Save PNG charts")
    p.add_argument("--chart-path", default="threshold_sweep",
                   help="Chart filename prefix (suffixes _discovery/_matching/_ranking are added)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    # core/, db/ and config.py must be importable: accept the current folder
    # (run it from backend/) as well as --backend-dir / this file's folder.
    for d in (args.backend_dir, "."):
        sys.path.insert(0, str(Path(d).resolve()))
    logging.disable(logging.WARNING - 1)   # keep the core.* INFO/DEBUG logs quiet

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"labels file not found: {labels_path}")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    names = [n for n, l in labels.items() if l.get("expected_skills")]
    print(f"{len(names)} labeled resumes with expected_skills "
          f"(of {len(labels)} entries in {labels_path.name}).")

    # ---- Stage 1 (once) ---------------------------------------------------
    print("\nStage 1: text acquisition")
    texts, missing, failed = _acquire_texts(
        names, Path(args.resumes), Path(args.text_cache))
    if missing:
        print(f"  [!] {len(missing)} resume(s) have no PDF in {args.resumes!r} "
              f"and no cached text -- skipped: {missing[:5]}"
              f"{' ...' if len(missing) > 5 else ''}")
    if failed:
        print(f"  [!] {len(failed)} resume(s) returned EMPTY text from the OCR "
              f"service -- skipped (infrastructure failure, not a threshold "
              f"effect): {failed[:5]}{' ...' if len(failed) > 5 else ''}")
    if not texts:
        raise SystemExit("No resume text available; nothing to sweep.")
    samples = {n: (texts[n][0], texts[n][1], labels[n]["expected_skills"])
               for n in names if n in texts}
    print(f"  {len(samples)} resumes ready.")

    # ---- shared setup -----------------------------------------------------
    corpus = _load_corpus(Path(args.corpus_cache) if args.corpus_cache else None)
    from core.scoring    import ITEM_SEMANTIC_THRESHOLD
    from core.validation import _SEMANTIC_THRESHOLD
    patches = _Patches()
    disc_prod = patches.production_discovery_threshold
    print(f"\nProduction thresholds: discovery={disc_prod:.2f}  "
          f"scoring={ITEM_SEMANTIC_THRESHOLD:.2f}  validation={_SEMANTIC_THRESHOLD:.2f}")

    disc_grid  = _parse_thresholds(args.thresholds)
    match_grid = _parse_thresholds(args.match_thresholds or args.thresholds)
    report: dict = dict(
        meta=dict(n_samples=len(samples), n_labeled=len(names),
                  production=dict(discovery=disc_prod,
                                  scoring=ITEM_SEMANTIC_THRESHOLD,
                                  validation=_SEMANTIC_THRESHOLD)))

    do_disc  = args.sweep in ("all", "both", "discovery")
    do_match = args.sweep in ("all", "both", "matching")
    do_rank  = args.sweep in ("all", "ranking")

    # ---- Sweep 1: discovery threshold ------------------------------------
    if do_disc:
        print(f"\nSweep 1 -- DISCOVERY threshold {disc_grid} "
              f"(matcher fixed at scoring={ITEM_SEMANTIC_THRESHOLD:.2f} / "
              f"validation={_SEMANTIC_THRESHOLD:.2f})")
        rows = run_discovery_sweep(samples, corpus, patches, disc_grid,
                                   ITEM_SEMANTIC_THRESHOLD, _SEMANTIC_THRESHOLD)
        _print_table(rows, "Sweep 1: discovery threshold "
                           "(extractor_base._SBERT_COSINE_THRESHOLD)", disc_prod)
        report["discovery_sweep"] = rows
        if args.charts:
            _save_chart(rows, f"{args.chart_path}_discovery.png",
                        "Skill discovery: SBERT threshold sweep",
                        "Discovery cosine similarity threshold")

    # Discovery at the production threshold: the real Stage 2 output that both
    # the matching and the ranking sweeps take as their input.
    prod_preds = (_discover_all(samples, corpus, patches, None)
                  if (do_match or do_rank) else None)

    # ---- Sweep 2: matching threshold -------------------------------------
    if do_match:
        print(f"\nSweep 2 -- MATCHING threshold {match_grid} "
              f"(discovery fixed at production {disc_prod:.2f})")
        rows = run_matching_sweep(samples, prod_preds, match_grid)
        _print_table(rows, "Sweep 2: matching threshold "
                           "(scoring.ITEM_SEMANTIC_THRESHOLD / "
                           "validation._SEMANTIC_THRESHOLD)",
                     ITEM_SEMANTIC_THRESHOLD)
        report["matching_sweep"] = rows

        zero = [n for n, p in prod_preds.items() if not p]
        if zero:
            print(f"\n[!] {len(zero)}/{len(samples)} resumes discovered ZERO skills "
                  f"at the production threshold:")
            for n in zero[:15]:
                print(f"      - {n}")
            if len(zero) > 15:
                print(f"      ... and {len(zero) - 15} more")
        report["production_samples"] = [
            dict(name=n, discovered=prod_preds[n],
                 **_item_metrics_at_threshold(prod_preds[n], samples[n][2],
                                              ITEM_SEMANTIC_THRESHOLD))
            for n in prod_preds]
        if args.charts:
            _save_chart(rows, f"{args.chart_path}_matching.png",
                        "Skill matching: SBERT threshold sweep",
                        "Matching cosine similarity threshold")

    # ---- Sweep 3: Stage 3 job ranking ------------------------------------
    if do_rank:
        from config import TOP_K
        top_k = args.top_k or TOP_K
        meta = {n: (labels[n]["expected_job_title"],
                    labels[n].get("min_match_score", 50))
                for n in samples if labels[n].get("expected_job_title")}
        no_title = [n for n in samples if n not in meta]
        if no_title:
            print(f"\n  [!] {len(no_title)} resume(s) have no expected_job_title "
                  f"and are excluded from the ranking sweep: {no_title[:5]}")
        if not meta:
            print("\nSweep 3 skipped: no labels have expected_job_title.")
        else:
            print(f"\nSweep 3 -- STAGE 3 RANKING {match_grid}, top_k={top_k}, "
                  f"{len(meta)} resumes (real calculate_cosine_similarity; "
                  f"discovery fixed at production; fit_reason LLM disabled; "
                  f"certs not passed)")
            rows = run_ranking_sweep(
                {n: samples[n][0] for n in meta}, meta,
                {n: prod_preds[n] for n in meta}, corpus, patches, match_grid,
                top_k)
            _print_ranking_table(rows, ITEM_SEMANTIC_THRESHOLD, top_k)
            report["ranking_sweep"] = rows
            if args.charts:
                _save_ranking_chart(rows, f"{args.chart_path}_ranking.png", top_k)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()