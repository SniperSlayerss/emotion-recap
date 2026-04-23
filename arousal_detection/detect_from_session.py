"""
detect_from_session.py

Load a recorded session and list every window flagged as high-arousal.

Works with both detector types:
    * Ensemble (recommended)     — a models/ensemble/ directory or
                                   manifest.json. Scores GSR and HRV
                                   independently, combines with
                                   --combine-mode.
    * Single-source (comparison) — a plain .pkl (an old merged model that
                                   expects both gsr_* and hrv_* features,
                                   or a per-source model). Uses the merge
                                   pipeline in session_scoring.py when
                                   appropriate.

Usage:
    python detect_from_session.py <session_dir> --model <path>

    --min-score      Only show rows with normalised score >= this (default: 0.0)
    --all            Show every scored row, not just flagged ones
    --combine-mode   For an ensemble: 'any' | 'all' | 'mean' (default: 'any')
    --mean-threshold For 'mean' mode: threshold on combined normalised score
    --pair-tolerance Seconds between GSR and HRV rows to consider paired
                     under 'all' / 'mean' modes (default: 30)
    --tolerance      Merge-based only: max seconds gap between GSR/HRV
                     for single-source merged models (default: 20)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from arousal_detector import (
    ArousalDetector,
    EnsembleDetector,
    load_detector,
)
from session_scoring import (
    score_session,
    combined_timeline,
    DEFAULT_MERGE_TOLERANCE_S,
)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:d}:{s:02d}"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_ensemble_report(
    results: pd.DataFrame,
    combined: pd.DataFrame,
    mode: str,
    min_score: float,
    show_all: bool,
) -> None:
    scored = results[results["scored"]]
    by_source = scored.groupby("source").size().to_dict()

    print()
    print(f"Scored rows : total={len(scored)} "
          f"({', '.join(f'{s}={n}' for s, n in by_source.items())})")
    print(f"Flagged rows: {int(combined['is_aroused'].sum())}  (mode='{mode}')")
    print("-" * 78)

    hits = combined if show_all else combined[combined["is_aroused"]
                                              & (combined["normalised"] >= min_score)]

    if hits.empty:
        print("No rows matched.")
        return

    title = "ALL SCORED ROWS" if show_all else "FLAGGED ROWS"
    print(f"{title} (sorted by time):")
    print()
    print(f"{'time':>7}  {'src':<4}  {'score':>8}  {'norm':>5}  {'pair':>6}  flag")
    print("-" * 78)

    for _, r in hits.sort_values("time_s").iterrows():
        flag = "AROUSED" if r["is_aroused"] else "baseline"
        pair = format_time(r["paired_with"]) if r.get("paired_with") == r.get("paired_with") else "—"
        # ^ NaN check (NaN != NaN)
        print(f"{format_time(r['time_s']):>7}  {r['source']:<4}  "
              f"{r['score']:>8.4f}  {r['normalised']:>5.2f}  {pair:>6}  {flag}")

    if not show_all:
        print()
        print("TOP 5 PEAKS (by normalised score):")
        print()
        top = hits.nlargest(min(5, len(hits)), "normalised")
        for _, r in top.iterrows():
            print(f"  {format_time(r['time_s']):>7}  {r['source']:<4}  "
                  f"norm={r['normalised']:.2f}  score={r['score']:.4f}")


def print_single_source_report(
    results: pd.DataFrame,
    min_score: float,
    show_all: bool,
    is_merged_model: bool,
) -> None:
    """Report shape for the merge-based / single-source case."""
    total   = len(results)
    scored  = int(results["scored"].sum())
    skipped = total - scored
    aroused = int(results["is_aroused"].sum())

    print()
    print(f"Rows    : {total} total | {scored} scored | "
          f"{aroused} flagged | {skipped} skipped")
    print("-" * 78)

    if show_all:
        hits = results[results["scored"]]
    else:
        hits = results[(results["scored"])
                       & (results["is_aroused"])
                       & (results["normalised"] >= min_score)]

    if hits.empty:
        print("No rows matched.")
        return

    title = "ALL SCORED ROWS" if show_all else "FLAGGED ROWS"
    print(f"{title} (sorted by time):")
    print()
    if is_merged_model:
        print(f"{'time':>7}  {'gap':>5}  {'score':>8}  {'norm':>5}  flag")
    else:
        print(f"{'time':>7}  {'score':>8}  {'norm':>5}  flag")
    print("-" * 78)

    for _, r in hits.sort_values("time_s").iterrows():
        flag = "AROUSED" if r["is_aroused"] else "baseline"
        if is_merged_model:
            gap = (f"{r['match_gap_s']:.0f}s"
                   if r.get("match_gap_s") == r.get("match_gap_s") else "—")
            print(f"{format_time(r['time_s']):>7}  {gap:>5}  "
                  f"{r['score']:>8.4f}  {r['normalised']:>5.2f}  {flag}")
        else:
            print(f"{format_time(r['time_s']):>7}  "
                  f"{r['score']:>8.4f}  {r['normalised']:>5.2f}  {flag}")

    if not show_all:
        print()
        print("TOP 5 PEAKS (by normalised score):")
        print()
        top = hits.nlargest(min(5, len(hits)), "normalised")
        for _, r in top.iterrows():
            print(f"  {format_time(r['time_s']):>7}  "
                  f"norm={r['normalised']:.2f}  score={r['score']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("session_dir", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--all", action="store_true")
    p.add_argument("--combine-mode", choices=("any", "all", "mean"), default="any",
                   help="Ensemble combination mode (default: any)")
    p.add_argument("--mean-threshold", type=float, default=None,
                   help="Ensemble 'mean' mode threshold override")
    p.add_argument("--pair-tolerance", type=float, default=30.0,
                   help="Seconds between GSR/HRV rows to consider paired (all/mean)")
    p.add_argument("--tolerance", type=float, default=DEFAULT_MERGE_TOLERANCE_S,
                   help="Merge-based single-source model only")
    args = p.parse_args()

    csv_path = args.session_dir / "features.csv"
    if not csv_path.exists():
        print(f"[ERROR] No features.csv in {args.session_dir}", file=sys.stderr)
        return 1

    detector = load_detector(
        args.model,
        ensemble_mode=args.combine_mode,
        mean_threshold=args.mean_threshold,
    )

    results = score_session(csv_path, detector, tolerance_s=args.tolerance)

    print(f"\nSession : {args.session_dir}")
    print(f"Model   : {args.model}")

    if isinstance(detector, EnsembleDetector):
        print(f"Mode    : ensemble ({args.combine_mode})")
        combined = combined_timeline(
            results,
            mode=args.combine_mode,
            pair_tolerance_s=args.pair_tolerance,
            mean_threshold=(args.mean_threshold
                            if args.mean_threshold is not None
                            else detector.mean_threshold),
        )
        print_ensemble_report(results, combined, args.combine_mode,
                              args.min_score, args.all)
    else:
        # ArousalDetector — decide whether it's a merged or single-source
        is_merged = (any(c.startswith("gsr_") for c in detector.feature_cols)
                     and any(c.startswith("hrv_") for c in detector.feature_cols))
        kind = "single (merged)" if is_merged else f"single ({detector.source})"
        print(f"Mode    : {kind}")
        if is_merged:
            print(f"Merge tol: ±{args.tolerance:.0f}s between GSR/HRV windows")
        print_single_source_report(results, args.min_score, args.all, is_merged)

    return 0


if __name__ == "__main__":
    sys.exit(main())
