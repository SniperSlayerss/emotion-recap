"""
detect_from_session.py

Load a recorded session, merge GSR + HRV windows temporally, and list every
merged window flagged as high-arousal.

Usage:
    python detect_from_session.py <session_dir> --model <path/to/iforest.pkl>

    --min-score     Only show windows with normalised score >= this (default: 0.0)
    --all           Show every scored window, not just flagged ones
    --tolerance     Max seconds between GSR and HRV windows to merge (default: 20)
"""

import argparse
import sys
from pathlib import Path

from arousal_detector import ArousalDetector
from session_scoring import score_session_file, DEFAULT_MERGE_TOLERANCE_S


def format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:d}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_MERGE_TOLERANCE_S,
                        help="Max seconds gap between GSR and HRV windows to pair")
    args = parser.parse_args()

    csv_path = args.session_dir / "features.csv"
    if not csv_path.exists():
        print(f"[ERROR] No features.csv in {args.session_dir}", file=sys.stderr)
        return 1

    detector = ArousalDetector(args.model)
    results = score_session_file(csv_path, detector, tolerance_s=args.tolerance)

    total    = len(results)
    scored   = int(results["scored"].sum())
    skipped  = total - scored
    aroused  = int(results["is_aroused"].sum())

    print()
    print(f"Session   : {args.session_dir}")
    print(f"Model     : {args.model}")
    print(f"Tolerance : ±{args.tolerance:.0f}s between GSR/HRV windows")
    print(f"Windows   : {total} merged | {scored} scored | {aroused} flagged | "
          f"{skipped} skipped (no matching HRV within tolerance)")
    print("-" * 78)

    if args.all:
        hits = results[results["scored"]]
    else:
        hits = results[(results["scored"]) & (results["is_aroused"])
                       & (results["normalised"] >= args.min_score)]

    if hits.empty:
        print("No windows matched.")
        return 0

    label = "ALL SCORED WINDOWS" if args.all else "FLAGGED WINDOWS"
    print(f"{label} (sorted by time):")
    print()
    print(f"{'time':>7}  {'gap':>5}  {'score':>8}  {'norm':>5}  flag")
    print("-" * 78)

    for _, r in hits.sort_values("time_s").iterrows():
        flag = "AROUSED" if r["is_aroused"] else "baseline"
        gap = f"{r['match_gap_s']:.0f}s" if r["match_gap_s"] == r["match_gap_s"] else "—"
        print(f"{format_time(r['time_s']):>7}  {gap:>5}  "
              f"{r['score']:>8.4f}  {r['normalised']:>5.2f}  {flag}")

    # Top 5 peaks
    if not args.all:
        print()
        print("TOP 5 PEAKS (by normalised score):")
        print()
        top = hits.nlargest(min(5, len(hits)), "normalised")
        for _, r in top.iterrows():
            print(f"  {format_time(r['time_s']):>7}  "
                  f"norm={r['normalised']:.2f}  score={r['score']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
