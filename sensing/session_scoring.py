"""
session_scoring.py

Shared helpers for scoring a recorded session against the trained detector.

The training pipeline built each row by combining GSR and HRV features within
a session (each raw row only carries one source's features; the trainer
imputes the other source's features with the training-set median). Live
inference in arousal_detector.py used zero-imputation, which creates
synthetic "impossible" combinations at detection time and over-flags.

The fix here: merge each GSR row with its nearest-in-time HRV row (within
`tolerance_s` seconds) to produce full 8-feature windows, then score once
per merged window. No imputation required because every merged row has real
values for every feature.

Used by:
    detect_from_session.py   — offline session analysis / CLI report
    extract_clips.py         — clip extraction around high-arousal moments
    plot_session.py          — arousal overlays on diagnostic plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from arousal_detector import ArousalDetector, ArousalResult


DEFAULT_MERGE_TOLERANCE_S = 20.0


# ---------------------------------------------------------------------------
# Row merging
# ---------------------------------------------------------------------------


def _to_float(v):
    if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def merge_gsr_hrv(
    df: pd.DataFrame,
    tolerance_s: float = DEFAULT_MERGE_TOLERANCE_S,
) -> pd.DataFrame:
    """
    For each GSR row, find the HRV row with the closest window_start_time
    (within `tolerance_s` seconds) and combine them into one row with
    both sets of features populated.

    Unmatched GSR rows are kept with HRV features as NaN (and the caller
    can decide to drop or skip them — the scorer will skip incomplete rows).

    Returns a DataFrame indexed 0..N-1 with columns:
        window_start_time  (from the GSR row)
        hrv_time_s         (from the matched HRV row, or NaN)
        match_gap_s        (abs time diff, or NaN if unmatched)
        gsr_*              (GSR features)
        hrv_*              (HRV features, NaN if unmatched)
    """
    if "source" not in df.columns or "window_start_time" not in df.columns:
        raise ValueError("DataFrame must have 'source' and 'window_start_time' columns")

    gsr = df[df["source"] == "gsr"].sort_values("window_start_time").reset_index(drop=True)
    hrv = df[df["source"] == "hrv"].sort_values("window_start_time").reset_index(drop=True)

    gsr_cols = [c for c in df.columns if c.startswith("gsr_")]
    hrv_cols = [c for c in df.columns if c.startswith("hrv_")]

    merged_rows = []

    if gsr.empty and not hrv.empty:
        # Fall back to HRV-only rows (no GSR to anchor)
        for _, row in hrv.iterrows():
            out = {
                "window_start_time": row["window_start_time"],
                "hrv_time_s":        row["window_start_time"],
                "match_gap_s":       0.0,
            }
            for c in gsr_cols:
                out[c] = np.nan
            for c in hrv_cols:
                out[c] = row[c]
            merged_rows.append(out)
        return pd.DataFrame(merged_rows)

    # Walk GSR rows and use a two-pointer scan through HRV (sorted)
    hrv_times = hrv["window_start_time"].to_numpy() if not hrv.empty else np.array([])

    for _, g in gsr.iterrows():
        out = {"window_start_time": g["window_start_time"]}
        for c in gsr_cols:
            out[c] = g[c]

        if hrv_times.size:
            idx = int(np.argmin(np.abs(hrv_times - g["window_start_time"])))
            gap = abs(hrv_times[idx] - g["window_start_time"])
            if gap <= tolerance_s:
                h = hrv.iloc[idx]
                out["hrv_time_s"] = h["window_start_time"]
                out["match_gap_s"] = float(gap)
                for c in hrv_cols:
                    out[c] = h[c]
            else:
                out["hrv_time_s"] = np.nan
                out["match_gap_s"] = np.nan
                for c in hrv_cols:
                    out[c] = np.nan
        else:
            out["hrv_time_s"] = np.nan
            out["match_gap_s"] = np.nan
            for c in hrv_cols:
                out[c] = np.nan

        merged_rows.append(out)

    return pd.DataFrame(merged_rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_merged_row(
    row: pd.Series,
    detector: ArousalDetector,
) -> Optional[ArousalResult]:
    """
    Score one merged row. Returns None if the row is missing any feature
    the detector expects (the whole point of merging is to avoid imputation
    — an incomplete row means the GSR window had no matching HRV window,
    so we don't score it rather than fake it).
    """
    features = {}
    for col in detector.feature_cols:
        v = _to_float(row.get(col))
        if v is None:
            return None  # any missing feature → skip
        features[col] = v

    # Merged rows have all prefixed keys; source="merged" tells the detector
    # not to try re-prefixing.
    return detector.score(features, source="merged")


def score_session_file(
    csv_path: Path,
    detector: ArousalDetector,
    tolerance_s: float = DEFAULT_MERGE_TOLERANCE_S,
) -> pd.DataFrame:
    """
    End-to-end: read a session's features.csv, merge GSR+HRV rows, score
    each merged row.

    Returns a DataFrame with columns:
        time_s          window_start_time of the GSR row (or HRV if no GSR)
        hrv_time_s      matched HRV window time (or NaN)
        match_gap_s     seconds between matched windows
        score           raw Isolation Forest score (NaN if unscored)
        normalised      0..1 normalised score (NaN if unscored)
        is_aroused      bool — True if flagged
        scored          bool — True if the row could be scored
    """
    df = pd.read_csv(csv_path)
    merged = merge_gsr_hrv(df, tolerance_s=tolerance_s)

    scores        = np.full(len(merged), np.nan)
    normalised_a  = np.full(len(merged), np.nan)
    aroused       = np.zeros(len(merged), dtype=bool)
    scored        = np.zeros(len(merged), dtype=bool)

    for i, row in merged.iterrows():
        result = score_merged_row(row, detector)
        if result is None:
            continue
        scores[i]       = result.score
        normalised_a[i] = result.normalised
        aroused[i]      = result.is_aroused
        scored[i]       = True

    out = pd.DataFrame({
        "time_s":       merged["window_start_time"],
        "hrv_time_s":   merged.get("hrv_time_s", np.nan),
        "match_gap_s":  merged.get("match_gap_s", np.nan),
        "score":        scores,
        "normalised":   normalised_a,
        "is_aroused":   aroused,
        "scored":       scored,
    })
    return out
