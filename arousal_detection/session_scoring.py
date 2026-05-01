from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from arousal_detector import (
    ArousalDetector,
    ArousalResult,
    EnsembleDetector,
)


DEFAULT_MERGE_TOLERANCE_S = 20.0


def _to_float(v) -> Optional[float]:
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

    if "source" not in df.columns or "window_start_time" not in df.columns:
        raise ValueError("DataFrame must have 'source' and 'window_start_time' columns")

    gsr = (
        df[df["source"] == "gsr"]
        .sort_values("window_start_time")
        .reset_index(drop=True)
    )
    hrv = (
        df[df["source"] == "hrv"]
        .sort_values("window_start_time")
        .reset_index(drop=True)
    )

    gsr_cols = [c for c in df.columns if c.startswith("gsr_")]
    hrv_cols = [c for c in df.columns if c.startswith("hrv_")]

    merged_rows = []

    if gsr.empty and not hrv.empty:
        for _, row in hrv.iterrows():
            out = {
                "window_start_time": row["window_start_time"],
                "hrv_time_s": row["window_start_time"],
                "match_gap_s": 0.0,
            }
            out.update({c: np.nan for c in gsr_cols})
            out.update({c: row[c] for c in hrv_cols})
            merged_rows.append(out)
        return pd.DataFrame(merged_rows)

    hrv_times = hrv["window_start_time"].to_numpy() if not hrv.empty else np.array([])

    for _, g in gsr.iterrows():
        out = {"window_start_time": g["window_start_time"]}
        out.update({c: g[c] for c in gsr_cols})

        if hrv_times.size:
            idx = int(np.argmin(np.abs(hrv_times - g["window_start_time"])))
            gap = abs(hrv_times[idx] - g["window_start_time"])
            if gap <= tolerance_s:
                h = hrv.iloc[idx]
                out["hrv_time_s"] = h["window_start_time"]
                out["match_gap_s"] = float(gap)
                out.update({c: h[c] for c in hrv_cols})
            else:
                out["hrv_time_s"] = np.nan
                out["match_gap_s"] = np.nan
                out.update({c: np.nan for c in hrv_cols})
        else:
            out["hrv_time_s"] = np.nan
            out["match_gap_s"] = np.nan
            out.update({c: np.nan for c in hrv_cols})

        merged_rows.append(out)

    return pd.DataFrame(merged_rows)


def score_merged_row(
    row: pd.Series,
    detector: ArousalDetector,
) -> Optional[ArousalResult]:

    features = {}
    for col in detector.feature_cols:
        v = _to_float(row.get(col))
        if v is None:
            return None
        features[col] = v
    return detector.score(features, source="merged")


def score_session_file(
    csv_path: Path,
    detector: ArousalDetector,
    tolerance_s: float = DEFAULT_MERGE_TOLERANCE_S,
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    merged = merge_gsr_hrv(df, tolerance_s=tolerance_s)

    n = len(merged)
    scores = np.full(n, np.nan)
    normalised_a = np.full(n, np.nan)
    aroused = np.zeros(n, dtype=bool)
    scored = np.zeros(n, dtype=bool)

    for i, row in merged.iterrows():
        result = score_merged_row(row, detector)
        if result is None:
            continue
        scores[i] = result.score
        normalised_a[i] = result.normalised
        aroused[i] = result.is_aroused
        scored[i] = True

    return pd.DataFrame(
        {
            "time_s": merged["window_start_time"],
            "hrv_time_s": merged.get("hrv_time_s", np.nan),
            "match_gap_s": merged.get("match_gap_s", np.nan),
            "score": scores,
            "normalised": normalised_a,
            "is_aroused": aroused,
            "scored": scored,
        }
    )


def _score_source_rows(
    df: pd.DataFrame,
    source: str,
    detector: ArousalDetector,
) -> pd.DataFrame:
    """Score every row where df['source'] == source. No imputation."""
    rows = (
        df[df["source"] == source]
        .sort_values("window_start_time")
        .reset_index(drop=True)
    )
    if rows.empty or detector is None:
        return pd.DataFrame(
            columns=[
                "time_s",
                "source",
                "score",
                "normalised",
                "is_aroused",
                "scored",
            ]
        )

    n = len(rows)
    scores = np.full(n, np.nan)
    normalised_a = np.full(n, np.nan)
    aroused = np.zeros(n, dtype=bool)
    scored = np.zeros(n, dtype=bool)

    for i, row in rows.iterrows():
        features = {}
        complete = True
        for col in detector.feature_cols:
            v = _to_float(row.get(col))
            if v is None:
                complete = False
                break
            features[col] = v
        if not complete:
            continue

        result = detector.score(features, source=source)
        if result is None:
            continue
        scores[i] = result.score
        normalised_a[i] = result.normalised
        aroused[i] = result.is_aroused
        scored[i] = True

    return pd.DataFrame(
        {
            "time_s": rows["window_start_time"].values,
            "source": source,
            "score": scores,
            "normalised": normalised_a,
            "is_aroused": aroused,
            "scored": scored,
        }
    )


def score_session_ensemble(
    csv_path: Path,
    detector: EnsembleDetector,
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    frames = []
    if detector.gsr is not None:
        frames.append(_score_source_rows(df, "gsr", detector.gsr))
    if detector.hrv is not None:
        frames.append(_score_source_rows(df, "hrv", detector.hrv))

    if not frames:
        return pd.DataFrame(
            columns=[
                "time_s",
                "source",
                "score",
                "normalised",
                "is_aroused",
                "scored",
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("time_s").reset_index(drop=True)


def combined_timeline(
    scored_rows: pd.DataFrame,
    mode: str = "any",
    pair_tolerance_s: float = 30.0,
    mean_threshold: float = 0.5,
) -> pd.DataFrame:

    if scored_rows.empty:
        return scored_rows.assign(paired_with=pd.Series(dtype=float))

    if mode == "any":
        out = scored_rows.copy()
        out["paired_with"] = np.nan
        return out

    if mode not in ("all", "mean"):
        raise ValueError(f"mode must be 'any' | 'all' | 'mean', got {mode!r}")

    gsr_rows = scored_rows[scored_rows["source"] == "gsr"].reset_index(drop=True)
    hrv_rows = scored_rows[scored_rows["source"] == "hrv"].reset_index(drop=True)

    results = []
    used_hrv: set[int] = set()
    hrv_times = hrv_rows["time_s"].to_numpy() if not hrv_rows.empty else np.array([])

    for _, g in gsr_rows.iterrows():
        if hrv_times.size == 0:
            results.append({**g.to_dict(), "is_aroused": False, "paired_with": np.nan})
            continue

        idx = int(np.argmin(np.abs(hrv_times - g["time_s"])))
        gap = abs(hrv_times[idx] - g["time_s"])

        if (
            gap <= pair_tolerance_s
            and bool(g["scored"])
            and bool(hrv_rows.iloc[idx]["scored"])
        ):
            h = hrv_rows.iloc[idx]
            used_hrv.add(idx)

            if mode == "all":
                flag = bool(g["is_aroused"]) and bool(h["is_aroused"])
            else:
                flag = ((g["normalised"] + h["normalised"]) / 2) >= mean_threshold

            results.append(
                {**g.to_dict(), "is_aroused": flag, "paired_with": float(h["time_s"])}
            )
            results.append(
                {**h.to_dict(), "is_aroused": flag, "paired_with": float(g["time_s"])}
            )
        else:
            results.append({**g.to_dict(), "is_aroused": False, "paired_with": np.nan})

    for i, h in hrv_rows.iterrows():
        if i in used_hrv:
            continue
        results.append({**h.to_dict(), "is_aroused": False, "paired_with": np.nan})

    return pd.DataFrame(results).sort_values("time_s").reset_index(drop=True)


def score_session(
    csv_path: Path,
    detector: Union[ArousalDetector, EnsembleDetector],
    **kwargs,
) -> pd.DataFrame:

    if isinstance(detector, EnsembleDetector):
        return score_session_ensemble(csv_path, detector)

    has_gsr = any(c.startswith("gsr_") for c in detector.feature_cols)
    has_hrv = any(c.startswith("hrv_") for c in detector.feature_cols)

    if has_gsr and has_hrv:
        tolerance_s = kwargs.get("tolerance_s", DEFAULT_MERGE_TOLERANCE_S)
        return score_session_file(csv_path, detector, tolerance_s=tolerance_s)

    source = "gsr" if has_gsr else "hrv"
    df = pd.read_csv(csv_path)
    return _score_source_rows(df, source, detector)
