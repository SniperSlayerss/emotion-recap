"""
train_model.py

Train an Isolation Forest on baseline session data and save the model artefacts.

Usage:
    python train_model.py <session_dir_or_glob> [<session_dir_2> ...] [options]

Examples:
    # Train on all sessions labelled 'rest'
    python train_model.py sessions/*/

    # Train only on specific label
    python train_model.py sessions/ --label rest

    # Set a custom contamination estimate (fraction of expected anomalies in training)
    python train_model.py sessions/ --label rest --contamination 0.05

    # Save model to a specific path
    python train_model.py sessions/ --out models/iforest_v1.pkl

Outputs (all to --out directory or alongside the .pkl):
    iforest.pkl         Trained IsolationForest + StandardScaler bundled together
    iforest_meta.json   Feature list, thresholds, training stats for the detector
    iforest_report.png  Diagnostic plot: score distributions, feature importances
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature columns — must match what collect_training_data.py writes
# ---------------------------------------------------------------------------

# Pairs of (csv_column, short_name)
GSR_FEATURES = [
    ("gsr_scl_mean",      "SCL mean"),
    ("gsr_scr_count",     "SCR/min"),
    ("gsr_phasic_std",    "Phasic std"),
    ("gsr_scr_mean_amp",  "SCR amp"),
]

HRV_FEATURES = [
    ("hrv_hr_mean",  "HR mean"),
    ("hrv_rmssd",    "RMSSD"),
    ("hrv_sdnn",     "SDNN"),
    ("hrv_pnn50",    "pNN50"),
]

ALL_FEATURE_COLS = [col for col, _ in GSR_FEATURES + HRV_FEATURES]
ALL_FEATURE_NAMES = [name for _, name in GSR_FEATURES + HRV_FEATURES]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sessions(paths: list[Path], label_filter: str | None) -> pd.DataFrame:
    """
    Load and concatenate features.csv from one or more session directories.
    Each path can be a session directory or a glob pattern parent.
    """
    dfs = []
    found_csvs = []

    for p in paths:
        p = Path(p)
        if p.is_dir():
            # Could be a single session dir or a parent of many
            csvs = list(p.glob("*/features.csv")) or [p / "features.csv"]
            found_csvs.extend([c for c in csvs if c.exists()])
        else:
            # Treat as direct CSV path
            if p.exists():
                found_csvs.append(p)

    if not found_csvs:
        sys.exit("[ERROR] No features.csv files found. Check your paths.")

    print(f"[TRAIN] Found {len(found_csvs)} session file(s):")
    for c in found_csvs:
        print(f"        {c}")

    for csv in found_csvs:
        df = pd.read_csv(csv)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if label_filter:
        before = len(df)
        df = df[df["label"] == label_filter]
        print(f"[TRAIN] Label filter '{label_filter}': {before} → {len(df)} rows")

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge GSR and HRV rows into a single row per time window.

    Each window has a window_start_time. GSR and HRV emit windows at
    different rates, so we merge them by nearest timestamp within a
    tolerance (half the HRV step = 15s).
    """
    gsr = df[df["source"] == "gsr"][["window_start_time"] + [c for c in ALL_FEATURE_COLS if c.startswith("gsr_")]].copy()
    hrv = df[df["source"] == "hrv"][["window_start_time"] + [c for c in ALL_FEATURE_COLS if c.startswith("hrv_")]].copy()

    gsr = gsr.sort_values("window_start_time").reset_index(drop=True)
    hrv = hrv.sort_values("window_start_time").reset_index(drop=True)

    # If we have both sources, merge on nearest timestamp
    if not gsr.empty and not hrv.empty:
        merged = pd.merge_asof(
            gsr, hrv,
            on="window_start_time",
            tolerance=15.0,          # seconds
            direction="nearest",
        )
    elif not gsr.empty:
        merged = gsr
    else:
        merged = hrv

    return merged


def validate_features(X: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with too many NaNs, warn about remaining ones."""
    missing_thresh = 0.5  # Drop rows missing >50% of features
    available_cols = [c for c in ALL_FEATURE_COLS if c in X.columns]

    row_missing = X[available_cols].isnull().mean(axis=1)
    n_drop = (row_missing > missing_thresh).sum()
    if n_drop:
        print(f"[TRAIN] Dropping {n_drop} rows with >{missing_thresh:.0%} missing features.")
        X = X[row_missing <= missing_thresh].copy()

    # For remaining NaNs, impute with column median
    for col in available_cols:
        n_nan = X[col].isnull().sum()
        if n_nan:
            median = X[col].median()
            X[col] = X[col].fillna(median)
            print(f"[TRAIN] Imputed {n_nan} NaN in '{col}' with median={median:.4f}")

    return X


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X: np.ndarray, contamination: float, n_estimators: int = 200) -> Pipeline:
    """Fit StandardScaler + IsolationForest pipeline."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X)
    return pipe


def compute_threshold(pipe: Pipeline, X: np.ndarray, percentile: float = 5.0) -> float:
    """
    Derive an anomaly score threshold from the training data.

    IsolationForest.score_samples() returns negative average path lengths;
    more negative = more anomalous. We set the flag threshold at the
    `percentile`-th percentile of training scores, so that ~percentile% of
    baseline windows are considered anomalous (expected false positive rate).
    """
    scores = pipe.score_samples(X)
    threshold = float(np.percentile(scores, percentile))
    return threshold


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def plot_report(
    pipe: Pipeline,
    X: np.ndarray,
    feature_cols: list[str],
    feature_names: list[str],
    threshold: float,
    out_path: Path,
) -> None:
    scores = pipe.score_samples(X)
    predictions = pipe.predict(X)   # +1 = inlier, -1 = outlier

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Isolation Forest — Training Diagnostic Report", fontsize=13, fontweight="bold")

    BLUE, RED, GREEN = "#4C78A8", "#E45756", "#54A24B"

    # 1. Score distribution
    ax = axes[0]
    ax.hist(scores, bins=40, color=BLUE, alpha=0.8, edgecolor="white", label="Baseline scores")
    ax.axvline(threshold, color=RED, lw=2, ls="--", label=f"Flag threshold ({threshold:.3f})")
    n_flagged = (scores < threshold).sum()
    ax.set_xlabel("Anomaly score (more negative = more anomalous)")
    ax.set_ylabel("Window count")
    ax.set_title(f"Score Distribution\n({n_flagged}/{len(scores)} training windows flagged)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Feature distributions (coloured by inlier/outlier)
    ax = axes[1]
    scaler = pipe.named_steps["scaler"]
    X_scaled = scaler.transform(X)

    inlier_means  = X_scaled[predictions ==  1].mean(axis=0)
    outlier_means = X_scaled[predictions == -1].mean(axis=0) if (predictions == -1).any() else np.zeros(len(feature_names))

    x = np.arange(len(feature_names))
    width = 0.35
    ax.barh(x - width/2, inlier_means,  width, color=GREEN, alpha=0.8, label="Inliers (mean z-score)")
    ax.barh(x + width/2, outlier_means, width, color=RED,   alpha=0.8, label="Outliers (mean z-score)")
    ax.set_yticks(x)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("Mean z-score")
    ax.set_title("Feature Means\n(inliers vs outliers in training set)")
    ax.axvline(0, color="black", lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    # 3. Anomaly score over time (window index as proxy)
    ax = axes[2]
    ax.plot(scores, color=BLUE, lw=1, alpha=0.7, label="Anomaly score")
    ax.axhline(threshold, color=RED, lw=1.5, ls="--", label="Flag threshold")
    ax.fill_between(range(len(scores)), scores, threshold,
                    where=(scores < threshold), color=RED, alpha=0.3, label="Flagged")
    ax.set_xlabel("Window index (chronological)")
    ax.set_ylabel("Anomaly score")
    ax.set_title("Score Trace\n(training windows in order)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[TRAIN] Report saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest on baseline sessions.")
    parser.add_argument("paths", nargs="+", help="Session directories or parent directory")
    parser.add_argument("--label", default=None, help="Filter to rows with this label (e.g. 'rest')")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Expected fraction of anomalies in training data (default: 0.05)")
    parser.add_argument("--threshold-pct", type=float, default=5.0,
                        help="Percentile of training scores to use as flag threshold (default: 5.0)")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--out", default="models/iforest.pkl",
                        help="Output path for the model .pkl (default: models/iforest.pkl)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    df_raw = load_sessions([Path(p) for p in args.paths], args.label)
    print(f"[TRAIN] Total raw rows: {len(df_raw)}")

    df_merged = build_feature_matrix(df_raw)
    print(f"[TRAIN] Merged feature rows: {len(df_merged)}")

    df_merged = validate_features(df_merged)

    available_cols  = [c for c in ALL_FEATURE_COLS  if c in df_merged.columns]
    available_names = [ALL_FEATURE_NAMES[i] for i, c in enumerate(ALL_FEATURE_COLS) if c in df_merged.columns]

    if len(available_cols) < 3:
        sys.exit(f"[ERROR] Too few features available: {available_cols}. Need at least 3.")

    X = df_merged[available_cols].values
    print(f"[TRAIN] Training on {X.shape[0]} windows × {X.shape[1]} features: {available_cols}")

    if X.shape[0] < 50:
        print(f"[WARN]  Only {X.shape[0]} windows — consider recording more baseline sessions.")

    # --- Train ---
    pipe = train(X, contamination=args.contamination, n_estimators=args.n_estimators)
    threshold = compute_threshold(pipe, X, percentile=args.threshold_pct)

    train_scores = pipe.score_samples(X)
    n_flagged = (train_scores < threshold).sum()

    print(f"\n[TRAIN] ── Results ─────────────────────────────────")
    print(f"         Windows trained on : {X.shape[0]}")
    print(f"         Features           : {len(available_cols)}")
    print(f"         Anomaly threshold  : {threshold:.4f}")
    print(f"         Training flagged   : {n_flagged} / {X.shape[0]}  ({n_flagged/X.shape[0]:.1%})")
    print(f"         Score range        : [{train_scores.min():.4f}, {train_scores.max():.4f}]")
    print(f"         Score mean ± std   : {train_scores.mean():.4f} ± {train_scores.std():.4f}")
    print(f"[TRAIN] ────────────────────────────────────────────\n")

    # --- Save model ---
    joblib.dump(pipe, out_path)
    print(f"[TRAIN] Model saved: {out_path}")

    # --- Save metadata (for detector to load) ---
    meta = {
        "feature_cols":     available_cols,
        "feature_names":    available_names,
        "threshold":        threshold,
        "contamination":    args.contamination,
        "n_estimators":     args.n_estimators,
        "n_training_windows": int(X.shape[0]),
        "score_mean":       float(train_scores.mean()),
        "score_std":        float(train_scores.std()),
        "score_min":        float(train_scores.min()),
        "score_max":        float(train_scores.max()),
        "trained_at":       datetime.now().isoformat(),
        "label_filter":     args.label,
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[TRAIN] Metadata saved: {meta_path}")

    # --- Diagnostic plot ---
    report_path = out_path.with_name(out_path.stem + "_report.png")
    plot_report(pipe, X, available_cols, available_names, threshold, report_path)


if __name__ == "__main__":
    main()
