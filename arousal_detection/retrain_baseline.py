"""
retrain_baseline.py

Retrain a single-merged-feature Isolation Forest on BASELINE-ONLY sessions.

Kept for comparison with the ensemble approach. For new work use:

    python train_ensemble.py sessions/ --label-prefix baseline \\
        --gsr-contamination 0.01 --hrv-contamination 0.01 \\
        --gsr-threshold-pct 1.0 --hrv-threshold-pct 1.0

Usage:
    python retrain_baseline.py sessions/ [options]

    --baseline-prefix  Label prefix to match (default: 'baseline')
    --contamination    Expected anomaly fraction in baseline data (default: 0.01)
                       NOTE: lower than the old 0.05 because baseline should
                       be almost entirely "normal".
    --threshold-pct    Percentile of training scores for flag threshold
                       (default: 1.0 — flag anything below 1st percentile)
    --out              Output .pkl path (default: models/iforest.pkl)

    --dry-run          Scan sessions, show what would be used, but don't train.

Produces the same three files as the original trainer:
    <out>.pkl, <out>_meta.json, <out>_report.png
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature columns — match collect_training_data.py output
# ---------------------------------------------------------------------------

GSR_FEATURES = [
    ("gsr_scl_mean",     "SCL mean"),
    ("gsr_scr_count",    "SCR/min"),
    ("gsr_phasic_std",   "Phasic std"),
    ("gsr_scr_mean_amp", "SCR amp"),
]

HRV_FEATURES = [
    ("hrv_hr_mean", "HR mean"),
    ("hrv_rmssd",   "RMSSD"),
    ("hrv_sdnn",    "SDNN"),
    ("hrv_pnn50",   "pNN50"),
]

ALL_FEATURE_COLS  = [c for c, _ in GSR_FEATURES + HRV_FEATURES]
ALL_FEATURE_NAMES = [n for _, n in GSR_FEATURES + HRV_FEATURES]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_baseline_sessions(parent: Path, prefix: str) -> list[Path]:
    """
    Return sorted list of session dirs whose label (from session.json or
    folder name suffix) starts with `prefix`.
    """
    if not parent.is_dir():
        sys.exit(f"[ERROR] Not a directory: {parent}")

    matches = []
    skipped = []

    for d in sorted(parent.iterdir()):
        if not d.is_dir() or not (d / "features.csv").exists():
            continue

        label = _read_label(d)
        if label is None:
            skipped.append((d.name, "no label found"))
            continue

        if label.startswith(prefix):
            matches.append(d)
        else:
            skipped.append((d.name, f"label='{label}'"))

    print(f"[SCAN] Parent: {parent}")
    print(f"[SCAN] Matched {len(matches)} baseline session(s) (prefix='{prefix}'):")
    for d in matches:
        print(f"         ✓ {d.name}")
    if skipped:
        print(f"[SCAN] Skipped {len(skipped)} non-baseline session(s):")
        for name, reason in skipped:
            print(f"         ✗ {name}  ({reason})")

    return matches


def _read_label(session_dir: Path) -> str | None:
    """Prefer session.json label, fall back to parsing folder name."""
    meta_path = session_dir / "session.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            lbl = meta.get("label")
            if lbl:
                return lbl
        except (json.JSONDecodeError, OSError):
            pass

    # Folder name format: YYYYMMDD_HHMMSS_<label>
    parts = session_dir.name.split("_", 2)
    if len(parts) >= 3:
        return parts[2]
    return None


# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------


def load_features(session_dirs: list[Path]) -> pd.DataFrame:
    """Concatenate features.csv from all matched sessions."""
    frames = []
    for d in session_dirs:
        df = pd.read_csv(d / "features.csv")
        df["_session"] = d.name
        frames.append(df)
    if not frames:
        sys.exit("[ERROR] No features.csv loaded.")
    return pd.concat(frames, ignore_index=True)


def build_feature_matrix(df: pd.DataFrame, tolerance_s: float = 20.0) -> pd.DataFrame:
    """
    Merge GSR and HRV rows into one row per window using temporal matching
    (±tolerance_s seconds). Unmatched rows are dropped. This matches what
    the detector does at inference time, eliminating the train/inference
    distribution mismatch caused by half-imputation.

    Sessions are merged independently (we never pair GSR from one session
    with HRV from another).
    """
    from session_scoring import merge_gsr_hrv

    if "source" not in df.columns:
        sys.exit("[ERROR] features.csv missing 'source' column")

    # Merge per-session, then concatenate
    session_col = "_session" if "_session" in df.columns else None
    if session_col is None:
        # Fall back: treat the whole df as one session
        merged = merge_gsr_hrv(df, tolerance_s=tolerance_s)
    else:
        frames = []
        for sess, sess_df in df.groupby(session_col):
            sess_merged = merge_gsr_hrv(sess_df, tolerance_s=tolerance_s)
            sess_merged[session_col] = sess
            frames.append(sess_merged)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if merged.empty:
        sys.exit("[ERROR] No rows remaining after GSR+HRV merge.")

    # Drop rows where any feature is NaN — we only train on fully-merged windows
    feature_cols = [c for c in merged.columns
                    if c.startswith("gsr_") or c.startswith("hrv_")]
    before = len(merged)
    merged = merged.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = before - len(merged)
    if dropped:
        print(f"[MERGE] Dropped {dropped} unmatched row(s) "
              f"({before} → {len(merged)} after merge+dropna)")

    return merged


def validate_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    After merging, rows should have no NaN in feature columns. This is now
    just a safety net — if anything slipped through, drop it.
    """
    available_cols = [c for c in ALL_FEATURE_COLS if c in X.columns]
    row_missing = X[available_cols].isnull().any(axis=1)
    n_drop = int(row_missing.sum())
    if n_drop:
        print(f"[CLEAN] Dropping {n_drop} row(s) with any missing features "
              f"(shouldn't happen after merge — check session data).")
        X = X[~row_missing].copy()
    return X


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(X: np.ndarray, contamination: float, n_estimators: int = 200) -> Pipeline:
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


def compute_threshold(pipe: Pipeline, X: np.ndarray, percentile: float) -> float:
    scores = pipe.score_samples(X)
    return float(np.percentile(scores, percentile))


# ---------------------------------------------------------------------------
# Diagnostic report
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
    is_inlier = scores >= threshold

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Baseline Isolation Forest — Trained on {X.shape[0]} windows",
        fontsize=13, fontweight="bold",
    )

    BLUE, RED, GREEN = "#4C78A8", "#E45756", "#54A24B"

    # 1. Score distribution
    ax = axes[0]
    ax.hist(scores, bins=40, color=BLUE, alpha=0.8, edgecolor="white")
    ax.axvline(threshold, color=RED, ls="--", lw=2, label=f"Threshold = {threshold:.4f}")
    ax.axvline(scores.mean(), color=GREEN, ls=":", lw=1.5, label=f"Mean = {scores.mean():.4f}")
    ax.set_xlabel("Anomaly score (lower = more anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Training Score Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Feature means: inliers vs outliers
    ax = axes[1]
    inlier_means  = X[is_inlier].mean(axis=0)  if is_inlier.any() else np.zeros(X.shape[1])
    outlier_means = X[~is_inlier].mean(axis=0) if (~is_inlier).any() else np.zeros(X.shape[1])
    x_pos = np.arange(len(feature_cols))
    width = 0.38
    ax.bar(x_pos - width/2, inlier_means,  width, color=BLUE, label="Inlier mean")
    ax.bar(x_pos + width/2, outlier_means, width, color=RED,  label="Outlier mean")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Raw feature value")
    ax.set_title("Feature means: inliers vs outliers")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Score trace over training rows
    ax = axes[2]
    ax.plot(scores, lw=0.8, color=BLUE, alpha=0.8)
    ax.axhline(threshold, color=RED, ls="--", lw=1.2, label="Threshold")
    ax.set_xlabel("Training window index")
    ax.set_ylabel("Score")
    ax.set_title("Score trace (ordered by session)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[TRAIN] Report saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parent", type=Path, help="Parent directory containing session folders")
    parser.add_argument("--baseline-prefix", default="baseline",
                        help="Label prefix that marks a baseline session (default: 'baseline')")
    parser.add_argument("--contamination", type=float, default=0.01,
                        help="Expected anomaly fraction in baseline data (default: 0.01)")
    parser.add_argument("--threshold-pct", type=float, default=1.0,
                        help="Percentile of training scores for flag threshold (default: 1.0)")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("models/iforest.pkl"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Just list matched sessions, don't train.")
    args = parser.parse_args()

    # --- Find baseline sessions ---
    sessions = find_baseline_sessions(args.parent, args.baseline_prefix)
    if not sessions:
        sys.exit(f"[ERROR] No sessions found with label prefix '{args.baseline_prefix}'")

    if args.dry_run:
        print("\n[DRY-RUN] No training performed.")
        return 0

    # --- Load + clean ---
    df_raw = load_features(sessions)
    print(f"\n[LOAD] Total raw rows across baseline sessions: {len(df_raw)}")

    df_merged = build_feature_matrix(df_raw)
    df_clean  = validate_features(df_merged)

    available_cols = [c for c in ALL_FEATURE_COLS if c in df_clean.columns]
    available_names = [
        ALL_FEATURE_NAMES[i]
        for i, c in enumerate(ALL_FEATURE_COLS)
        if c in df_clean.columns
    ]
    if len(available_cols) < 3:
        sys.exit(f"[ERROR] Too few features available: {available_cols}")

    X = df_clean[available_cols].values
    print(f"[TRAIN] Training matrix: {X.shape[0]} windows × {X.shape[1]} features")

    if X.shape[0] < 100:
        print(f"[WARN] Only {X.shape[0]} windows — more baseline data strongly recommended.")

    # --- Train ---
    pipe = train(X, contamination=args.contamination, n_estimators=args.n_estimators)
    threshold = compute_threshold(pipe, X, percentile=args.threshold_pct)

    scores = pipe.score_samples(X)
    n_flagged = int((scores < threshold).sum())

    print(f"\n[TRAIN] ── Results ─────────────────────────────────")
    print(f"         Baseline sessions  : {len(sessions)}")
    print(f"         Windows trained on : {X.shape[0]}")
    print(f"         Features           : {len(available_cols)}")
    print(f"         Contamination      : {args.contamination}")
    print(f"         Threshold (pct {args.threshold_pct}) : {threshold:.4f}")
    print(f"         Training flagged   : {n_flagged} / {X.shape[0]}  ({n_flagged/X.shape[0]:.1%})")
    print(f"         Score range        : [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"         Score mean ± std   : {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"[TRAIN] ────────────────────────────────────────────\n")

    # --- Save ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"[TRAIN] Model saved:    {args.out}")

    meta = {
        "feature_cols":       available_cols,
        "feature_names":      available_names,
        "threshold":          threshold,
        "contamination":      args.contamination,
        "n_estimators":       args.n_estimators,
        "n_training_windows": int(X.shape[0]),
        "score_mean":         float(scores.mean()),
        "score_std":          float(scores.std()),
        "score_min":          float(scores.min()),
        "score_max":          float(scores.max()),
        "trained_at":         datetime.now().isoformat(),
        "label_filter":       f"{args.baseline_prefix}*",
        "training_sessions":  [s.name for s in sessions],
    }
    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[TRAIN] Metadata saved: {meta_path}")

    report_path = args.out.with_name(args.out.stem + "_report.png")
    plot_report(pipe, X, available_cols, available_names, threshold, report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
