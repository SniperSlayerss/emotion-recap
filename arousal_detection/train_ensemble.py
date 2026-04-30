"""
train_ensemble.py

Train two independent Isolation Forest models — one per source — and bundle
them into an ensemble artifact. No GSR/HRV window merging is performed.

Each sub-model learns the distribution of its own source's features only,
so there is no train-time vs inference-time distribution mismatch caused
by imputing a missing modality.

Usage:
    # Train on all sessions
    python train_ensemble.py sessions/

    # Train on sessions with a specific label
    python train_ensemble.py sessions/ --label baseline_classical

    # Train on sessions whose label starts with a prefix (e.g. baseline_*)
    python train_ensemble.py sessions/ --label-prefix baseline

    # Custom output directory (contamination/threshold can differ per source)
    python train_ensemble.py sessions/ \\
        --out models/ensemble \\
        --gsr-contamination 0.05 \\
        --hrv-contamination 0.02

Outputs to the --out directory:
    manifest.json        Bundle manifest — list of sub-models, combination mode
    gsr.pkl              GSR StandardScaler + IsolationForest pipeline
    gsr_meta.json        GSR metadata (features, threshold, training stats)
    gsr_report.png       GSR diagnostic plot
    hrv.pkl              HRV pipeline
    hrv_meta.json        HRV metadata
    hrv_report.png       HRV diagnostic plot
    ensemble_report.png  Combined figure showing both training distributions

Load with:
    detector = load_detector("models/ensemble", ensemble_mode="any")
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

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
# Feature columns — must match collect_training_data.py output
# ---------------------------------------------------------------------------

# GSR_FEATURES = [
#     ("gsr_scl_mean",     "SCL mean"),
#     ("gsr_scr_count",    "SCR/min"),
#     ("gsr_phasic_std",   "Phasic std"),
#     ("gsr_scr_mean_amp", "SCR amp"),
# ]

GSR_FEATURES = [
    ("gsr_scl_mean", "SCL Mean"),
    ("gsr_scr_count", "SCR Count/min"),
    ("gsr_scr_mean_amp", "SCR Mean Amp"),
    ("gsr_phasic_std", "Phasic Std"),

    ("gsr_scl_std", "SCL Std"),
    # ("gsr_scr_rise_time", "SCR Rise Time"),
    # ("gsr_scr_recovery_time", "SCR Recovery Time"),
    # ("gsr_phasic_mean", "Phasic Mean"),
]

# HRV_FEATURES = [
#     ("hrv_hr_mean", "HR mean"),
#     ("hrv_rmssd", "RMSSD"),
#     ("hrv_sdnn", "SDNN"),
#     ("hrv_pnn50", "pNN50"),
# ]

HRV_FEATURES = [
    ("hrv_hr_mean", "HR mean"),
    ("hrv_hr_std", "HR Std"),
    ("hrv_hr_min", "HR Min"),
    ("hrv_hr_max", "HR Max"),
    ("hrv_rmssd", "RMSSD"),
    ("hrv_sdnn", "SDNN"),
    ("hrv_pnn50", "PNN50"),
    ("hrv_mean_rr", "Mean RR"),
]

SOURCES = {
    "gsr": GSR_FEATURES,
    "hrv": HRV_FEATURES,
}


# ---------------------------------------------------------------------------
# Session discovery + loading
# ---------------------------------------------------------------------------


def find_sessions(
    parent: Path,
    label: Optional[str],
    label_prefix: Optional[str],
) -> list[Path]:
    """Discover session directories matching an optional label / label prefix."""
    if not parent.is_dir():
        sys.exit(f"[ERROR] Not a directory: {parent}")

    matches: list[Path] = []
    skipped: list[tuple[str, str]] = []

    for d in sorted(parent.iterdir()):
        if not d.is_dir() or not (d / "features.csv").exists():
            continue

        sess_label = _read_label(d)
        if sess_label is None:
            skipped.append((d.name, "no label found"))
            continue

        if label is not None and sess_label != label:
            skipped.append((d.name, f"label='{sess_label}' != '{label}'"))
            continue
        if label_prefix is not None and not sess_label.startswith(label_prefix):
            skipped.append(
                (d.name, f"label='{sess_label}' not under '{label_prefix}*'")
            )
            continue

        matches.append(d)

    print(f"[SCAN] Parent: {parent}")
    print(f"[SCAN] Matched {len(matches)} session(s):")
    for d in matches:
        print(f"         + {d.name}")
    if skipped:
        print(f"[SCAN] Skipped {len(skipped)} session(s):")
        for name, reason in skipped[:20]:
            print(f"         - {name}  ({reason})")
        if len(skipped) > 20:
            print(f"         ... and {len(skipped) - 20} more")

    return matches


def _read_label(session_dir: Path) -> Optional[str]:
    """Prefer session.json label, fall back to parsing the folder name."""
    meta_path = session_dir / "session.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                lbl = json.load(f).get("label")
                if lbl:
                    return lbl
        except (json.JSONDecodeError, OSError):
            pass

    parts = session_dir.name.split("_", 2)
    if len(parts) >= 3:
        return parts[2]
    return None


def load_features(session_dirs: list[Path]) -> pd.DataFrame:
    """Concatenate features.csv across sessions, tagged with __session__."""
    frames = []
    for d in session_dirs:
        df = pd.read_csv(d / "features.csv")
        df["__session__"] = d.name
        frames.append(df)
    if not frames:
        sys.exit("[ERROR] No features.csv loaded.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-source training
# ---------------------------------------------------------------------------


def build_source_matrix(
    df_all: pd.DataFrame,
    source: str,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """
    Extract rows where df['source'] == source, return (X, feature_cols,
    feature_names, session_ids).

    No merging, no imputation — every row is a genuine measurement.
    Rows with any NaN in this source's feature columns are dropped.
    """
    feature_defs = SOURCES[source]
    feature_cols = [c for c, _ in feature_defs]
    feature_names = [n for _, n in feature_defs]

    if "source" not in df_all.columns:
        sys.exit(f"[ERROR] features.csv missing 'source' column")

    rows = df_all[df_all["source"] == source].copy()
    if rows.empty:
        return (np.empty((0, 0)), feature_cols, feature_names, np.array([]))

    # Keep only feature columns that are actually present (graceful if older
    # CSVs are missing some)
    available = [c for c in feature_cols if c in rows.columns]
    missing = [c for c in feature_cols if c not in rows.columns]
    if missing:
        print(f"[{source.upper()}] Dropping columns not in data: {missing}")

    n_before = len(rows)
    rows = rows.dropna(subset=available).reset_index(drop=True)
    n_dropped = n_before - len(rows)
    if n_dropped:
        print(
            f"[{source.upper()}] Dropped {n_dropped} row(s) with NaN features "
            f"({n_before} -> {len(rows)})"
        )

    X = rows[available].values
    sessions = (
        rows["__session__"].values if "__session__" in rows.columns else np.array([])
    )

    # Preserve the order of the canonical definition
    sel_feature_names = [feature_names[feature_cols.index(c)] for c in available]
    return X, available, sel_feature_names, sessions


def train_pipeline(
    X: np.ndarray,
    contamination: float,
    n_estimators: int = 200,
) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    max_samples="auto",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X)
    return pipe


def compute_threshold(pipe: Pipeline, X: np.ndarray, percentile: float) -> float:
    scores = pipe.score_samples(X)
    return float(np.percentile(scores, percentile))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


BLUE, RED, GREEN = "#4C78A8", "#E45756", "#54A24B"


def plot_source_report(
    pipe: Pipeline,
    X: np.ndarray,
    feature_cols: list[str],
    feature_names: list[str],
    threshold: float,
    source: str,
    out_path: Path,
) -> None:
    scores = pipe.score_samples(X)
    is_inlier = scores >= threshold

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"{source.upper()} Isolation Forest — Trained on {X.shape[0]} windows",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Score distribution
    ax = axes[0]
    ax.hist(scores, bins=40, color=BLUE, alpha=0.8, edgecolor="white")
    ax.axvline(
        threshold, color=RED, ls="--", lw=2, label=f"Threshold = {threshold:.4f}"
    )
    ax.axvline(
        scores.mean(), color=GREEN, ls=":", lw=1.5, label=f"Mean = {scores.mean():.4f}"
    )
    ax.set_xlabel("Anomaly score (lower = more anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Training score distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Feature means: inliers vs outliers
    ax = axes[1]
    inlier_means = (
        X[is_inlier].mean(axis=0) if is_inlier.any() else np.zeros(X.shape[1])
    )
    outlier_means = (
        X[~is_inlier].mean(axis=0) if (~is_inlier).any() else np.zeros(X.shape[1])
    )
    x_pos = np.arange(len(feature_cols))
    width = 0.38
    ax.bar(x_pos - width / 2, inlier_means, width, color=BLUE, label="Inlier mean")
    ax.bar(x_pos + width / 2, outlier_means, width, color=RED, label="Outlier mean")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Raw feature value")
    ax.set_title("Feature means: inliers vs outliers")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Score trace
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
    print(f"[{source.upper()}] Report saved: {out_path}")


def plot_ensemble_report(
    per_source: dict,
    out_path: Path,
) -> None:
    """Side-by-side training score distributions for each sub-model."""
    fig, axes = plt.subplots(
        1, len(per_source), figsize=(7 * len(per_source), 5), squeeze=False
    )
    fig.suptitle(
        "Ensemble training — per-source score distributions",
        fontsize=13,
        fontweight="bold",
    )

    for i, (source, data) in enumerate(per_source.items()):
        ax = axes[0, i]
        scores = data["scores"]
        threshold = data["threshold"]
        ax.hist(scores, bins=40, color=BLUE, alpha=0.8, edgecolor="white")
        ax.axvline(
            threshold, color=RED, ls="--", lw=2, label=f"Threshold={threshold:.4f}"
        )
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{source.upper()}  "
            f"(n={len(scores)}, flagged={int((scores < threshold).sum())})"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ENSEMBLE] Combined report saved: {out_path}")


# ---------------------------------------------------------------------------
# Training a single source end-to-end
# ---------------------------------------------------------------------------


def train_one_source(
    df_all: pd.DataFrame,
    source: str,
    out_dir: Path,
    contamination: float,
    threshold_pct: float,
    n_estimators: int,
    session_list: list[str],
) -> Optional[dict]:
    """
    Train, threshold, plot and save one source model. Returns a summary dict
    suitable for the ensemble manifest, or None if there was no usable data.
    """
    print(f"\n[{source.upper()}] ── Training ──────────────────────────────")
    X, feature_cols, feature_names, _ = build_source_matrix(df_all, source)

    if X.shape[0] < 20:
        print(f"[{source.upper()}] Only {X.shape[0]} window(s) — skipping this source.")
        return None

    if X.shape[0] < 100:
        print(
            f"[{source.upper()}] WARN: only {X.shape[0]} windows — more data recommended."
        )

    pipe = train_pipeline(X, contamination=contamination, n_estimators=n_estimators)
    threshold = compute_threshold(pipe, X, percentile=threshold_pct)
    scores = pipe.score_samples(X)
    n_flagged = int((scores < threshold).sum())

    print(f"[{source.upper()}]   Windows       : {X.shape[0]}")
    print(f"[{source.upper()}]   Features      : {feature_cols}")
    print(f"[{source.upper()}]   Contamination : {contamination}")
    print(f"[{source.upper()}]   Threshold     : {threshold:.4f} (pct {threshold_pct})")
    print(
        f"[{source.upper()}]   Training flag : {n_flagged}/{X.shape[0]} "
        f"({n_flagged / X.shape[0]:.1%})"
    )
    print(
        f"[{source.upper()}]   Score range   : [{scores.min():.4f}, {scores.max():.4f}]"
    )

    # Save model
    model_path = out_dir / f"{source}.pkl"
    meta_path = out_dir / f"{source}_meta.json"
    report_path = out_dir / f"{source}_report.png"

    joblib.dump(pipe, model_path)
    meta = {
        "source": source,
        "feature_cols": feature_cols,
        "feature_names": feature_names,
        "threshold": threshold,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "n_training_windows": int(X.shape[0]),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "trained_at": datetime.now().isoformat(),
        "training_sessions": session_list,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    plot_source_report(
        pipe, X, feature_cols, feature_names, threshold, source, report_path
    )

    print(f"[{source.upper()}] Saved model:    {model_path}")
    print(f"[{source.upper()}] Saved metadata: {meta_path}")

    return {
        "source": source,
        "model_rel": f"{source}.pkl",
        "threshold": threshold,
        "scores": scores,
        "n_windows": int(X.shape[0]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "parent", type=Path, help="Parent directory containing session folders"
    )
    p.add_argument(
        "--label", default=None, help="Exact label filter (e.g. 'baseline_classical')"
    )
    p.add_argument(
        "--label-prefix",
        default=None,
        help="Label prefix filter (e.g. 'baseline' matches baseline_*)",
    )
    p.add_argument("--gsr-contamination", type=float, default=0.05)
    p.add_argument("--hrv-contamination", type=float, default=0.05)
    p.add_argument("--gsr-threshold-pct", type=float, default=5.0)
    p.add_argument("--hrv-threshold-pct", type=float, default=5.0)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument(
        "--combine-mode",
        choices=("any", "all", "mean"),
        default="any",
        help="Default ensemble combination mode written to manifest",
    )
    p.add_argument("--out", type=Path, default=Path("models/ensemble"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Discover sessions
    sessions = find_sessions(args.parent, args.label, args.label_prefix)
    if not sessions:
        filters = [
            f"label={args.label!r}" if args.label else "",
            f"label_prefix={args.label_prefix!r}" if args.label_prefix else "",
        ]
        sys.exit(
            f"[ERROR] No sessions matched ({', '.join(filter(None, filters)) or 'no filter'})"
        )

    if args.dry_run:
        print("\n[DRY-RUN] No training performed.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    df_all = load_features(sessions)
    print(f"\n[LOAD] Total raw rows across sessions: {len(df_all)}")
    session_list = [s.name for s in sessions]

    # Train each source independently
    per_source = {}
    for source, contamination, threshold_pct in (
        ("gsr", args.gsr_contamination, args.gsr_threshold_pct),
        ("hrv", args.hrv_contamination, args.hrv_threshold_pct),
    ):
        result = train_one_source(
            df_all=df_all,
            source=source,
            out_dir=args.out,
            contamination=contamination,
            threshold_pct=threshold_pct,
            n_estimators=args.n_estimators,
            session_list=session_list,
        )
        if result is not None:
            per_source[source] = result

    if not per_source:
        sys.exit(
            "[ERROR] No sub-models were trained (insufficient data for both sources)."
        )

    # Write manifest
    manifest = {
        "type": "ensemble",
        "combine_mode": args.combine_mode,
        "trained_at": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "training_sessions": session_list,
    }
    for source, data in per_source.items():
        manifest[f"{source}_model"] = data["model_rel"]

    manifest_path = args.out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[ENSEMBLE] Manifest saved: {manifest_path}")

    plot_ensemble_report(per_source, args.out / "ensemble_report.png")

    print(f"\n[DONE] Ensemble bundle: {args.out.resolve()}")
    print(f"       Load in code with:")
    print(f"           from arousal_detector import load_detector")
    print(
        f"           detector = load_detector('{args.out}', ensemble_mode='{args.combine_mode}')"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
