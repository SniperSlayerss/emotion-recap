"""
tune_threshold_ensemble.py

Sweep contamination values for the ensemble — train both sub-models at each
contamination and report per-session flag rate, split by source.

For each (contamination, session), this prints:
    gsr_flag_rate  : % of GSR windows in the session flagged by the GSR model
    hrv_flag_rate  : % of HRV windows in the session flagged by the HRV model
    any_rate       : % of all windows (either source) flagged under 'any' mode
    all_rate       : % of timepoints flagged under 'all' mode (both sources
                     agree within `window_pair_s` of each other)

A good contamination gives a wide gap between the mean baseline flag rate
and the mean non-baseline flag rate for a given combine mode.

Usage:
    python tune_threshold_ensemble.py sessions/ \\
        --baseline-prefix baseline \\
        --contaminations 0.005,0.01,0.02,0.05,0.1
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_ensemble import (
    SOURCES,
    find_sessions,
    _read_label,
    load_features,
    build_source_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def all_session_dirs(parent: Path) -> list[Path]:
    return sorted(
        d for d in parent.iterdir()
        if d.is_dir() and (d / "features.csv").exists()
    )


def train_pipeline(X: np.ndarray, contamination: float, n_estimators: int = 200) -> Pipeline:
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


def score_session_source(
    pipe: Pipeline,
    session_dir: Path,
    source: str,
    threshold: float,
    feature_cols: list[str],
) -> tuple[int, int]:
    """Return (n_flagged, n_total) for one source in one session."""
    csv_path = session_dir / "features.csv"
    df = pd.read_csv(csv_path)
    rows = df[df["source"] == source].copy()
    if rows.empty:
        return 0, 0

    available = [c for c in feature_cols if c in rows.columns]
    if len(available) < len(feature_cols):
        return 0, 0

    rows = rows.dropna(subset=available)
    if rows.empty:
        return 0, 0

    scores = pipe.score_samples(rows[available].values)
    return int((scores < threshold).sum()), len(rows)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    parent: Path,
    baseline_prefix: str,
    contaminations: list[float],
    threshold_pct: Optional[float],
    n_estimators: int,
) -> pd.DataFrame:
    baseline_dirs = find_sessions(parent, label=None, label_prefix=baseline_prefix)
    if not baseline_dirs:
        sys.exit(f"[ERROR] No baseline sessions found with prefix '{baseline_prefix}'")

    df_train = load_features(baseline_dirs)

    # Build one training matrix per source up front
    train_matrices = {}
    for source in SOURCES:
        X, feature_cols, _names, _ = build_source_matrix(df_train, source)
        if X.shape[0] < 20:
            print(f"[SWEEP] Skipping {source}: too few training windows ({X.shape[0]})")
            continue
        train_matrices[source] = {"X": X, "feature_cols": feature_cols}
        print(f"[SWEEP] {source.upper()} training: {X.shape[0]} windows × {X.shape[1]} features")

    if not train_matrices:
        sys.exit("[ERROR] No source had enough baseline data to train.")

    # Score every session against each (contamination, source) combo
    all_dirs = all_session_dirs(parent)
    rows = []

    for c in contaminations:
        pct = threshold_pct if threshold_pct is not None else c * 100
        print(f"\n[SWEEP] contamination={c:.4f}, threshold_pct={pct:.2f}")

        for source, data in train_matrices.items():
            pipe = train_pipeline(data["X"], contamination=c, n_estimators=n_estimators)
            threshold = float(np.percentile(pipe.score_samples(data["X"]), pct))

            for d in all_dirs:
                label = _read_label(d) or "?"
                is_base = label.startswith(baseline_prefix)
                n_flag, n_tot = score_session_source(
                    pipe, d, source, threshold, data["feature_cols"],
                )
                rate = n_flag / n_tot if n_tot else 0.0
                rows.append({
                    "contamination": c,
                    "source":        source,
                    "threshold":     threshold,
                    "session":       d.name,
                    "label":         label,
                    "is_baseline":   is_base,
                    "n_flagged":     n_flag,
                    "n_total":       n_tot,
                    "flag_rate":     rate,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(df: pd.DataFrame) -> None:
    sessions = sorted(df["session"].unique())
    contaminations = sorted(df["contamination"].unique())
    sources = sorted(df["source"].unique())

    for source in sources:
        sub = df[df["source"] == source]
        print("\n" + "=" * 110)
        print(f"{source.upper()} FLAG RATE PER SESSION (%)")
        print("=" * 110)

        header = f"{'session':<44} {'label':<22} {'base?':<6} "
        header += " ".join(f"c={c:<7.3f}" for c in contaminations)
        print(header)
        print("-" * len(header))

        for s in sessions:
            sess = sub[sub["session"] == s]
            if sess.empty:
                continue
            label = sess["label"].iloc[0]
            is_base = sess["is_baseline"].iloc[0]
            line = f"{s[:44]:<44} {label[:22]:<22} {'yes' if is_base else 'no':<6} "
            for c in contaminations:
                r = sess[sess["contamination"] == c]["flag_rate"]
                line += f"{r.iloc[0] * 100:>7.1f} " if not r.empty else "    --- "
            print(line)

        # Mean / gap row
        print("-" * len(header))
        base_means    = [sub[(sub.contamination == c) & sub.is_baseline]["flag_rate"].mean() * 100
                         for c in contaminations]
        nonbase_means = [sub[(sub.contamination == c) & ~sub.is_baseline]["flag_rate"].mean() * 100
                         for c in contaminations]
        gaps = [n - b for b, n in zip(base_means, nonbase_means)]

        print(f"{'mean BASELINE':<44} {'':<22} {'':<6} " +
              "".join(f"{v:>7.1f} " for v in base_means))
        print(f"{'mean NON-BASELINE':<44} {'':<22} {'':<6} " +
              "".join(f"{v:>7.1f} " for v in nonbase_means))
        print(f"{'GAP (non − base)':<44} {'':<22} {'':<6} " +
              "".join(f"{v:>+7.1f} " for v in gaps))
        print("=" * 110)

    # Recommend per source
    MAX_BASELINE_RATE = 10.0
    MIN_GAP           = 10.0
    print("\n[RECOMMEND]")
    for source in sources:
        sub = df[df["source"] == source]
        base_means    = [sub[(sub.contamination == c) & sub.is_baseline]["flag_rate"].mean() * 100
                         for c in contaminations]
        nonbase_means = [sub[(sub.contamination == c) & ~sub.is_baseline]["flag_rate"].mean() * 100
                         for c in contaminations]
        gaps = [n - b for b, n in zip(base_means, nonbase_means)]

        valid = [(c, b, n, g) for c, b, n, g
                 in zip(contaminations, base_means, nonbase_means, gaps)
                 if b <= MAX_BASELINE_RATE]

        if not valid:
            print(f"    {source.upper()}: no contamination keeps baseline below "
                  f"{MAX_BASELINE_RATE}%. Try cleaner baselines or fewer features.")
            continue

        best_c, best_b, best_n, best_g = max(valid, key=lambda r: r[3])
        note = "" if best_g >= MIN_GAP else "  (gap small — features may not discriminate)"
        print(f"    {source.upper()}: best c={best_c} "
              f"(baseline {best_b:.1f}%, non-baseline {best_n:.1f}%, "
              f"gap +{best_g:.1f} pts){note}")


def plot_sweep(df: pd.DataFrame, out_path: Path) -> None:
    contaminations = sorted(df["contamination"].unique())
    sources = sorted(df["source"].unique())

    fig, axes = plt.subplots(1, len(sources), figsize=(7 * len(sources), 6),
                             squeeze=False)
    fig.suptitle("Ensemble contamination sweep — per source",
                 fontsize=13, fontweight="bold")

    BASE, OTHER = "#4C78A8", "#E45756"

    for i, source in enumerate(sources):
        ax = axes[0, i]
        sub = df[df["source"] == source]
        base_means  = [sub[(sub.contamination == c) & sub.is_baseline]["flag_rate"].mean() * 100
                       for c in contaminations]
        other_means = [sub[(sub.contamination == c) & ~sub.is_baseline]["flag_rate"].mean() * 100
                       for c in contaminations]
        gaps = [o - b for b, o in zip(base_means, other_means)]

        ax.plot(contaminations, base_means,  "o-", color=BASE,  lw=2, ms=7, label="Baseline mean")
        ax.plot(contaminations, other_means, "o-", color=OTHER, lw=2, ms=7, label="Non-baseline mean")
        ax.fill_between(contaminations, base_means, other_means,
                        alpha=0.15, color="green", label="Gap (bigger = better)")
        if gaps:
            best_idx = int(np.argmax(gaps))
            ax.axvline(contaminations[best_idx], color="green", ls="--", lw=1,
                       label=f"Best gap @ c={contaminations[best_idx]}")
        ax.set_xscale("log")
        ax.set_xlabel("Contamination")
        ax.set_ylabel("Mean flag rate (%)")
        ax.set_title(f"{source.upper()}")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SWEEP] Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("parent", type=Path)
    p.add_argument("--baseline-prefix", default="baseline")
    p.add_argument("--contaminations", type=str,
                   default="0.005,0.01,0.02,0.05,0.1,0.15,0.2")
    p.add_argument("--threshold-pct", type=float, default=None,
                   help="Fixed threshold percentile (default: scales with contamination)")
    p.add_argument("--n-estimators", type=int, default=200)
    args = p.parse_args()

    try:
        contaminations = [float(c.strip()) for c in args.contaminations.split(",")]
    except ValueError:
        sys.exit("[ERROR] --contaminations must be comma-separated floats")

    df = run_sweep(
        parent=args.parent,
        baseline_prefix=args.baseline_prefix,
        contaminations=contaminations,
        threshold_pct=args.threshold_pct,
        n_estimators=args.n_estimators,
    )

    print_table(df)

    plot_sweep(df, args.parent / "tune_sweep_ensemble.png")

    csv_path = args.parent / "tune_sweep_ensemble.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SWEEP] Per-session data: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
