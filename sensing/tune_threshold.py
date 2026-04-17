"""
tune_threshold.py

Sweep contamination values, retrain on baseline sessions, and report the
flag rate per session (baseline AND non-baseline) for each value.

The goal is to find a contamination that:
  - Flags ~0–10% of BASELINE sessions (low false positive rate)
  - Flags significantly more of your non-baseline sessions (e.g. horror)

A big gap between the two columns = a discriminating model.
A small gap = the model can't tell them apart (more data / better features
needed).

Usage:
    python tune_threshold.py sessions/ [options]

    --baseline-prefix  Label prefix for training sessions (default: 'baseline')
    --contaminations   Comma-separated list (default: 0.005,0.01,0.02,0.05,0.1)
    --threshold-pct    Percentile for threshold (default: matches contamination)
    --save-best        If set, save the model with the best gap to --out
    --out              Where to save best model (default: models/iforest.pkl)
"""

import argparse
import json
import sys
import warnings
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

# Reuse helpers from the retrain script
from retrain_baseline import (
    ALL_FEATURE_COLS, ALL_FEATURE_NAMES,
    find_baseline_sessions, _read_label,
    load_features, build_feature_matrix, validate_features,
)

from session_scoring import merge_gsr_hrv, score_merged_row, DEFAULT_MERGE_TOLERANCE_S

warnings.filterwarnings("ignore", category=UserWarning)


def all_sessions(parent: Path) -> list[Path]:
    return sorted(
        d for d in parent.iterdir()
        if d.is_dir() and (d / "features.csv").exists()
    )


def train_model(X: np.ndarray, contamination: float, n_estimators: int = 200) -> Pipeline:
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


def score_session(pipe: Pipeline, session_dir: Path, feature_cols: list[str],
                  threshold: float) -> tuple[int, int]:
    """
    Return (n_flagged, n_total) for this session using temporal GSR+HRV
    merging — matches what detect_from_session.py does at inference time.
    """
    df = pd.read_csv(session_dir / "features.csv")
    merged = merge_gsr_hrv(df, tolerance_s=DEFAULT_MERGE_TOLERANCE_S)

    if merged.empty:
        return 0, 0

    # Build score matrix from fully-merged rows only (skip rows with any NaN
    # in the feature columns — these are GSR rows with no HRV match)
    usable = merged.dropna(subset=feature_cols)
    if usable.empty:
        return 0, 0

    X = usable[feature_cols].values
    scores = pipe.score_samples(X)
    n_flagged = int((scores < threshold).sum())
    return n_flagged, len(usable)


def run_sweep(
    parent: Path,
    baseline_prefix: str,
    contaminations: list[float],
    threshold_pct: float | None,
) -> tuple[pd.DataFrame, dict]:
    """
    Train one model per contamination value; score every session against each.
    Returns:
        df_results  — long-form DataFrame: contamination, session, label,
                      is_baseline, n_flagged, n_total, flag_rate
        artefacts   — dict mapping contamination → (pipe, threshold)
    """
    print(f"\n[SWEEP] Discovering sessions under {parent}")

    # ---- Build training matrix once (from baseline sessions) ----
    baseline_dirs = find_baseline_sessions(parent, baseline_prefix)
    if not baseline_dirs:
        sys.exit(f"[ERROR] No baseline sessions found with prefix '{baseline_prefix}'")

    df_raw   = load_features(baseline_dirs)
    df_merged = build_feature_matrix(df_raw)
    df_clean  = validate_features(df_merged)

    feature_cols  = [c for c in ALL_FEATURE_COLS  if c in df_clean.columns]
    feature_names = [ALL_FEATURE_NAMES[i]
                     for i, c in enumerate(ALL_FEATURE_COLS) if c in df_clean.columns]
    X = df_clean[feature_cols].values
    print(f"[SWEEP] Training matrix: {X.shape[0]} windows × {X.shape[1]} features\n")

    # ---- All sessions to score against each model ----
    all_dirs = all_sessions(parent)
    all_meta = []
    for d in all_dirs:
        lbl = _read_label(d) or "?"
        all_meta.append((d, lbl, lbl.startswith(baseline_prefix)))

    # ---- Sweep ----
    rows = []
    artefacts: dict[float, tuple[Pipeline, float]] = {}

    for c in contaminations:
        pct = threshold_pct if threshold_pct is not None else c * 100
        print(f"[SWEEP] Training with contamination={c:.4f}, threshold_pct={pct:.2f}")
        pipe = train_model(X, contamination=c)
        scores = pipe.score_samples(X)
        threshold = float(np.percentile(scores, pct))
        artefacts[c] = (pipe, threshold)

        for d, label, is_baseline in all_meta:
            n_flag, n_tot = score_session(pipe, d, feature_cols, threshold)
            rate = n_flag / n_tot if n_tot else 0.0
            rows.append({
                "contamination": c,
                "threshold":     threshold,
                "session":       d.name,
                "label":         label,
                "is_baseline":   is_baseline,
                "n_flagged":     n_flag,
                "n_total":       n_tot,
                "flag_rate":     rate,
            })

    return pd.DataFrame(rows), artefacts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(df: pd.DataFrame) -> None:
    sessions = df["session"].unique().tolist()
    contaminations = sorted(df["contamination"].unique())

    # Header
    print("\n" + "=" * 110)
    print("FLAG RATE PER SESSION (%)")
    print("=" * 110)
    header = f"{'session':<44} {'label':<22} {'base?':<6} "
    header += " ".join(f"c={c:<7.3f}" for c in contaminations)
    print(header)
    print("-" * len(header))

    for s in sessions:
        sub = df[df["session"] == s]
        label = sub["label"].iloc[0]
        is_base = sub["is_baseline"].iloc[0]
        row = f"{s[:44]:<44} {label[:22]:<22} {'yes' if is_base else 'no':<6} "
        rates = []
        for c in contaminations:
            r = sub[sub["contamination"] == c]["flag_rate"]
            rates.append(f"{r.iloc[0]*100:>7.1f} " if not r.empty else "    --- ")
        row += "".join(rates)
        print(row)

    # Summary row: mean baseline vs mean non-baseline at each contamination
    print("-" * len(header))
    base_means = [
        df[(df.contamination == c) & df.is_baseline]["flag_rate"].mean() * 100
        for c in contaminations
    ]
    nonbase_means = [
        df[(df.contamination == c) & ~df.is_baseline]["flag_rate"].mean() * 100
        for c in contaminations
    ]
    gaps = [n - b for n, b in zip(nonbase_means, base_means)]

    print(f"{'mean BASELINE flag rate':<44} {'':<22} {'':<6} " +
          "".join(f"{v:>7.1f} " for v in base_means))
    print(f"{'mean NON-BASELINE flag rate':<44} {'':<22} {'':<6} " +
          "".join(f"{v:>7.1f} " for v in nonbase_means))
    print(f"{'GAP (non-baseline − baseline)':<44} {'':<22} {'':<6} " +
          "".join(f"{v:>+7.1f} " for v in gaps))
    print("=" * 110)

    # Recommend — pick the contamination that maximises gap while keeping
    # baseline flag rate below a sensible ceiling. A huge gap is useless if
    # baselines are also flagging heavily.
    MAX_BASELINE_RATE = 10.0  # percent — don't recommend configs above this
    MIN_GAP           = 10.0  # percent — below this, features aren't separating

    valid = [
        (c, b, n, g) for c, b, n, g
        in zip(contaminations, base_means, nonbase_means, gaps)
        if b <= MAX_BASELINE_RATE
    ]

    if not valid:
        print(f"\n[RECOMMEND] No contamination keeps baseline flag rate below {MAX_BASELINE_RATE}%.")
        print(f"            Consider cleaner baselines, or retrain on a narrower subset.")
    else:
        best_c, best_b, best_n, best_g = max(valid, key=lambda x: x[3])
        print(f"\n[RECOMMEND] Best config: contamination={best_c} "
              f"(baseline {best_b:.1f}%, non-baseline {best_n:.1f}%, gap +{best_g:.1f} pts)")
        if best_g < MIN_GAP:
            print("[RECOMMEND] Gap is small — features may not discriminate well.")
        print(f"[RECOMMEND] Retrain with: "
              f"python retrain_baseline.py <parent> --contamination {best_c} "
              f"--threshold-pct {best_c * 100:.2f}")


def plot_sweep(df: pd.DataFrame, out_path: Path) -> None:
    contaminations = sorted(df["contamination"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Contamination Sweep — Flag Rate by Session Type",
                 fontsize=13, fontweight="bold")

    BASE, OTHER = "#4C78A8", "#E45756"

    # (a) Per-session lines
    ax = axes[0]
    for session, sub in df.groupby("session"):
        sub = sub.sort_values("contamination")
        is_base = sub["is_baseline"].iloc[0]
        colour = BASE if is_base else OTHER
        label = f"{sub['label'].iloc[0]}" + (" (base)" if is_base else "")
        ax.plot(sub["contamination"], sub["flag_rate"] * 100, "o-",
                color=colour, alpha=0.5, lw=1.2, ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("Contamination")
    ax.set_ylabel("Flag rate (%)")
    ax.set_title("Per-session flag rates\n(blue = baseline, red = other)")
    ax.grid(True, alpha=0.3)

    # (b) Mean baseline vs non-baseline + gap
    base_means = [df[(df.contamination == c) & df.is_baseline]["flag_rate"].mean() * 100
                  for c in contaminations]
    other_means = [df[(df.contamination == c) & ~df.is_baseline]["flag_rate"].mean() * 100
                   for c in contaminations]
    gaps = [o - b for o, b in zip(other_means, base_means)]

    ax = axes[1]
    ax.plot(contaminations, base_means,  "o-", color=BASE,  lw=2, ms=7, label="Baseline mean")
    ax.plot(contaminations, other_means, "o-", color=OTHER, lw=2, ms=7, label="Non-baseline mean")
    ax.fill_between(contaminations, base_means, other_means,
                    alpha=0.15, color="green", label="Gap (bigger = better)")
    best_idx = int(np.argmax(gaps))
    ax.axvline(contaminations[best_idx], color="green", ls="--", lw=1,
               label=f"Best gap @ c={contaminations[best_idx]}")
    ax.set_xscale("log")
    ax.set_xlabel("Contamination")
    ax.set_ylabel("Mean flag rate (%)")
    ax.set_title("Mean flag rates — you want blue low, red high")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SWEEP] Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--baseline-prefix", default="baseline")
    parser.add_argument("--contaminations", type=str, default="0.005,0.01,0.02,0.05,0.1,0.12,0.15,0.2,0.25",
                        help="Comma-separated list of contamination values")
    parser.add_argument("--threshold-pct", type=float, default=None,
                        help="Fixed threshold percentile (default: scales with contamination)")
    parser.add_argument("--save-best", action="store_true",
                        help="Save the model with the best baseline/non-baseline gap")
    parser.add_argument("--out", type=Path, default=Path("models/iforest.pkl"))
    args = parser.parse_args()

    try:
        contaminations = [float(c.strip()) for c in args.contaminations.split(",")]
    except ValueError:
        sys.exit("[ERROR] --contaminations must be comma-separated floats")

    df, artefacts = run_sweep(
        args.parent,
        args.baseline_prefix,
        contaminations,
        args.threshold_pct,
    )

    print_table(df)

    # Save plot
    plot_path = args.parent / "tune_sweep.png"
    plot_sweep(df, plot_path)

    # Save CSV for reference
    csv_path = args.parent / "tune_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SWEEP] Per-session data: {csv_path}")

    # Optionally save best model
    if args.save_best:
        base_means = [df[(df.contamination == c) & df.is_baseline]["flag_rate"].mean()
                      for c in contaminations]
        other_means = [df[(df.contamination == c) & ~df.is_baseline]["flag_rate"].mean()
                       for c in contaminations]
        gaps = [o - b for o, b in zip(other_means, base_means)]
        best_c = contaminations[int(np.argmax(gaps))]
        pipe, threshold = artefacts[best_c]

        args.out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, args.out)
        print(f"\n[SAVE] Best model saved: {args.out}  (contamination={best_c})")
        print(f"[SAVE] Run retrain_baseline.py to generate proper _meta.json / _report.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
