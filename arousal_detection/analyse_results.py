"""
analyse_results.py

End-to-end analysis for the arousal-detection results chapter.

Inputs:
    sessions/      Parent directory of recorded sessions. Each session is
                   labelled by prefix (default: 'baseline' = negative class,
                   anything else = positive class — but see --positive-prefix).
    --model        Path to a trained detector (ensemble bundle or .pkl).

Outputs to <out>/ :
    figures/
        02_feature_violins.png         baseline characterisation (§5.2)
        02_pca_projection.png          2D feature-space projection (§5.2)
        02_loso_shifts.png             leave-one-session-out bar chart (§5.2)
        03_score_distributions.png     HEADLINE: baseline vs aroused overlay (§5.3)
        03_roc_curve.png               ROC with AUC (§5.3)
        03_pr_curve.png                PR with AUC (§5.3)
        03_ablation_roc.png            GSR-only / HRV-only / combined (§5.3)
        03_threshold_sweep.png         precision / recall vs threshold (§5.3)
        03_per_session_breakdown.png   per-session flag-rate / score bars (§5.3)
        03_hero_timeseries.png         score over time for best aroused session (§5.3)
        04_latency_hist.png            score() call latency (§5.4)

    tables/
        02_data_summary.csv            per-session row counts, data-quality stats
        02_loso_shifts.csv
        03_metrics.csv                 AUC / PR-AUC / precision@recall / latency
        03_ablation_metrics.csv        AUC etc. for GSR-only / HRV-only / combined
        03_per_session.csv             per-session: flag rate, score stats
        03_confusion_at_operating_pt.csv

    scored_windows.csv                 every window with label + score (for reproducibility)
    summary.json                       headline metrics + bullet points (§5.6)
    report.md                          human-readable summary that embeds the figures

Usage:
    python analyse_results.py sessions/ --model models/ensemble
    python analyse_results.py sessions/ --model models/iforest.pkl
    python analyse_results.py sessions/ --model models/ensemble \\
        --baseline-prefix baseline --out results/run_2026_04

By default every session whose label does NOT start with --baseline-prefix is
treated as positive (aroused). Use --positive-prefix to restrict that further
(e.g. 'horror' to exclude 'unknown' / test sessions from the positive class).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from arousal_detector import (
    ArousalDetector,
    EnsembleDetector,
    load_detector,
)
from session_scoring import (
    score_session_ensemble,
    score_session_file,
    combined_timeline,
)
from train_ensemble import (
    GSR_FEATURES,
    HRV_FEATURES,
    SOURCES,
    _read_label,
    build_source_matrix,
    load_features,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BLUE   = "#4C78A8"
RED    = "#E45756"
GREEN  = "#54A24B"
ORANGE = "#F58518"
GREY   = "#8C8C8C"
PURPLE = "#B279A2"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """One session on disk, with its ground-truth label."""
    dir:      Path
    label:    str
    is_aroused: bool  # ground truth at session level

    @property
    def name(self) -> str:
        return self.dir.name


def discover_sessions(
    parent: Path,
    baseline_prefix: str,
    positive_prefix: Optional[str],
) -> list[Session]:
    """Walk the parent dir, classify each session by its label prefix."""
    if not parent.is_dir():
        sys.exit(f"[ERROR] Not a directory: {parent}")

    sessions: list[Session] = []
    for d in sorted(parent.iterdir()):
        if not d.is_dir() or not (d / "features.csv").exists():
            continue
        label = _read_label(d)
        if label is None:
            print(f"[SCAN] Skipping {d.name} (no label)")
            continue

        if label.startswith(baseline_prefix):
            is_aroused = False
        elif positive_prefix is not None:
            if not label.startswith(positive_prefix):
                print(f"[SCAN] Skipping {d.name} (label='{label}' is neither "
                      f"'{baseline_prefix}*' nor '{positive_prefix}*')")
                continue
            is_aroused = True
        else:
            is_aroused = True

        sessions.append(Session(dir=d, label=label, is_aroused=is_aroused))

    n_base = sum(1 for s in sessions if not s.is_aroused)
    n_pos  = sum(1 for s in sessions if s.is_aroused)
    print(f"[SCAN] {len(sessions)} sessions: {n_base} baseline, {n_pos} aroused")
    return sessions


# ---------------------------------------------------------------------------
# Scoring all sessions under the main detector
# ---------------------------------------------------------------------------


def score_all_sessions(
    sessions: list[Session],
    detector,
    combine_mode: str,
) -> pd.DataFrame:
    """
    Score every window of every session with the supplied detector.

    Returns a long-form DataFrame with columns:
        session, label, y_true (0|1), source, time_s,
        score, normalised, is_aroused (the detector's own flag),
        scored
    """
    frames = []

    is_ensemble = isinstance(detector, EnsembleDetector)

    for s in sessions:
        csv_path = s.dir / "features.csv"
        if is_ensemble:
            scored = score_session_ensemble(csv_path, detector)
            # Apply the configured combine mode so 'is_aroused' reflects
            # the user-chosen decision rule (important for 'all' / 'mean')
            if combine_mode in ("all", "mean"):
                scored = combined_timeline(
                    scored,
                    mode=combine_mode,
                    pair_tolerance_s=30.0,
                    mean_threshold=detector.mean_threshold,
                )
                # combined_timeline adds 'paired_with' — harmless extra col
        else:
            # Single-source (possibly old merged) model
            if (any(c.startswith("gsr_") for c in detector.feature_cols)
                    and any(c.startswith("hrv_") for c in detector.feature_cols)):
                scored = score_session_file(csv_path, detector)
                # Harmonise columns — merge-based returns 'time_s' already
                scored["source"] = "merged"
            else:
                # Per-source single model — use the ensemble-style helper path
                raw = pd.read_csv(csv_path)
                from session_scoring import _score_source_rows
                src = "gsr" if any(c.startswith("gsr_") for c in detector.feature_cols) else "hrv"
                scored = _score_source_rows(raw, src, detector)

        scored = scored.copy()
        scored["session"]   = s.name
        scored["label"]     = s.label
        scored["y_true"]    = int(s.is_aroused)
        frames.append(scored)

    if not frames:
        sys.exit("[ERROR] No sessions produced scored windows.")

    out = pd.concat(frames, ignore_index=True)
    # Drop unscored rows — they can't contribute to metrics
    before = len(out)
    out = out[out["scored"]].reset_index(drop=True)
    print(f"[SCORE] {len(out)} scored windows "
          f"({before - len(out)} unscored dropped)")
    return out


# ---------------------------------------------------------------------------
# §5.2  Baseline characterisation
# ---------------------------------------------------------------------------


ALL_FEATURE_COLS = [c for c, _ in GSR_FEATURES + HRV_FEATURES]
ALL_FEATURE_NAMES = [n for _, n in GSR_FEATURES + HRV_FEATURES]


def plot_feature_violins(
    sessions: list[Session],
    out_path: Path,
) -> None:
    """One violin per feature, split by session, coloured by class."""
    frames = []
    for s in sessions:
        df = pd.read_csv(s.dir / "features.csv")
        df["__session"] = s.name
        df["__label"]   = s.label
        df["__y"]       = int(s.is_aroused)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    available = [c for c in ALL_FEATURE_COLS if c in all_df.columns]
    available_names = [ALL_FEATURE_NAMES[ALL_FEATURE_COLS.index(c)]
                       for c in available]

    n = len(available)
    fig, axes = plt.subplots((n + 3) // 4, 4, figsize=(16, 3.5 * ((n + 3) // 4)))
    axes = axes.flatten()
    fig.suptitle("Baseline characterisation — feature distributions per session",
                 fontsize=13, fontweight="bold")

    session_names = [s.name for s in sessions]
    session_y     = [int(s.is_aroused) for s in sessions]
    colours = [RED if y else BLUE for y in session_y]

    for i, (col, name) in enumerate(zip(available, available_names)):
        ax = axes[i]
        data = []
        for sess_name in session_names:
            vals = all_df[all_df["__session"] == sess_name][col].dropna().values
            data.append(vals)

        # Violins with per-session colour
        parts = ax.violinplot(
            [d if len(d) else np.array([0.0]) for d in data],
            showmeans=True, showmedians=False,
        )
        for pc, c in zip(parts["bodies"], colours):
            pc.set_facecolor(c)
            pc.set_alpha(0.55)

        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(1, len(session_names) + 1))
        short_labels = [nm[-18:] for nm in session_names]
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    # Hide leftover axes
    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    # Legend
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color=BLUE, label="baseline"),
                 Patch(color=RED,  label="aroused")],
        loc="lower center", ncol=2, fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] Feature violins → {out_path.name}")


def plot_pca_projection(
    sessions: list[Session],
    out_path: Path,
) -> None:
    """2D PCA of the feature space, coloured by session label class."""
    # Collect per-source matrices, then plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Baseline characterisation — 2D PCA projection per source",
                 fontsize=13, fontweight="bold")

    for ax, source in zip(axes, ["gsr", "hrv"]):
        frames = []
        for s in sessions:
            df = pd.read_csv(s.dir / "features.csv")
            rows = df[df["source"] == source].copy()
            if rows.empty:
                continue
            rows["__session"] = s.name
            rows["__y"]       = int(s.is_aroused)
            frames.append(rows)
        if not frames:
            ax.text(0.5, 0.5, f"No {source.upper()} data",
                    ha="center", transform=ax.transAxes)
            continue

        d = pd.concat(frames, ignore_index=True)
        feature_cols = [c for c, _ in SOURCES[source]]
        available = [c for c in feature_cols if c in d.columns]
        d = d.dropna(subset=available)
        if d.empty:
            continue

        X = StandardScaler().fit_transform(d[available].values)
        pcs = PCA(n_components=2).fit_transform(X)

        for is_pos, colour, marker in [(0, BLUE, "o"), (1, RED, "^")]:
            mask = d["__y"].values == is_pos
            if mask.any():
                ax.scatter(pcs[mask, 0], pcs[mask, 1],
                           c=colour, marker=marker, s=36, alpha=0.55,
                           edgecolors="white", linewidths=0.5,
                           label="baseline" if is_pos == 0 else "aroused")

        ax.set_title(f"{source.upper()} feature space")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] PCA projection → {out_path.name}")


def compute_loso_shifts(
    baseline_sessions: list[Session],
    n_estimators: int = 200,
) -> pd.DataFrame:
    """
    For each baseline session, retrain on the remaining baseline sessions
    and compute the held-out mean score shift in sigma units.
    Runs per-source (gsr, hrv) independently.
    """
    if len(baseline_sessions) < 2:
        return pd.DataFrame()

    rows = []
    frames = []
    for s in baseline_sessions:
        df = pd.read_csv(s.dir / "features.csv")
        df["__session"] = s.name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    for source in ("gsr", "hrv"):
        src_df = all_df[all_df["source"] == source].copy()
        if src_df.empty:
            continue
        feature_cols = [c for c, _ in SOURCES[source] if c in src_df.columns]
        src_df = src_df.dropna(subset=feature_cols)
        if src_df.empty:
            continue

        # Reference distribution: train on all baseline sessions
        X_all = src_df[feature_cols].values
        ref_pipe = _fit_iforest(X_all, 0.05, n_estimators)
        ref_scores = ref_pipe.score_samples(X_all)
        ref_mean = float(ref_scores.mean())
        ref_std  = float(ref_scores.std()) or 1e-9

        for held_out in baseline_sessions:
            train_mask = src_df["__session"] != held_out.name
            test_mask  = src_df["__session"] == held_out.name
            if train_mask.sum() < 10 or test_mask.sum() < 1:
                continue

            pipe = _fit_iforest(
                src_df.loc[train_mask, feature_cols].values,
                0.05, n_estimators,
            )
            held_scores = pipe.score_samples(
                src_df.loc[test_mask, feature_cols].values
            )

            rows.append({
                "source":   source,
                "session":  held_out.name,
                "label":    held_out.label,
                "n_windows": int(test_mask.sum()),
                "heldout_mean":  float(held_scores.mean()),
                "heldout_std":   float(held_scores.std()),
                "shift_sigma":   float((held_scores.mean() - ref_mean) / ref_std),
            })

    return pd.DataFrame(rows)


def plot_loso_shifts(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print("[02] LOSO skipped (need ≥2 baseline sessions)")
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.35)))
    fig.suptitle("Leave-one-session-out score shift (baseline sessions only)",
                 fontsize=13, fontweight="bold")

    df = df.sort_values(["source", "shift_sigma"]).reset_index(drop=True)
    y_pos = np.arange(len(df))
    colours = [BLUE if s == "gsr" else ORANGE for s in df["source"]]

    ax.barh(y_pos, df["shift_sigma"], color=colours, alpha=0.8, edgecolor="white")
    labels = [f"{r.session[-28:]}  [{r.source}]" for r in df.itertuples()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0,  color="black", lw=0.8)
    ax.axvline(-1, color=RED, ls="--", lw=1, alpha=0.6, label="−1σ")
    ax.axvline( 1, color=RED, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Shift in mean score when held out (σ units) — "
                  "|shift| > 1σ = distinct from the rest")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=BLUE,   label="GSR"),
        Patch(color=ORANGE, label="HRV"),
    ], fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] LOSO shifts → {out_path.name}")


def compute_data_summary(sessions: list[Session]) -> pd.DataFrame:
    rows = []
    for s in sessions:
        df = pd.read_csv(s.dir / "features.csv")
        n_gsr = int((df["source"] == "gsr").sum())
        n_hrv = int((df["source"] == "hrv").sum())

        gsr_cols = [c for c, _ in GSR_FEATURES if c in df.columns]
        hrv_cols = [c for c, _ in HRV_FEATURES if c in df.columns]
        gsr_nan = int(df.loc[df["source"] == "gsr", gsr_cols].isna().any(axis=1).sum()) if gsr_cols else 0
        hrv_nan = int(df.loc[df["source"] == "hrv", hrv_cols].isna().any(axis=1).sum()) if hrv_cols else 0

        duration_s = float(df["window_start_time"].max()) if not df.empty else 0.0

        # Try to read session.json for duration/meta
        meta_path = s.dir / "session.json"
        meta_dur = None
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta_dur = float(json.load(f).get("duration_s", 0)) or None
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        rows.append({
            "session":     s.name,
            "label":       s.label,
            "y_true":      int(s.is_aroused),
            "n_gsr_windows": n_gsr,
            "n_hrv_windows": n_hrv,
            "gsr_rows_with_nan": gsr_nan,
            "hrv_rows_with_nan": hrv_nan,
            "duration_s":  meta_dur if meta_dur is not None else duration_s,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §5.3  Arousal detection performance
# ---------------------------------------------------------------------------


def plot_score_distributions(scored: pd.DataFrame, out_path: Path) -> None:
    """THE headline plot — baseline score distribution vs aroused overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))

    base = scored[scored["y_true"] == 0]["normalised"].values
    pos  = scored[scored["y_true"] == 1]["normalised"].values

    bins = np.linspace(0, 1, 40)
    ax.hist(base, bins=bins, alpha=0.65, color=BLUE, edgecolor="white",
            label=f"Baseline (n={len(base)})", density=True)
    ax.hist(pos,  bins=bins, alpha=0.65, color=RED,  edgecolor="white",
            label=f"Aroused (n={len(pos)})",  density=True)

    ax.axvline(np.mean(base), color=BLUE, ls="--", lw=1.5,
               label=f"Baseline mean ({np.mean(base):.2f})")
    ax.axvline(np.mean(pos),  color=RED,  ls="--", lw=1.5,
               label=f"Aroused mean ({np.mean(pos):.2f})")

    ax.set_xlabel("Normalised arousal score (0 = deep baseline, 1 = max)")
    ax.set_ylabel("Density")
    ax.set_title("Score distributions — baseline vs aroused sessions "
                 "(HEADLINE RESULT)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Score distributions → {out_path.name}")


def compute_roc_pr(scored: pd.DataFrame) -> dict:
    """Compute ROC/PR curves, AUCs, and precision@recall metrics."""
    y = scored["y_true"].values
    # Higher score should correspond to more-aroused. We use normalised
    # (already in [0, 1] where 1 = aroused).
    s = scored["normalised"].values

    if len(np.unique(y)) < 2:
        return {"error": "Only one class present, ROC/PR not defined"}

    fpr, tpr, thr_roc = roc_curve(y, s)
    roc_auc_val = roc_auc_score(y, s)
    prec, rec, thr_pr = precision_recall_curve(y, s)
    pr_auc_val = average_precision_score(y, s)

    # Precision @ recall >= 0.8 — pick the highest precision among all
    # thresholds that still give at least 80% recall. `precision_recall_curve`
    # returns them sorted by threshold (recall descending), so we scan.
    target_recall = 0.8
    eligible = rec >= target_recall
    if eligible.any():
        p_at_r = float(prec[eligible].max())
    else:
        p_at_r = float("nan")

    return {
        "n":          int(len(y)),
        "n_pos":      int(y.sum()),
        "n_neg":      int((1 - y).sum()),
        "roc_auc":    float(roc_auc_val),
        "pr_auc":     float(pr_auc_val),
        "precision_at_recall_0p8": p_at_r,
        "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr_roc": thr_roc.tolist(),
        "prec": prec.tolist(), "rec": rec.tolist(), "thr_pr": thr_pr.tolist(),
    }


def plot_roc(metrics: dict, out_path: Path) -> None:
    if "error" in metrics:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(metrics["fpr"], metrics["tpr"], color=BLUE, lw=2,
            label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — arousal classification at window level",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] ROC → {out_path.name}")


def plot_pr(metrics: dict, out_path: Path) -> None:
    if "error" in metrics:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(metrics["rec"], metrics["prec"], color=RED, lw=2,
            label=f"PR (AUC = {metrics['pr_auc']:.3f})")
    base_rate = metrics["n_pos"] / metrics["n"]
    ax.axhline(base_rate, color=GREY, ls="--", lw=1,
               label=f"Base rate ({base_rate:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve — arousal classification",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] PR → {out_path.name}")


# ----- Ablation: train GSR-only, HRV-only, both from baseline ----------

def _fit_iforest(X: np.ndarray, contamination: float,
                 n_estimators: int = 200) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            max_samples="auto", random_state=42, n_jobs=-1,
        )),
    ])
    pipe.fit(X)
    return pipe


def _normalise_scores(raw: np.ndarray, train_scores: np.ndarray) -> np.ndarray:
    """Mirror ArousalDetector._normalise_score vectorised."""
    lo = train_scores.min() - train_scores.std()
    hi = train_scores.max()
    if hi == lo:
        return np.zeros_like(raw)
    out = (hi - raw) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def run_ablation(
    sessions: list[Session],
    contamination: float,
) -> tuple[dict, pd.DataFrame]:
    """
    Train GSR-only, HRV-only on baseline sessions; score all session windows;
    compute per-variant ROC/PR and return a metrics dict + long-form scored
    frame for the 'combined' variant (average of per-source normalised
    scores on rows where both available).
    """
    baselines = [s for s in sessions if not s.is_aroused]

    # Train one pipeline per source
    frames = []
    for s in sessions:
        df = pd.read_csv(s.dir / "features.csv")
        df["__session"] = s.name
        df["__y"]       = int(s.is_aroused)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    # Baseline-only training frame
    base_frames = []
    for s in baselines:
        df = pd.read_csv(s.dir / "features.csv")
        df["__session"] = s.name
        base_frames.append(df)
    base_df = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()

    variants: dict[str, dict] = {}

    for source in ("gsr", "hrv"):
        feature_cols = [c for c, _ in SOURCES[source]]
        base_rows = base_df[base_df["source"] == source].dropna(subset=feature_cols)
        if base_rows.empty:
            continue

        X_train = base_rows[feature_cols].values
        pipe = _fit_iforest(X_train, contamination=contamination)
        train_scores = pipe.score_samples(X_train)

        all_rows = all_df[all_df["source"] == source].dropna(subset=feature_cols).copy()
        raw = pipe.score_samples(all_rows[feature_cols].values)
        norm = _normalise_scores(raw, train_scores)

        all_rows["raw_score"]   = raw
        all_rows["normalised"]  = norm
        all_rows["y_true"]      = all_rows["__y"]
        variants[source] = {
            "pipe":          pipe,
            "train_scores":  train_scores,
            "scored_rows":   all_rows.reset_index(drop=True),
        }

    # Build the 'combined' variant = for each time window, if both a GSR and
    # HRV row exist within 30s in the same session, use the mean of their
    # normalised scores; otherwise use whichever is present.
    combined_rows = _build_combined_variant(variants, all_df)

    # Compute metrics for each variant
    metrics_rows = []
    for name in ("gsr", "hrv"):
        if name not in variants:
            continue
        v = variants[name]
        m = _roc_pr_from(v["scored_rows"]["y_true"].values,
                         v["scored_rows"]["normalised"].values)
        m["variant"] = name
        m["n_windows"] = len(v["scored_rows"])
        metrics_rows.append(m)

    if not combined_rows.empty:
        m = _roc_pr_from(combined_rows["y_true"].values,
                         combined_rows["normalised"].values)
        m["variant"] = "combined"
        m["n_windows"] = len(combined_rows)
        metrics_rows.append(m)

    return {
        "variants":    variants,
        "combined":    combined_rows,
        "metrics":     metrics_rows,
    }, pd.DataFrame(metrics_rows)


def _build_combined_variant(variants: dict, all_df: pd.DataFrame) -> pd.DataFrame:
    """Pair GSR and HRV scored rows within 30s per session, average scores."""
    if "gsr" not in variants or "hrv" not in variants:
        # Fall back: whichever is present
        if "gsr" in variants:
            return variants["gsr"]["scored_rows"]
        if "hrv" in variants:
            return variants["hrv"]["scored_rows"]
        return pd.DataFrame()

    gsr_rows = variants["gsr"]["scored_rows"]
    hrv_rows = variants["hrv"]["scored_rows"]

    combined = []
    for sess_name in pd.concat([gsr_rows["__session"], hrv_rows["__session"]]).unique():
        g = gsr_rows[gsr_rows["__session"] == sess_name].sort_values("window_start_time")
        h = hrv_rows[hrv_rows["__session"] == sess_name].sort_values("window_start_time")
        if g.empty and h.empty:
            continue

        g_times = g["window_start_time"].values
        h_times = h["window_start_time"].values
        used_h: set[int] = set()

        for _, gr in g.iterrows():
            if h_times.size == 0:
                combined.append({
                    "__session":       sess_name,
                    "window_start_time": gr["window_start_time"],
                    "normalised":      gr["normalised"],
                    "y_true":          gr["y_true"],
                })
                continue
            idx = int(np.argmin(np.abs(h_times - gr["window_start_time"])))
            gap = abs(h_times[idx] - gr["window_start_time"])
            if gap <= 30.0 and idx not in used_h:
                used_h.add(idx)
                hr = h.iloc[idx]
                combined.append({
                    "__session":       sess_name,
                    "window_start_time": gr["window_start_time"],
                    "normalised":      (gr["normalised"] + hr["normalised"]) / 2,
                    "y_true":          gr["y_true"],
                })
            else:
                combined.append({
                    "__session":       sess_name,
                    "window_start_time": gr["window_start_time"],
                    "normalised":      gr["normalised"],
                    "y_true":          gr["y_true"],
                })

        # Unpaired HRV rows
        for i, hr in h.reset_index(drop=True).iterrows():
            if i in used_h:
                continue
            combined.append({
                "__session":       sess_name,
                "window_start_time": hr["window_start_time"],
                "normalised":      hr["normalised"],
                "y_true":          hr["y_true"],
            })

    return pd.DataFrame(combined)


def _roc_pr_from(y, s) -> dict:
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "precision_at_recall_0p8": float("nan")}
    prec, rec, _ = precision_recall_curve(y, s)
    eligible = rec >= 0.8
    p_at_r = float(prec[eligible].max()) if eligible.any() else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc":  float(average_precision_score(y, s)),
        "precision_at_recall_0p8": p_at_r,
    }


def plot_ablation_roc(ablation: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    colours = {"gsr": BLUE, "hrv": ORANGE, "combined": GREEN}

    for name, colour in colours.items():
        if name == "combined":
            df = ablation["combined"]
            if df.empty:
                continue
            y = df["y_true"].values
            s = df["normalised"].values
        else:
            if name not in ablation["variants"]:
                continue
            rows = ablation["variants"][name]["scored_rows"]
            y = rows["y_true"].values
            s = rows["normalised"].values
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc_val = roc_auc_score(y, s)
        ax.plot(fpr, tpr, lw=2, color=colour,
                label=f"{name.upper()} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Ablation — GSR-only vs HRV-only vs combined",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Ablation ROC → {out_path.name}")


def plot_threshold_sweep(
    scored: pd.DataFrame,
    operating_threshold: float,
    out_path: Path,
) -> None:
    if len(scored["y_true"].unique()) < 2:
        return
    y = scored["y_true"].values
    s = scored["normalised"].values

    thresholds = np.linspace(0.0, 1.0, 101)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        yhat = (s >= t).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        precs.append(p); recs.append(r); f1s.append(f)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precs, color=BLUE,  lw=2, label="Precision")
    ax.plot(thresholds, recs,  color=RED,   lw=2, label="Recall")
    ax.plot(thresholds, f1s,   color=GREEN, lw=2, label="F1")
    ax.axvline(operating_threshold, color="black", ls="--", lw=1.5,
               label=f"Operating threshold ({operating_threshold:.2f})")
    ax.set_xlabel("Decision threshold (normalised score)")
    ax.set_ylabel("Metric value")
    ax.set_title("Threshold sweep — precision / recall / F1",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Threshold sweep → {out_path.name}")


def plot_per_session(scored: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Per-session flag rate and mean score — grouped baseline vs aroused."""
    rows = []
    for (sess, label, y), sub in scored.groupby(["session", "label", "y_true"]):
        rows.append({
            "session":      sess,
            "label":        label,
            "y_true":       int(y),
            "n_windows":    len(sub),
            "mean_norm":    float(sub["normalised"].mean()),
            "std_norm":     float(sub["normalised"].std()),
            "flag_rate":    float(sub["is_aroused"].mean()),
        })
    df = pd.DataFrame(rows).sort_values(["y_true", "mean_norm"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.35)))
    fig.suptitle("Per-session breakdown", fontsize=13, fontweight="bold")

    colours = [RED if y else BLUE for y in df["y_true"]]
    labels = [r.session[-30:] + f"  ({r.label[-14:]})" for r in df.itertuples()]
    y_pos = np.arange(len(df))

    ax = axes[0]
    ax.barh(y_pos, df["mean_norm"], xerr=df["std_norm"],
            color=colours, alpha=0.85, edgecolor="white",
            error_kw={"ecolor": "black", "alpha": 0.4, "capsize": 3})
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean normalised score ± 1 SD")
    ax.set_title("Mean arousal score per session")
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[1]
    ax.barh(y_pos, df["flag_rate"] * 100, color=colours, alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos); ax.set_yticklabels(["" for _ in y_pos])
    ax.set_xlabel("Flag rate (%)")
    ax.set_title("Flagged windows per session")
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis="x")

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(color=BLUE, label="baseline"),
        Patch(color=RED,  label="aroused"),
    ], loc="lower center", ncol=2, fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Per-session → {out_path.name}")

    return df


def plot_hero_timeseries(scored: pd.DataFrame, out_path: Path) -> None:
    """One good aroused session's score trace over time."""
    positives = scored[scored["y_true"] == 1]
    if positives.empty:
        return

    # Pick the positive session with the highest flag rate — the model's
    # best answer. If you want a specific session, edit here.
    per_sess = (positives.groupby("session")["is_aroused"].mean()
                         .sort_values(ascending=False))
    if per_sess.empty:
        return
    target = per_sess.index[0]
    sub = scored[scored["session"] == target].sort_values("time_s").copy()
    sub["t_min"] = sub["time_s"] / 60.0

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(f"Hero timeseries — {target}  (label: {sub['label'].iloc[0]})",
                 fontsize=12, fontweight="bold")

    for src, colour in (("gsr", BLUE), ("hrv", ORANGE), ("merged", PURPLE)):
        part = sub[sub["source"] == src]
        if part.empty:
            continue
        ax.plot(part["t_min"], part["normalised"], "o-",
                color=colour, lw=1.3, ms=4, label=f"{src.upper()} normalised", alpha=0.85)

    flagged = sub[sub["is_aroused"]]
    if not flagged.empty:
        ax.scatter(flagged["t_min"], flagged["normalised"],
                   facecolors="none", edgecolors=RED, s=100, lw=2, zorder=5,
                   label=f"Flagged (n={len(flagged)})")

    # Smoothed trend across all sources on this session
    if len(sub) >= 5:
        window = max(3, min(7, len(sub) // 4))
        smoothed = sub["normalised"].rolling(window=window, center=True, min_periods=1).mean()
        ax.plot(sub["t_min"], smoothed, color="black", lw=2.2,
                label=f"Smoothed ({window}-win)")

    ax.set_xlabel("Session time (minutes)")
    ax.set_ylabel("Normalised arousal score")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Hero timeseries → {out_path.name}")


def compute_confusion_at(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    y = scored["y_true"].values
    yhat = (scored["normalised"].values >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return pd.DataFrame([
        {"threshold": threshold,
         "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
         "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
         "recall":    float(tp / (tp + fn)) if (tp + fn) else 0.0,
         "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
         "f1":        float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0},
    ])


# ---------------------------------------------------------------------------
# §5.4  Real-time system performance (latency)
# ---------------------------------------------------------------------------


def measure_latency(sessions: list[Session], detector, n_calls: int = 1000) -> dict:
    """
    Replay recorded session windows through detector.score() and measure
    wall-clock time per call.
    """
    timings = []
    count = 0

    def score_features(features: dict, source: str) -> None:
        nonlocal count
        t0 = time.perf_counter()
        detector.score(features, source=source)
        timings.append((time.perf_counter() - t0) * 1000.0)  # ms
        count += 1

    is_ensemble = isinstance(detector, EnsembleDetector)

    for s in sessions:
        if count >= n_calls:
            break
        df = pd.read_csv(s.dir / "features.csv")
        for _, row in df.iterrows():
            if count >= n_calls:
                break
            source = row.get("source", "unknown")
            if is_ensemble:
                feature_cols = (detector.gsr.feature_cols if source == "gsr"
                                and detector.gsr is not None
                                else detector.hrv.feature_cols if source == "hrv"
                                and detector.hrv is not None
                                else [])
            else:
                feature_cols = detector.feature_cols
            if not feature_cols:
                continue
            features = {}
            complete = True
            for c in feature_cols:
                v = row.get(c)
                if pd.isna(v):
                    complete = False
                    break
                try:
                    features[c] = float(v)
                except (TypeError, ValueError):
                    complete = False
                    break
            if not complete:
                continue
            score_features(features, source)

    if not timings:
        return {}
    arr = np.array(timings)
    return {
        "n_calls":   int(len(arr)),
        "mean_ms":   float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "max_ms":    float(arr.max()),
        "timings_ms": arr.tolist(),
    }


def plot_latency(stats: dict, out_path: Path) -> None:
    if not stats or not stats.get("timings_ms"):
        return
    arr = np.array(stats["timings_ms"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(arr, bins=50, color=BLUE, alpha=0.85, edgecolor="white")
    for label, val, colour in [
        ("mean",   stats["mean_ms"],   GREEN),
        ("median", stats["median_ms"], ORANGE),
        ("p95",    stats["p95_ms"],    RED),
        ("max",    stats["max_ms"],    "black"),
    ]:
        ax.axvline(val, color=colour, ls="--", lw=1.5,
                   label=f"{label} = {val:.2f} ms")
    ax.set_xlabel("score() call latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Inference latency — {stats['n_calls']} calls",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[04] Latency histogram → {out_path.name}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_report(
    out_dir: Path,
    sessions: list[Session],
    scored: pd.DataFrame,
    detector,
    metrics: dict,
    ablation_metrics: pd.DataFrame,
    loso: pd.DataFrame,
    latency: dict,
    operating_threshold: float,
    per_session: pd.DataFrame,
    combine_mode: str,
) -> None:
    report_path = out_dir / "report.md"
    lines = []

    is_ensemble = isinstance(detector, EnsembleDetector)
    n_base = sum(1 for s in sessions if not s.is_aroused)
    n_pos  = sum(1 for s in sessions if s.is_aroused)

    lines.append(f"# Arousal detection — results summary")
    lines.append(f"")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_")
    lines.append(f"")
    lines.append(f"## Setup")
    lines.append(f"- Detector: **{'ensemble (' + combine_mode + ')' if is_ensemble else 'single-source / merged'}**")
    lines.append(f"- Sessions: **{len(sessions)}** total — {n_base} baseline, {n_pos} aroused")
    lines.append(f"- Scored windows: **{len(scored)}**")
    lines.append(f"")
    lines.append(f"## §5.3 Headline metrics")
    if "error" not in metrics:
        lines.append(f"- **ROC-AUC**: {metrics['roc_auc']:.3f}")
        lines.append(f"- **PR-AUC**:  {metrics['pr_auc']:.3f}")
        lines.append(f"- **Precision @ recall=0.8**: {metrics['precision_at_recall_0p8']:.3f}")
        lines.append(f"- Operating threshold (normalised): {operating_threshold:.3f}")
    else:
        lines.append(f"- {metrics['error']}")
    lines.append(f"")
    lines.append(f"![Score distributions](figures/03_score_distributions.png)")
    lines.append(f"")
    lines.append(f"![ROC](figures/03_roc_curve.png)")
    lines.append(f"![PR](figures/03_pr_curve.png)")
    lines.append(f"")
    lines.append(f"## §5.3 Ablation (trained on baseline only)")
    if not ablation_metrics.empty:
        lines.append(f"")
        lines.append(ablation_metrics[["variant", "n_windows", "roc_auc",
                                       "pr_auc", "precision_at_recall_0p8"]]
                     .to_markdown(index=False, floatfmt=".3f"))
        lines.append(f"")
    lines.append(f"![Ablation ROC](figures/03_ablation_roc.png)")
    lines.append(f"")
    lines.append(f"## §5.3 Per-session breakdown")
    lines.append(per_session.to_markdown(index=False, floatfmt=".3f"))
    lines.append(f"")
    lines.append(f"![Per-session](figures/03_per_session_breakdown.png)")
    lines.append(f"")
    lines.append(f"## §5.3 Hero timeseries")
    lines.append(f"![Hero](figures/03_hero_timeseries.png)")
    lines.append(f"")
    lines.append(f"## §5.2 Baseline characterisation")
    lines.append(f"![Violins](figures/02_feature_violins.png)")
    lines.append(f"![PCA](figures/02_pca_projection.png)")
    lines.append(f"")
    if not loso.empty:
        lines.append(f"### LOSO shifts")
        lines.append(loso[["source", "session", "n_windows", "heldout_mean",
                           "shift_sigma"]].to_markdown(index=False, floatfmt=".3f"))
        lines.append(f"")
        lines.append(f"![LOSO](figures/02_loso_shifts.png)")
        lines.append(f"")
    if latency:
        lines.append(f"## §5.4 Real-time performance")
        lines.append(f"- Calls timed: {latency['n_calls']}")
        lines.append(f"- mean: {latency['mean_ms']:.2f} ms, "
                     f"p95: {latency['p95_ms']:.2f} ms, "
                     f"max: {latency['max_ms']:.2f} ms")
        lines.append(f"")
        lines.append(f"![Latency](figures/04_latency_hist.png)")
        lines.append(f"")

    lines.append(f"## §5.6 Summary bullets")
    if "error" not in metrics:
        base_mean = float(scored[scored.y_true == 0]["normalised"].mean())
        pos_mean  = float(scored[scored.y_true == 1]["normalised"].mean())
        lines.append(f"- Mean normalised score is {pos_mean:.2f} for aroused "
                     f"windows vs {base_mean:.2f} for baseline (gap = "
                     f"{pos_mean - base_mean:+.2f}).")
        lines.append(f"- The detector achieves ROC-AUC = {metrics['roc_auc']:.3f} at "
                     f"the window level (n = {metrics['n']}).")
        lines.append(f"- Precision reaches {metrics['precision_at_recall_0p8']:.2f} "
                     f"at recall = 0.8.")
    if not ablation_metrics.empty:
        row = ablation_metrics.set_index("variant")
        if "combined" in row.index and "gsr" in row.index and "hrv" in row.index:
            lines.append(f"- Combining modalities improves ROC-AUC from "
                         f"{row.loc['gsr', 'roc_auc']:.3f} (GSR-only) and "
                         f"{row.loc['hrv', 'roc_auc']:.3f} (HRV-only) "
                         f"to {row.loc['combined', 'roc_auc']:.3f}.")
    if latency:
        lines.append(f"- Mean inference latency: {latency['mean_ms']:.1f} ms "
                     f"(p95 {latency['p95_ms']:.1f} ms).")

    report_path.write_text("\n".join(lines))
    print(f"[REPORT] Written: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("parent", type=Path,
                   help="Parent directory containing session folders")
    p.add_argument("--model", type=Path, required=True,
                   help="Ensemble directory / manifest.json / single .pkl")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--baseline-prefix", default="baseline",
                   help="Label prefix for the negative class (default: 'baseline')")
    p.add_argument("--positive-prefix", default=None,
                   help="Optional label prefix to restrict the positive class")
    p.add_argument("--combine-mode", choices=("any", "all", "mean"), default="any",
                   help="Ensemble combination mode (default: any)")
    p.add_argument("--operating-threshold", type=float, default=0.5,
                   help="Normalised score threshold for confusion matrix (default: 0.5)")
    p.add_argument("--ablation-contamination", type=float, default=0.05,
                   help="Contamination for ablation models trained from scratch")
    p.add_argument("--latency-calls", type=int, default=1000)
    args = p.parse_args()

    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # --- Discover + classify sessions ---
    sessions = discover_sessions(args.parent, args.baseline_prefix,
                                 args.positive_prefix)
    if len(sessions) < 2:
        sys.exit("[ERROR] Need at least 2 sessions to analyse")

    # --- Load detector ---
    detector = load_detector(args.model, ensemble_mode=args.combine_mode)

    # --- §5.2 Baseline characterisation ---
    print("\n=== §5.2 Baseline characterisation ===")
    plot_feature_violins(sessions, out / "figures/02_feature_violins.png")
    plot_pca_projection(sessions, out / "figures/02_pca_projection.png")

    data_summary = compute_data_summary(sessions)
    data_summary.to_csv(out / "tables/02_data_summary.csv", index=False)
    print(f"[02] Data summary → tables/02_data_summary.csv")

    baseline_sessions = [s for s in sessions if not s.is_aroused]
    loso = compute_loso_shifts(baseline_sessions)
    loso.to_csv(out / "tables/02_loso_shifts.csv", index=False)
    plot_loso_shifts(loso, out / "figures/02_loso_shifts.png")

    # --- Score all windows with main detector ---
    print("\n=== Scoring all sessions ===")
    scored = score_all_sessions(sessions, detector, args.combine_mode)
    scored.to_csv(out / "scored_windows.csv", index=False)

    # --- §5.3 Arousal detection ---
    print("\n=== §5.3 Arousal detection ===")
    plot_score_distributions(scored, out / "figures/03_score_distributions.png")

    metrics = compute_roc_pr(scored)
    plot_roc(metrics, out / "figures/03_roc_curve.png")
    plot_pr(metrics,  out / "figures/03_pr_curve.png")

    # Save metrics (without the full curve arrays that bloat the CSV)
    metrics_compact = {k: v for k, v in metrics.items()
                       if k not in ("fpr", "tpr", "thr_roc",
                                    "prec", "rec", "thr_pr")}
    pd.DataFrame([metrics_compact]).to_csv(
        out / "tables/03_metrics.csv", index=False,
    )

    # Ablation (trains its own models)
    print("\n=== §5.3 Ablation ===")
    ablation, ablation_metrics = run_ablation(
        sessions, contamination=args.ablation_contamination,
    )
    ablation_metrics.to_csv(out / "tables/03_ablation_metrics.csv", index=False)
    plot_ablation_roc(ablation, out / "figures/03_ablation_roc.png")

    # Threshold sweep + confusion
    plot_threshold_sweep(scored, args.operating_threshold,
                         out / "figures/03_threshold_sweep.png")
    cm_df = compute_confusion_at(scored, args.operating_threshold)
    cm_df.to_csv(out / "tables/03_confusion_at_operating_pt.csv", index=False)

    per_session = plot_per_session(scored, out / "figures/03_per_session_breakdown.png")
    per_session.to_csv(out / "tables/03_per_session.csv", index=False)

    plot_hero_timeseries(scored, out / "figures/03_hero_timeseries.png")

    # --- §5.4 Latency ---
    print("\n=== §5.4 Latency ===")
    latency = measure_latency(sessions, detector, n_calls=args.latency_calls)
    plot_latency(latency, out / "figures/04_latency_hist.png")
    if latency:
        compact = {k: v for k, v in latency.items() if k != "timings_ms"}
        pd.DataFrame([compact]).to_csv(
            out / "tables/04_latency.csv", index=False,
        )

    # --- Summary JSON ---
    summary = {
        "generated":          datetime.now().isoformat(),
        "n_sessions":         len(sessions),
        "n_baseline":         sum(1 for s in sessions if not s.is_aroused),
        "n_aroused":          sum(1 for s in sessions if s.is_aroused),
        "n_scored_windows":   int(len(scored)),
        "combine_mode":       args.combine_mode if isinstance(detector, EnsembleDetector) else "single",
        "main_metrics":       metrics_compact,
        "ablation":           ablation_metrics.to_dict(orient="records"),
        "operating_threshold": args.operating_threshold,
        "operating_confusion": cm_df.iloc[0].to_dict(),
        "latency":            {k: v for k, v in latency.items()
                               if k != "timings_ms"} if latency else None,
        "sessions": [
            {"name": s.name, "label": s.label,
             "y_true": int(s.is_aroused)} for s in sessions
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[SUMMARY] Written: {out / 'summary.json'}")

    # --- Markdown report ---
    try:
        write_report(
            out, sessions, scored, detector, metrics, ablation_metrics,
            loso, latency, args.operating_threshold, per_session,
            args.combine_mode,
        )
    except Exception as e:
        # to_markdown needs tabulate; report is a nice-to-have, don't die for it
        print(f"[REPORT] Skipped markdown report: {e}")
        print(f"[REPORT] (Install 'tabulate' for markdown tables)")

    print(f"\n[DONE] All results in {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
