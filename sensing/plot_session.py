"""
plot_session.py

Generate diagnostic plots for a recorded session, optionally overlaying
live arousal detection from the trained Isolation Forest.

Usage:
    python plot_session.py <path> [--model <path/to/iforest.pkl>]

    <path> can be:
      - A single session directory  (contains features.csv)
      - A parent directory whose sub-folders are sessions

    --model  Optional. Path to trained iforest.pkl. If supplied, every
             plot gets arousal flag overlays and a new plot 07 is added
             showing the arousal timeline.

Reads:
    <session_dir>/features.csv
    <session_dir>/session.json   (optional, for metadata)

Saves to:
    <session_dir>/plots/
        01_gsr_overview.png      — SCL mean, SCR count, phasic std over time
        02_gsr_quality.png       — SCR amplitude, rise time, recovery time
        03_hrv_overview.png      — HR mean/min/max, RMSSD, SDNN over time
        04_hrv_poincare.png      — Poincaré-style SD1 vs SD2 scatter
        05_data_quality.png      — Feature counts, source timeline, missing data heatmap
        06_combined_arousal.png  — Key GSR + HRV features on shared time axis
        07_arousal_timeline.png  — (with --model) arousal score + flagged windows
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#B279A2"
GREY = "#9D9D9D"

# Colour used for AROUSED markers and shading across every plot
AROUSED = "#D62728"


def load_session(session_dir: Path) -> tuple[pd.DataFrame, dict]:
    csv_path = session_dir / "features.csv"
    meta_path = session_dir / "session.json"

    if not csv_path.exists():
        sys.exit(f"[ERROR] features.csv not found in {session_dir}")

    df = pd.read_csv(csv_path)
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return df, meta


def split_sources(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gsr = df[df["source"] == "gsr"].copy().reset_index(drop=True)
    hrv = df[df["source"] == "hrv"].copy().reset_index(drop=True)
    return gsr, hrv


def time_col(df: pd.DataFrame) -> np.ndarray:
    """Return window_start_time in minutes."""
    return df["window_start_time"].values / 60.0


def savefig(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Arousal scoring — runs the session through the detector
# ---------------------------------------------------------------------------


def score_session(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """
    Run every row of features.csv through ArousalDetector.
    Returns a copy of df with extra columns:
        _score       raw iforest score_samples()
        _normalised  [0, 1] normalised score
        _is_aroused  bool, True if flagged
        _threshold   the detector's threshold (same for every row)
    """
    from arousal_detector import ArousalDetector

    detector = ArousalDetector(model_path)

    out = df.copy()
    out["_score"] = np.nan
    out["_normalised"] = np.nan
    out["_is_aroused"] = False
    out["_threshold"] = detector.threshold

    skip = {
        "source", "label", "session_start", "window_start_time",
        "anomaly_score", "anomaly_normalised", "is_aroused",
    }

    for i, row in df.iterrows():
        source = row.get("source", "unknown")
        features = {}
        for k, v in row.items():
            if k in skip:
                continue
            if pd.isna(v):
                continue
            try:
                features[k] = float(v)
            except (TypeError, ValueError):
                continue

        result = detector.score(features, source=source)
        if result is not None:
            out.at[i, "_score"] = result.score
            out.at[i, "_normalised"] = result.normalised
            out.at[i, "_is_aroused"] = result.is_aroused

    n_flagged = int(out["_is_aroused"].sum())
    n_scored  = int(out["_score"].notna().sum())
    print(f"  Scored {n_scored}/{len(out)} windows — {n_flagged} flagged as aroused")
    return out


def shade_aroused(ax, scored: pd.DataFrame, source: Optional[str] = None) -> int:
    """
    Add a light red vertical band at every flagged window on the given axis.
    If `source` is given, only shade flags from that source (gsr/hrv).
    Returns the number of bands drawn (so the caller can decide whether to
    add a legend entry).
    """
    if scored is None or scored.empty:
        return 0

    rows = scored[scored["_is_aroused"] == True]  # noqa: E712
    if source is not None:
        rows = rows[rows["source"] == source]
    if rows.empty:
        return 0

    # Figure out a sensible band width. For GSR we typically score every
    # window (~60s); HRV windows step every 30s. Use half the median gap.
    if len(rows) > 1:
        times_min = rows["window_start_time"].values / 60.0
        median_gap = float(np.median(np.diff(np.sort(times_min))))
        half_width = max(median_gap / 2.0, 0.25)
    else:
        half_width = 0.25  # 15 seconds either side for a single flag

    for _, row in rows.iterrows():
        t_min = row["window_start_time"] / 60.0
        ax.axvspan(
            t_min - half_width,
            t_min + half_width,
            color=AROUSED,
            alpha=0.12,
            zorder=0,
        )
    return len(rows)


def add_arousal_legend_entry(ax, count: int, source_label: str = "all") -> None:
    """Append an 'aroused' legend patch to an axis that already has a legend."""
    if count == 0:
        return
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=AROUSED, alpha=0.25,
                         label=f"AROUSED ({count} win, {source_label})"))
    ax.legend(handles=handles, fontsize=8, loc="upper right")


# ---------------------------------------------------------------------------
# Plot 1 — GSR overview
# ---------------------------------------------------------------------------


def plot_gsr_overview(
    gsr: pd.DataFrame,
    plots_dir: Path,
    meta: dict,
    scored: Optional[pd.DataFrame] = None,
) -> None:
    if gsr.empty:
        print("  [SKIP] No GSR rows found.")
        return

    t = time_col(gsr)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        "GSR Overview — Skin Conductance Level & Responses",
        fontsize=14,
        fontweight="bold",
    )

    # --- SCL ---
    ax = axes[0]
    ax.plot(t, gsr["gsr_scl_mean"], color=BLUE, lw=2, label="SCL mean (µS)")
    ax.fill_between(
        t,
        gsr["gsr_scl_mean"] - gsr["gsr_scl_std"],
        gsr["gsr_scl_mean"] + gsr["gsr_scl_std"],
        alpha=0.2,
        color=BLUE,
        label="±1 SD",
    )
    ax.set_ylabel("Skin Conductance\nLevel (µS)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _add_normal_band(ax, 0.5, 20.0, "Typical range (0.5–20 µS)")
    n = shade_aroused(ax, scored, source="gsr")
    add_arousal_legend_entry(ax, n, "GSR")

    # --- SCR count ---
    ax = axes[1]
    ax.bar(
        t,
        gsr["gsr_scr_count"],
        width=0.4,
        color=ORANGE,
        alpha=0.8,
        label="SCR count / min",
    )
    ax.set_ylabel("SCR Count\n(per minute)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    n = shade_aroused(ax, scored, source="gsr")
    add_arousal_legend_entry(ax, n, "GSR")

    # --- Phasic std ---
    ax = axes[2]
    ax.plot(t, gsr["gsr_phasic_std"], color=GREEN, lw=2, label="Phasic std (µS)")
    ax.plot(
        t,
        gsr["gsr_phasic_mean"],
        color=GREEN,
        lw=1.5,
        ls="--",
        alpha=0.6,
        label="Phasic mean (µS)",
    )
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_ylabel("Phasic Component\n(µS)")
    ax.set_xlabel("Session time (minutes)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    n = shade_aroused(ax, scored, source="gsr")
    add_arousal_legend_entry(ax, n, "GSR")

    fig.tight_layout()
    savefig(fig, plots_dir, "01_gsr_overview.png")


# ---------------------------------------------------------------------------
# Plot 2 — GSR quality
# ---------------------------------------------------------------------------


def plot_gsr_quality(
    gsr: pd.DataFrame,
    plots_dir: Path,
    meta: dict,
    scored: Optional[pd.DataFrame] = None,
) -> None:
    if gsr.empty:
        return

    t = time_col(gsr)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        "GSR Data Quality — SCR Timing & Amplitude", fontsize=14, fontweight="bold"
    )

    # SCR amplitude
    ax = axes[0]
    ax.bar(t, gsr["gsr_scr_mean_amp"], width=0.4, color=PURPLE, alpha=0.85,
           label="Mean SCR amplitude")
    ax.set_ylabel("Mean SCR Amplitude\n(µS)")
    ax.axhline(0.02, color=RED, ls="--", lw=1.2, label="Detection threshold (0.02 µS)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    n = shade_aroused(ax, scored, source="gsr")
    add_arousal_legend_entry(ax, n, "GSR")

    # SCR rise time
    ax = axes[1]
    ax.plot(
        t,
        gsr["gsr_scr_rise_time"],
        "o-",
        color=ORANGE,
        lw=1.5,
        ms=5,
        label="Rise time (s)",
    )
    ax.set_ylabel("Mean SCR Rise Time (s)")
    _add_normal_band(ax, 1.0, 3.0, "Typical 1–3 s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    n = shade_aroused(ax, scored, source="gsr")
    add_arousal_legend_entry(ax, n, "GSR")

    # SCR recovery time
    ax = axes[2]
    ax.plot(
        t,
        gsr["gsr_scr_recovery_time"],
        "o-",
        color=BLUE,
        lw=1.5,
        ms=5,
        label="Half-recovery time (s)",
    )
    ax.set_ylabel("Mean SCR Recovery Time (s)")
    _add_normal_band(ax, 5.0, 30.0, "Typical 5–30 s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Session time (minutes)")
    n = shade_aroused(ax, scored, source="gsr")
    add_arousal_legend_entry(ax, n, "GSR")

    fig.tight_layout()
    savefig(fig, plots_dir, "02_gsr_quality.png")


# ---------------------------------------------------------------------------
# Plot 3 — HRV overview
# ---------------------------------------------------------------------------


def plot_hrv_overview(
    hrv: pd.DataFrame,
    plots_dir: Path,
    meta: dict,
    scored: Optional[pd.DataFrame] = None,
) -> None:
    if hrv.empty:
        print("  [SKIP] No HRV rows found.")
        return

    t = time_col(hrv)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        "HRV Overview — Heart Rate & Variability", fontsize=14, fontweight="bold"
    )

    # HR band
    ax = axes[0]
    ax.fill_between(
        t,
        hrv["hrv_hr_min"],
        hrv["hrv_hr_max"],
        alpha=0.15,
        color=RED,
        label="HR range (min–max)",
    )
    ax.plot(t, hrv["hrv_hr_mean"], color=RED, lw=2, label="HR mean (bpm)")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    n = shade_aroused(ax, scored, source="hrv")
    add_arousal_legend_entry(ax, n, "HRV")

    # RMSSD
    ax = axes[1]
    ax.plot(t, hrv["hrv_rmssd"], color=BLUE, lw=2, label="RMSSD (ms)")
    ax.plot(t, hrv["hrv_sdnn"], color=ORANGE, lw=2, ls="--", label="SDNN (ms)")
    ax.set_ylabel("HRV (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _add_normal_band(ax, 20.0, 80.0, "Healthy RMSSD range")
    n = shade_aroused(ax, scored, source="hrv")
    add_arousal_legend_entry(ax, n, "HRV")

    # pNN50
    ax = axes[2]
    ax.plot(t, hrv["hrv_pnn50"] * 100, color=GREEN, lw=2, label="pNN50 (%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("pNN50 (%)")
    ax.set_xlabel("Session time (minutes)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    n = shade_aroused(ax, scored, source="hrv")
    add_arousal_legend_entry(ax, n, "HRV")

    fig.tight_layout()
    savefig(fig, plots_dir, "03_hrv_overview.png")


# ---------------------------------------------------------------------------
# Plot 4 — HRV Poincaré scatter
# ---------------------------------------------------------------------------


def plot_hrv_poincare(
    hrv: pd.DataFrame,
    plots_dir: Path,
    scored: Optional[pd.DataFrame] = None,
) -> None:
    if hrv.empty or len(hrv) < 2:
        return

    t = time_col(hrv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("HRV Poincaré Analysis — SD1 vs SD2", fontsize=14, fontweight="bold")

    # Figure out which HRV rows were flagged (same index as `hrv`)
    flagged_mask = np.zeros(len(hrv), dtype=bool)
    if scored is not None and not scored.empty:
        hrv_scored = scored[scored["source"] == "hrv"].reset_index(drop=True)
        if len(hrv_scored) == len(hrv):
            flagged_mask = hrv_scored["_is_aroused"].fillna(False).to_numpy()

    # Scatter SD1 vs SD2 — colour by time, highlight flagged ones with a red ring
    ax = axes[0]
    sc = ax.scatter(hrv["hrv_sd1"], hrv["hrv_sd2"], c=t, cmap="viridis", s=60, zorder=3)
    if flagged_mask.any():
        ax.scatter(
            hrv["hrv_sd1"][flagged_mask],
            hrv["hrv_sd2"][flagged_mask],
            facecolors="none",
            edgecolors=AROUSED,
            s=180,
            lw=2.0,
            zorder=4,
            label=f"AROUSED ({flagged_mask.sum()})",
        )
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("SD1 — Short-term variability (ms)")
    ax.set_ylabel("SD2 — Long-term variability (ms)")
    ax.set_title("SD1 vs SD2 (colour = session time)")
    fig.colorbar(sc, ax=ax, label="Time (min)")
    ax.grid(True, alpha=0.3)

    # SD1 and SD2 over time
    ax = axes[1]
    ax.plot(t, hrv["hrv_sd1"], color=BLUE, lw=2, label="SD1 (short-term)")
    ax.plot(t, hrv["hrv_sd2"], color=ORANGE, lw=2, label="SD2 (long-term)")
    ax.set_xlabel("Session time (minutes)")
    ax.set_ylabel("SD (ms)")
    ax.set_title("SD1 & SD2 over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    n = shade_aroused(ax, scored, source="hrv")
    add_arousal_legend_entry(ax, n, "HRV")

    fig.tight_layout()
    savefig(fig, plots_dir, "04_hrv_poincare.png")


# ---------------------------------------------------------------------------
# Plot 5 — Data quality dashboard
# ---------------------------------------------------------------------------


def plot_data_quality(
    df: pd.DataFrame, gsr: pd.DataFrame, hrv: pd.DataFrame, plots_dir: Path, meta: dict
) -> None:
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Data Quality Dashboard", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # --- 1. Window count bar ---
    ax = fig.add_subplot(gs[0, 0])
    counts = df["source"].value_counts()
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[BLUE if s == "gsr" else RED for s in counts.index],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Feature Windows per Source")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts.values) * 1.2)
    ax.grid(True, alpha=0.2, axis="y")

    # --- 2. Source timeline ---
    ax = fig.add_subplot(gs[0, 1:])
    source_colours = {"gsr": BLUE, "hrv": RED}
    for i, row in df.iterrows():
        ax.barh(
            row["source"],
            width=1.0,
            left=row["window_start_time"] / 60.0,
            color=source_colours.get(row["source"], GREY),
            alpha=0.7,
            edgecolor="white",
            height=0.4,
        )
    ax.set_xlabel("Session time (minutes)")
    ax.set_title("Feature Window Timeline")
    ax.grid(True, alpha=0.2, axis="x")

    # --- 3. Missing values heatmap ---
    ax = fig.add_subplot(gs[1, :2])
    gsr_cols = [c for c in df.columns if c.startswith("gsr_")]
    hrv_cols = [c for c in df.columns if c.startswith("hrv_")]
    heat_cols = gsr_cols[:8] + hrv_cols[:8]  # Keep it readable
    if heat_cols:
        heat = df[heat_cols].isnull().astype(int)
        im = ax.imshow(
            heat.T,
            aspect="auto",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(heat_cols)))
        ax.set_yticklabels(
            [c.replace("gsr_", "G:").replace("hrv_", "H:") for c in heat_cols],
            fontsize=7,
        )
        ax.set_xlabel("Feature window index")
        ax.set_title("Missing Values (green = present, red = absent)")
        fig.colorbar(im, ax=ax, ticks=[0, 1], label="Missing")

    # --- 4. Inter-window gap distribution ---
    ax = fig.add_subplot(gs[1, 2])
    for src, col, c in [("gsr", gsr, BLUE), ("hrv", hrv, RED)]:
        if len(col) > 1:
            gaps = np.diff(col["window_start_time"].values) / 60.0
            ax.hist(
                gaps, bins=15, alpha=0.6, color=c, label=src.upper(), edgecolor="white"
            )
    ax.set_xlabel("Gap between windows (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-window Gap Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    savefig(fig, plots_dir, "05_data_quality.png")


# ---------------------------------------------------------------------------
# Plot 6 — Combined arousal proxy
# ---------------------------------------------------------------------------


def plot_combined(
    gsr: pd.DataFrame,
    hrv: pd.DataFrame,
    plots_dir: Path,
    scored: Optional[pd.DataFrame] = None,
) -> None:
    if gsr.empty and hrv.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        "Combined GSR + HRV — Arousal Proxy Overview", fontsize=14, fontweight="bold"
    )

    # SCL + HR on same axis (dual y)
    ax1 = axes[0]
    ax2 = ax1.twinx()
    if not gsr.empty:
        tg = time_col(gsr)
        ax1.plot(tg, gsr["gsr_scl_mean"], color=BLUE, lw=2, label="SCL mean (µS)")
        ax1.set_ylabel("SCL (µS)", color=BLUE)
        ax1.tick_params(axis="y", labelcolor=BLUE)
    if not hrv.empty:
        th = time_col(hrv)
        ax2.plot(
            th, hrv["hrv_hr_mean"], color=RED, lw=2, ls="--", label="HR mean (bpm)"
        )
        ax2.set_ylabel("HR (bpm)", color=RED)
        ax2.tick_params(axis="y", labelcolor=RED)
    ax1.set_title("Skin Conductance Level vs Heart Rate")
    ax1.grid(True, alpha=0.3)
    # Shade with flags from both sources on combined plots
    n = shade_aroused(ax1, scored, source=None)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    handles = lines1 + lines2
    labels  = labels1 + labels2
    if n > 0:
        handles.append(Patch(facecolor=AROUSED, alpha=0.25,
                             label=f"AROUSED ({n} win)"))
        labels.append(f"AROUSED ({n} win)")
    ax1.legend(handles, labels, fontsize=8, loc="upper right")

    # SCR count + RMSSD
    ax = axes[1]
    if not gsr.empty:
        ax.bar(
            tg,
            gsr["gsr_scr_count"],
            width=0.35,
            color=ORANGE,
            alpha=0.7,
            label="SCR count/min (GSR)",
        )
    ax3 = ax.twinx()
    if not hrv.empty:
        ax3.plot(th, hrv["hrv_rmssd"], color=PURPLE, lw=2, label="RMSSD (ms)")
        ax3.set_ylabel("RMSSD (ms)", color=PURPLE)
        ax3.tick_params(axis="y", labelcolor=PURPLE)
    ax.set_ylabel("SCR count/min", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title("SCR Rate vs RMSSD (both ↑ = higher arousal from sympathetic drive)")
    ax.grid(True, alpha=0.3, axis="y")
    n = shade_aroused(ax, scored, source=None)
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax3.get_legend_handles_labels()
    handles = lines_a + lines_b
    labels  = labels_a + labels_b
    if n > 0:
        handles.append(Patch(facecolor=AROUSED, alpha=0.25,
                             label=f"AROUSED ({n} win)"))
        labels.append(f"AROUSED ({n} win)")
    ax.legend(handles, labels, fontsize=8, loc="upper right")

    # Phasic std + SD1
    ax = axes[2]
    if not gsr.empty:
        ax.plot(tg, gsr["gsr_phasic_std"], color=GREEN, lw=2, label="Phasic std (µS)")
        ax.set_ylabel("Phasic std (µS)", color=GREEN)
        ax.tick_params(axis="y", labelcolor=GREEN)
    ax4 = ax.twinx()
    if not hrv.empty:
        ax4.plot(th, hrv["hrv_sd1"], color=BLUE, lw=2, ls="--", label="SD1 (ms)")
        ax4.set_ylabel("SD1 (ms)", color=BLUE)
        ax4.tick_params(axis="y", labelcolor=BLUE)
    ax.set_title("GSR Phasic Std vs HRV SD1 (short-term vagal activity)")
    ax.set_xlabel("Session time (minutes)")
    ax.grid(True, alpha=0.3)
    n = shade_aroused(ax, scored, source=None)
    lines_c, labels_c = ax.get_legend_handles_labels()
    lines_d, labels_d = ax4.get_legend_handles_labels()
    handles = lines_c + lines_d
    labels  = labels_c + labels_d
    if n > 0:
        handles.append(Patch(facecolor=AROUSED, alpha=0.25,
                             label=f"AROUSED ({n} win)"))
        labels.append(f"AROUSED ({n} win)")
    ax.legend(handles, labels, fontsize=8, loc="upper right")

    fig.tight_layout()
    savefig(fig, plots_dir, "06_combined_arousal.png")


# ---------------------------------------------------------------------------
# Plot 7 — Arousal timeline (only when a model is supplied)
# ---------------------------------------------------------------------------


def plot_arousal_timeline(scored: pd.DataFrame, plots_dir: Path, meta: dict) -> None:
    """
    A dedicated arousal plot:
      - Top:   raw anomaly score over time, per source, with threshold
      - Mid:   normalised [0,1] score, smoothed, flagged windows shaded
      - Bottom: flag strip + peak markers
    """
    if scored is None or scored.empty:
        return

    valid = scored[scored["_score"].notna()].copy()
    if valid.empty:
        print("  [SKIP] No scored windows (model rejected all — likely missing features).")
        return

    valid["t_min"] = valid["window_start_time"] / 60.0
    valid = valid.sort_values("t_min").reset_index(drop=True)

    threshold = float(valid["_threshold"].iloc[0])
    n_flagged = int(valid["_is_aroused"].sum())
    duration_min = valid["t_min"].max()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Arousal Detection Timeline — {n_flagged} flagged window(s) over {duration_min:.1f} min",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3, 3, 1], hspace=0.35)

    # --- (a) Raw score by source ---
    ax = fig.add_subplot(gs[0])
    for src, colour in [("gsr", BLUE), ("hrv", RED)]:
        sub = valid[valid["source"] == src]
        if sub.empty:
            continue
        ax.plot(sub["t_min"], sub["_score"], "o-",
                color=colour, lw=1.5, ms=4, label=f"{src.upper()} score")
        # Highlight flagged points
        flagged = sub[sub["_is_aroused"] == True]  # noqa: E712
        if not flagged.empty:
            ax.scatter(flagged["t_min"], flagged["_score"],
                       s=80, facecolors="none", edgecolors=AROUSED,
                       lw=2, zorder=5)

    ax.axhline(threshold, color=AROUSED, ls="--", lw=1.5,
               label=f"Threshold ({threshold:.3f})")
    ax.axhspan(valid["_score"].min() - 0.02, threshold,
               color=AROUSED, alpha=0.06)
    ax.set_ylabel("Raw anomaly score\n(lower = more aroused)")
    ax.set_title("Raw Isolation Forest score — points below the dashed line are flagged")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # So "higher" on the plot = higher arousal

    # --- (b) Normalised score with smoothing ---
    ax = fig.add_subplot(gs[1])

    # Shade flagged windows
    shade_aroused(ax, valid, source=None)

    for src, colour in [("gsr", BLUE), ("hrv", RED)]:
        sub = valid[valid["source"] == src]
        if sub.empty:
            continue
        ax.plot(sub["t_min"], sub["_normalised"], "o-",
                color=colour, lw=1.2, ms=3, alpha=0.6, label=f"{src.upper()}")

    # Combined smoothed trend across both sources
    if len(valid) >= 5:
        window = max(3, min(7, len(valid) // 4))
        smoothed = valid["_normalised"].rolling(window=window, center=True, min_periods=1).mean()
        ax.plot(valid["t_min"], smoothed, color="black", lw=2.5,
                label=f"Smoothed ({window}-win rolling mean)")

    ax.axhline(0.5, color=GREY, ls=":", lw=1, label="Mid-point (0.5)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Normalised arousal\n(0 = baseline, 1 = peak)")
    ax.set_xlabel("Session time (minutes)")
    ax.set_title("Normalised arousal score — red shading marks flagged windows")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- (c) Flag strip + top 5 peak annotations ---
    ax = fig.add_subplot(gs[2])
    ax.set_xlim(0, duration_min * 1.02 if duration_min > 0 else 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Session time (minutes)")
    ax.set_title("Flagged window strip (↓ arrows mark top 5 peaks)")

    for _, row in valid.iterrows():
        if row["_is_aroused"]:
            src_colour = BLUE if row["source"] == "gsr" else RED
            ax.axvspan(row["t_min"] - 0.1, row["t_min"] + 0.1,
                       color=src_colour, alpha=0.6)

    # Top 5 peaks by normalised score
    flagged = valid[valid["_is_aroused"] == True]  # noqa: E712
    if not flagged.empty:
        top = flagged.nlargest(min(5, len(flagged)), "_normalised")
        for _, row in top.iterrows():
            ax.annotate(
                f"{int(row['t_min']):d}:{int((row['t_min'] % 1) * 60):02d}\n({row['source']})",
                xy=(row["t_min"], 0.9),
                xytext=(row["t_min"], 0.3),
                ha="center", va="top", fontsize=8,
                arrowprops=dict(arrowstyle="->", color=AROUSED, lw=1.2),
            )

    # Legend for the strip
    ax.legend(handles=[
        Patch(color=BLUE,  alpha=0.6, label="GSR flag"),
        Patch(color=RED,   alpha=0.6, label="HRV flag"),
    ], fontsize=8, loc="upper right")

    savefig(fig, plots_dir, "07_arousal_timeline.png")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _add_normal_band(ax, lo, hi, label):
    ax.axhspan(lo, hi, alpha=0.08, color=GREEN, label=label)
    ax.legend(fontsize=8, loc="upper right")


def print_summary(df, gsr, hrv, meta, scored: Optional[pd.DataFrame] = None):
    print("\n  ── Session Summary ──────────────────────────────")
    if meta:
        print(f"  Label      : {meta.get('label', '?')}")
        print(f"  Start      : {meta.get('start_time', '?')}")
        dur = meta.get('duration_s', 0)
        if isinstance(dur, (int, float)):
            print(f"  Duration   : {dur:.0f} s  ({dur / 60:.1f} min)")
    print(f"  Total rows : {len(df)}  (GSR: {len(gsr)}, HRV: {len(hrv)})")
    if not gsr.empty:
        print(f"  GSR SCL    : {gsr['gsr_scl_mean'].mean():.3f} µS  (mean across windows)")
        print(f"  GSR SCR/min: {gsr['gsr_scr_count'].mean():.2f}")
    if not hrv.empty:
        print(f"  HR mean    : {hrv['hrv_hr_mean'].mean():.1f} bpm")
        print(f"  RMSSD mean : {hrv['hrv_rmssd'].mean():.1f} ms")
        print(f"  pNN50 mean : {hrv['hrv_pnn50'].mean() * 100:.1f} %")
    if scored is not None and not scored.empty:
        n_flagged = int(scored["_is_aroused"].sum())
        n_scored  = int(scored["_score"].notna().sum())
        gsr_flagged = int(((scored["source"] == "gsr") & scored["_is_aroused"]).sum())
        hrv_flagged = int(((scored["source"] == "hrv") & scored["_is_aroused"]).sum())
        pct = 100 * n_flagged / n_scored if n_scored else 0
        print(f"  Arousal    : {n_flagged}/{n_scored} windows flagged ({pct:.1f}%)")
        print(f"               GSR: {gsr_flagged}   HRV: {hrv_flagged}")
    print("  ─────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------


def plot_session(session_dir: Path, model_path: Optional[Path] = None) -> None:
    """
    Entry point for a single session directory.
    Generates all plots and saves them to <session_dir>/plots/.
    If `model_path` is given, also runs arousal detection and adds overlays.
    """

    plots_dir = session_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\n[PLOT] Session: {session_dir.resolve()}")
    print(f"[PLOT] Output:  {plots_dir.resolve()}")
    if model_path:
        print(f"[PLOT] Model:   {model_path}")
    print()

    df, meta = load_session(session_dir)
    gsr, hrv = split_sources(df)

    scored: Optional[pd.DataFrame] = None
    if model_path is not None:
        print("[PLOT] Scoring session against model...")
        try:
            scored = score_session(df, model_path)
        except Exception as e:
            print(f"  [WARN] Scoring failed: {e}. Continuing without overlays.")
            scored = None

    print_summary(df, gsr, hrv, meta, scored)

    print("[PLOT] Generating plots...")
    plot_gsr_overview(gsr, plots_dir, meta, scored)
    plot_gsr_quality(gsr, plots_dir, meta, scored)
    plot_hrv_overview(hrv, plots_dir, meta, scored)
    plot_hrv_poincare(hrv, plots_dir, scored)
    plot_data_quality(df, gsr, hrv, plots_dir, meta)
    plot_combined(gsr, hrv, plots_dir, scored)
    if scored is not None:
        plot_arousal_timeline(scored, plots_dir, meta)

    print(
        f"\n[PLOT] Done — {len(list(plots_dir.glob('*.png')))} plots saved to {plots_dir}/"
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def plot_all_sessions(parent_dir: Path, model_path: Optional[Path] = None) -> None:
    """
    Find every sub-directory that contains features.csv and plot it.
    Plots are saved inside each session's own plots/ folder.
    """
    session_dirs = sorted(
        d for d in parent_dir.iterdir()
        if d.is_dir() and (d / "features.csv").exists()
    )

    if not session_dirs:
        sys.exit(
            f"[ERROR] No session directories (with features.csv) found in {parent_dir}"
        )

    print(f"\n[BATCH] Found {len(session_dirs)} session(s) in {parent_dir.resolve()}\n")
    for i, sd in enumerate(session_dirs, 1):
        print(f"{'─' * 60}")
        print(f"[BATCH] ({i}/{len(session_dirs)}) {sd.name}")
        print(f"{'─' * 60}")
        try:
            plot_session(sd, model_path=model_path)
        except Exception as e:
            print(f"  [ERROR] Failed on {sd.name}: {e}")

    print(f"\n[BATCH] Finished — processed {len(session_dirs)} session(s).\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot a session (with optional arousal overlays).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", type=Path,
                        help="Session directory, or parent containing sessions.")
    parser.add_argument("--model", type=Path, default=None,
                        help="Optional path to trained iforest.pkl for arousal overlays.")
    args = parser.parse_args()

    target = args.path
    if not target.is_dir():
        sys.exit(f"[ERROR] Not a directory: {target}")

    if args.model is not None and not args.model.exists():
        sys.exit(f"[ERROR] Model file not found: {args.model}")

    # Single session — features.csv lives directly in the given dir
    if (target / "features.csv").exists():
        plot_session(target, model_path=args.model)
    else:
        plot_all_sessions(target, model_path=args.model)


if __name__ == "__main__":
    main()
