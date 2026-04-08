"""
plot_session.py

Generate diagnostic plots for a recorded session.

Usage:
    python plot_session.py <session_dir>

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
"""

import sys
import json
from pathlib import Path

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

BLUE   = "#4C78A8"
ORANGE = "#F58518"
GREEN  = "#54A24B"
RED    = "#E45756"
PURPLE = "#B279A2"
GREY   = "#9D9D9D"

def load_session(session_dir: Path) -> tuple[pd.DataFrame, dict]:
    csv_path  = session_dir / "features.csv"
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
# Plot 1 — GSR overview
# ---------------------------------------------------------------------------

def plot_gsr_overview(gsr: pd.DataFrame, plots_dir: Path, meta: dict) -> None:
    if gsr.empty:
        print("  [SKIP] No GSR rows found.")
        return

    t = time_col(gsr)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("GSR Overview — Skin Conductance Level & Responses", fontsize=14, fontweight="bold")

    # --- SCL ---
    ax = axes[0]
    ax.plot(t, gsr["gsr_scl_mean"], color=BLUE, lw=2, label="SCL mean (µS)")
    ax.fill_between(
        t,
        gsr["gsr_scl_mean"] - gsr["gsr_scl_std"],
        gsr["gsr_scl_mean"] + gsr["gsr_scl_std"],
        alpha=0.2, color=BLUE, label="±1 SD"
    )
    ax.set_ylabel("Skin Conductance\nLevel (µS)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _add_normal_band(ax, 0.5, 20.0, "Typical range (0.5–20 µS)")

    # --- SCR count ---
    ax = axes[1]
    ax.bar(t, gsr["gsr_scr_count"], width=0.4, color=ORANGE, alpha=0.8, label="SCR count / min")
    ax.set_ylabel("SCR Count\n(per minute)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Phasic std ---
    ax = axes[2]
    ax.plot(t, gsr["gsr_phasic_std"], color=GREEN, lw=2, label="Phasic std (µS)")
    ax.plot(t, gsr["gsr_phasic_mean"], color=GREEN, lw=1.5, ls="--", alpha=0.6, label="Phasic mean (µS)")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_ylabel("Phasic Component\n(µS)")
    ax.set_xlabel("Session time (minutes)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, plots_dir, "01_gsr_overview.png")


# ---------------------------------------------------------------------------
# Plot 2 — GSR quality
# ---------------------------------------------------------------------------

def plot_gsr_quality(gsr: pd.DataFrame, plots_dir: Path, meta: dict) -> None:
    if gsr.empty:
        return

    t = time_col(gsr)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("GSR Data Quality — SCR Timing & Amplitude", fontsize=14, fontweight="bold")

    # SCR amplitude
    ax = axes[0]
    ax.bar(t, gsr["gsr_scr_mean_amp"], width=0.4, color=PURPLE, alpha=0.85)
    ax.set_ylabel("Mean SCR Amplitude\n(µS)")
    ax.axhline(0.02, color=RED, ls="--", lw=1.2, label="Detection threshold (0.02 µS)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # SCR rise time
    ax = axes[1]
    ax.plot(t, gsr["gsr_scr_rise_time"], "o-", color=ORANGE, lw=1.5, ms=5, label="Rise time (s)")
    ax.set_ylabel("Mean SCR Rise Time (s)")
    _add_normal_band(ax, 1.0, 3.0, "Typical 1–3 s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # SCR recovery time
    ax = axes[2]
    ax.plot(t, gsr["gsr_scr_recovery_time"], "o-", color=BLUE, lw=1.5, ms=5, label="Half-recovery time (s)")
    ax.set_ylabel("Mean SCR Recovery Time (s)")
    _add_normal_band(ax, 5.0, 30.0, "Typical 5–30 s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Session time (minutes)")

    fig.tight_layout()
    savefig(fig, plots_dir, "02_gsr_quality.png")


# ---------------------------------------------------------------------------
# Plot 3 — HRV overview
# ---------------------------------------------------------------------------

def plot_hrv_overview(hrv: pd.DataFrame, plots_dir: Path, meta: dict) -> None:
    if hrv.empty:
        print("  [SKIP] No HRV rows found.")
        return

    t = time_col(hrv)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("HRV Overview — Heart Rate & Variability", fontsize=14, fontweight="bold")

    # HR band
    ax = axes[0]
    ax.fill_between(t, hrv["hrv_hr_min"], hrv["hrv_hr_max"], alpha=0.15, color=RED, label="HR range (min–max)")
    ax.plot(t, hrv["hrv_hr_mean"], color=RED, lw=2, label="HR mean (bpm)")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # RMSSD
    ax = axes[1]
    ax.plot(t, hrv["hrv_rmssd"], color=BLUE, lw=2, label="RMSSD (ms)")
    ax.plot(t, hrv["hrv_sdnn"],  color=ORANGE, lw=2, ls="--", label="SDNN (ms)")
    ax.set_ylabel("HRV (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _add_normal_band(ax, 20.0, 80.0, "Healthy RMSSD range")

    # pNN50
    ax = axes[2]
    ax.plot(t, hrv["hrv_pnn50"] * 100, color=GREEN, lw=2, label="pNN50 (%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("pNN50 (%)")
    ax.set_xlabel("Session time (minutes)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, plots_dir, "03_hrv_overview.png")


# ---------------------------------------------------------------------------
# Plot 4 — HRV Poincaré scatter
# ---------------------------------------------------------------------------

def plot_hrv_poincare(hrv: pd.DataFrame, plots_dir: Path) -> None:
    if hrv.empty or len(hrv) < 2:
        return

    t = time_col(hrv)
    cmap = plt.cm.viridis
    norm = plt.Normalize(t.min(), t.max())
    colours = cmap(norm(t))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("HRV Poincaré Analysis — SD1 vs SD2", fontsize=14, fontweight="bold")

    # Scatter SD1 vs SD2
    ax = axes[0]
    sc = ax.scatter(hrv["hrv_sd1"], hrv["hrv_sd2"], c=t, cmap="viridis", s=60, zorder=3)
    ax.set_xlabel("SD1 — Short-term variability (ms)")
    ax.set_ylabel("SD2 — Long-term variability (ms)")
    ax.set_title("SD1 vs SD2 (colour = session time)")
    cbar = fig.colorbar(sc, ax=ax, label="Time (min)")
    ax.grid(True, alpha=0.3)

    # SD1 and SD2 over time
    ax = axes[1]
    ax.plot(t, hrv["hrv_sd1"], color=BLUE,   lw=2, label="SD1 (short-term)")
    ax.plot(t, hrv["hrv_sd2"], color=ORANGE,  lw=2, label="SD2 (long-term)")
    ax.set_xlabel("Session time (minutes)")
    ax.set_ylabel("SD (ms)")
    ax.set_title("SD1 & SD2 over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, plots_dir, "04_hrv_poincare.png")


# ---------------------------------------------------------------------------
# Plot 5 — Data quality dashboard
# ---------------------------------------------------------------------------

def plot_data_quality(df: pd.DataFrame, gsr: pd.DataFrame, hrv: pd.DataFrame,
                      plots_dir: Path, meta: dict) -> None:
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Data Quality Dashboard", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # --- 1. Window count bar ---
    ax = fig.add_subplot(gs[0, 0])
    counts = df["source"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[BLUE if s == "gsr" else RED for s in counts.index],
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
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
    hrv_cols  = [c for c in df.columns if c.startswith("hrv_")]
    heat_cols = gsr_cols[:8] + hrv_cols[:8]  # Keep it readable
    if heat_cols:
        heat = df[heat_cols].isnull().astype(int)
        im = ax.imshow(heat.T, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_yticks(range(len(heat_cols)))
        ax.set_yticklabels([c.replace("gsr_", "G:").replace("hrv_", "H:") for c in heat_cols],
                           fontsize=7)
        ax.set_xlabel("Feature window index")
        ax.set_title("Missing Values (green = present, red = absent)")
        fig.colorbar(im, ax=ax, ticks=[0, 1], label="Missing")

    # --- 4. Inter-window gap distribution ---
    ax = fig.add_subplot(gs[1, 2])
    for src, col, c in [("gsr", gsr, BLUE), ("hrv", hrv, RED)]:
        if len(col) > 1:
            gaps = np.diff(col["window_start_time"].values) / 60.0
            ax.hist(gaps, bins=15, alpha=0.6, color=c, label=src.upper(), edgecolor="white")
    ax.set_xlabel("Gap between windows (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-window Gap Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    savefig(fig, plots_dir, "05_data_quality.png")


# ---------------------------------------------------------------------------
# Plot 6 — Combined arousal proxy
# ---------------------------------------------------------------------------

def plot_combined(gsr: pd.DataFrame, hrv: pd.DataFrame, plots_dir: Path) -> None:
    if gsr.empty and hrv.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Combined GSR + HRV — Arousal Proxy Overview", fontsize=14, fontweight="bold")

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
        ax2.plot(th, hrv["hrv_hr_mean"], color=RED, lw=2, ls="--", label="HR mean (bpm)")
        ax2.set_ylabel("HR (bpm)", color=RED)
        ax2.tick_params(axis="y", labelcolor=RED)
    ax1.set_title("Skin Conductance Level vs Heart Rate")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # SCR count + RMSSD
    ax = axes[1]
    if not gsr.empty:
        ax.bar(tg, gsr["gsr_scr_count"], width=0.35, color=ORANGE, alpha=0.7, label="SCR count/min (GSR)")
    ax3 = ax.twinx()
    if not hrv.empty:
        ax3.plot(th, hrv["hrv_rmssd"], color=PURPLE, lw=2, label="RMSSD (ms)")
        ax3.set_ylabel("RMSSD (ms)", color=PURPLE)
        ax3.tick_params(axis="y", labelcolor=PURPLE)
    ax.set_ylabel("SCR count/min", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title("SCR Rate vs RMSSD (both ↑ = higher arousal from sympathetic drive)")
    ax.grid(True, alpha=0.3, axis="y")
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax3.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper right")

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
    lines_c, labels_c = ax.get_legend_handles_labels()
    lines_d, labels_d = ax4.get_legend_handles_labels()
    ax.legend(lines_c + lines_d, labels_c + labels_d, fontsize=8, loc="upper right")

    fig.tight_layout()
    savefig(fig, plots_dir, "06_combined_arousal.png")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _add_normal_band(ax, lo, hi, label):
    ax.axhspan(lo, hi, alpha=0.08, color=GREEN, label=label)
    ax.legend(fontsize=8, loc="upper right")


def print_summary(df, gsr, hrv, meta):
    print("\n  ── Session Summary ──────────────────────────────")
    if meta:
        print(f"  Label      : {meta.get('label', '?')}")
        print(f"  Start      : {meta.get('start_time', '?')}")
        print(f"  Duration   : {meta.get('duration_s', '?'):.0f} s  ({meta.get('duration_s', 0)/60:.1f} min)")
    print(f"  Total rows : {len(df)}  (GSR: {len(gsr)}, HRV: {len(hrv)})")
    if not gsr.empty:
        print(f"  GSR SCL    : {gsr['gsr_scl_mean'].mean():.3f} µS  (mean across windows)")
        print(f"  GSR SCR/min: {gsr['gsr_scr_count'].mean():.2f}")
    if not hrv.empty:
        print(f"  HR mean    : {hrv['hrv_hr_mean'].mean():.1f} bpm")
        print(f"  RMSSD mean : {hrv['hrv_rmssd'].mean():.1f} ms")
        print(f"  pNN50 mean : {hrv['hrv_pnn50'].mean()*100:.1f} %")
    print("  ─────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_session.py <session_dir>")

    session_dir = Path(sys.argv[1])
    if not session_dir.is_dir():
        sys.exit(f"[ERROR] Not a directory: {session_dir}")

    plots_dir = session_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\n[PLOT] Session: {session_dir.resolve()}")
    print(f"[PLOT] Output:  {plots_dir.resolve()}\n")

    df, meta = load_session(session_dir)
    gsr, hrv = split_sources(df)

    print_summary(df, gsr, hrv, meta)

    print("[PLOT] Generating plots...")
    plot_gsr_overview(gsr, plots_dir, meta)
    plot_gsr_quality(gsr, plots_dir, meta)
    plot_hrv_overview(hrv, plots_dir, meta)
    plot_hrv_poincare(hrv, plots_dir)
    plot_data_quality(df, gsr, hrv, plots_dir, meta)
    plot_combined(gsr, hrv, plots_dir)

    print(f"\n[PLOT] Done — {len(list(plots_dir.glob('*.png')))} plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
