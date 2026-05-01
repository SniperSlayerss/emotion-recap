from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)

VALENCE_CLASSES = ["negative", "neutral", "positive"]

# Sentiment-word fallback for P2-style descriptive outputs.
POS_WORDS = [
    "calm",
    "calmness",
    "tranquil",
    "tranquility",
    "peace",
    "peaceful",
    "serene",
    "serenity",
    "content",
    "contentment",
    "happy",
    "happiness",
    "joy",
    "relax",
    "relaxed",
    "comfort",
    "comfortable",
    "pleasant",
    "satisfaction",
    "satisfied",
    "appreciation",
    "wonder",
    "awe",
    "enjoyment",
    "enjoy",
    "productive",
    "productivity",
    "ease",
    "interest",
]
NEG_WORDS = [
    "anxious",
    "anxiety",
    "stress",
    "stressed",
    "frustrat",
    "fear",
    "afraid",
    "scared",
    "tense",
    "tension",
    "worried",
    "worry",
    "uncomfort",
    "distress",
    "anger",
    "angry",
    "sad",
    "sadness",
    "upset",
    "panic",
    "dread",
    "horror",
    "terror",
    "frighten",
    "unease",
    "nervous",
    "agitat",
    "alarm",
]

NEU_WORDS = [
    "neutral",
    "observ",
    "ordinary",
    "routine",
    "everyday",
    "mundane",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_valence_strict(text):
    if not isinstance(text, str):
        return None
    t = text.lower()
    hits = []
    for label in VALENCE_CLASSES:
        m = re.search(rf"\b{label}\b", t)
        if m:
            hits.append((m.start(), label))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def parse_valence_with_fallback(text):
    """Strict parser first; if it fails, fall back to sentiment-word voting."""
    strict = parse_valence_strict(text)
    if strict is not None:
        return strict, "strict"
    if not isinstance(text, str):
        return None, "none"
    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    neu = sum(1 for w in NEU_WORDS if w in t)
    if pos == 0 and neg == 0 and neu == 0:
        return None, "none"
    counts = {"negative": neg, "positive": pos, "neutral": neu}
    label = max(
        counts,
        key=lambda k: (counts[k], -["negative", "positive", "neutral"].index(k)),
    )
    return label, "fallback"


# ---------------------------------------------------------------------------
# Label inference
# ---------------------------------------------------------------------------


CATEGORY_VALENCE = {
    "horror": "negative",
    "classical": "neutral",
    "reading": "neutral",
    "walking": "neutral",
    "youtube": "neutral",
}

CATEGORY_INTENSITY = {
    "horror": 7,
    "classical": 4,
    "reading": 3,
    "walking": 4,
    "youtube": 4,
}


def category_of(clip: str):
    name = clip.lower()
    for cat in CATEGORY_VALENCE:
        if name.startswith(cat):
            return cat
    return None


def infer_labels(clips):
    """Build a labels DataFrame by matching clip name prefixes."""
    rows = []
    for c in clips:
        cat = category_of(c)
        rows.append(
            {
                "clip": c,
                "valence": CATEGORY_VALENCE.get(cat) if cat else None,
                "intensity": CATEGORY_INTENSITY.get(cat) if cat else None,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_latency(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="prompt_id", y="latency_s", ax=ax, color="#4C72B0")
    sns.stripplot(
        data=df, x="prompt_id", y="latency_s", ax=ax, color="black", size=3, alpha=0.6
    )
    ax.set_title("Inference latency per prompt")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Latency (s)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_per_prompt.png", dpi=200)
    plt.close(fig)


def plot_output_length(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x="prompt_id", y="n_tokens", ax=axes[0], color="#55A868")
    axes[0].set_title("Output length (tokens)")
    axes[0].tick_params(axis="x", rotation=20)
    sns.boxplot(data=df, x="prompt_id", y="n_chars", ax=axes[1], color="#C44E52")
    axes[1].set_title("Output length (characters)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "output_length.png", dpi=200)
    plt.close(fig)


def plot_consistency(df, out_dir):
    cv_df = (
        df[df.prompt_id != "P3_intensity_rating"]
        .groupby("prompt_id")["n_tokens"]
        .agg(["mean", "std"])
        .assign(cv=lambda d: d["std"] / d["mean"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=cv_df, x="prompt_id", y="cv", ax=ax, color="#8172B2")
    ax.set_title("Output-length consistency (P3 excluded — single-token output)")
    ax.set_ylabel("Coefficient of variation (tokens)")
    ax.set_xlabel("Prompt")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "consistency_cv.png", dpi=200)
    plt.close(fig)


def plot_valence(merged, out_dir):
    val_prompts = ["P1_open_emotional_state", "P2_valence_only", "P4_structured"]
    sub_all = merged[merged.prompt_id.isin(val_prompts)].copy()

    rows = []
    for pid, sub in sub_all.groupby("prompt_id"):
        sub = sub.dropna(subset=["pred_valence_v2"])
        if sub.empty:
            rows.append(
                {
                    "prompt_id": pid,
                    "accuracy": np.nan,
                    "kappa": np.nan,
                    "n": 0,
                    "n_strict": 0,
                    "n_fallback": 0,
                }
            )
            continue
        acc = accuracy_score(sub["valence"], sub["pred_valence_v2"])
        try:
            k = cohen_kappa_score(
                sub["valence"], sub["pred_valence_v2"], labels=VALENCE_CLASSES
            )
        except ValueError:
            k = np.nan
        rows.append(
            {
                "prompt_id": pid,
                "accuracy": acc,
                "kappa": k,
                "n": len(sub),
                "n_strict": (sub.valence_parse_method == "strict").sum(),
                "n_fallback": (sub.valence_parse_method == "fallback").sum(),
            }
        )
    agreement = pd.DataFrame(rows)
    agreement.to_csv(out_dir / "valence_agreement_v2.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agreement))
    w = 0.38
    ax.bar(x - w / 2, agreement["accuracy"], w, label="Accuracy", color="#4C72B0")
    ax.bar(x + w / 2, agreement["kappa"], w, label="Cohen's κ", color="#DD8452")
    for i, row in agreement.iterrows():
        if pd.notna(row.accuracy):
            ax.text(
                i,
                max(row.accuracy, row.kappa) + 0.04,
                f"n={int(row.n)}\n({int(row.n_strict)} strict, "
                f"{int(row.n_fallback)} fallback)",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(agreement["prompt_id"], rotation=20)
    ax.set_ylim(-0.3, 1.15)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("Valence agreement with ground-truth labels")
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "valence_agreement.png", dpi=200)
    plt.close(fig)

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, pid in zip(axes, val_prompts):
        sub = sub_all[sub_all.prompt_id == pid].dropna(subset=["pred_valence_v2"])
        if sub.empty:
            ax.set_title(f"{pid}\n(no parseable preds)")
            ax.axis("off")
            continue
        cm = confusion_matrix(
            sub["valence"], sub["pred_valence_v2"], labels=VALENCE_CLASSES
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=VALENCE_CLASSES,
            yticklabels=VALENCE_CLASSES,
            ax=ax,
            cbar=False,
        )
        ax.set_title(pid)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / "valence_confusion_matrices.png", dpi=200)
    plt.close(fig)

    return agreement


def plot_intensity(df, labels, out_dir):
    if "intensity" not in labels.columns:
        print("No intensity column in labels — skipping intensity plot.")
        return None
    int_sub = (
        df[df.prompt_id == "P3_intensity_rating"]
        .merge(labels[["clip", "intensity"]], on="clip", how="inner")
        .dropna(subset=["pred_intensity", "intensity"])
    )
    if int_sub.empty:
        print("No matched P3 + intensity rows — skipping intensity plot.")
        return None

    mae = mean_absolute_error(int_sub["intensity"], int_sub["pred_intensity"])
    r = int_sub[["intensity", "pred_intensity"]].corr().iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    rng = np.random.default_rng(0)
    jx = int_sub["intensity"] + rng.uniform(-0.12, 0.12, len(int_sub))
    jy = int_sub["pred_intensity"] + rng.uniform(-0.12, 0.12, len(int_sub))
    ax.scatter(jx, jy, color="#4C72B0", s=70, alpha=0.85, edgecolor="white")
    ax.plot([0, 10], [0, 10], "--", color="grey", label="perfect agreement")
    z = np.polyfit(int_sub["intensity"], int_sub["pred_intensity"], 1)
    xs = np.linspace(0, 10, 50)
    ax.plot(xs, np.polyval(z, xs), color="#C44E52", lw=1.5, label="best fit")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Ground-truth intensity")
    ax.set_ylabel("Model intensity (P3)")
    ax.set_title(f"Intensity agreement   MAE={mae:.2f}   r={r:.2f}   n={len(int_sub)}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_agreement.png", dpi=200)
    plt.close(fig)

    summary = pd.DataFrame([{"mae": mae, "pearson_r": r, "n": len(int_sub)}])
    summary.to_csv(out_dir / "intensity_agreement_v2.csv", index=False)
    return summary


def plot_latency_vs_length(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for pid, sub in df.groupby("prompt_id"):
        ax.scatter(sub["n_tokens"], sub["latency_s"], label=pid, s=40, alpha=0.7)
    ax.set_xlabel("Tokens generated")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency vs output length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_length.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results_dir",
        required=True,
        help="Folder containing raw_results.csv from the first stage",
    )
    ap.add_argument(
        "--labels",
        default=None,
        help="Optional CSV with clip,valence[,intensity]. "
        "If omitted, looks for labels.csv in --results_dir, "
        "otherwise infers labels from clip-name prefixes.",
    )
    ap.add_argument(
        "--out_subdir",
        default="reanalysis",
        help="Subdirectory under --results_dir for new outputs (default: reanalysis)",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    raw_csv = results_dir / "raw_results.csv"
    if not raw_csv.exists():
        raise SystemExit(
            f"Could not find {raw_csv}. "
            f"Pass the folder produced by evaluate_smolvlm2.py."
        )

    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {out_dir}")

    df = pd.read_csv(raw_csv)
    print(f"Loaded {len(df)} rows, prompts: {sorted(df.prompt_id.unique())}")

    # Re-parse valence with fallback
    parsed = df["output_text"].apply(parse_valence_with_fallback)
    df["pred_valence_v2"] = parsed.apply(lambda x: x[0])
    df["valence_parse_method"] = parsed.apply(lambda x: x[1])

    # Labels: explicit file > labels.csv next to raw_results > prefix inference
    if args.labels:
        labels_path = Path(args.labels)
    else:
        candidate = results_dir / "labels.csv"
        labels_path = candidate if candidate.exists() else None

    if labels_path and labels_path.exists():
        print(f"Using labels from: {labels_path}")
        labels = pd.read_csv(labels_path)
    else:
        print("No labels file, inferring from clip filename prefixes.")
        labels = infer_labels(df["clip"].unique().tolist())
        labels = labels.dropna(subset=["valence"])
        print(f"Inferred labels for {len(labels)} / {df['clip'].nunique()} clips")

    labels["valence"] = labels["valence"].astype(str).str.lower().str.strip()

    print("\nGround-truth valence distribution:")
    print(labels["valence"].value_counts().to_string())

    merged = df.merge(labels[["clip", "valence"]], on="clip", how="inner")

    sns.set_theme(style="whitegrid")
    plot_latency(df, out_dir)
    plot_output_length(df, out_dir)
    plot_consistency(df, out_dir)
    agreement = plot_valence(merged, out_dir)
    intensity_summary = plot_intensity(df, labels, out_dir)
    plot_latency_vs_length(df, out_dir)

    df.to_csv(out_dir / "raw_results_v2.csv", index=False)

    print("\nValence agreement (with fallback parser):")
    print(agreement.to_string(index=False))
    if intensity_summary is not None:
        print("\nIntensity agreement:")
        print(intensity_summary.to_string(index=False))

    n_files = len(list(out_dir.iterdir()))
    print(f"\nWrote {n_files} files to {out_dir}")


if __name__ == "__main__":
    main()
