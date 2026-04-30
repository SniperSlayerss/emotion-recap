"""
SmolVLM2 prompt evaluation across multiple body-cam clips.

For each (prompt, clip) pair this script:
  - runs the model once
  - records latency, output text, output length (chars + tokens)
  - parses a valence label and intensity rating where applicable

It then compares predictions against retrospective ground-truth labels
(valence + intensity) provided in a CSV, and produces dissertation-ready
plots plus a results CSV.

Expected ground-truth CSV (--labels):
    clip,valence,intensity
    clip01.mp4,positive,7
    clip02.mp4,neutral,3
    ...

Usage:
    python evaluate_smolvlm2.py \
        --clips_dir vlm/clips \
        --labels vlm/labels.csv \
        --out_dir results/
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)
from transformers import AutoModelForImageTextToText, AutoProcessor


# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

PROMPTS: dict[str, str] = {
    "P1_open_emotional_state": (
        "Describe the likely emotional state of the person wearing this camera, "
        "focusing on whether their experience appears positive, negative, or neutral, "
        "and how intense it seems."
    ),
    "P2_valence_only": (
        "Describe the emotional valence the camera wearer would be experiencing"
    ),
    "P3_intensity_rating": (
        "Rate on a scale of 1-10 how emotional the camera wearer is"
    ),
    "P4_structured": (
        "You are watching a 30-second clip recorded from a body-worn camera. "
        "Based on the visual scene, answer in this format:\n"
        "Likely activity: [one phrase]\n"
        "Emotional valence: [positive / neutral / negative]\n"
        "Confidence: [low / medium / high]\n"
        "One-sentence justification."
    ),
}

VALENCE_CLASSES = ["negative", "neutral", "positive"]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    clip: str
    prompt_id: str
    output_text: str
    latency_s: float
    n_chars: int
    n_tokens: int
    pred_valence: Optional[str]
    pred_intensity: Optional[float]
    pred_confidence: Optional[str]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_valence(text: str) -> Optional[str]:
    """Pick the first valence keyword that appears."""
    t = text.lower()
    # Search for whole words; preference order doesn't actually matter because
    # we take the earliest occurrence.
    hits: list[tuple[int, str]] = []
    for label in VALENCE_CLASSES:
        m = re.search(rf"\b{label}\b", t)
        if m:
            hits.append((m.start(), label))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def parse_intensity(text: str) -> Optional[float]:
    """Extract a 1-10 rating. Accepts '7', '7/10', '7 out of 10', etc."""
    # Try X/10 or X out of 10 first
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*10", text, flags=re.I)
    if m:
        v = float(m.group(1))
        return v if 0 <= v <= 10 else None
    # Otherwise grab the first standalone integer 1-10
    for m in re.finditer(r"\b(\d{1,2})\b", text):
        v = float(m.group(1))
        if 1 <= v <= 10:
            return v
    return None


def parse_confidence(text: str) -> Optional[str]:
    m = re.search(r"confidence\s*[:\-]\s*(low|medium|high)", text, flags=re.I)
    return m.group(1).lower() if m else None


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------


def build_messages(video_path: str, prompt_text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def run_clip_prompt(
    model,
    processor,
    device: torch.device,
    dtype: torch.dtype,
    clip_path: str,
    prompt_id: str,
    prompt_text: str,
    max_new_tokens: int = 128,
) -> RunResult:
    messages = build_messages(clip_path, prompt_text)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=dtype)

    input_len = inputs["input_ids"].shape[-1]

    t0 = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    new_token_ids = generated_ids[0, input_len:]
    text = processor.decode(new_token_ids, skip_special_tokens=True).strip()

    result = RunResult(
        clip=os.path.basename(clip_path),
        prompt_id=prompt_id,
        output_text=text,
        latency_s=round(latency, 3),
        n_chars=len(text),
        n_tokens=int(new_token_ids.shape[-1]),
        pred_valence=parse_valence(text),
        pred_intensity=parse_intensity(text) if prompt_id == "P3_intensity_rating" else parse_intensity(text),
        pred_confidence=parse_confidence(text),
    )

    # Free per-iteration tensors
    del inputs, generated_ids, new_token_ids
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_latency(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="prompt_id", y="latency_s", ax=ax, color="#4C72B0")
    sns.stripplot(
        data=df, x="prompt_id", y="latency_s",
        ax=ax, color="black", size=3, alpha=0.6,
    )
    ax.set_title("Inference latency per prompt")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Latency (s)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_per_prompt.png", dpi=200)
    plt.close(fig)


def plot_output_length(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x="prompt_id", y="n_tokens", ax=axes[0], color="#55A868")
    axes[0].set_title("Output length (tokens)")
    axes[0].set_xlabel("Prompt")
    axes[0].set_ylabel("Tokens generated")
    axes[0].tick_params(axis="x", rotation=20)

    sns.boxplot(data=df, x="prompt_id", y="n_chars", ax=axes[1], color="#C44E52")
    axes[1].set_title("Output length (characters)")
    axes[1].set_xlabel("Prompt")
    axes[1].set_ylabel("Characters")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_dir / "output_length.png", dpi=200)
    plt.close(fig)


def plot_consistency(df: pd.DataFrame, out_dir: Path) -> None:
    """Coefficient of variation of output length per prompt, as a proxy for
    response-shape consistency across clips."""
    stats = (
        df.groupby("prompt_id")["n_tokens"]
        .agg(["mean", "std"])
        .assign(cv=lambda d: d["std"] / d["mean"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=stats, x="prompt_id", y="cv", ax=ax, color="#8172B2")
    ax.set_title("Output-length consistency (lower = more consistent)")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Coefficient of variation (tokens)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "consistency_cv.png", dpi=200)
    plt.close(fig)


def plot_valence_agreement(
    df: pd.DataFrame, labels: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    merged = df.merge(labels, on="clip", how="inner")
    rows = []
    for prompt_id, sub in merged.groupby("prompt_id"):
        sub = sub.dropna(subset=["pred_valence"])
        if sub.empty:
            rows.append({"prompt_id": prompt_id, "accuracy": np.nan, "kappa": np.nan, "n": 0})
            continue
        acc = accuracy_score(sub["valence"], sub["pred_valence"])
        try:
            kappa = cohen_kappa_score(
                sub["valence"], sub["pred_valence"], labels=VALENCE_CLASSES
            )
        except ValueError:
            kappa = np.nan
        rows.append(
            {"prompt_id": prompt_id, "accuracy": acc, "kappa": kappa, "n": len(sub)}
        )
    agreement = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agreement))
    w = 0.38
    ax.bar(x - w / 2, agreement["accuracy"], w, label="Accuracy", color="#4C72B0")
    ax.bar(x + w / 2, agreement["kappa"], w, label="Cohen's κ", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(agreement["prompt_id"], rotation=20)
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("Valence agreement with retrospective labels")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "valence_agreement.png", dpi=200)
    plt.close(fig)

    # Per-prompt confusion matrices
    n_prompts = merged["prompt_id"].nunique()
    fig, axes = plt.subplots(1, n_prompts, figsize=(4 * n_prompts, 4), squeeze=False)
    for ax, (prompt_id, sub) in zip(axes[0], merged.groupby("prompt_id")):
        sub = sub.dropna(subset=["pred_valence"])
        if sub.empty:
            ax.set_title(f"{prompt_id}\n(no parseable preds)")
            ax.axis("off")
            continue
        cm = confusion_matrix(
            sub["valence"], sub["pred_valence"], labels=VALENCE_CLASSES
        )
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=VALENCE_CLASSES, yticklabels=VALENCE_CLASSES, ax=ax, cbar=False,
        )
        ax.set_title(prompt_id)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / "valence_confusion_matrices.png", dpi=200)
    plt.close(fig)

    return agreement


def plot_intensity_agreement(
    df: pd.DataFrame, labels: pd.DataFrame, out_dir: Path
) -> Optional[pd.DataFrame]:
    if "intensity" not in labels.columns:
        return None
    sub = (
        df[df["prompt_id"] == "P3_intensity_rating"]
        .merge(labels[["clip", "intensity"]], on="clip", how="inner")
        .dropna(subset=["pred_intensity", "intensity"])
    )
    if sub.empty:
        return None

    mae = mean_absolute_error(sub["intensity"], sub["pred_intensity"])
    corr = sub[["intensity", "pred_intensity"]].corr().iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub["intensity"], sub["pred_intensity"], color="#4C72B0", s=60)
    ax.plot([0, 10], [0, 10], "--", color="grey")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Retrospective intensity")
    ax.set_ylabel("Model intensity (P3)")
    ax.set_title(f"Intensity agreement  MAE={mae:.2f}  r={corr:.2f}  n={len(sub)}")
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_agreement.png", dpi=200)
    plt.close(fig)

    return pd.DataFrame([{"mae": mae, "pearson_r": corr, "n": len(sub)}])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", required=True, help="Folder of .mp4 clips")
    ap.add_argument("--labels", required=True, help="CSV with clip,valence,intensity")
    ap.add_argument("--out_dir", default="results", help="Where to save outputs")
    ap.add_argument("--model_path", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device)
    model.eval()

    clips = sorted(
        p for p in Path(args.clips_dir).iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
    )
    if not clips:
        raise SystemExit(f"No video clips found in {args.clips_dir}")
    print(f"Found {len(clips)} clips, {len(PROMPTS)} prompts → "
          f"{len(clips) * len(PROMPTS)} runs")

    results: list[RunResult] = []
    for clip in clips:
        for prompt_id, prompt_text in PROMPTS.items():
            print(f"  {clip.name} | {prompt_id} ...", end=" ", flush=True)
            try:
                r = run_clip_prompt(
                    model, processor, device, dtype,
                    str(clip), prompt_id, prompt_text,
                    max_new_tokens=args.max_new_tokens,
                )
                results.append(r)
                print(f"{r.latency_s:.2f}s, {r.n_tokens} tok")
            except Exception as e:  # keep going if one clip fails
                print(f"FAILED ({e})")

    # Persist raw results
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(out_dir / "raw_results.csv", index=False)
    print(f"\nSaved {len(df)} rows → {out_dir/'raw_results.csv'}")

    # Load labels and normalise
    labels = pd.read_csv(args.labels)
    labels["clip"] = labels["clip"].astype(str)
    labels["valence"] = labels["valence"].str.lower().str.strip()

    # Plots
    sns.set_theme(style="whitegrid")
    plot_latency(df, out_dir)
    plot_output_length(df, out_dir)
    plot_consistency(df, out_dir)
    valence_summary = plot_valence_agreement(df, labels, out_dir)
    intensity_summary = plot_intensity_agreement(df, labels, out_dir)

    valence_summary.to_csv(out_dir / "valence_agreement.csv", index=False)
    if intensity_summary is not None:
        intensity_summary.to_csv(out_dir / "intensity_agreement.csv", index=False)

    # Summary table for the dissertation
    summary = (
        df.groupby("prompt_id")
        .agg(
            n=("clip", "count"),
            mean_latency_s=("latency_s", "mean"),
            std_latency_s=("latency_s", "std"),
            mean_tokens=("n_tokens", "mean"),
            std_tokens=("n_tokens", "std"),
        )
        .round(3)
        .reset_index()
        .merge(valence_summary, on="prompt_id", how="left")
    )
    summary.to_csv(out_dir / "summary_per_prompt.csv", index=False)
    print("\n=== Summary per prompt ===")
    print(summary.to_string(index=False))
    print(f"\nAll outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
