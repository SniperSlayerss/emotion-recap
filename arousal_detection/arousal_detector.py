"""
arousal_detector.py

Inference wrappers for the trained Isolation Forest models.

Two detector types are supported:

    ArousalDetector
        Single-source model (GSR-only or HRV-only). Loads a sklearn Pipeline
        and a metadata JSON and scores feature dicts of one kind.

    EnsembleDetector
        Two ArousalDetectors (one per source) with configurable combination
        ('any', 'all', 'mean'). Scores each source independently — no
        GSR/HRV window merging required.

Use the `load_detector()` factory to load either artifact type from a path;
it inspects the metadata and returns the right class.

Both classes are thread-safe (inference happens under a lock) so they can
be called from the GSR thread and the HRV BLE coroutine in parallel.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import joblib
except ImportError:
    raise ImportError("Install joblib: pip install joblib")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ArousalResult:
    """
    Result from one single-source inference call.

    score       : Raw anomaly score from IsolationForest.score_samples().
                  More negative = more anomalous. Typical range: [-0.7, 0.0]
    normalised  : Score re-mapped to [0, 1] using training distribution.
    is_aroused  : True if score < threshold.
    source      : 'gsr' or 'hrv' — which model produced this score.
    """
    score:         float
    normalised:    float
    is_aroused:    bool
    threshold:     float
    source:        str = "unknown"
    features_used: list[str] = field(default_factory=list)
    missing:       list[str] = field(default_factory=list)

    def summary(self) -> str:
        flag = "AROUSED" if self.is_aroused else "baseline"
        return (
            f"[{self.source}] {flag} | score={self.score:.4f} "
            f"(norm={self.normalised:.2f}, thresh={self.threshold:.4f})"
        )


@dataclass
class EnsembleResult:
    """
    Result from one ensemble inference call.

    gsr, hrv  : Per-source results. Either may be None if that source's
                features weren't supplied this window.
    combined_normalised : Combined score in [0, 1] per the configured mode.
    is_aroused          : Final flag after combining per-source flags.
    mode                : 'any' | 'all' | 'mean' — how the decision was made.
    """
    gsr:                  Optional[ArousalResult]
    hrv:                  Optional[ArousalResult]
    combined_normalised:  float
    is_aroused:           bool
    mode:                 str

    def summary(self) -> str:
        parts = []
        if self.gsr is not None:
            parts.append(self.gsr.summary())
        if self.hrv is not None:
            parts.append(self.hrv.summary())
        flag = "AROUSED" if self.is_aroused else "baseline"
        head = (
            f"[ensemble:{self.mode}] {flag} "
            f"combined_norm={self.combined_normalised:.2f}"
        )
        return head + " | " + " | ".join(parts) if parts else head


# ---------------------------------------------------------------------------
# Single-source detector
# ---------------------------------------------------------------------------


class ArousalDetector:
    """
    Load one trained Isolation Forest and run live inference.

    A detector is tied to a single feature source (GSR or HRV). Use the
    EnsembleDetector wrapper if you have separate GSR and HRV models.
    """

    def __init__(self, model_path: Union[str, Path]):
        model_path = Path(model_path)
        meta_path  = model_path.with_name(model_path.stem + "_meta.json")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        self._pipe = joblib.load(model_path)
        self._lock = threading.Lock()

        with open(meta_path) as f:
            meta = json.load(f)

        self.feature_cols:  list[str] = meta["feature_cols"]
        self.feature_names: list[str] = meta["feature_names"]
        self.threshold:     float     = meta["threshold"]
        self.source:        str       = meta.get("source", "unknown")
        self._score_std:    float     = meta["score_std"]
        self._score_min:    float     = meta["score_min"]
        self._score_max:    float     = meta["score_max"]

        print(
            f"[DETECTOR] Loaded {self.source} model from {model_path}\n"
            f"           Features  : {self.feature_cols}\n"
            f"           Threshold : {self.threshold:.4f}\n"
            f"           Trained on: {meta['n_training_windows']} windows "
            f"({meta.get('trained_at', '?')})"
        )

        self._recent_scores: list[float] = []
        self._max_recent = 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        features: dict,
        source: Optional[str] = None,
    ) -> Optional[ArousalResult]:
        """
        Score one feature window.

        `features` accepts both shapes:
          - Bare keys ('scl_mean', 'rmssd') — pass `source` so we can
            prefix them to match feature_cols.
          - Prefixed keys ('gsr_scl_mean', 'hrv_rmssd') — `source` ignored.

        Returns None if more than half of expected features are missing.
        """
        source = source or self.source
        normalised = self._normalise_keys(features, source)

        present = [c for c in self.feature_cols
                   if c in normalised and normalised[c] is not None]
        missing = [c for c in self.feature_cols if c not in present]

        if len(present) < max(1, len(self.feature_cols) // 2):
            return None

        x = np.array([[
            float(normalised.get(c, 0.0) or 0.0)
            for c in self.feature_cols
        ]])

        with self._lock:
            raw_score = float(self._pipe.score_samples(x)[0])
            self._recent_scores.append(raw_score)
            if len(self._recent_scores) > self._max_recent:
                self._recent_scores.pop(0)

        return ArousalResult(
            score=raw_score,
            normalised=self._normalise_score(raw_score),
            is_aroused=raw_score < self.threshold,
            threshold=self.threshold,
            source=self.source,
            features_used=present,
            missing=missing,
        )

    def trend(self) -> Optional[str]:
        """Simple slope over the last ~10 windows: 'rising', 'falling', 'stable'."""
        with self._lock:
            recent = list(self._recent_scores)

        if len(recent) < 4:
            return None

        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if abs(slope) < 0.005:
            return "stable"
        # Score falls → arousal rises
        return "falling" if slope < 0 else "rising"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalise_keys(self, features: dict, source: str) -> dict:
        """Accept both prefixed ('gsr_scl_mean') and bare ('scl_mean') keys."""
        out = {}
        for k, v in features.items():
            if k in self.feature_cols:
                out[k] = v
            else:
                prefixed = f"{source}_{k}"
                if prefixed in self.feature_cols:
                    out[prefixed] = v
        return out

    def _normalise_score(self, score: float) -> float:
        """Map raw score to [0, 1] using the training distribution."""
        lo = self._score_min - self._score_std
        hi = self._score_max
        if hi == lo:
            return 0.0
        normalised = (hi - score) / (hi - lo)
        return float(np.clip(normalised, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Ensemble detector
# ---------------------------------------------------------------------------


COMBINE_MODES = ("any", "all", "mean")


class EnsembleDetector:
    """
    Combines an independent GSR detector and HRV detector into one decision.

    Combination modes:
        'any'  : flag if either source flags (most sensitive)
        'all'  : flag only if both sources flag (most specific)
        'mean' : flag if the mean of normalised scores crosses a threshold

    For 'mean' mode, the flag threshold defaults to the mean of the two
    models' per-source normalised thresholds, but can be overridden via
    `mean_threshold`.
    """

    def __init__(
        self,
        gsr_detector: Optional[ArousalDetector],
        hrv_detector: Optional[ArousalDetector],
        mode: str = "any",
        mean_threshold: Optional[float] = None,
    ):
        if gsr_detector is None and hrv_detector is None:
            raise ValueError("EnsembleDetector needs at least one sub-detector")
        if mode not in COMBINE_MODES:
            raise ValueError(f"mode must be one of {COMBINE_MODES}, got {mode!r}")

        self.gsr = gsr_detector
        self.hrv = hrv_detector
        self.mode = mode
        self._lock = threading.Lock()
        self._last_gsr: Optional[ArousalResult] = None
        self._last_hrv: Optional[ArousalResult] = None

        # Default 'mean' threshold: the average of each sub-model's own
        # normalised threshold — i.e. what each one would flag at on its
        # own scale. Overridable for hand-tuning.
        if mean_threshold is None:
            thresholds = []
            if gsr_detector is not None:
                thresholds.append(gsr_detector._normalise_score(gsr_detector.threshold))
            if hrv_detector is not None:
                thresholds.append(hrv_detector._normalise_score(hrv_detector.threshold))
            self.mean_threshold = float(np.mean(thresholds)) if thresholds else 0.5
        else:
            self.mean_threshold = float(mean_threshold)

        print(
            f"[ENSEMBLE] Loaded: gsr={'yes' if gsr_detector else 'no'}, "
            f"hrv={'yes' if hrv_detector else 'no'}, mode='{mode}', "
            f"mean_threshold={self.mean_threshold:.3f}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_cols(self) -> list[str]:
        """Union of all expected feature columns across sub-detectors."""
        cols = []
        if self.gsr is not None:
            cols.extend(self.gsr.feature_cols)
        if self.hrv is not None:
            cols.extend(self.hrv.feature_cols)
        return cols

    def score(
        self,
        features: dict,
        source: str = "unknown",
    ) -> Optional[EnsembleResult]:
        """
        Score features from one source (or both, if the dict contains both).

        Typical usage from live capture: one thread calls with source='gsr'
        and only GSR features; another with source='hrv' and HRV features.
        In that case only the matching sub-detector scores, and the other
        source's last cached result is reused so the combined decision
        reflects the most recent state of both modalities.

        For offline batch scoring a single dict may contain both sources'
        features — both sub-detectors then score simultaneously.
        """
        gsr_result: Optional[ArousalResult] = None
        hrv_result: Optional[ArousalResult] = None

        gsr_features = self._extract_for("gsr", features, source)
        hrv_features = self._extract_for("hrv", features, source)

        if self.gsr is not None and gsr_features:
            gsr_result = self.gsr.score(gsr_features, source="gsr")
        if self.hrv is not None and hrv_features:
            hrv_result = self.hrv.score(hrv_features, source="hrv")

        with self._lock:
            if gsr_result is not None:
                self._last_gsr = gsr_result
            if hrv_result is not None:
                self._last_hrv = hrv_result
            cached_gsr = self._last_gsr
            cached_hrv = self._last_hrv

        if cached_gsr is None and cached_hrv is None:
            return None

        combined_norm, is_aroused = self._combine(cached_gsr, cached_hrv)

        return EnsembleResult(
            gsr=cached_gsr,
            hrv=cached_hrv,
            combined_normalised=combined_norm,
            is_aroused=is_aroused,
            mode=self.mode,
        )

    def trend(self) -> Optional[str]:
        """Combined trend: 'rising' if either source is rising, else 'falling', else 'stable'."""
        trends = []
        if self.gsr is not None:
            t = self.gsr.trend()
            if t:
                trends.append(t)
        if self.hrv is not None:
            t = self.hrv.trend()
            if t:
                trends.append(t)
        if not trends:
            return None
        if "rising" in trends:
            return "rising"
        if "falling" in trends:
            return "falling"
        return "stable"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_for(target_source: str, features: dict, source_hint: str) -> dict:
        """
        Pick out features relevant to `target_source`.

        If the dict has prefixed keys ('gsr_scl_mean'), pick those out.
        Otherwise if the hint matches target_source, pass everything through.
        """
        prefix = f"{target_source}_"
        prefixed_keys = {k: v for k, v in features.items() if k.startswith(prefix)}
        if prefixed_keys:
            return prefixed_keys
        if source_hint == target_source:
            return dict(features)
        return {}

    def _combine(
        self,
        gsr_result: Optional[ArousalResult],
        hrv_result: Optional[ArousalResult],
    ) -> tuple[float, bool]:
        norms = []
        flags = []
        if gsr_result is not None:
            norms.append(gsr_result.normalised)
            flags.append(gsr_result.is_aroused)
        if hrv_result is not None:
            norms.append(hrv_result.normalised)
            flags.append(hrv_result.is_aroused)

        combined_norm = float(np.mean(norms)) if norms else 0.0

        if self.mode == "any":
            is_aroused = any(flags)
        elif self.mode == "all":
            # 'all' only flags when both sub-detectors have fired AND flagged
            # — a single-source-only 'all' would be misleadingly equivalent
            # to that source's own flag.
            is_aroused = len(flags) == 2 and all(flags)
        else:  # 'mean'
            is_aroused = combined_norm >= self.mean_threshold

        return combined_norm, is_aroused


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_detector(
    model_path: Union[str, Path],
    ensemble_mode: str = "any",
    mean_threshold: Optional[float] = None,
) -> Union[ArousalDetector, EnsembleDetector]:
    """
    Load either a single-source model or an ensemble bundle.

    Ensemble bundles live in a directory with a manifest:

        models/ensemble/
            manifest.json    (points at gsr.pkl, hrv.pkl)
            gsr.pkl
            gsr_meta.json
            hrv.pkl
            hrv_meta.json

    Accepts the directory, the manifest JSON, or a plain .pkl (the latter
    returns a single-source ArousalDetector).
    """
    path = Path(model_path)

    if path.is_dir():
        manifest = path / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"No manifest.json in {path}")
        return _load_ensemble(manifest, ensemble_mode, mean_threshold)

    if path.suffix == ".json":
        return _load_ensemble(path, ensemble_mode, mean_threshold)

    return ArousalDetector(path)


def _load_ensemble(
    manifest_path: Path,
    mode: str,
    mean_threshold: Optional[float],
) -> EnsembleDetector:
    with open(manifest_path) as f:
        manifest = json.load(f)

    root = manifest_path.parent
    gsr_rel = manifest.get("gsr_model")
    hrv_rel = manifest.get("hrv_model")

    gsr_det = ArousalDetector(root / gsr_rel) if gsr_rel else None
    hrv_det = ArousalDetector(root / hrv_rel) if hrv_rel else None

    return EnsembleDetector(
        gsr_detector=gsr_det,
        hrv_detector=hrv_det,
        mode=mode,
        mean_threshold=mean_threshold,
    )
