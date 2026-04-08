"""
arousal_detector.py

Lightweight inference wrapper for the trained Isolation Forest.
Designed to run on the Raspberry Pi during live capture.

Loaded once at session start, then called per feature window.
Thread-safe: score() can be called from the GSR thread, the HRV
BLE coroutine, or the merge_and_save() path — all protected by a lock.
"""

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import joblib
except ImportError:
    raise ImportError("Install joblib: pip install joblib")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArousalResult:
    """
    Result from one inference call.

    score       : Raw anomaly score from IsolationForest.score_samples().
                  More negative = more anomalous. Typical range: [-0.7, 0.0]

    normalised  : Score re-mapped to [0, 1] using training distribution.
                  0 = deep baseline, 1 = maximally anomalous.
                  Useful for logging and display.

    is_aroused  : True if score < threshold (i.e. flagged as non-baseline).

    features_used : Which feature columns were present in this window.
    missing       : Which expected feature columns were absent (NaN / not provided).
    """
    score:         float
    normalised:    float
    is_aroused:    bool
    threshold:     float
    features_used: list[str] = field(default_factory=list)
    missing:       list[str] = field(default_factory=list)

    def summary(self) -> str:
        flag = "⚠  AROUSED" if self.is_aroused else "✓  baseline"
        return (
            f"{flag} | score={self.score:.4f} "
            f"(norm={self.normalised:.2f}, thresh={self.threshold:.4f})"
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ArousalDetector:
    """
    Load a trained model and run live inference on the Pi.

    Usage:
        detector = ArousalDetector("models/iforest.pkl")

        # From merge_and_save() after each window:
        result = detector.score(features_dict, source="gsr")
        if result:
            print(result.summary())
    """

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        meta_path  = model_path.with_name(model_path.stem + "_meta.json")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        self._pipe      = joblib.load(model_path)
        self._lock      = threading.Lock()

        with open(meta_path) as f:
            meta = json.load(f)

        self.feature_cols:  list[str] = meta["feature_cols"]
        self.feature_names: list[str] = meta["feature_names"]
        self.threshold:     float     = meta["threshold"]
        self._score_mean:   float     = meta["score_mean"]
        self._score_std:    float     = meta["score_std"]
        self._score_min:    float     = meta["score_min"]
        self._score_max:    float     = meta["score_max"]

        print(
            f"[DETECTOR] Loaded model from {model_path}\n"
            f"           Features  : {self.feature_cols}\n"
            f"           Threshold : {self.threshold:.4f}\n"
            f"           Trained on: {meta['n_training_windows']} windows "
            f"({meta.get('trained_at', '?')})"
        )

        # Rolling buffer of recent results for trend detection
        self._recent_scores: list[float] = []
        self._max_recent = 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        features: dict,
        source: str = "unknown",
    ) -> Optional[ArousalResult]:
        """
        Score one feature window.

        `features` is the dict returned by GSRFeatureExtractor.extract_features()
        or HRVFeatureExtractor.extract_features() — keys are plain names like
        'scl_mean', 'rmssd'. The detector will also accept the prefixed form
        used in features.csv ('gsr_scl_mean', 'hrv_rmssd').

        Returns None if too many expected features are missing to be reliable.
        """
        # Normalise keys: strip gsr_/hrv_ prefixes if present, then re-add
        # based on feature_cols (which always have prefixes from the CSV).
        # This lets us accept both dict shapes.
        normalised = {}
        for k, v in features.items():
            # Already prefixed (e.g. from a merged CSV row)
            if k in self.feature_cols:
                normalised[k] = v
            else:
                # Bare key — try attaching source prefix
                prefixed = f"{source}_{k}"
                if prefixed in self.feature_cols:
                    normalised[prefixed] = v

        present  = [c for c in self.feature_cols if c in normalised and normalised[c] is not None]
        missing  = [c for c in self.feature_cols if c not in present]

        # Require at least half the expected features
        if len(present) < max(1, len(self.feature_cols) // 2):
            return None

        # Build input vector; impute missing with 0 (mean after scaling)
        x = np.array([[
            float(normalised.get(c, 0.0) or 0.0)
            for c in self.feature_cols
        ]])

        with self._lock:
            raw_score = float(self._pipe.score_samples(x)[0])

        normalised_score = self._normalise(raw_score)
        is_aroused = raw_score < self.threshold

        with self._lock:
            self._recent_scores.append(raw_score)
            if len(self._recent_scores) > self._max_recent:
                self._recent_scores.pop(0)

        return ArousalResult(
            score=raw_score,
            normalised=normalised_score,
            is_aroused=is_aroused,
            threshold=self.threshold,
            features_used=present,
            missing=missing,
        )

    def trend(self) -> Optional[str]:
        """
        Simple trend over the last N windows:
        'rising', 'falling', or 'stable'.
        Returns None if not enough history.
        """
        with self._lock:
            recent = list(self._recent_scores)

        if len(recent) < 4:
            return None

        # Linear slope of scores (more negative = more aroused, so slope < 0 = rising arousal)
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if abs(slope) < 0.005:
            return "stable"
        return "falling" if slope < 0 else "rising"   # score falls → arousal rises

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalise(self, score: float) -> float:
        """
        Map raw score to [0, 1] using the training distribution.
        Clipped so values outside the training range stay in [0, 1].
        """
        lo = self._score_min - self._score_std
        hi = self._score_max
        if hi == lo:
            return 0.0
        normalised = (hi - score) / (hi - lo)   # invert: lower score → higher normalised
        return float(np.clip(normalised, 0.0, 1.0))
