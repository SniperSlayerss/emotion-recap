import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

SERIAL_CALIBRATION = 509
GROVE_FIXED_RESISTOR_OHMS = 10_000

# Typical Grove ADC resolution (CHECK)
ADC_RESOLUTION = 1024

# Butterworth filter settings
LOWPASS_CUTOFF_HZ = 1.0  # Remove noise above 1 Hz
PHASIC_HIGHPASS_CUTOFF_HZ = 0.05  # Separate phasic from tonic

# SCR (skin conductance response) peak detection
SCR_MIN_AMPLITUDE_US = 0.02  # Minimum SCR amplitude to count as a response (µS)
SCR_MIN_DISTANCE_SAMPLES = (
    10  # Minimum samples between SCR peaks (avoids double-counting)
)

DEFAULT_WINDOW_SECONDS = 60
DEFAULT_SAMPLE_RATE_HZ = 3.33  # 1 sample per 0.3s


def adc_to_resistance_kohms(
    adc_reading: float, calibration: float = SERIAL_CALIBRATION
) -> Optional[float]:
    """
    Convert a raw ADC reading to skin resistance in kΩ.
    """
    div = calibration - adc_reading
    if div == 0:
        return None
    resistance_ohms = (
        (ADC_RESOLUTION + 2 * adc_reading) * GROVE_FIXED_RESISTOR_OHMS
    ) / div
    return resistance_ohms / 1000.0  # Convert to kΩ


def resistance_to_conductance_us(resistance_kohms: float) -> float:
    """
    Convert skin resistance (kΩ) to skin conductance (µS = microsiemens).

    conductance_µS = 1000 / resistance_kΩ
    """
    if resistance_kohms <= 0:
        return 0.0
    return 1000.0 / resistance_kohms


def adc_to_conductance_us(
    adc_reading: float, calibration: float = SERIAL_CALIBRATION
) -> Optional[float]:
    resistance = adc_to_resistance_kohms(adc_reading, calibration)
    if resistance is None or resistance <= 0:
        return None
    return resistance_to_conductance_us(resistance)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
def lowpass_filter(
    signal: np.ndarray, cutoff_hz: float, sample_rate_hz: float, order: int = 4
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter.

    Zero-phase (filtfilt) avoids phase distortion, which matters for
    accurate SCR peak timing. filtfilt is not suitable for real-time
    use, see GSRFeatureExtractor for the online alternative.
    """
    nyquist = sample_rate_hz / 2.0
    if cutoff_hz >= nyquist:
        return signal  # Nothing to filter
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, signal)


def decompose_tonic_phasic(
    conductance: np.ndarray,
    sample_rate_hz: float,
    method: str = "highpass",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose the conductance signal into tonic (SCL) and phasic (SCR) components.
    - Tonic:  Slow-varying baseline skin conductance level (SCL) in µS
    - Phasic: Fast-varying skin conductance responses (SCR) in µS

    Two methods are available:
    - "highpass" (default, no extra dependencies):
       - Tonic  = low-pass filtered signal  (slow baseline drift)
       - Phasic = high-pass filtered signal (fast SCR responses)
       - Simple and robust. Slightly less clean separation than cvxEDA.

    - "cvxeda" (optional, requires cvxeda package):
       - Uses the convex optimisation approach from Greco et al. (2016)
    """
    if method == "cvxeda":
        try:
            import cvxeda

            # cvxEDA expects signal sampled at 25 Hz — resample if needed
            # For simplicity at 3.33 Hz we use the highpass fallback
            # TODO: upsample to 25 Hz with scipy.signal.resample before calling cvxEDA
            print(
                "[gsr] cvxEDA requested but signal is at low sample rate — using highpass fallback."
            )
        except ImportError:
            print(
                "[gsr] cvxeda not installed — using highpass decomposition. "
                "Install with: pip install cvxeda"
            )

    # Highpass decomposition
    nyquist = sample_rate_hz / 2.0
    cutoff = PHASIC_HIGHPASS_CUTOFF_HZ

    if cutoff >= nyquist:
        # Sample rate too low for clean HP filtering; use moving average subtraction instead
        window = max(
            1, int(sample_rate_hz * 20)
        )  # 20-second moving average as tonic estimate
        tonic = np.convolve(conductance, np.ones(window) / window, mode="same")
    else:
        b, a = butter(4, cutoff / nyquist, btype="low")
        tonic = filtfilt(b, a, conductance)

    phasic = conductance - tonic
    return tonic, phasic


def detect_scr_peaks(
    phasic: np.ndarray,
    sample_rate_hz: float,
    min_amplitude_us: float = SCR_MIN_AMPLITUDE_US,
) -> tuple[np.ndarray, dict]:
    """
    Detect skin conductance response (SCR) peaks in the phasic component.

    SCRs are transient increases in conductance caused by sympathetic
    nervous system activation. They typically:
    - Rise in 1–3 seconds
    - Have amplitudes of 0.01–3 µS
    - Recover (half-decay) in 5–30 seconds

    Returned are the
    - peak_indices: Array of sample indices of detected SCR peaks
    - peak_properties: Dictionary of peak properties from scipy.find_peaks
    """
    min_distance = max(1, int(SCR_MIN_DISTANCE_SAMPLES))

    # Only look at positive deflections in phasic signal
    phasic_pos = np.clip(phasic, 0, None)

    peaks, properties = find_peaks(
        phasic_pos,
        height=min_amplitude_us,
        distance=min_distance,
        prominence=min_amplitude_us * 0.5,
    )
    return peaks, properties


def compute_scr_timings(
    phasic: np.ndarray,
    peak_indices: np.ndarray,
    sample_rate_hz: float,
) -> tuple[list[float], list[float]]:
    """
    Estimate rise time and half-recovery time for each detected SCR.
    - Rise time: Samples from onset (local minimum before peak) to peak.
    - Recovery time: Samples from peak to 50% amplitude decay after peak.

    Both are converted to seconds.
    """
    rise_times = []
    recovery_times = []

    for peak_idx in peak_indices:
        peak_val = phasic[peak_idx]

        # Rise time
        # Walk backwards from peak to find onset (local minimum or zero-crossing)
        onset_idx = peak_idx
        for i in range(peak_idx - 1, max(0, peak_idx - int(sample_rate_hz * 5)), -1):
            if phasic[i] <= 0 or phasic[i] >= phasic[i + 1]:
                onset_idx = i
                break
        rise_time_s = (peak_idx - onset_idx) / sample_rate_hz
        rise_times.append(rise_time_s)

        # Half-recovery time
        half_amp = peak_val * 0.5
        recovery_idx = None
        for i in range(
            peak_idx + 1, min(len(phasic), peak_idx + int(sample_rate_hz * 60))
        ):
            if phasic[i] <= half_amp:
                recovery_idx = i
                break
        if recovery_idx is not None:
            recovery_time_s = (recovery_idx - peak_idx) / sample_rate_hz
            recovery_times.append(recovery_time_s)

    return rise_times, recovery_times


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@dataclass
class GSRFeatures:
    """
    Extracted GSR features for one window.
    Field names match GSR_FEATURES in train_arousal_model.py exactly.
    """

    scl_mean: float  # Mean tonic skin conductance level (µS)
    scl_std: float  # Std dev of tonic SCL
    scr_count: float  # Number of SCRs per minute
    scr_mean_amp: float  # Mean SCR amplitude (µS)
    scr_rise_time: float  # Mean SCR rise time (s)
    scr_recovery_time: float  # Mean SCR half-recovery time (s)
    phasic_mean: float  # Mean phasic component (µS)
    phasic_std: float  # Std dev of phasic component

    # Debug info
    n_samples: int = 0
    window_duration_s: float = 0.0
    mean_conductance_us: float = 0.0
    conductance_range_us: float = 0.0
    n_scr_peaks: int = 0

    def to_dict(self) -> dict:
        """Return model features"""
        return {
            "scl_mean": self.scl_mean,
            "scl_std": self.scl_std,
            "scr_count": self.scr_count,
            "scr_mean_amp": self.scr_mean_amp,
            "scr_rise_time": self.scr_rise_time,
            "scr_recovery_time": self.scr_recovery_time,
            "phasic_mean": self.phasic_mean,
            "phasic_std": self.phasic_std,
        }

    def to_full_dict(self) -> dict:
        """Return all fields including diagnostics."""
        return self.__dict__


def extract_gsr_features(
    conductance_us: np.ndarray,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    decomp_method: str = "highpass",
) -> GSRFeatures:
    """
    Extract all GSR features from a window of conductance values (µS).

    Args:
    - conductance_us:  1-D array of skin conductance values in µS.
    - sample_rate_hz:  Sampling rate of the signal.
    - decomp_method:   "highpass" or "cvxeda"

    Returns
    - GSRFeatures dataclass with all model-ready features.
    """
    n = len(conductance_us)
    if n < 4:
        raise ValueError(
            f"Window too short for feature extraction: {n} samples (need at least 4)."
        )

    window_duration_s = n / sample_rate_hz

    # Low-pass filter to remove high-freq noise
    filtered = lowpass_filter(conductance_us, LOWPASS_CUTOFF_HZ, sample_rate_hz)

    # Decompose into tonic (SCL) and phasic (SCR) components
    tonic, phasic = decompose_tonic_phasic(
        filtered, sample_rate_hz, method=decomp_method
    )

    # Detect SCR peaks
    peak_indices, peak_props = detect_scr_peaks(phasic, sample_rate_hz)

    # Compute SCR timings
    rise_times, recovery_times = compute_scr_timings(
        phasic, peak_indices, sample_rate_hz
    )

    # Aggregate features
    scr_count_per_min = (len(peak_indices) / window_duration_s) * 60.0

    scr_amplitudes = peak_props.get("peak_heights", np.array([]))
    scr_mean_amp = float(np.mean(scr_amplitudes)) if len(scr_amplitudes) > 0 else 0.0
    mean_rise_time = float(np.mean(rise_times)) if rise_times else 0.0
    mean_recovery_time = float(np.mean(recovery_times)) if recovery_times else 0.0

    return GSRFeatures(
        scl_mean=float(np.mean(tonic)),
        scl_std=float(np.std(tonic)),
        scr_count=float(scr_count_per_min),
        scr_mean_amp=scr_mean_amp,
        scr_rise_time=mean_rise_time,
        scr_recovery_time=mean_recovery_time,
        phasic_mean=float(np.mean(phasic)),
        phasic_std=float(np.std(phasic)),
        # Diagnostics
        n_samples=n,
        window_duration_s=window_duration_s,
        mean_conductance_us=float(np.mean(conductance_us)),
        conductance_range_us=float(np.max(conductance_us) - np.min(conductance_us)),
        n_scr_peaks=len(peak_indices),
    )


# ---------------------------------------------------------------------------
# Rolling window extractor (for real-time use)
# ---------------------------------------------------------------------------
class GSRFeatureExtractor:
    """
    Stateful rolling-window feature extractor for real-time use.

    The window uses a sliding approach: once full, each new reading shifts
    the window forward by step_seconds (default: window_seconds / 2 = 50% overlap).
    """

    def __init__(
        self,
        sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        step_seconds: Optional[float] = None,
        calibration: float = SERIAL_CALIBRATION,
        decomp_method: str = "highpass",
    ):
        self.sample_rate_hz = sample_rate_hz
        self.window_size = int(window_seconds * sample_rate_hz)
        self.step_size = int((step_seconds or window_seconds / 2) * sample_rate_hz)
        self.calibration = calibration
        self.decomp_method = decomp_method

        self._buffer: deque = deque(maxlen=self.window_size)
        self._samples_since_last_extract: int = 0
        self._rejected_samples: int = 0

    def add_reading(self, adc_value: float) -> bool:
        """
        Add a raw ADC reading to the buffer.

        Returns True if the reading was accepted (sensor contact detected), False if it was rejected (no skin contact or out-of-range value).
        """
        conductance = adc_to_conductance_us(adc_value, self.calibration)

        if conductance is None or conductance <= 0 or conductance > 100:
            # Typical human skin conductance is 0.5–50 µS at rest.
            # Values outside this range indicate poor contact or sensor error.
            self._rejected_samples += 1
            return False

        self._buffer.append(conductance)
        self._samples_since_last_extract += 1
        return True

    def window_ready(self) -> bool:
        """
        Returns True if the buffer is full and enough new samples have accumulated since the last extraction (i.e. the step has elapsed).
        """
        return (
            len(self._buffer) >= self.window_size
            and self._samples_since_last_extract >= self.step_size
        )

    def extract_features(self) -> dict:
        """
        Extract features from the current window.
        Resets the step counter. Does NOT clear the buffer (sliding window).

        Returns a dict with keys matching GSR_FEATURES in train_arousal_model.py.
        """
        if len(self._buffer) < self.window_size:
            raise RuntimeError(
                f"Buffer not full yet ({len(self._buffer)}/{self.window_size} samples). "
                f"Call window_ready() before extract_features()."
            )

        signal = np.array(self._buffer)
        features = extract_gsr_features(signal, self.sample_rate_hz, self.decomp_method)
        self._samples_since_last_extract = 0

        if self._rejected_samples > 0:
            rejection_rate = self._rejected_samples / (
                len(self._buffer) + self._rejected_samples
            )
            if rejection_rate > 0.2:
                print(
                    f"[gsr] Warning: {rejection_rate:.0%} of samples were rejected "
                    f"(poor sensor contact?). Consider checking electrode placement."
                )

        return features.to_dict()

    @property
    def buffer_fill_fraction(self) -> float:
        """How full the buffer is (0.0 to 1.0). Useful for progress logging."""
        return len(self._buffer) / self.window_size

    @property
    def n_buffered(self) -> int:
        return len(self._buffer)


def collect_and_extract(adc_channel: int, window_seconds: float = 60.0):
    """
    Read from the Grove GSR sensor and extract features in real time.
    """
    try:
        from grove.adc import ADC

        adc = ADC()

        class _Sensor:
            def read(self):
                return adc.read(adc_channel)
    except ImportError:
        print("[gsr] grove.adc not available — using mock sensor for testing.")

        class _Sensor:
            """Mock sensor that simulates a slowly varying conductance signal."""

            def __init__(self):
                self._t = 0

            def read(self):
                self._t += 1
                base = 300 + 20 * np.sin(self._t / 50)
                noise = np.random.normal(0, 3)
                return int(np.clip(base + noise, 0, 1023))

    sensor = _Sensor()
    extractor = GSRFeatureExtractor(
        sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ,
        window_seconds=window_seconds,
    )

    print(
        f"[gsr] Collecting... (window: {window_seconds}s, "
        f"need {extractor.window_size} samples)"
    )

    while True:
        raw = sensor.read()
        accepted = extractor.add_reading(raw)

        # Progress indicator
        fill = extractor.buffer_fill_fraction
        print(fill)

        if extractor.window_ready():
            print("\n[gsr] --- Window complete, extracting features ---")
            features = extractor.extract_features()
            for k, v in features.items():
                print(f"  {k:<22} {v:.4f}")
            print()

        time.sleep(1.0 / DEFAULT_SAMPLE_RATE_HZ)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} <adc_channel> [window_seconds]".format(sys.argv[0]))
        sys.exit(1)

    channel = int(sys.argv[1])
    window = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_WINDOW_SECONDS
    collect_and_extract(channel, window_seconds=window)
