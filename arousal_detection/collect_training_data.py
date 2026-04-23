"""
collect_training_data.py

Capture synchronised GSR + HRV features, audio, and video for arousal model
training. Optionally runs live Isolation Forest inference on each feature
window.

Algorithmically unchanged from the original: same sample rates, same window
sizes, same CSV schema. This cleanup fixes the picam2/cam naming bug,
removes dead imports, and makes --model accept either a single-source
.pkl OR an ensemble directory / manifest.

Usage:
    python collect_training_data.py <gsr_adc_channel> [--label <label>] [--model <path>]

    --model  Path to a trained model artefact. Accepts:
               - a single-source .pkl  (single ArousalDetector)
               - an ensemble directory (containing manifest.json)
               - an ensemble manifest.json directly
             If omitted, inference is skipped (capture only).

    --combine-mode  When --model is an ensemble: 'any' | 'all' | 'mean'
                    (default: 'any').

Outputs to ./sessions/<timestamp>_<label>/
    features.csv   one row per feature window
                   (GSR or HRV, with live-inference columns if --model set)
    audio.wav
    video.mp4      (converted from h264 on exit)
    session.json   metadata

Ctrl+C to stop cleanly.
"""

from __future__ import annotations

import asyncio
import csv
import json
import signal
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyaudio
from bleak import BleakClient
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

from arousal_detector import (
    ArousalDetector,
    EnsembleDetector,
    EnsembleResult,
    ArousalResult,
    load_detector,
)
from gsr_features import GSRFeatureExtractor, DEFAULT_SAMPLE_RATE_HZ as GSR_RATE
from hrv_features import HRVFeatureExtractor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BLE_ADDRESS = "00:22:D0:47:9C:DE"
HR_CHAR     = "00002a37-0000-1000-8000-00805f9b34fb"

AUDIO_RATE     = 48_000
AUDIO_CHANNELS = 2
AUDIO_CHUNK    = 1024
AUDIO_FORMAT   = pyaudio.paInt16

VIDEO_WIDTH     = 1280
VIDEO_HEIGHT    = 720
VIDEO_FRAMERATE = 10
# Picamera2 FrameDurationLimits are in microseconds. 100_000 µs = 10 fps.
VIDEO_FRAME_DURATION_US = 100_000

GSR_WINDOW_S = 60.0
HRV_WINDOW_S = 60.0
HRV_STEP_S   = 30.0


DetectorLike = Union[ArousalDetector, EnsembleDetector]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def make_session_dir(label: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("sessions") / f"{ts}_{label}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


@dataclass
class SessionState:
    label:       str
    session_dir: Path
    start_time:  float = field(default_factory=time.time)
    running:     bool  = True

    # Feature rows are written from multiple threads — protect with a lock
    features_lock: threading.Lock = field(default_factory=threading.Lock)
    feature_rows:  list           = field(default_factory=list)

    # Optional live detector — None if --model isn't supplied
    detector: Optional[DetectorLike] = None


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _log_result(source: str, result: Union[ArousalResult, EnsembleResult]) -> None:
    """Print a single-line summary of a scored window."""
    print(f"[DETECTOR/{source}] {result.summary()}")


def _flatten_result(result: Union[ArousalResult, EnsembleResult, None]) -> dict:
    """
    Collapse a result into a flat dict of CSV-bound columns. Keeps the same
    top-level column names the older data used (anomaly_score, is_aroused)
    while adding per-source ensemble columns where applicable.
    """
    if result is None:
        return {
            "anomaly_score":      None,
            "anomaly_normalised": None,
            "is_aroused":         None,
        }

    if isinstance(result, ArousalResult):
        return {
            "anomaly_score":      result.score,
            "anomaly_normalised": result.normalised,
            "is_aroused":         int(result.is_aroused),
        }

    # EnsembleResult
    out = {
        "anomaly_score":      None,  # no single 'raw' score for an ensemble
        "anomaly_normalised": result.combined_normalised,
        "is_aroused":         int(result.is_aroused),
        "ensemble_mode":      result.mode,
    }
    if result.gsr is not None:
        out["gsr_anomaly_score"]      = result.gsr.score
        out["gsr_anomaly_normalised"] = result.gsr.normalised
        out["gsr_is_aroused"]         = int(result.gsr.is_aroused)
    if result.hrv is not None:
        out["hrv_anomaly_score"]      = result.hrv.score
        out["hrv_anomaly_normalised"] = result.hrv.normalised
        out["hrv_is_aroused"]         = int(result.hrv.is_aroused)
    return out


def merge_and_save(
    state: SessionState,
    source: str,
    features: dict,
) -> None:
    """Run live inference (if a detector is loaded) and append one CSV row."""
    result: Union[ArousalResult, EnsembleResult, None] = None

    if state.detector is not None:
        result = state.detector.score(features, source=source)
        if result is not None:
            trend = state.detector.trend()
            trend_str = f" trend={trend}" if trend else ""
            _log_result(source, result)
            if trend:
                print(f"[DETECTOR/{source}]{trend_str}")
        else:
            print(
                f"[DETECTOR/{source}] Skipped — too few features for this window."
            )

    row = {
        "source":            source,
        "label":             state.label,
        "session_start":     state.start_time,
        "window_start_time": time.time() - state.start_time,
        **{f"{source}_{k}": v for k, v in features.items()},
        **_flatten_result(result),
    }

    with state.features_lock:
        state.feature_rows.append(row)

    print(
        f"[{source.upper()}] window saved "
        f"t={row['window_start_time']:.1f}s label={state.label}"
    )


def flush_csv(state: SessionState) -> None:
    csv_path = state.session_dir / "features.csv"
    with state.features_lock:
        rows = list(state.feature_rows)

    if not rows:
        print("[SESSION] No features to save")
        return

    # Union of all keys across rows, preserving first-seen order
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[SESSION] Features saved at {csv_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# GSR thread
# ---------------------------------------------------------------------------


def gsr_thread(state: SessionState, gsr_channel: int) -> None:
    try:
        from grove.adc import ADC
        adc = ADC()

        def read_adc() -> int:
            return adc.read(gsr_channel)
    except ImportError:
        print("[GSR] grove.adc not available — using mock sensor.")
        counter = [0]

        def read_adc() -> int:
            counter[0] += 1
            base = 300 + 20 * np.sin(counter[0] / 50)
            return int(np.clip(base + np.random.normal(0, 3), 0, 1023))

    extractor = GSRFeatureExtractor(
        sample_rate_hz=GSR_RATE,
        window_seconds=GSR_WINDOW_S,
    )

    print("[GSR] Starting GSR capture...")
    sleep_s = 1.0 / GSR_RATE

    while state.running:
        raw = read_adc()
        extractor.add_reading(raw)

        fill = extractor.buffer_fill_fraction
        if int(fill * 10) % 2 == 0:
            print(f"[GSR] buffer: {fill:.0%}", end="\r")

        if extractor.window_ready():
            features = extractor.extract_features()
            merge_and_save(state, "gsr", features)

        time.sleep(sleep_s)

    print("\n[GSR] Thread stopped.")


# ---------------------------------------------------------------------------
# Audio thread
# ---------------------------------------------------------------------------


def audio_thread(state: SessionState) -> None:
    wav_path = state.session_dir / "audio.wav"
    p = pyaudio.PyAudio()

    stream = p.open(
        format=AUDIO_FORMAT,
        channels=AUDIO_CHANNELS,
        rate=AUDIO_RATE,
        frames_per_buffer=AUDIO_CHUNK,
        input=True,
    )

    frames: list[bytes] = []
    print("[AUDIO] Recording started...")

    try:
        while state.running:
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
            wf.setframerate(AUDIO_RATE)
            wf.writeframes(b"".join(frames))
        print(f"[AUDIO] Saved as {wav_path}")


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def start_video(state: SessionState) -> tuple[Picamera2, H264Encoder]:
    video_path = state.session_dir / "video.h264"

    cam = Picamera2()
    config = cam.create_preview_configuration(
        main={"size": (VIDEO_WIDTH, VIDEO_HEIGHT)},
        sensor={"output_size": (4608, 2592)},
        controls={"FrameDurationLimits": (VIDEO_FRAME_DURATION_US,
                                          VIDEO_FRAME_DURATION_US)},
    )
    cam.configure(config)

    encoder = H264Encoder(bitrate=10_000_000)
    output  = FileOutput(str(video_path))
    cam.start_recording(encoder, output)

    print(f"[VIDEO] Recording to {video_path}")
    return cam, encoder


def stop_video(cam: Picamera2, session_dir: Path) -> None:
    cam.stop_recording()
    cam.close()
    print("[VIDEO] Stopped")

    h264_path = session_dir / "video.h264"
    mp4_path  = session_dir / "video.mp4"
    print("[VIDEO] Converting to MP4...")

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(VIDEO_FRAMERATE),
            "-i", str(h264_path),
            "-c", "copy",
            str(mp4_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        h264_path.unlink()
        print(f"[VIDEO] Saved as {mp4_path}")
    else:
        print(f"[VIDEO] ffmpeg conversion failed, keeping raw .h264\n{result.stderr}")


# ---------------------------------------------------------------------------
# BLE / HRV task
# ---------------------------------------------------------------------------


async def ble_task(state: SessionState) -> None:
    extractor = HRVFeatureExtractor(
        window_seconds=HRV_WINDOW_S,
        step_seconds=HRV_STEP_S,
    )

    def handle_hr(_sender, data: bytes) -> None:
        flags = data[0]
        if flags & 0x01:
            hr = int.from_bytes(data[1:3], byteorder="little")
            rr_offset = 3
        else:
            hr = data[1]
            rr_offset = 2

        rr_values: list[float] = []
        if flags & 0x10:
            while rr_offset + 1 < len(data):
                rr_raw = int.from_bytes(data[rr_offset:rr_offset + 2],
                                         byteorder="little")
                rr_values.append(round(rr_raw * 1000 / 1024, 1))
                rr_offset += 2

        extractor.add_reading(hr, rr_values)
        print(
            f"[HRV] HR: {hr} bpm | RR: {rr_values} | "
            f"buffer: {extractor.buffer_fill_fraction:.0%}"
        )

        if extractor.window_ready():
            features = extractor.extract_features()
            merge_and_save(state, "hrv", features)

    print(f"[HRV] Connecting to {BLE_ADDRESS}...")
    async with BleakClient(BLE_ADDRESS, timeout=60.0) as client:
        print(f"[HRV] Connected: {client.is_connected}")
        await client.start_notify(HR_CHAR, handle_hr)

        while state.running:
            await asyncio.sleep(1.0)

        await client.stop_notify(HR_CHAR)
    print("[HRV] BLE disconnected")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def describe_detector(detector: Optional[DetectorLike]) -> Optional[dict]:
    if detector is None:
        return None
    if isinstance(detector, EnsembleDetector):
        return {
            "type":           "ensemble",
            "mode":           detector.mode,
            "mean_threshold": detector.mean_threshold,
            "sub_models":     [d.source for d in (detector.gsr, detector.hrv) if d is not None],
        }
    return {
        "type":         "single",
        "source":       detector.source,
        "feature_cols": detector.feature_cols,
    }


def save_metadata(state: SessionState, duration_s: float) -> None:
    meta = {
        "label":             state.label,
        "start_time":        datetime.fromtimestamp(state.start_time).isoformat(),
        "duration_s":        round(duration_s, 2),
        "gsr_sample_rate":   GSR_RATE,
        "gsr_window_s":      GSR_WINDOW_S,
        "hrv_window_s":      HRV_WINDOW_S,
        "hrv_step_s":        HRV_STEP_S,
        "audio_rate":        AUDIO_RATE,
        "video_fps":         VIDEO_FRAMERATE,
        "video_resolution":  [VIDEO_WIDTH, VIDEO_HEIGHT],
        "ble_address":       BLE_ADDRESS,
        "n_feature_rows":    len(state.feature_rows),
        "model_used":        describe_detector(state.detector),
    }
    meta_path = state.session_dir / "session.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SESSION] Metadata saved as {meta_path}")


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------


def _arg_value(flag: str) -> Optional[str]:
    if flag not in sys.argv:
        return None
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return None
    return sys.argv[idx + 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit("Usage: python collect_training_data.py <gsr_adc_channel> "
                 "[--label LBL] [--model PATH] [--combine-mode any|all|mean]")

    gsr_channel = int(sys.argv[1])
    label       = (_arg_value("--label") or "unlabelled").replace(" ", "_")
    model_path  = _arg_value("--model")
    combine_mode = _arg_value("--combine-mode") or "any"

    detector: Optional[DetectorLike] = None
    if model_path:
        try:
            detector = load_detector(model_path, ensemble_mode=combine_mode)
        except FileNotFoundError as e:
            print(f"[WARN] Could not load model: {e}. Running capture-only.")
    else:
        print("[SESSION] No --model supplied. Capture-only mode.")

    state = SessionState(
        label=label,
        session_dir=make_session_dir(label),
        detector=detector,
    )
    print(f"[SESSION] Starting label='{label}' saved at {state.session_dir}")

    def _shutdown(_sig, _frame):
        print("\n[SESSION] Shutting down...")
        state.running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    cam, _encoder = start_video(state)

    t_gsr   = threading.Thread(target=gsr_thread,
                               args=(state, gsr_channel), daemon=True)
    t_audio = threading.Thread(target=audio_thread, args=(state,), daemon=True)
    t_gsr.start()
    t_audio.start()

    try:
        await ble_task(state)
    except Exception as e:
        print(f"[HRV] BLE error: {e}")
    finally:
        state.running = False

    t_gsr.join(timeout=5)
    t_audio.join(timeout=10)

    stop_video(cam, state.session_dir)

    duration_s = time.time() - state.start_time
    flush_csv(state)
    save_metadata(state, duration_s)

    print(f"\n[SESSION] Done. Duration: {duration_s:.1f}s")
    print(f"[SESSION] Output: {state.session_dir.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
