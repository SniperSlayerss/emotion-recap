import asyncio
from bleak import BleakClient
from collections import deque
from dataclasses import dataclass
import numpy as np

# --- Device ---
ADDRESS = "00:22:D0:47:9C:DE"

# --- Characteristics ---
HR_CHAR           = "00002a37-0000-1000-8000-00805f9b34fb"
BATTERY_CHAR      = "00002a19-0000-1000-8000-00805f9b34fb"
MANUFACTURER_CHAR = "00002a29-0000-1000-8000-00805f9b34fb"
MODEL_CHAR        = "00002a24-0000-1000-8000-00805f9b34fb"
SERIAL_CHAR       = "00002a25-0000-1000-8000-00805f9b34fb"
FIRMWARE_CHAR     = "00002a26-0000-1000-8000-00805f9b34fb"

# --- Window settings (mirror GSR pipeline) ---
DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_STEP_SECONDS   = 30.0   # 50% overlap

# RR interval plausibility range (ms)
RR_MIN_MS = 300.0   # ~200 bpm max
RR_MAX_MS = 2000.0  # ~30 bpm min


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------
@dataclass
class HRVFeatures:
    """
    HRV features for one window.
    Time-domain features match standard HRV guidelines (Task Force, 1996).
    """
    hr_mean: float
    hr_std: float
    hr_min: float
    hr_max: float

    rmssd: float
    sdnn: float
    pnn50: float
    mean_rr: float

    sd1: float
    sd2: float

    # Diagnostics
    n_hr_samples: int = 0
    n_rr_samples: int = 0
    n_rejected_rr: int = 0
    window_duration_s: float = 0.0

    def to_dict(self) -> dict:
        """Model-ready features"""
        return {
            "hr_mean":   self.hr_mean,
            "hr_std":    self.hr_std,
            "hr_min":    self.hr_min,
            "hr_max":    self.hr_max,
            "rmssd":     self.rmssd,
            "sdnn":      self.sdnn,
            "pnn50":     self.pnn50,
            "mean_rr":   self.mean_rr,
            "sd1":       self.sd1,
            "sd2":       self.sd2,
        }


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_hrv_features(
    hr_samples: np.ndarray,
    rr_samples: np.ndarray,
    window_duration_s: float,
    n_rejected_rr: int,
) -> HRVFeatures:
    """Compute all HRV features from buffered HR and RR arrays."""
    if len(rr_samples) < 2:
        raise ValueError(f"Need at least 2 RR intervals, got {len(rr_samples)}.")

    successive_diffs = np.diff(rr_samples)

    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
    sdnn  = float(np.std(rr_samples))
    pnn50 = float(np.mean(np.abs(successive_diffs) > 50.0))

    # Poincare: plot RR[n] vs RR[n+1], SD1 = short-term, SD2 = long-term
    sd1 = float(np.std(successive_diffs) / np.sqrt(2))
    sd2 = float(np.sqrt(max(0.0, 2 * sdnn**2 - sd1**2)))

    return HRVFeatures(
        hr_mean=float(np.mean(hr_samples)),
        hr_std=float(np.std(hr_samples)),
        hr_min=float(np.min(hr_samples)),
        hr_max=float(np.max(hr_samples)),
        rmssd=rmssd,
        sdnn=sdnn,
        pnn50=pnn50,
        mean_rr=float(np.mean(rr_samples)),
        sd1=sd1,
        sd2=sd2,
        n_hr_samples=len(hr_samples),
        n_rr_samples=len(rr_samples),
        n_rejected_rr=n_rejected_rr,
        window_duration_s=window_duration_s,
    )


# ---------------------------------------------------------------------------
# Rolling window extractor
# ---------------------------------------------------------------------------
class HRVFeatureExtractor:
    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        step_seconds: float = DEFAULT_STEP_SECONDS,
        hr_sample_rate_hz: float = 1.0,   # H7 sends ~1 HR packet/s
    ):
        self.window_seconds    = window_seconds
        self.step_seconds      = step_seconds
        self.hr_sample_rate_hz = hr_sample_rate_hz

        self._hr_window_size  = int(window_seconds * hr_sample_rate_hz)
        self._hr_step_size    = int(step_seconds   * hr_sample_rate_hz)

        self._hr_buffer: deque  = deque(maxlen=self._hr_window_size)
        self._rr_buffer: list   = []   # All RR intervals (grows; trimmed on extract)
        self._rejected_rr: int  = 0
        self._samples_since_last_extract: int = 0

    def add_reading(self, hr_bpm: int, rr_intervals_ms: list[float]) -> None:
        """
        Feed one BLE notification packet.
        """
        self._hr_buffer.append(float(hr_bpm))
        self._samples_since_last_extract += 1

        for rr in rr_intervals_ms:
            if RR_MIN_MS <= rr <= RR_MAX_MS:
                self._rr_buffer.append(rr)
            else:
                self._rejected_rr += 1

    def window_ready(self) -> bool:
        """True when the HR buffer is full and the step has elapsed."""
        return (
            len(self._hr_buffer) >= self._hr_window_size
            and self._samples_since_last_extract >= self._hr_step_size
            and len(self._rr_buffer) >= 10   # Need a minimum of RR points
        )

    def extract_features(self) -> dict:
        if not self.window_ready():
            raise RuntimeError("Window not ready. Call window_ready() first.")

        hr_arr = np.array(self._hr_buffer)
        rr_arr = np.array(self._rr_buffer)

        features = compute_hrv_features(
            hr_arr, rr_arr,
            window_duration_s=self.window_seconds,
            n_rejected_rr=self._rejected_rr,
        )

        # Slide: keep RR intervals from the latter half of the window
        keep = len(self._rr_buffer) // 2
        self._rr_buffer = self._rr_buffer[-keep:]
        self._samples_since_last_extract = 0

        rejection_rate = self._rejected_rr / max(1, self._rejected_rr + len(rr_arr))
        if rejection_rate > 0.2:
            print(f"[hrv] Warning: {rejection_rate:.0%} of RR intervals rejected "
                  f"(artefacts or poor contact?).")
        self._rejected_rr = 0

        return features.to_dict()

    @property
    def buffer_fill_fraction(self) -> float:
        return len(self._hr_buffer) / self._hr_window_size


async def read_battery(client):
    data = await client.read_gatt_char(BATTERY_CHAR)
    print(f"Battery: {data[0]}%")

async def read_device_info(client):
    for label, uuid in [
        ("Manufacturer", MANUFACTURER_CHAR),
        ("Model",        MODEL_CHAR),
        ("Serial",       SERIAL_CHAR),
        ("Firmware",     FIRMWARE_CHAR),
    ]:
        data = await client.read_gatt_char(uuid)
        print(f"{label}: {data.decode('utf-8', errors='replace')}")


if __name__ == "__main__":
    # Build the extractor and BLE notification handler locally so they
    # don't exist as module globals when hrv_features is imported as a
    # library (e.g. by collect_training_data.py).
    extractor = HRVFeatureExtractor()

    def handle_hr(sender, data):
        flags = data[0]
        if flags & 0x01:
            hr = int.from_bytes(data[1:3], byteorder="little")
            rr_offset = 3
        else:
            hr = data[1]
            rr_offset = 2

        rr_values = []
        if flags & 0x10:
            while rr_offset + 1 < len(data):
                rr_raw = int.from_bytes(data[rr_offset:rr_offset + 2], byteorder="little")
                rr_values.append(round(rr_raw * 1000 / 1024, 1))
                rr_offset += 2

        extractor.add_reading(hr, rr_values)
        fill = extractor.buffer_fill_fraction
        print(f"HR: {hr} bpm | RR: {rr_values} | buffer: {fill:.0%}")

        if extractor.window_ready():
            print("\n[hrv] --- Window complete, extracting features ---")
            features = extractor.extract_features()
            for k, v in features.items():
                print(f"  {k:<12} {v:.4f}")
            print()

    async def main():
        print(f"Connecting to {ADDRESS}...")
        async with BleakClient(ADDRESS, timeout=60.0) as client:
            print(f"Connected: {client.is_connected}")
            await read_device_info(client)
            await read_battery(client)
            await client.start_notify(HR_CHAR, handle_hr)
            await asyncio.sleep(120)   # 2 min to get two 60s windows

    asyncio.run(main())
