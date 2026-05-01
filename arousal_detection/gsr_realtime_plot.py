import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe — script writes PNGs, no display
import matplotlib.pyplot as plt

from gsr_features import (
    GSRFeatureExtractor,
    lowpass_filter,
    decompose_tonic_phasic,
    detect_scr_peaks,
    adc_to_conductance_us,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_WINDOW_SECONDS,
)

from grove.adc import ADC

adc = ADC()

class Sensor:
    def __init__(self, channel):
        self.channel = channel

    def read(self):
        return adc.read(self.channel)

# ---------------------------------------------------------------------
# Real-time plotter
# ---------------------------------------------------------------------
class RealTimeGSRPlotter:
    def __init__(self, sample_rate, window_seconds, calibration):
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_seconds)
        self.buffer = deque(maxlen=self.window_size)
        self.calibration = calibration

    def add_adc(self, adc_value):
        c = adc_to_conductance_us(adc_value, self.calibration)
        if c is not None and 0 < c < 100:
            self.buffer.append(c)

    def plot(self, filename="gsr_live.png"):
        if len(self.buffer) < 5:
            return

        signal = np.array(self.buffer)
        t = np.arange(len(signal)) / self.sample_rate

        # ------------------------------------------------------------
        # Only filter when enough history exists for filtfilt
        # ------------------------------------------------------------
        if len(signal) < 20:
            filtered = signal
            tonic = signal
            phasic = np.zeros_like(signal)
            peaks = np.array([], dtype=int)
        else:
            filtered = lowpass_filter(signal, 1.0, self.sample_rate)
            tonic, phasic = decompose_tonic_phasic(filtered, self.sample_rate)
            peaks, _ = detect_scr_peaks(phasic, self.sample_rate)

        # ------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------
        plt.figure(figsize=(12, 6))

        # Raw + tonic
        plt.subplot(2, 1, 1)
        plt.plot(t, signal, label="Raw Conductance (µS)")
        plt.plot(t, tonic, label="Tonic (SCL)", linewidth=2)
        plt.legend()
        plt.ylabel("µS")

        # Phasic + peaks
        plt.subplot(2, 1, 2)
        plt.plot(t, phasic, label="Phasic (SCR)")
        if len(peaks) > 0:
            plt.scatter(
                t[peaks],
                phasic[peaks],
                color="red",
                label="SCR Peaks",
                zorder=5,
            )
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("µS")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main(adc_channel):
    sensor = Sensor(adc_channel)

    extractor = GSRFeatureExtractor(
        sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ,
        window_seconds=DEFAULT_WINDOW_SECONDS,
    )

    plotter = RealTimeGSRPlotter(
        DEFAULT_SAMPLE_RATE_HZ,
        DEFAULT_WINDOW_SECONDS,
        extractor.calibration,
    )

    last_plot = time.time()

    print("[gsr] Real-time plotting started gsr_live.png")

    while True:
        raw = sensor.read()

        # Feed both systems
        extractor.add_reading(raw)
        plotter.add_adc(raw)

        # Update PNG every 3 seconds
        if time.time() - last_plot > 3:
            plotter.plot("gsr_live.png")
            last_plot = time.time()

        # Also show features when window ready
        if extractor.window_ready():
            features = extractor.extract_features()
            print("\n[gsr] Features:")
            for k, v in features.items():
                print(f"  {k:<20} {v:.4f}")

        time.sleep(1.0 / DEFAULT_SAMPLE_RATE_HZ)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gsr_realtime_plot.py <adc_channel>")
        sys.exit(1)

    main(int(sys.argv[1]))
