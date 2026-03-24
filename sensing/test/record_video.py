import time
import math
import sys

from picamera2 import Picamera2, Preview
from libcamera import Transform
from grove.adc import ADC


def preview_picamera():
    picam2 = Picamera2()

    # for m in picam2.sensor_modes:
    #     print(m)

    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080)},
        sensor={"output_size": (4608, 2592)},
        # main={"size": (1920, 1080)}, sensor={"output_size": (1920, 1080)}
    )

    picam2.configure(config)
    picam2.start_preview(Preview.QTGL, transform=Transform(hflip=1))
    picam2.start()

    time.sleep(100)


class GroveGSRSensor:
    def __init__(self, channel):
        self.channel = channel
        self.adc = ADC()

    @property
    def GSR(self):
        value = self.adc.read(self.channel)
        return value


def calibrate_gsr_sensor():
    if len(sys.argv) < 2:
        print("Usage: {} adc_channel".format(sys.argv[0]))
        sys.exit(1)

    sensor = GroveGSRSensor(int(sys.argv[1]))

    print("Detecting...")
    while True:
        avg = 0
        for i in range(10):
            avg += sensor.GSR
            time.sleep(0.3)
        print(f"Human resistance: {avg / 10}")

def read_gsr_sensor():
    if len(sys.argv) < 2:
        print("Usage: {} adc_channel".format(sys.argv[0]))
        sys.exit(1)

    sensor = GroveGSRSensor(int(sys.argv[1]))
    serial_calibration = 509

    print("Detecting...")
    while True:
        reading = sensor.GSR
        div = serial_calibration - reading
        if (div != 0):
            human_resistance_ohms = ((1024 + 2 * reading) * 10000) / div
            print(reading)
            print(f"Human resistance: {human_resistance_ohms/1000}KOhms")
        time.sleep(0.3)


if __name__ == "__main__":
    # calibrate_gsr_sensor()
    read_gsr_sensor()
