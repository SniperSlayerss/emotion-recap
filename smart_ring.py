from colmi_r02_client import real_time
from colmi_r02_client.client import Client
from colmi_r02_client.hr_settings import hr_log_settings_packet, HeartRateLogSettings

from datetime import datetime
from pathlib import Path
import asyncio
import time


from collections.abc import AsyncGenerator

async def stream_real_time_reading(
    client, reading_type: real_time.RealTimeReading
) -> AsyncGenerator[int, None]:
    await client.send_packet(real_time.get_start_packet(reading_type))
    consecutive_timeouts = 0

    try:
        while consecutive_timeouts < 5:  # stop after 5 consecutive timeouts (~10s of silence)
            try:
                data: real_time.Reading | real_time.ReadingError = await asyncio.wait_for(
                    client.queues[real_time.CMD_START_REAL_TIME].get(),
                    timeout=2.0,
                )
                if isinstance(data, real_time.ReadingError):
                    print("Reading error")
                    return
                if data.value != 0:
                    consecutive_timeouts = 0  # reset on any valid packet
                    yield data.value

            except TimeoutError:
                consecutive_timeouts += 1
                print(f"Timeout waiting for reading ({consecutive_timeouts}/5)")
    finally:
        await client.send_packet(real_time.get_stop_packet(reading_type))

async def main():
    client = Client("32:32:41:36:2B:04", Path("smart_ring.log"))
    await client.connect()

    async for value in stream_real_time_reading(
        client, real_time.RealTimeReading.HEART_RATE
    ):
        print(f"Got value: {value}")
    # process each reading immediately as it arrives
    # await asyncio.sleep(2)

    # packet = hr_log_settings_packet(HeartRateLogSettings(enabled = True, interval = 2))
    # await client.send_packet(packet)

    # await client.set_time(datetime.now())
    # print(await client.get_device_info())
    # print(await client.get_heart_rate_log(datetime(2026, 3, 12)))

    # battery = await client.get_battery()
    # print(battery)

    # reading = await client.get_realtime_reading(RealTimeReading.HEART_RATE)
    # print(reading)
    # reading = await client.get_realtime_reading(RealTimeReading.BLOOD_PRESSURE)
    # print(reading)


# asyncio.run(main())
# with open("smart_ring.log", "r") as f:
#     for line in f:
#         packet = bytearray.fromhex(line.strip().replace(" ", ""))
#         result = parse_real_time_reading(packet)
#         print(result)
asyncio.run(main())
