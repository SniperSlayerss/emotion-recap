import asyncio
from bleak import BleakClient

ADDRESS = "00:22:D0:47:9C:DE"

HR_CHAR = "00002a37-0000-1000-8000-00805f9b34fb"
BATTERY_CHAR = "00002a19-0000-1000-8000-00805f9b34fb"
MANUFACTURER_CHAR = "00002a29-0000-1000-8000-00805f9b34fb"
MODEL_CHAR = "00002a24-0000-1000-8000-00805f9b34fb"
SERIAL_CHAR = "00002a25-0000-1000-8000-00805f9b34fb"
FIRMWARE_CHAR = "00002a26-0000-1000-8000-00805f9b34fb"


def handle_hr(sender, data):
    flags = data[0]
    if flags & 0x01:
        hr = int.from_bytes(data[1:3], byteorder="little")
        rr_offset = 3
    else:
        hr = data[1]
        rr_offset = 2

    print(f"HR: {hr} bpm")

    if flags & 0x10:
        rr_values = []
        while rr_offset + 1 < len(data):
            rr_raw = int.from_bytes(data[rr_offset : rr_offset + 2], byteorder="little")
            rr_values.append(round(rr_raw * 1000 / 1024, 1))
            rr_offset += 2
        print(f"RR intervals (ms): {rr_values}")


async def read_battery(client):
    data = await client.read_gatt_char(BATTERY_CHAR)
    print(f"Battery: {data[0]}%")


async def read_device_info(client):
    for label, uuid in [
        ("Manufacturer", MANUFACTURER_CHAR),
        ("Model", MODEL_CHAR),
        ("Serial", SERIAL_CHAR),
        ("Firmware", FIRMWARE_CHAR),
    ]:
        data = await client.read_gatt_char(uuid)
        print(f"{label}: {data.decode('utf-8', errors='replace')}")


async def main():
    print(f"Connecting to {ADDRESS}...")
    async with BleakClient(ADDRESS, timeout=60.0) as client:
        print(f"Connected: {client.is_connected}")

        # await read_device_info(client)
        # await read_battery(client)

        await client.start_notify(HR_CHAR, handle_hr)
        await asyncio.sleep(30)


asyncio.run(main())
