from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import serial
import sounddevice as sd
from serial.tools import list_ports

from audio import read_wav


BAUDRATE = 115200

SERVO_1_START = 0
SERVO_1_END = 130

SERVO_2_START = 180
SERVO_2_END = 50

MOVE_DELAY = 0.4
SHORT_PAUSE = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECOGNIZED_PATH = PROJECT_ROOT / "data" / "parrot_voice" / "dance_recognized.wav"
SONG_PATH = PROJECT_ROOT / "data" / "parrot_voice" / "Dancing_in_september.wav"


def detect_default_port() -> str:
    env_port = os.environ.get("PARROT_SERIAL_PORT")
    if env_port:
        return env_port

    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    if not ports:
        raise RuntimeError(
            "No serial ports found. Connect the device or set PARROT_SERIAL_PORT."
        )

    preferred_tokens = {
        "Darwin": ["usbmodem", "usbserial", "tty.usb", "cu.usb"],
        "Linux": ["ttyUSB", "ttyACM", "usbserial"],
        "Windows": ["COM"],
    }.get(platform.system(), [])

    for port in ports:
        device = port.device.lower()
        description = (port.description or "").lower()

        if any(token.lower() in device for token in preferred_tokens):
            return port.device

        if any(
            token in description
            for token in ("arduino", "usb serial", "cp210", "ch340", "esp32")
        ):
            return port.device

    return ports[0].device


def open_serial(port: str | None = None, baudrate: int = BAUDRATE) -> serial.Serial:
    selected_port = port or detect_default_port()
    ser = serial.Serial(selected_port, baudrate, timeout=1)
    time.sleep(2.0)
    return ser


def play_wav_blocking(path: Path) -> None:
    samples, sample_rate = read_wav(path)
    sd.play(samples, samplerate=sample_rate, blocking=True)
    sd.stop()


def play_wav_async(path: Path) -> None:
    samples, sample_rate = read_wav(path)
    sd.play(samples, samplerate=sample_rate, blocking=False)


def stop_wav_async() -> None:
    sd.stop()


def move(ser: serial.Serial, servo_name: str, angle: int) -> None:
    safe_angle = max(0, min(180, int(angle)))
    ser.write(f"{servo_name}:{safe_angle}\n".encode())


def move_both(ser: serial.Serial, s1_angle: int, s2_angle: int) -> None:
    move(ser, "s1", s1_angle)
    move(ser, "s2", s2_angle)


def cycle_single(
    ser: serial.Serial,
    servo_name: str,
    start_angle: int,
    end_angle: int,
    pause: float = MOVE_DELAY,
) -> None:
    move(ser, servo_name, end_angle)
    time.sleep(pause)
    move(ser, servo_name, start_angle)
    time.sleep(pause)


def cycle_both(
    ser: serial.Serial,
    s1_start: int = SERVO_1_START,
    s1_end: int = SERVO_1_END,
    s2_start: int = SERVO_2_START,
    s2_end: int = SERVO_2_END,
    pause: float = MOVE_DELAY,
) -> None:
    move_both(ser, s1_end, s2_end)
    time.sleep(pause)
    move_both(ser, s1_start, s2_start)
    time.sleep(pause)


def perform_dance_moves(
    ser: serial.Serial,
    *,
    short_pause: float = SHORT_PAUSE,
) -> None:
    move_both(ser, SERVO_1_START, SERVO_2_START)
    time.sleep(short_pause)

    cycle_both(ser, pause=MOVE_DELAY)
    time.sleep(short_pause)

    move(ser, "s1", SERVO_1_END)
    time.sleep(MOVE_DELAY)
    move(ser, "s1", SERVO_1_START)

    move(ser, "s2", SERVO_2_END)
    time.sleep(MOVE_DELAY)
    move(ser, "s2", SERVO_2_START)

    move(ser, "s1", SERVO_1_END)
    time.sleep(MOVE_DELAY)
    move(ser, "s1", SERVO_1_START)

    time.sleep(short_pause)
    cycle_both(ser, pause=MOVE_DELAY)
    time.sleep(short_pause)

    move(ser, "s1", SERVO_1_END)
    time.sleep(MOVE_DELAY)
    move(ser, "s1", SERVO_1_START)


def dance_sequence(
    ser: serial.Serial,
    recognized_path: Path = RECOGNIZED_PATH,
    song_path: Path = SONG_PATH,
    short_pause: float = SHORT_PAUSE,
) -> None:
    play_wav_blocking(recognized_path)
    play_wav_async(song_path)

    try:
        perform_dance_moves(ser, short_pause=short_pause)
        sd.wait()
    finally:
        stop_wav_async()


def run_dance_movement(port: str | None = None, baudrate: int = BAUDRATE) -> None:
    ser = open_serial(port=port, baudrate=baudrate)
    try:
        perform_dance_moves(ser)
    finally:
        ser.close()


def run_dance_sequence(port: str | None = None, baudrate: int = BAUDRATE) -> None:
    ser = open_serial(port=port, baudrate=baudrate)
    try:
        dance_sequence(ser)
    finally:
        ser.close()


def main() -> None:
    run_dance_sequence()


if __name__ == "__main__":
    main()