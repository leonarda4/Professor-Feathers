import time
import shutil
import subprocess
from pathlib import Path

import serial


PORT = "/dev/cu.SLAB_USBtoUART"
BAUDRATE = 115200

SERVO_1_START = 50
SERVO_1_END = 130

SERVO_2_START = 130
SERVO_2_END = 50

MOVE_DELAY = 0.4
SHORT_PAUSE = 1
LONG_PAUSE = 2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECOGNIZED_PATH = PROJECT_ROOT / "data" / "parrot_voice" / "dance_recognized.wav"
SONG_PATH = PROJECT_ROOT / "data" / "parrot_voice" / "Dancing_in_september.mp3"


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
    afplay_path = shutil.which("afplay")
    if afplay_path is None:
        raise RuntimeError("macOS 'afplay' is required to play the dance song.")
    if not recognized_path.is_file():
        raise FileNotFoundError(f"Recognized clip not found: {recognized_path}")
    if not song_path.is_file():
        raise FileNotFoundError(f"Song file not found: {song_path}")

    subprocess.run(
        [afplay_path, str(recognized_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    song_process = subprocess.Popen(
        [afplay_path, str(song_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        perform_dance_moves(ser, short_pause=short_pause)
        song_process.wait()

    finally:
        if song_process.poll() is None:
            song_process.terminate()
            song_process.wait(timeout=1.0)


def run_dance_movement(port: str = PORT, baudrate: int = BAUDRATE) -> None:
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2.0)
    try:
        perform_dance_moves(ser)
    finally:
        ser.close()


def run_dance_sequence(port: str = PORT, baudrate: int = BAUDRATE) -> None:
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2.0)
    try:
        dance_sequence(ser)
    finally:
        ser.close()


def main() -> None:
    run_dance_sequence()


if __name__ == "__main__":
    main()
