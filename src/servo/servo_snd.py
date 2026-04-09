import serial
import time

class ServoController:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # allow ESP32 reset

    def move(self, servo: str, angle: int):
        cmd = f"{servo}:{angle}\n"
        self.ser.write(cmd.encode())

    def move_pair(self, s1: int = None, s2: int = None):
        if s1 is not None:
            self.move("s1", s1)
        if s2 is not None:
            self.move("s2", s2)

    def close(self):
        self.ser.close()



def main():
    servo = ServoController()