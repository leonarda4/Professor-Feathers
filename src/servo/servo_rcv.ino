#include <ESP32Servo.h>

Servo servo1;
Servo servo2;

const int pin1 = 18;
const int pin2 = 19;

String input = "";

void setup() {
  Serial.begin(115200);

  servo1.attach(pin1);
  servo2.attach(pin2);

  Serial.println("READY");
}

void handleCommand(String cmd) {
  // Expected: s1:45 or s2:120

  int sep = cmd.indexOf(':');
  if (sep == -1) return;

  String id = cmd.substring(0, sep);
  int angle = cmd.substring(sep + 1).toInt();

  angle = constrain(angle, 0, 180);

  if (id == "s1") {
    servo1.write(angle);
  } else if (id == "s2") {
    servo2.write(angle);
  }

  Serial.println("OK");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n') {
      handleCommand(input);
      input = "";
    } else {
      input += c;
    }
  }
}