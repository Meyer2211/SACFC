#include <Servo.h>

Servo servos[4];

int servoPins[4] = {3, 5, 6, 9};

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 4; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].write(0);
  }
}

void loop() {

  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');

    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {

      int carril = data.substring(0, commaIndex).toInt();
      int estado = data.substring(commaIndex + 1).toInt();

      if (carril >= 0 && carril < 4) {

        if (estado == 1) {
          servos[carril].write(90);
          delay(5000);
          servos[carril].write(0);
        }
      }
    }
  }
}
