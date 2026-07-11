#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

#define SERVO_REPOSO 120
#define SERVO_ACTIVO 300

#define TIEMPO_ESPERA 100      // <-- CAMBIAR AQUI (ms)
#define TIEMPO_ACTIVACION 2000   // <-- tiempo que empuja el servo

#define NUM_CARRILES 4

String comando = "";

void setup() {

  Serial.begin(115200);

  pca.begin();
  pca.setPWMFreq(50);

  delay(10);

  // poner todos los servos en reposo
  for(int i=0;i<NUM_CARRILES;i++){
    pca.setPWM(i,0,SERVO_REPOSO);
  }

  Serial.println("Sistema listo");
}

void loop() {

  if(Serial.available()){

    comando = Serial.readStringUntil('\n');

    int coma = comando.indexOf(',');

    if(coma > 0){

      int carril = comando.substring(0,coma).toInt();
      int accion = comando.substring(coma+1).toInt();

      if(carril >=0 && carril < NUM_CARRILES){

        if(accion == 1){

          Serial.print("Activando carril ");
          Serial.println(carril);

          // tiempo desde deteccion hasta que llega al mecanismo
          delay(TIEMPO_ESPERA);

          // activar servo
          pca.setPWM(carril,0,SERVO_ACTIVO);

          delay(TIEMPO_ACTIVACION);

          // volver a reposo
          pca.setPWM(carril,0,SERVO_REPOSO);
        }
      }
    }
  }
}
