// =============================================================================
// servo_control.ino — Ejecutor inmediato de servos con máquina de estados
// =============================================================================
// Hardware:
//   - Arduino UNO
//   - 2 servomotores en pines PWM D9 (izquierdo) y D6 (derecho)
//   - Alimentación de servos desde fuente externa 5V/3-5A con GND común
//
// Protocolo serial recibido desde Python:
//   "L\n" → accionar servo carril izquierdo (D9)
//   "R\n" → accionar servo carril derecho  (D6)
//
// Arduino NO gestiona delays largos ni colas.
// Todo el timing de tránsito vive en actuator.py (Python).
//
// Máquina de estados por servo:
//   CERRADO → recibe comando → ABRIENDO → millis() >= t_cierre → CERRANDO → CERRADO
//   Mientras el servo está abierto, el loop sigue corriendo y puede
//   recibir comandos del otro carril sin bloqueos.
//
// Notas:
//   SERVO_LEFT_PIN y SERVO_RIGHT_PIN son configuración de hardware.
//   Si cambian los pines físicos, modificar únicamente esas constantes.
// =============================================================================

#include <Servo.h>

// =============================================================================
// CONFIGURACIÓN
// =============================================================================

#define SERVO_LEFT_PIN   9    // D9 → servo carril izquierdo
#define SERVO_RIGHT_PIN  6    // D6 → servo carril derecho

#define ANGULO_CERRADO_L   13
#define ANGULO_ABIERTO_L  100

#define ANGULO_CERRADO_R   12
#define ANGULO_ABIERTO_R  100

#define TIEMPO_ABIERTO_MS  400   // ms que la compuerta permanece abierta

#define BAUDRATE  115200

// =============================================================================
// ESTADOS
// =============================================================================

enum EstadoServo { CERRADO, ABIERTO };

struct CanalServo {
  Servo        servo;
  EstadoServo  estado;
  unsigned long t_cierre;   // millis() en que debe cerrarse
  int          ang_abierto;
  int          ang_cerrado;
  char         id;
};

CanalServo canales[2];

// =============================================================================
// BUFFER SERIAL
// =============================================================================

char serial_buffer[16];
int  serial_idx = 0;

// =============================================================================
// SETUP
// =============================================================================

void setup() {
  Serial.begin(BAUDRATE);

  // Carril izquierdo (L)
  canales[0].servo.attach(SERVO_LEFT_PIN);
  canales[0].estado     = CERRADO;
  canales[0].t_cierre   = 0;
  canales[0].ang_abierto  = ANGULO_ABIERTO_L;
  canales[0].ang_cerrado  = ANGULO_CERRADO_L;
  canales[0].id           = 'L';
  canales[0].servo.write(ANGULO_CERRADO_L);

  // Carril derecho (R)
  canales[1].servo.attach(SERVO_RIGHT_PIN);
  canales[1].estado     = CERRADO;
  canales[1].t_cierre   = 0;
  canales[1].ang_abierto  = ANGULO_ABIERTO_R;
  canales[1].ang_cerrado  = ANGULO_CERRADO_R;
  canales[1].id           = 'R';
  canales[1].servo.write(ANGULO_CERRADO_R);

  delay(300);
  Serial.println("READY");
}

// =============================================================================
// LOOP PRINCIPAL — no bloqueante
// =============================================================================

void loop() {
  leer_serial();
  actualizar_servos();
}

// =============================================================================
// LECTURA SERIAL NO BLOQUEANTE
// =============================================================================

void leer_serial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      serial_buffer[serial_idx] = '\0';
      parsear_comando(serial_buffer);
      serial_idx = 0;
    } else if (serial_idx < 15) {
      serial_buffer[serial_idx++] = c;
    }
  }
}

// =============================================================================
// PARSING
// =============================================================================

void parsear_comando(char* cmd) {
  if (strlen(cmd) == 0) return;

  if (strcmp(cmd, "L") == 0) {
    abrir_canal(0);
  } else if (strcmp(cmd, "R") == 0) {
    abrir_canal(1);
  } else {
    Serial.print("ERR:cmd_desconocido:");
    Serial.println(cmd);
  }
}

void abrir_canal(int idx) {
  CanalServo& c = canales[idx];
  c.servo.write(c.ang_abierto);
  c.estado   = ABIERTO;
  c.t_cierre = millis() + TIEMPO_ABIERTO_MS;
  Serial.print("FIRE:");
  Serial.println(c.id);
}

// =============================================================================
// MÁQUINA DE ESTADOS — se llama cada iteración del loop
// =============================================================================

void actualizar_servos() {
  unsigned long ahora = millis();
  for (int i = 0; i < 2; i++) {
    CanalServo& c = canales[i];
    if (c.estado == ABIERTO && ahora >= c.t_cierre) {
      c.servo.write(c.ang_cerrado);
      c.estado = CERRADO;
      Serial.print("CLOSE:");
      Serial.println(c.id);
    }
  }
}
