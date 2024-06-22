const int IN1_PIN = 9; // The Arduino Nano pin connected to the IN1 pin L298N
const int IN2_PIN = 10; // The Arduino Nano pin connected to the IN2 pin L298N
const int IN3_PIN = 11; // The Arduino Nano pin connected to the IN1 pin L298N
const int IN4_PIN = 12;  

void setup() {
  // Initialize digital pins as outputs
  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);
  pinMode(IN3_PIN, OUTPUT);
  pinMode(IN4_PIN, OUTPUT);
  Serial.begin (9600);

}

void endavant(int pin1, int pin2, int temps) {
  digitalWrite(pin1, HIGH); 
  digitalWrite(pin2, LOW);

  if(temps != 0){
    delay(temps);
  }
}

void enrrera(int pin1, int pin2, int temps) {
  digitalWrite(pin1, LOW); 
  digitalWrite(pin2, HIGH);

  if(temps != 0){
    delay(temps);
  }
}

void pararMotors(int pin1, int pin2){
  digitalWrite(pin1, LOW); 
  digitalWrite(pin2, LOW);
}

void endavantv2(int pin1, int pin2, int pin3, int pin4, int temps) {
  digitalWrite(pin1, HIGH); 
  digitalWrite(pin2, LOW);
  digitalWrite(pin3, HIGH); 
  digitalWrite(pin4, LOW);

  if(temps != 0){
    delay(temps);
  }
}

void enrrerav2(int pin1, int pin2, int pin3, int pin4, int temps) {
  digitalWrite(pin1, LOW); 
  digitalWrite(pin2, HIGH);
  digitalWrite(pin3, LOW);
  digitalWrite(pin4, HIGH);

  if(temps != 0){
    delay(temps);
  }
}

void pararMotorsv2(int pin1, int pin2, int pin3, int pin4){
  digitalWrite(pin1, LOW); 
  digitalWrite(pin2, LOW);
  digitalWrite(pin3, LOW); 
  digitalWrite(pin4, LOW);
}

/*
Robar "num" cartes de un dels dos motors.
*/
void robar(int pin1, int pin2, int num){
  int var = 0;
  while(var < num){
    enrrera(pin1, pin2, 80);
    endavant(pin1, pin2, 120);
    enrrera(pin1, pin2, 80);
    pararMotors(pin1, pin2);
    delay(1000);
    var++;
  }
}

void loop() {
    if (Serial.available() > 0) {
      String message = Serial.readStringUntil('\n'); // Lee el mensaje hasta un salto de l�nea
      int numero = message.toInt(); // Convierte el mensaje a un n�mero entero
      if (numero == 0) { //BARREJAR
        endavantv2(IN1_PIN, IN2_PIN, IN3_PIN, IN4_PIN, 1000);
        pararMotorsv2(IN1_PIN, IN2_PIN, IN3_PIN, IN4_PIN);
      }
      else {
        robar(IN1_PIN, IN2_PIN, numero);
      }
    }
}



//Altres funcions
/*
void barrejarv2(int pin1, int pin2, int pin3, int pin4){
  int var = 0;
  while(var < 20){
    enrrera(IN1_PIN, IN2_PIN, 100);
    endavant(IN1_PIN, IN2_PIN, 200);
    enrrera(IN1_PIN, IN2_PIN, 100);
    pararMotorsv2(IN1_PIN, IN2_PIN, IN3_PIN, IN4_PIN);
    delay(100);
    enrrera(IN3_PIN, IN4_PIN, 100);
    endavant(IN3_PIN, IN4_PIN, 200);
    enrrera(IN3_PIN, IN4_PIN, 100);
    pararMotorsv2(IN1_PIN, IN2_PIN, IN3_PIN, IN4_PIN);
    delay(100);
    var++;
  }
}


// La idea es barrejar traient 2 o 3 cartes d'una banda cada cop
// De moment per provar es repetiria 10 vegades

void barrejar(int pin1, int pin2, int pin3, int pin4){
  int var = 0;
  while(var < 10){
    //Nomes treu cartes una banda
    digitalWrite(pin1, HIGH); 
    digitalWrite(pin2, LOW);
    digitalWrite(pin3, LOW); 
    digitalWrite(pin4, LOW);
    delay(200);
    //Nomes treu cartes l'altra banda
    digitalWrite(pin1, LOW); 
    digitalWrite(pin2, LOW);
    digitalWrite(pin3, HIGH); 
    digitalWrite(pin4, LOW);
    delay(200);
    var++;
  }
}
*/
