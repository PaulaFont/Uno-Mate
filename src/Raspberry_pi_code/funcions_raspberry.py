import serial
import time
import funcions_logica as funcions
import comunicacio_Arduino

# port = '/dev/ttyUSB0'
# ser = serial.Serial(port,9600, timeout=1)


def enviar_comanda(comanda):
#    ser.write(comanda.encode())
    return 0

def repartir_cartes_inicials(num_jugadors):
    funcions.executar_mp3_sin_borrar("audios/repartir_inicials.mp3")
    print("Repartir cartes a " + str(num_jugadors) + " jugadors")
    
    for jugador in range(num_jugadors):
        comunicacio_Arduino.enviar_arduino("7\n")
        time.sleep(10)
    

def repartir_n_cartes(n_cartes):
    funcions.executar_mp3_sin_borrar("audios/repartir_cartas.mp3")
    print("Repartir " + str(n_cartes) + " cartes")

    comunicacio_Arduino.enviar_arduino(str(n_cartes) + "\n")  
    
    time.sleep(n_cartes * 2)  
