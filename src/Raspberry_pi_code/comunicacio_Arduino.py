import serial
import time

# Configuracion del puerto serie


def enviar_arduino(data_to_send):
    arduino_port = "/dev/ttyUSB0"  # Reemplaza esto con el puerto serie correcto en tu sistema
    baudrate = 9600
    timeout = 1
    
    try:
        # Iniciar la conexion serie
        ser = serial.Serial(arduino_port, baudrate, timeout=timeout)
        print()
        print("INCIANDO COMUNIACION ARDUINO")
        print(f"Conectado a {arduino_port} a {baudrate} baudios.")
        
        # Esperar un momento para asegurarse de que la conexion se ha establecido
        time.sleep(2)
        
        # Enviar datos al Arduino
        ser.write(data_to_send.encode('utf-8'))
        
        # Cerrar la conexion serie
        ser.close()
        print("ACABANDO COMUNIACION ARDUINO")
        print()
                
    except serial.SerialException as e:
        print(f"Error al comunicarse con el Arduino: {e}")
        return None

# Ejemplo de uso
# communicate_with_arduino("3\n")
