import socket
import bluetooth
import subprocess

# Configura el socket Bluetooth
server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

# Puerto RFCOMM arbitrario
port = 1
uuid = "00001101-0000-1000-8000-00805F9B34FB"  # UUID estandar para Bluetooth SPP

# Enlaza el socket al puerto
server_socket.bind(("", port))

# Escucha hasta 1 conexion entrante
server_socket.listen(1)

# Espera a que se establezca la conexion
client_socket, client_address = server_socket.accept()
print("Conexion establecida con", client_address)

while True:
    try:
        # Recibe datos del cliente
        data = client_socket.recv(1024)
        decoded_data = data.decode()
        if decoded_data == "CLOSE_CONNECTION":
            print("Cliente se ha desconectado de forma deliberada")
            break
        elif "USER_" in decoded_data:
            user_name = decoded_data.replace("USER_", "")
            with open("usuarios.txt", "a") as file:
                file.write(user_name + "\n")
        elif decoded_data == "ORDEN_INICIAR":
            subprocess.run(["python3", "main2.py"])

        print("Mensaje recibido:", data.decode())


    except Exception as e:
        if "Connection reset by peer" in str(e):
            print("Cliente se ha desconectado de forma abrupta")
        else:
            print("Error:", e)
        break

# Cierra la conexion
client_socket.close()
server_socket.close()