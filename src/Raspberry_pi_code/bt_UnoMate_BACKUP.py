import socket
import bluetooth
import subprocess
import os
import comunicacio_Arduino


if 'XDG_RUNTIME_DIR' not in os.environ:
    os.environ['XDG_RUNTIME_DIR'] = f'/run/user/{os.getuid()}'
    
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

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
			with open("user_names.txt", "a") as file:
				file.write(user_name + "\n")

		elif decoded_data == "ORDEN_INICIAR":
			env = os.environ.copy()
			command = ["xterm", "-e", "python3", "main2.py"]
			subprocess.Popen(command, env=os.environ)
			#subprocess.run(["python3", "main2.py"])
		elif decoded_data == "ORDEN_MEZCLAR":
			comunicacio_Arduino.enviar_arduino("0\n")
		elif decoded_data == "ORDEN_2":
			comunicacio_Arduino.enviar_arduino("2\n")
		elif decoded_data == "ORDEN_4":
			comunicacio_Arduino.enviar_arduino("4\n")
		elif decoded_data == "ORDEN_REPARTIR":
			comunicacio_Arduino.enviar_arduino("7\n")
		print("Mensaje recibido:", decoded_data)
	except Exception as e:
		if "Connection reset by peer" in str(e):
			print("Cliente se ha desconectado de forma abrupta")
		else:
			print("Error:", e)
		break

# Cierra la conexion
client_socket.close()
server_socket.close()
