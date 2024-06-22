import cv2
from datetime import datetime
import speech_recognition as sr
import time
import firebase
import funcions_visio_computador
from google.cloud import texttospeech
from google.oauth2 import service_account
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

os.environ['PYTHONWARNINGS'] = 'ignore:ResourceWarning'
f = open(os.devnull, 'w')
os.dup2(f.fileno(), 2)

import pygame


def moviment_legal_inicial(carta, partida):
    legal = False
    ultima_carta = partida.pila_joc[-1]
    # Si estas en una cadena de +2 o +4, nomes podras tirar +2 o +4, qualsevol altra carta es ilegal
    if partida.cadena_robar is True:
        if carta.valor == ultima_carta.valor:
            legal = True
    else:
        # Si es el la primera jugada del torn, podem cambiar de color si el numero es el mateix, tirar una del
        # mateix color sense importar el numero, o puc tirar una wild
        if carta.color == ultima_carta.color or carta.valor == ultima_carta.valor or carta.color == "Wild"\
                or carta.color.lower() == ultima_carta.color_wild.lower():
            legal = True
    return legal


def moviment_legal_continuacio(carta, partida):
    legal = False
    ultima_carta = partida.pila_joc[-1]
    # En el cas de ser ja mínim la segona carta que tirem i per tant ja hem tirat una abans, només podem tirar
    # cartes on la figura sigui la mateixa que abans.
    # Ex: si tiro un 7, només podré tirar 7 en les següents
    if carta.valor == ultima_carta.valor:
        legal = True
    return legal


def get_noms_jugadors():
    jugadors = ["Misisuko", "Paula", "marc “TheMarcartero” artero", "Carles"]
    mails = ["misisuko1@gmail.com", "paulafsola@gmail.com", "marc.artero.pons@gmail.com", "carlesfornes003@mail.com"]
    return jugadors, mails


def nou_frame(cap):
    
    cap = cv2.VideoCapture(0) 

    
    if not cap.isOpened():
        print("Error: No s'ha pogut obrir la camera")
        exit()

    ret, frame = cap.read()
    
    cap.release()
    if not ret:
        print("Error: No es pot llegir el frame de la camera")

    frame_filename = "frame.jpg"
    cv2.imwrite(frame_filename, frame)
    return frame


def list_cameras(max_cameras=10):
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras


def check_victoria(jugador):
    victoria = False
    if jugador.num_cartes <= 0:
        victoria = True
    return victoria


def dades_partida(partida, mails):
    fecha_actual = datetime.now()
    game_data = {
        'fecha': fecha_actual.strftime('%d/%m/%Y'),
        'hora': fecha_actual.strftime('%H:%M'),
        'nombre': 'Partida',
        'userId': mails[partida.torn_actual],
        'turnos': partida.torns_totals + 1,
        'jugadores': len(partida.get_jugadors()),
        'ganador': partida.jugadors[partida.torn_actual].nom,
        'partida': [carta.to_dict() for carta in partida.historial],
        'usuarios': mails
    }
    return game_data


def act_player_data(partida, mails):
    for i, mail in enumerate(mails):
        dades_actuals = firebase.get_document("usuarios", mail)
        if dades_actuals is None:
            if partida.jugadors[i].ganador:
                dades_noves = {
                    'ganadas': 1,
                    'nombre': partida.jugadors[i].nom,
                    'partidas': 1
                }
            else:
                dades_noves = {
                    'ganadas': 0,
                    'nombre': partida.jugadors[i].nom,
                    'partidas': 1
                }
        else:
            if partida.jugadors[i].ganador:
                dades_noves = {
                    'ganadas': dades_actuals.get('ganadas', 0) + 1,
                    'nombre': partida.jugadors[i].nom,
                    'partidas': dades_actuals.get('partidas', 0) + 1
                }
            else:
                dades_noves = {
                    'ganadas': dades_actuals.get('ganadas', 0),
                    'nombre': partida.jugadors[i].nom,
                    'partidas': dades_actuals.get('partidas', 0) + 1
                }
        firebase.actualitzar_jugadors(mail, dades_noves)

    print("JUGADORS ACTUALITZATS")


def escoltar_color():
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=1) as source:
        print("Dir color")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source)
            color_detectat = recognizer.recognize_google_cloud(audio, language="es-ES")
            print("COLOR DETECTCAT " + str(color_detectat))
            time.sleep(5)
        except sr.UnknownValueError:
            print("No s'ha entés l'audio")
            color_detectat = "null"
        # color_detectat = color_detectat.replace(" ", "")
        if "rojo" in color_detectat:
            color_detectat = "red"
        elif "azul" in color_detectat:
            color_detectat = "blue"
        elif "verde" in color_detectat:
            color_detectat = "green"
        elif "amarillo" in color_detectat:
            color_detectat = "yellow"
        print(color_detectat)
        return color_detectat


def escoltar_robar(parar_escoltar, robar_carta):
    executar_mp3_sin_borrar("audios/tirar_o_robar.mp3")
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=1) as source:
        print("En cas de no tenir carta per tirar dir 'robar'")
        while not parar_escoltar[0]:  # Accede al valor dentro de la lista
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=5)
                robar_detectat = recognizer.recognize_google_cloud(audio, language="es-ES")
                if "robar" in robar_detectat.lower() or "probar" in robar_detectat.lower() or "roba" in robar_detectat.lower() :  # Si el color es valid:
                    robar_carta[0] = True
            except sr.UnknownValueError:
                if robar_carta[0] == True:
                    print("No s'ha entés l'audio")
                    executar_mp3_sin_borrar("audios/no_entiendo.mp3")
                    #robar_detectat = "null"
            except sr.WaitTimeoutError:
                print("Escoltant...")
                #robar_detectat = "null"
            #robar_detectat = robar_detectat.replace(" ", "")
            print(robar_detectat)


def wait_and_stop(parar_escoltar):
    time.sleep(5)
    parar_escoltar[0] = True  # Modifica el valor dentro de la lista


def actualitzar_torns(partida):
    for i in range(partida.skips_acumulats):
        if partida.direccio == 0:
            partida.torn_actual += 1
            if partida.torn_actual == len(partida.get_jugadors()):
                partida.torn_actual = 0
        else:
            partida.torn_actual -= 1
            if partida.torn_actual == -1:
                partida.torn_actual = len(partida.get_jugadors()) - 1

    partida.torns_totals += 1
    partida.skips_acumulats = 1


def actualitzar_skip(partida):
    if partida.direccio == 0:
        partida.torn_actual += 1
        if partida.torn_actual == len(partida.get_jugadors()):
            partida.torn_actual = 0
    else:
        partida.torn_actual -= 1
        if partida.torn_actual == -1:
            partida.torn_actual = len(partida.get_jugadors()) - 1


def listar_dispositivos():
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))


def comprovar_coordenades_i_carta(coordenades_anteriors, coordenades, valor, color, partida):
    mateixa_carta = True
    if valor != partida.pila_joc[-1].valor or color != partida.pila_joc[-1].color:
        return False
    print(coordenades_anteriors, coordenades)
    for i in range(len(coordenades_anteriors)):
        distancia = funcions_visio_computador.distanciaEntrePunts(coordenades_anteriors[i], coordenades[i])
        if distancia > 80:
            return False
    return mateixa_carta


def ordenar_puntos(puntos):
    # Ordenar los puntos primero por y (de menor a mayor) y luego por x (de menor a mayor)
    puntos_ordenados = sorted(puntos, key=lambda punto: (punto[1], punto[0]))

    # Verificar que hay al menos cuatro puntos
    if len(puntos_ordenados) < 4:
        print(puntos)
        print(puntos_ordenados)
        raise ValueError("La lista de puntos debe contener al menos cuatro puntos")

    # Seleccionar los dos puntos más arriba
    arriba = puntos_ordenados[:2]
    # Seleccionar los dos puntos más abajo
    abajo = puntos_ordenados[2:]

    # Ordenar los puntos de arriba de izquierda a derecha
    arriba = sorted(arriba, key=lambda punto: punto[0])
    # Ordenar los puntos de abajo de izquierda a derecha
    abajo = sorted(abajo, key=lambda punto: punto[0])

    # Combinar los resultados en el orden especificado
    orden_final = [arriba[0], arriba[1], abajo[0], abajo[1]]

    return orden_final


def get_mails():
    mails = []
    with open("user_names.txt", "r") as file:
        for line in file:
            mails.append(line.strip())

    return mails


def borrar_txt():
    with open("user_names.txt", "w") as file:
        pass

import os

from google.cloud import texttospeech


# Función para convertir texto a audio
# Credenciales y cliente
client_file = 'speechToTextPrivateKey.json'
credentials = service_account.Credentials.from_service_account_file(client_file)
client = texttospeech.TextToSpeechClient(credentials=credentials)

# Lista de textos para convertir a audio
textos = ["Repartiendo 4 cartas"]

# Función para convertir texto a audio
def texto_a_audio(texto, nombre_archivo):
    # Configuración de la solicitud de síntesis
    input_text = texttospeech.SynthesisInput(text=texto)

    # Configuración de la voz
    voz = texttospeech.VoiceSelectionParams(
        language_code="es-ES",  # Cambia el código de idioma si es necesario
        name="es-ES-Wavenet-C",  # Cambia esto por el nombre de la voz que prefieras
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    # Configuración del tipo de audio
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Realizar la solicitud para sintetizar el texto en audio
    response = client.synthesize_speech(
        input=input_text, voice=voz, audio_config=audio_config
    )

    # Guardar el audio en un archivo
    with open(nombre_archivo, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio guardado como {nombre_archivo}")


def executar_mp3(ruta):
    pygame.mixer.init()

    pygame.mixer.music.load(ruta)

    pygame.mixer.music.play()

    # Mantén el programa en ejecución hasta que termine la reproducción
    while pygame.mixer.music.get_busy():
        time.sleep(1)

    # Detiene la música y desinicializa el mixer
    pygame.mixer.music.stop()
    pygame.mixer.quit()


    # Verifica si el archivo existe
    if os.path.exists(ruta):
        # Borra el archivo
        os.remove(ruta)
        print(f"{ruta} ha sido borrado.")
    else:
        print(f"{ruta} no existe.")


def executar_mp3_sin_borrar(ruta):
    pygame.mixer.init()

    pygame.mixer.music.load(ruta)

    pygame.mixer.music.play()

    # Mantén el programa en ejecución hasta que termine la reproducción
    while pygame.mixer.music.get_busy():
        time.sleep(1)

    # Detiene la música y desinicializa el mixer
    pygame.mixer.music.stop()
    pygame.mixer.quit()
