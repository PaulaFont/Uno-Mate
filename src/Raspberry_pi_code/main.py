import clases
import firebase
import funcions_logica as funcions
import algorisme_final
import funcions_raspberry
import time
import cv2
import os
import threading

COLORS_DISPONIBLES = ["Red", "Blue", "Green", "Yellow"]

# Inicialitzar la partida

funcions.listar_dispositivos()

# Pillar els jugadors que ens passa l'arduino que rep per bluetooth
jugadors = funcions.get_noms_jugadors()
partida = clases.Partida(jugadors)  # Crear la partida

for jugador in partida.jugadors:
    print(jugador.torn_jugador)
    print(jugador.nom)


# Repartir les cartes
funcions_raspberry.repartir_cartes_inicials(len(partida.get_jugadors()))

# Primera carta
cap = cv2.VideoCapture(0)  # Mantenir la càmera oberta tota la estona ja que tarda molt en obrir i tancar
print("Colocar carta inicial")
carta_tirada = False
while not carta_tirada:  # Mentre que no s'ha posat i detectat la primera carta del tauler no seguim
    funcions.nou_frame(cap)  # Trec una nova captura de la camara per buscar la carta cada 3 segons
    figura, color, coordenades = algorisme_final.detectar_carta("prova_partida_entrega/WIN_20240517_11_27_43_Pro.jpg")
    if figura != "Null" and color != "Null":
        carta_tirada = True  # Un cop tenim la primera carta detectada ja seguim
        carta = clases.Carta(color, figura, partida.jugadors[partida.torn_actual].nom)
        partida.pila_joc.append(carta)  # Guardar la primera carta al stack
    else:
        time.sleep(3)

print("Valor de la carta: " + str(partida.pila_joc[0].valor))
print("Color de la carta: " + str(partida.pila_joc[0].color))

# Comença l'algorisme

acabar_partida = False

while not acabar_partida:
    finalitzar_torn = False
    print("Torn de " + str(partida.jugadors[partida.torn_actual].nom))
    # posar aqui un while de la part d'escoltar 5 segons i veure si es la mateixa carta, fins que no digui robar o tiri
    # no seguira amb la resta de torns que sera el bucle d'abaix normal com fins ara
    while not finalitzar_torn:
        # Aqui tiraria la carta el jugador o dir "Robar" si no te carta, ho faig amb threads per escoltar a la vegada
        robar_carta = [False]
        parar_escoltar = [False]  # Lista para pasar el valor por referencia
        thread2 = threading.Thread(target=funcions.escoltar_robar, args=(parar_escoltar, robar_carta,))
        thread1 = threading.Thread(target=funcions.wait_and_stop, args=(parar_escoltar,))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        if robar_carta[0]:  # En cas d'haver de robar una carta passem de torn
            finalitzar_torn = True
            funcions.actualitzar_torns(partida)
            funcions_raspberry.repartir_n_cartes(1)
            partida.jugadors[partida.torn_actual].num_cartes += 1  # Li sumem les cartes necessaries
            carta = clases.Carta("", "", str(partida.jugadors[partida.torn_actual].nom))
            partida.pila_joc.append(carta)
            break

        # Haure de revisar que la carta no es la mateixa d'abans i per tant el jugador no ha tirat res
        figura, color, coordenades = algorisme_final.detectar_carta("prova_partida_entrega/WIN_20240517_11_27_43_Pro.jpg") # Si te carta i tira
        carta = clases.Carta(color, figura, str(partida.jugadors[partida.torn_actual].nom))
        print("Valor de la carta: " + str(carta.valor))
        print("Color de la carta: " + str(carta.color))
        moviment_valid = funcions.moviment_legal(carta, partida)  # Mirar si la carta tirada es valida
        if moviment_valid:  # Si el moviment es valid
            print("MOVIMENT VALID")
            finalitzar_torn = True
            partida.jugadors[partida.torn_actual].num_cartes -= 1  # Li restem una carta de la mà
            print("N cartes: " + str(partida.jugadors[partida.torn_actual].num_cartes))
            if carta.color == "Wild":  # Si la carta tirada es Wild hem d'especificar quin color volem
                ha_dit_color = False
                while not ha_dit_color:  # Fins que no es digui un color valid preguntarem pel color de la carta
                    color_detectat = funcions.escoltar_color()
                    if color_detectat.lower() in COLORS_DISPONIBLES:  # Si el color es valid:
                        carta.color_wild = color_detectat.lower()  # Assignar el color de la carta wild
                        ha_dit_color = True
                    else:
                        print("Color no vàlid. Repeteix el color")
            if carta.valor == "Reverse":
                partida.direccio = 1 - partida.direccio
            if carta.valor == "Skip":
                funcions.actualitzar_skip(partida)

            partida.pila_joc.append(carta)  # Si el moviment es valid, guardem la carta al stack

        else:
            print("MOVIMENT ILEGAL")
            # si el moviment no es valid el robot pitara i s'haura de treure la carta
            break
        # Si la carta es +num haurem de repartir le
        # s cartes que toquen si e
        # l jugador no pot continuar la cadena de +4/+2
        ha_guanyat = funcions.check_victoria(partida.jugadors[partida.torn_actual])
        if ha_guanyat:
            id_joc = "exemple_hist"
            data = funcions.dades_partida(partida)
            firebase.save_game(id_joc, data)
            acabar_partida = True
        funcions.actualitzar_torns(partida)

print("PARTIDA FINALITZADA")

