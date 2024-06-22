import clases
import firebase
import funcions_logica as funcions
import algorisme_final
import funcions_raspberry
import time
import cv2
import os
import threading

COLORS_DISPONIBLES = ["rojo", "azul", "verde", "amarillo"]

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
    figura, color = algorisme_final.detectar_carta("prova_partida_entrega/WIN_20240517_11_27_43_Pro.jpg")
    if figura != "Null" and color != "Null":
        carta_tirada = True  # Un cop tenim la primera carta detectada ja seguim
        carta = clases.Carta(color, figura)
        partida.pila_joc.append(carta)  # Guardar la primera carta al stack
    else:
        time.sleep(3)

print("Valor de la carta: " + str(partida.pila_joc[0].valor))
print("Color de la carta: " + str(partida.pila_joc[0].color))

# Comença l'algorisme

acabar_partida = False

while not acabar_partida:  # INICIAR PARTIDA
    finalitzar_torn = False
    print("Torn de " + str(partida.jugadors[partida.torn_actual].nom))
    primera_jugada = True
    while not finalitzar_torn: # COMENÇA UN JUGADOR
        acabar_jugada = False
        while not acabar_jugada:
            # Aqui tiraria la carta el jugador o dir "Robar" si no te carta, ho faig amb threads per escoltar a la vegada
            robar_carta = [False]
            parar_escoltar = [False]  # Lista para pasar el valor por referencia
            thread2 = threading.Thread(target=funcions.escoltar_robar, args=(parar_escoltar, robar_carta,))
            thread1 = threading.Thread(target=funcions.wait_and_stop, args=(parar_escoltar,))
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()

            if robar_carta[0]:  # En cas d'haver de robar una carta acaba una jugada
                # TODO: Comprovar que sigui la primera jugada del jugador. SINO NO ES PERMET ROBAR
                acabar_jugada = True
                primera_jugada = False
                #funcions.actualitzar_torns(partida)
                funcions_raspberry.repartir_n_cartes(1)
                partida.jugadors[partida.torn_actual].num_cartes += 1  # Li sumem les cartes necessaries
                carta = clases.Carta("", "", str(partida.jugadors[partida.torn_actual].nom))
                partida.pila_joc.append(carta)
                # TODO: Aquí es pot fer lo del +2/+4 (robar quan et donen +2 o +4)
                break #sortim del bucle de la jugada.

            # funcions.nou_frame(cap)  # Trec una nova captura de la camara per buscar la carta que s'ha tirat
            # TODO: Haure de revisar que la carta no es la mateixa d'abans i per tant el jugador no ha tirat res
            canvi = False
            figura, color = algorisme_final.detectar_carta("prova_partida_entrega/WIN_20240517_11_50_01_Pro.jpg")
            # SI HA CANVIAT LA CARTA
            # TODO: posar bé el if següent (if ha canviat la carta)
            if canvi:
                primera_jugada = False
                carta = clases.Carta(color, figura, str(partida.jugadors[partida.torn_actual].nom))
                print("Valor de la carta: " + str(carta.valor))
                print("Color de la carta: " + str(carta.color))
                moviment_valid = funcions.moviment_legal(carta, partida)  # Mirar si la carta tirada es valida
                if moviment_valid:  # Si el moviment és vàlid
                    print("MOVIMENT VALID")
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

                    partida.pila_joc.append(carta)  # Si el moviment és vàlid, guardem la carta al stack
                    acabar_jugada = True

                else:
                    print("MOVIMENT ILEGAL")
                    #TODO: si el moviment no es valid el robot pitarà i s'haurà de treure la carta
                    break

                ha_guanyat = funcions.check_victoria(partida.jugadors[partida.torn_actual])
                if ha_guanyat:
                    id_joc = "exemple"
                    data = funcions.dades_partida(partida)
                    firebase.save_game(id_joc, data)
                    acabar_partida = True
                    finalitzar_torn = True

            elif not primera_jugada:
            #Han passat 5 segons, no és la primera jugada i NO HA CANVIAT RES: canvi de torn
                finalitzar_torn = True
                #TODO: actualitzar torns tenint en compta el SKIP
                funcions.actualitzar_torns(partida)
            else:
            #Han passat 5 segons, és la primera jugada i NO HA CANVIAT RES
                #TODO: controlar el cas del +2 o +4 i canviar directament de torn (mirar els ifs que estan malament:)
                # ara està fet de manera que si et tiren un +2/+4 i passen 5 segons sense que facis res te les chupes.
                # potser es pot canviar perque no hi hagi temps. Llavors s'hauràn de demanar per veu les cartes. "ROBAR"
                #
                # Si la carta es +num haurem de repartir les cartes que toquen si
                # el jugador no pot continuar la cadena de +4/+2
                chupa = False
                carta_previa = partida.pila_joc[-1]
                if carta_previa.valor == "+2":
                    funcions_raspberry.repartir_n_cartes(2)
                    partida.jugadors[partida.torn_actual].num_cartes += 2
                    finalitzar_torn = True
                    funcions.actualitzar_torns(partida)

                elif carta_previa.valor == "+4":
                    funcions_raspberry.repartir_n_cartes(4)
                    partida.jugadors[partida.torn_actual].num_cartes += 4
                    finalitzar_torn = True
                    funcions.actualitzar_torns(partida)
        # Estem fora el while de jugada
        # Ara pot fer la següent jugada o no depenent de finalitzar_torn
    # Estem fora del torn. Aixo vol dir que juga un nou jugador
    # Es posa primera_jugada a true

print("PARTIDA FINALITZADA")

