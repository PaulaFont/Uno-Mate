import clases
import firebase
import funcions_logica as funcions
import algorisme_final
print("HOLA antes")
import funcions_raspberry
print("HOLA despues")
import time
import cv2
import threading
import random


COLORS_DISPONIBLES = ["red", "blue", "green", "yellow"]
# funcions.texto_a_audio("Repartiendo cartas iniciales", "audios/repartir_inicials.mp3")
# funcions.executar_mp3_sin_borrar("audios/repartir_inicials.mp3")

# Pillar els jugadors que ens passa l'arduino que rep per bluetooth
mails = funcions.get_mails()
jugadors = []
for mail in mails:
    jugador = firebase.get_nom("usuarios", mail)
    jugadors.append(jugador)

partida = clases.Partida(jugadors)  # Crear la partida

# funcions.borrar_txt()

for jugador in partida.jugadors:
    print(jugador.torn_jugador)
    print(jugador.nom)
print("IMPORTS HECHOS")

# Repartir les cartes
funcions_raspberry.repartir_cartes_inicials(len(partida.get_jugadors()))
print("IMPORTS HECHOS2")

funcions.texto_a_audio("Los jugadores son.", 'nom.mp3')
funcions.executar_mp3("nom.mp3")
for jugador in partida.jugadors:
    funcions.texto_a_audio(str(jugador.nom), 'nom.mp3')
    funcions.executar_mp3("nom.mp3")

# Primera carta
cap = cv2.VideoCapture(0)  # Mantenir la cÃ mera oberta tota la estona ja que tarda molt en obrir i tancar
cap.release()
funcions.nou_frame(cap)
print("Colocar carta inicial")
carta_tirada = False
coordenades = []
while not carta_tirada:  # Mentre que no s'ha posat i detectat la primera carta del tauler no seguim
    funcions.executar_mp3_sin_borrar("audios/colocar_inicial.mp3")
    frame = funcions.nou_frame(cap)  # Trec una nova captura de la camara per buscar la carta cada 3 segons
    figura, color, coordenades = algorisme_final.detectar_carta(frame)
    if figura != "Null" and color != "Null":
        carta_tirada = True  # Un cop tenim la primera carta detectada ja seguim
        carta = clases.Carta(color, figura, "Carta Inicial")
        partida.pila_joc.append(carta)  # Guardar la primera carta al stack
        partida.historial.append(carta)  # l'Historial es el que penjarem al firebase i la pila de joc per controlar
    else:
        time.sleep(3)

print("Valor de la carta: " + str(partida.pila_joc[0].valor))
print("Color de la carta: " + str(partida.pila_joc[0].color))

# ComenÃ§a l'algorisme

acabar_partida = False
finalitzar_primera_jugada = False
coordenades_anteriors = coordenades

while not acabar_partida:
    finalitzar_torn = False
    finalitzar_primera_jugada = False
    print("Torn de " + str(partida.jugadors[partida.torn_actual].nom))
    funcions.texto_a_audio("Turno de " + str(partida.jugadors[partida.torn_actual].nom), 'nom.mp3')
    funcions.executar_mp3("nom.mp3")
    print("primer torn")
    # posar aqui un while de la part d'escoltar 5 segons i veure si es la mateixa carta, fins que no digui robar o tiri
    # no seguira amb la resta de torns que sera el bucle d'abaix normal com fins ara
    while not finalitzar_primera_jugada:
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
            finalitzar_primera_jugada = True
            finalitzar_torn = True
            if partida.cartes_a_robar == 0:
                partida.cartes_a_robar += 1
            partida.cadena_robar = False
            funcions_raspberry.repartir_n_cartes(partida.cartes_a_robar)
            partida.jugadors[partida.torn_actual].num_cartes += partida.cartes_a_robar  # Li sumem les cartes a robar
            carta = clases.Carta("", "", str(partida.jugadors[partida.torn_actual].nom))
            partida.historial.append(carta)  # NomÃ©s l'afegim al historial ja que sino ens afecta a la logica
            partida.cartes_a_robar = 0
            break

        carta_tirada = False
        while not carta_tirada:  # Mentre que no s'ha posat i detectat la primera carta del tauler no seguim
            frame = funcions.nou_frame(cap)  # Trec una nova captura de la camara per buscar la carta cada 3 segons
            figura, color, coordenades = algorisme_final.detectar_carta(frame)
            if figura != "Null" and color != "Null":
                carta_tirada = True  # Un cop tenim la primera carta detectada ja seguim
                carta = clases.Carta(color, figura, str(partida.jugadors[partida.torn_actual].nom))
            else:
                funcions.executar_mp3_sin_borrar("audios/carta_no_identificada.mp3")
                time.sleep(3)
                print("No s'ha detectat la carta")

        print("Valor de la carta: " + str(carta.valor))
        print("Color de la carta: " + str(carta.color))
        # Comprovo que no sigui la mateixa carta d'abans
        mateixes_coordenades = funcions.comprovar_coordenades_i_carta(coordenades_anteriors, coordenades, carta.valor,
                                                                      carta.color, partida)
        if mateixes_coordenades:  # Si les coordenades son les mateixes repetim el bucle, aixi fins que tiri o robi
            funcions.executar_mp3_sin_borrar("audios/carta_no_tirada.mp3")
            print("NO HA TIRAT CARTA, REPETIM BUCLE")
            continue
        print("HA TIRAT CARTA, MIREM SI ES VALID")
        moviment_valid = funcions.moviment_legal_inicial(carta, partida)  # Mirar si la carta tirada es valida
        if moviment_valid:  # Si el moviment es valid
            print("MOVIMENT VALID")
            funcions.executar_mp3_sin_borrar("audios/moviment_valid.mp3")
            coordenades_anteriors = coordenades  # Si les coordenades no son les mateixes, posem les noves com anteriors
            finalitzar_primera_jugada = True
            partida.jugadors[partida.torn_actual].num_cartes -= 1  # Li restem una carta de la mÃ 
            if carta.color == "Wild":  # Si la carta tirada es Wild hem d'especificar quin color volem
                funcions.executar_mp3_sin_borrar("audios/dir_color_wild.mp3")
                ha_dit_color = False
                while not ha_dit_color:  # Fins que no es digui un color valid preguntarem pel color de la carta
                    color_detectat = funcions.escoltar_color()
                    if color_detectat.lower() in COLORS_DISPONIBLES:  # Si el color es valid:
                        funcions.executar_mp3_sin_borrar("audios/color_valid.mp3")
                        carta.color_wild = color_detectat.lower()  # Assignar el color de la carta wild
                        ha_dit_color = True
                    else:
                        print("Color no vàlid. Repeteix el color")
                        funcions.executar_mp3_sin_borrar("audios/color_no_valid.mp3")
            if carta.valor == "Reverse":
                partida.direccio = 1 - partida.direccio
            if carta.valor == "Skip":
                partida.skips_acumulats += 1
            if carta.valor == "Draw4":
                partida.cadena_robar = True
                partida.cartes_a_robar += 4
            if carta.valor == "Draw2":
                partida.cadena_robar = True
                partida.cartes_a_robar += 2

            partida.pila_joc.append(carta)  # Si el moviment es valid, guardem la carta al stack
            partida.historial.append(carta)

        else:
            print("MOVIMENT ILEGAL")
            funcions.executar_mp3_sin_borrar("audios/moviment_ilegal.mp3")
            # si el moviment no es valid el robot pitara i s'haura de treure la carta
            funcions_raspberry.repartir_n_cartes(partida.cartes_a_robar + 2)
            partida.cartes_a_robar = 0
            restaurar_carta = False
            while not restaurar_carta:
                print("carta no restaurada")
                funcions.executar_mp3_sin_borrar("audios/quitar_carta_ilegal.mp3")
                time.sleep(1)
                frame = funcions.nou_frame(cap)
                figura, color, coordenades = algorisme_final.detectar_carta(frame)
                carta = clases.Carta(color, figura, str(partida.jugadors[partida.torn_actual].nom))
                if carta.valor == partida.pila_joc[-1].valor and carta.color == partida.pila_joc[-1].color:
                    restaurar_carta = True
            finalitzar_torn = True
            finalitzar_primera_jugada = True
            partida.cadena_robar = False
            print("carta restaurada")
            funcions.executar_mp3_sin_borrar("audios/ilegal_restaurat.mp3")

        ha_guanyat = funcions.check_victoria(partida.jugadors[partida.torn_actual])
        if ha_guanyat:
            partida.jugadors[partida.torn_actual].ganador = True
            document_existeix = True
            id_joc = "Null"
            while document_existeix:
                id_joc = "id_" + str(random.randint(0, 100000))

                
                existencia = firebase.get_document('partidas', id_joc)

                if existencia is None:
                    document_existeix = False
            data = funcions.dades_partida(partida, mails)
            firebase.save_game(id_joc, data)
            funcions.act_player_data(partida, mails)
            acabar_partida = True
            funcions.texto_a_audio("Partida finalizada. El jugador "
                                   + str(partida.jugadors[partida.torn_actual].nom) + " ha ganado la partida", 'fin.mp3')
            funcions.executar_mp3("fin.mp3")
            funcions.texto_a_audio("Partida guardada con id "
                                   + str(id_joc), 'id.mp3')
            funcions.executar_mp3("id.mp3")
            finalitzar_torn = True

    if finalitzar_torn is True:
        funcions.actualitzar_torns(partida)
        continue
    while not finalitzar_torn:
        funcions.executar_mp3_sin_borrar("audios/tirar_segon_torn.mp3")
        time.sleep(1)  # esperem 2 segons

        carta_tirada = False
        print("segona jugada")
        while not carta_tirada:  # Mentre que no s'ha posat i detectat la primera carta del tauler no seguim
            frame = funcions.nou_frame(cap)  # Trec una nova captura de la camara per buscar la carta cada 3 segons
            figura, color, coordenades = algorisme_final.detectar_carta(frame)
            if figura != "Null" and color != "Null":
                carta_tirada = True  # Un cop tenim la primera carta detectada ja seguim
                carta = clases.Carta(color, figura, str(partida.jugadors[partida.torn_actual].nom))
            else:
                funcions.executar_mp3_sin_borrar("audios/carta_no_identificada.mp3")
                time.sleep(3)
                print("No s'ha detectat la carta")

        # si les coordenades son les mateixes cambio de torn, sino pot seguir tirant
        mateixes_coordenades = funcions.comprovar_coordenades_i_carta(coordenades_anteriors, coordenades, carta.valor,
                                                                      carta.color, partida)
        if mateixes_coordenades:  # Si les coordenades son les mateixes no hem tirat carta i per tant passa de torn
            print("NO HA TIRAT RES MES PER TANT S'ACABA EL TORN")
            funcions.executar_mp3_sin_borrar("audios/carta_no_tirada.mp3")
            finalitzar_torn = True
            break
        print("NOVA CARTA, CONTINUA EL SEU TORN")

        print("Valor de la carta: " + str(carta.valor))
        print("Color de la carta: " + str(carta.color))
        moviment_valid = funcions.moviment_legal_continuacio(carta, partida)  # Mirar si la carta tirada es valida
        if moviment_valid:  # Si el moviment Ã©s vÃ lid
            print("MOVIMENT VALID")
            funcions.executar_mp3_sin_borrar("audios/moviment_valid.mp3")
            coordenades_anteriors = coordenades  # Si les coordenades no son les mateixes, posem les noves com anteriors
            partida.jugadors[partida.torn_actual].num_cartes -= 1  # Li restem una carta de la mÃ 
            print("N cartes: " + str(partida.jugadors[partida.torn_actual].num_cartes))
            if carta.color == "Wild":  # Si la carta tirada es Wild hem d'especificar quin color volem
                funcions.executar_mp3_sin_borrar("audios/dir_color_wild.mp3")
                ha_dit_color = False
                while not ha_dit_color:  # Fins que no es digui un color valid preguntarem pel color de la carta
                    color_detectat = funcions.escoltar_color()
                    if color_detectat.lower() in COLORS_DISPONIBLES:  # Si el color es valid:
                        funcions.executar_mp3_sin_borrar("audios/color_valid.mp3")
                        carta.color_wild = color_detectat.lower()  # Assignar el color de la carta wild
                        ha_dit_color = True
                    else:
                        funcions.executar_mp3_sin_borrar("audios/color_no_valid.mp3")
                        print("Color no vÃ lid. Repeteix el color")
            if carta.valor == "Reverse":
                partida.direccio = 1 - partida.direccio
            if carta.valor == "Skip":
                partida.skips_acumulats += 1
            if carta.valor == "Draw4":
                partida.cadena_robar = True
                partida.cartes_a_robar += 4
            if carta.valor == "Draw2":
                partida.cadena_robar = True
                partida.cartes_a_robar += 2

            partida.pila_joc.append(carta)  # Si el moviment es valid, guardem la carta al stack
            partida.historial.append(carta)

        else:
            print("MOVIMENT ILEGAL")
            funcions.executar_mp3_sin_borrar("audios/moviment_ilegal.mp3")
            # si el moviment no es valid el robot pitarÃ  i s'haura de treure la carta
            funcions_raspberry.repartir_n_cartes(2)
            partida.cartes_a_robar = 0
            restaurar_carta = False
            while not restaurar_carta:
                print("carta no restaurada")
                funcions.executar_mp3_sin_borrar("audios/quitar_carta_ilegal.mp3")
                time.sleep(1)
                frame = funcions.nou_frame(cap)
                figura, color, coordenades = algorisme_final.detectar_carta(frame)
                carta = clases.Carta(color, figura, str(partida.jugadors[partida.torn_actual].nom))
                if carta.valor == partida.pila_joc[-1].valor and carta.color == partida.pila_joc[-1].color:
                    restaurar_carta = True
            finalitzar_torn = True
            finalitzar_primera_jugada = True
            partida.cadena_robar = False
            print("carta restaurada")
            funcions.executar_mp3_sin_borrar("audios/ilegal_restaurat.mp3")

        ha_guanyat = funcions.check_victoria(partida.jugadors[partida.torn_actual])
        if ha_guanyat:
            
            partida.jugadors[partida.torn_actual].ganador = True
            document_existeix = True
            id_joc = "Null"
            while document_existeix:
                id_joc = "id_" + str(random.randint(0, 100000))
                existencia = firebase.get_document('partidas', id_joc)
                if existencia is None:
                    document_existeix = False
            data = funcions.dades_partida(partida, mails)
            firebase.save_game(id_joc, data)
            funcions.act_player_data(partida, mails)
            finalitzar_torn = True
            funcions.texto_a_audio("Partida finalizada. El jugador "
                                   + str(partida.jugadors[partida.torn_actual].nom) + " ha ganado la partida", 'fin.mp3')
            funcions.executar_mp3("fin.mp3")
            funcions.texto_a_audio("Partida guardada con id "
                                   + str(id_joc), 'id.mp3')
            funcions.executar_mp3("id.mp3")
            acabar_partida = True
    funcions.actualitzar_torns(partida)

print("PARTIDA FINALITZADA")
print("PARTIDA GUARDADA AMB ID: " + str(id_joc))
