import funcions_visio_computador as funcions
import cv2
import Modelo_Entrenado
from joblib import dump, load
import numberPrediction
import time


def detectar_carta(frame):
    figura_final = "Null"
    color_final = "Null"

    # LLegir la imatge
    tauler = frame
    #cv2.imshow('Imagen Original',tauler)
    #cv2.waitKey(0)

    #cv2.destroyAllWindows()
    tempsTotal = time.time()

    tempsHomo = time.time()
    # Fer la homografia del tauler
    tauler_final = funcions.HomografiaTaulerCoords(tauler, 9, 9)  # amplada, altura

    # Fer la homografia de la carta
    imatge = tauler_final
    homografies = []
    imatge_final, coordenades = funcions.HomografiaCarta(imatge, homografies, nombre_carta="imatge_final.jpg")

    if imatge_final == "Null":
        return figura_final, color_final, coordenades


    #print("Temps Homografia: " + str(time.time()-tempsHomo))

    tempsColor = time.time()
    # Cargar el model entrenat i normalizacion del misisuko
    modelo_cargado = load('modelo_entrenado_svm.joblib')
    d_min = load('scaler_data_min.joblib')
    d_max = load('scaler_data_max.joblib')

    prediccions_color = {"Blue": 0, "Green": 0, "Yellow": 0, "Red": 0, "Wild": 0}
    for imatge in homografies:
        color = Modelo_Entrenado.predecirImg(imatge, modelo_cargado, d_min, d_max)
        prediccions_color[color] += 1

    #print(prediccions_color)

    color_final = max(prediccions_color, key=prediccions_color.get)

    #print("Temps Predir Color: " + str(time.time()-tempsColor))

    tempsNumero = time.time()
    prediccions_figura = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "Draw2": 0
                          , "Draw4": 0, "Reverse": 0, "Skip": 0, "Color": 0}

    for imatge in homografies:
        figura = numberPrediction.predecir_imagen(imatge)
        prediccions_figura[figura] += 1


    #print(prediccions_figura)

    figura_final = max(prediccions_figura, key=prediccions_figura.get)

    #print(figura_final)
    #print(color_final)

    #print("Temps Predir Numero: " + str(time.time()-tempsNumero))
    #print("Ha tardat: " + str(time.time() - tempsTotal))

    print("retorno les coordenades: " + str(coordenades))

    return figura_final, color_final, coordenades
