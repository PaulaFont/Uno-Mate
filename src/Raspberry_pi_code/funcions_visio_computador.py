import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def distanciaEntrePunts(p1,p2):
    x1, y1 = p1
    x2, y2 = p2
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distancia


def eliminarLiniesProperes(linies, threshold):
    indices_eliminar = []
    for i, linia1 in enumerate(linies):
        puntInici1 = linia1[0][:2]
        puntFinal1 = linia1[0][2:]
        for j in range(i + 1, len(linies)):
            linia2 = linies[j]
            puntInici2 = linia2[0][:2]
            puntFinal2 = linia2[0][2:]
            distanciaInici = distanciaEntrePunts(puntInici1, puntInici2)
            distanciaFinal = distanciaEntrePunts(puntFinal1, puntFinal2)

            if distanciaInici < threshold and distanciaFinal < threshold:
                indices_eliminar.append(j)

    linies = np.delete(linies, indices_eliminar, axis=0)
    return linies

def EliminarPuntsPropers(coord1, coord2):
    eliminar = False
    distancia = distanciaEntrePunts(coord1, coord2)
    if distancia < 50:
        eliminar = True
    return eliminar


def augmentarContrast(img, alpha, beta):
    new_img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
    return new_img

def imadjust(img, low, high):
    return np.uint8((img - low) / (high - low) * 255)

def InterseccioDosRectes(linea1, linea2, graus):

    p1 = np.array([linea1[0][0], linea1[0][1], 1])
    p2 = np.array([linea1[0][2], linea1[0][3], 1])
    p3 = np.array([linea2[0][0], linea2[0][1], 1])
    p4 = np.array([linea2[0][2], linea2[0][3], 1])

    l12 = np.cross(p1, p2)
    l34 = np.cross(p3, p4)

    c = np.cross(l12, l34)
    c = c.astype(float)

    if c[2] == 0:
        return [-1000, -1000]

    c /= c[2]

    # Calcular los vectores direccionales de las rectas
    v1 = p2 - p1
    v2 = p4 - p3

    # Calcular el ángulo entre los vectores
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) # Ajustar el valor de cos_theta para evitar errores de redondeo

    angle = np.degrees(theta)
    #print(angle)

    if angle > 96 or angle < 85:
        return [-1000, -1000]

    graus.append(angle)
    return c[:2]


def InterseccioDosRectesTauler(linea1, linea2, graus):

    p1 = np.array([linea1[0][0], linea1[0][1], 1])
    p2 = np.array([linea1[0][2], linea1[0][3], 1])
    p3 = np.array([linea2[0][0], linea2[0][1], 1])
    p4 = np.array([linea2[0][2], linea2[0][3], 1])

    l12 = np.cross(p1, p2)
    l34 = np.cross(p3, p4)

    c = np.cross(l12, l34)
    c = c.astype(float)
    c /= c[2]

    # Calcular los vectores direccionales de las rectas
    v1 = p2 - p1
    v2 = p4 - p3

    return c[:2]

def HomografiaTauler(imatge, ample_carta_cm, alt_carta_cm):
    # Cv2 llegeix en BGR
    image_rgb = cv2.cvtColor(imatge, cv2.COLOR_BGR2RGB)

    # Aplicar canny edges

    imatge_bw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    contorns = cv2.Canny(imatge_bw, 100, 200)

    # Dilatar imatge

    kernel = np.ones((5, 5), np.uint8)

    contorns = cv2.dilate(contorns, kernel, iterations=1)
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 1, 1)
    plt.imshow(contorns, cmap='gray')
    plt.title('Contorns')
    plt.show()
    """
    # Trobar les linies rectes del tauler

    # Transformada de Hough per detectar linies rectes de la carta
    lineas = cv2.HoughLinesP(contorns, 1, np.pi / 180, threshold=350, minLineLength=100, maxLineGap=10)

    # Eliminar linies repetides

    lineasfinal = eliminarLiniesProperes(lineas, 500)

    # trobar els punts de tall de les linees

    llista_coordenades = []
    graus = []
    for i, linea in enumerate(lineasfinal):
        for j, linea1 in enumerate(lineasfinal):
            if i != j:
                llista_coordenades.append(InterseccioDosRectesTauler(linea, linea1, graus))


    # Dibuixar les linies rectes sobre el tauler
    imagen_con_lineas = image_rgb.copy()
    if lineas is not None:
        for linea in lineasfinal:
            x1, y1, x2, y2 = linea[0]
            cv2.line(imagen_con_lineas, (x1, y1), (x2, y2), (0, 0, 255), 5)

    #plt.figure(figsize=(10, 5))
    #plt.imshow(imagen_con_lineas)
    #plt.title('Líneas rectas detectadas')
    #plt.show()



    # Printejar els punts de tall de les linees que seran les cantonades del tauler
    imatge_amb_punts = image_rgb.copy()

    llista_coordenades_rep = np.round(llista_coordenades).astype(int)

    # Eliminar els punts que no son cantonades del tauler ja que tallen asi a l'infinit i repetits

    llista_coordenades = []

    for coord in llista_coordenades_rep:
        # Convertir a tupla pq amb numpy array no va
        coord_tuple = tuple(coord)
        if (coord[0] > 0 and coord[1] > 0) and (
                coord[0] < 1200 and coord[1] < 1600):  # Verificar que les coordenades no son a fora de la imatge
            if coord_tuple not in llista_coordenades:  # Verificar que no es repetit
                llista_coordenades.append(coord_tuple)

    # Eliminar punts que estan molt propers entre ells

    llista_eliminar = []

    for i in range(len(llista_coordenades)):
        for j in range(i + 1, len(llista_coordenades)):
            eliminar = EliminarPuntsPropers(llista_coordenades[i], llista_coordenades[j])
            if eliminar:
                llista_eliminar.append(llista_coordenades[j])

    llista_coordenades = [elemento for elemento in llista_coordenades if elemento not in llista_eliminar]
    """
    for punt in llista_coordenades:
        cv2.circle(imatge_amb_punts, (punt[0], punt[1]), 10, (0, 0, 255), -1)

    plt.figure(figsize=(10, 5))
    plt.imshow(imatge_amb_punts)
    plt.title('Cantonades de la carta')
    plt.show()
    """

    # Ordenar els punts per fer la homografia

    coordenades_ordenadas = sorted(llista_coordenades, key=lambda coord: coord[1])


    coordenades_mitat1 = coordenades_ordenadas[:2]
    coordenades_mitat2 = coordenades_ordenadas[2:]

    coordenades_ordenadas1 = sorted(coordenades_mitat1, key=lambda coord: coord[0])
    coordenades_ordenadas2 = sorted(coordenades_mitat2, key=lambda coord: coord[0])

    coordenades_ordenades = np.concatenate((coordenades_ordenadas1, coordenades_ordenadas2))

    # Calcular la homografia a partir de les coordenades

    # Convertir la mida de la carta a píxels
    ample_carta_px = ample_carta_cm * 100
    alt_carta_px = alt_carta_cm * 100

    # Definir els punts de destinació
    desti = np.array([[0, 0],  # Cantó superior esquerra
                      [ample_carta_px, 0],  # Cantó superior dreta
                      [0, alt_carta_px],  # Cantó inferior dreta
                      [ample_carta_px, alt_carta_px]])  # Cantó inferior esquerra

    # Definir els punts d'origen

    origen = np.array(coordenades_ordenades)

    # Calcular la matriu Homografia

    H, _ = cv2.findHomography(origen, desti)

    # Aplicar la Homografia

    imagen_original = imatge

    # Aplicar la homografía a la imagen original
    imatge_final = cv2.warpPerspective(imagen_original, H, (ample_carta_px, alt_carta_px))

    """
    cv2.imshow('Homografia final', imatge_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    cv2.imwrite("Imatge_inicial.jpg", imatge_final)


def HomografiaCarta(imatge, homografies, nombre_carta):
    altura, amplada, _ = imatge.shape
    x = 100
    y = 100
    amplada_final = amplada - 150
    altura_final = altura - 150

    imatge = imatge[y:y + altura_final, x:x + amplada_final]

    """
    cv2.imshow('Imagen Original', imatge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # Augmento el contrast de la imatge per una millor detecció dels contorns de les cartes

    alpha = 1.5  # Contrast
    beta = 2  # Brillo

    imatge = augmentarContrast(imatge, alpha, beta)

    """
    cv2.imshow('Mes contrast', imatge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # Vull aplicar un filtre gaussià per eliminar el soroll de la imatge
    image_rgb = cv2.cvtColor(imatge, cv2.COLOR_BGR2RGB)

    kernel = (5, 5)
    sigma = 2
    imatge_filtrada = cv2.GaussianBlur(image_rgb, kernel, sigma)

    """
    # Display original and smoothed images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb, cmap='gray')
    plt.title('Imatge original')

    plt.subplot(1, 2, 2)
    plt.imshow(imatge_filtrada, cmap='gray')
    plt.title('Filtre gaussià'.format(sigma))

    plt.show()
    """

    # Bucle principal de l'algorisme, els passos són:
    # 1 - Aplicar un threshold binari i detectar els contorns de la carta amb canny edges
    # 2 - Dibuixar els linees rectes que trobem a la imatge mitjançant les Hough lines
    # 3 - Fer un tractament de les linees detectades, com per exemple aquelles que estan repetides o son molt similars
    # 4 - Trobar els creuaments de les linees que contindrà les cantonades de la carta + altres punts que son soroll
    # 5 - Eliminar tots els punts que no son cantonades de la carta (angles de 90 graus, distancia entre punts)
    # 6 - Si no s'han trobat 4 punts exactament que serien les cantonades de la carta, cambiar threshold i repetir algorisme

    llista_coordeandes = []
    quatre_punts = False
    threshold = 150
    acabar_algorisme = False
    origen = []
    while not acabar_algorisme:

        # Aplicar threshold binari, aquest es dinamic i va augmentant amb cada iteració per trobar el màxim nombre
        # d'homografies
        #print(threshold)
        umbral, imatge_threshold = cv2.threshold(imatge_filtrada, threshold, 255, cv2.THRESH_BINARY)

        """
        cv2.imshow('Imagen binarizada', imatge_threshold)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        # Aplicar canny edges per trobar tots els contorns de la imatge
        imatge_bw = cv2.cvtColor(imatge_threshold, cv2.COLOR_RGB2GRAY)
        contorns = cv2.Canny(imatge_bw, 10, 100)

        # Dilatar la imatge per que els contorns es vegin mes clarament i es puguin detectar facilment amb les linees de
        # Hough
        kernel = np.ones((5, 5), np.uint8)
        contorns = cv2.dilate(contorns, kernel, iterations=1)

        """
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 1, 1)
        plt.imshow(contorns, cmap='gray')
        plt.title('Contorns')
        plt.show()
        """

        # Trobar les linees rectes de la imatge mitjançant la transformada de Hough
        lineas = cv2.HoughLinesP(contorns, 1, np.pi / 180, threshold=20, minLineLength=120, maxLineGap=10)

        if lineas is None:  # En el cas de no trobar mes linees (threshold > 255) acabem l'algorisme
            acabar_algorisme = True
            break

        # Eliminem les linees repetides , son aquelles que els seus punts d'inici i final son tan propers que bàsicament
        # es la mateixa linea i per tant no ens interessa
        lineasfinal = eliminarLiniesProperes(lineas, 110)
        #print(lineasfinal)

        # trobar els punts de tall de les linees
        llista_coordenades = []
        graus = []

        for i, linea in enumerate(lineasfinal):
            for j, linea1 in enumerate(lineasfinal):
                if i != j:
                    llista_coordenades.append(InterseccioDosRectes(linea, linea1, graus))

        graus = list(set(graus))
        #print(graus)
        #print(lineasfinal)

        """
        # Dibuixar les linies rectes sobre la carta
        imagen_con_lineas = image_rgb.copy()
        if lineas is not None:
            for linea in lineasfinal:
                x1, y1, x2, y2 = linea[0]
                cv2.line(imagen_con_lineas, (x1, y1), (x2, y2), (0, 0, 255), 5)

        plt.figure(figsize=(10, 5))
        plt.imshow(imagen_con_lineas)
        plt.title('Líneas rectas detectadas')
        plt.show()
        """

        llista_coordenades_rep = np.round(llista_coordenades).astype(int)  # Arrodonir que sino no va

        # Eliminar els punts que no son cantonades de la carta ja que tallen fora de la imatge i repetits
        llista_coordenades = []

        for coord in llista_coordenades_rep:
            # Convertir a tupla pq amb numpy array no va
            coord_tuple = tuple(coord)
            if (coord[0] > 0 and coord[1] > 0) and (
                    coord[0] < 800 and coord[1] < 800):  # Verificar que les coordenades no son a fora de la imatge
                if coord_tuple not in llista_coordenades:  # Verificar que no es repetit
                    llista_coordenades.append(coord_tuple)

        # Eliminar punts que estan molt propers entre ells ja que nomes ens serveix un d'ells
        llista_eliminar = []

        for i in range(len(llista_coordenades)):
            for j in range(i + 1, len(llista_coordenades)):
                eliminar = EliminarPuntsPropers(llista_coordenades[i], llista_coordenades[j])
                if eliminar:
                    dist_centre1 = distanciaEntrePunts(llista_coordenades[i], [0, 0])
                    dist_centre2 = distanciaEntrePunts(llista_coordenades[j], [0, 0])
                    if dist_centre1 < dist_centre2:  # Elimino el punt que está mes lluny del punt central de la imatge
                        llista_eliminar.append(llista_coordenades[i])
                    else:
                        llista_eliminar.append(llista_coordenades[j])

        llista_coordenades = [elemento for elemento in llista_coordenades if elemento not in llista_eliminar]

        # Eliminar els punts que no estan a distancia de carta amb dos altres punts

        # · --- ·
        # |     | ==> En aquest cas cadascun dels punts esta a una distancia del costat petit i una distancia del costat
        # |     |     gran que ve determinada per la carta, els punts son valids
        # · --- ·

        # · --- ·
        # |     | ==> En aquest els punts no respecten les mides de la carta i pert tant no son vàlids
        # · ------ ·

        llista_final1 = []
        llista_final = []
        # Un punt no pot estar a distancia de carta de mes de dos punts

        rectes_finals = []

        for i, punt1 in enumerate(llista_coordenades):
            for j, punt2 in enumerate(llista_coordenades):
                if i != j:
                    distancia = distanciaEntrePunts(punt1, punt2)
                    if 160 < distancia < 225:  # Rang de distàncies del costat petit de la carta
                        llista_final1.append(punt1)
                        break


        for i, punt1 in enumerate(llista_final1):
            for j, punt2 in enumerate(llista_final1):
                if i != j:
                    distancia = distanciaEntrePunts(punt1, punt2)
                    if 270 < distancia < 340:  # Rang de distàncies del costat gran de la carta
                        llista_final.append(punt1)
                        break

        #print(rectes_finals)

        llista_coordenades = llista_final

        #print(llista_coordenades)

        # Augmentem el threshold per una nova iteració del bucle per intentar aconseguir una millor homografia
        threshold = threshold + 10

        # En cas de no trobar exactament quatre punts que serien les cantonades de la carta, saltem la iteració i ho tornem
        # a intentar amb un nou valor del threshold
        if len(llista_coordenades) != 4:
            continue
        """
        imatge_amb_punts = image_rgb.copy()
        for punt in llista_coordenades:
            cv2.circle(imatge_amb_punts, (punt[0], punt[1]), 10, (0, 0, 255), -1)

        plt.figure(figsize=(10, 5))
        plt.imshow(imatge_amb_punts)
        plt.title('Cantonades de la carta')
        plt.show()
        """
        print(threshold)
        # Ordenar els punts per fer la homografia
        coordenades_ordenadas = sorted(llista_coordenades, key=lambda coord: coord[1])

        distancia_per_h = distanciaEntrePunts(coordenades_ordenadas[0],coordenades_ordenadas[1])
        distancia_per_h1 = distanciaEntrePunts(coordenades_ordenadas[2],coordenades_ordenadas[3])

        if distancia_per_h > 225 and distancia_per_h1 > 225:
            coordenades_ordenadas = sorted(llista_coordenades, key=lambda coord: coord[0])
            coordenades_mitat1 = coordenades_ordenadas[:2]
            coordenades_mitat2 = coordenades_ordenadas[2:]

            coordenades_ordenadas1 = sorted(coordenades_mitat1, key=lambda coord: coord[1])
            coordenades_ordenadas2 = sorted(coordenades_mitat2, key=lambda coord: coord[1])

            coordenades_ordenades = np.concatenate((coordenades_ordenadas2, coordenades_ordenadas1))

        else:
            coordenades_mitat1 = coordenades_ordenadas[:2]
            coordenades_mitat2 = coordenades_ordenadas[2:]

            coordenades_ordenadas1 = sorted(coordenades_mitat1, key=lambda coord: coord[0])
            coordenades_ordenadas2 = sorted(coordenades_mitat2, key=lambda coord: coord[0])

            coordenades_ordenades = np.concatenate((coordenades_ordenadas1, coordenades_ordenadas2))

        # Calcular la homografia a partir de les coordenades

        # Mida de la carta en centímetres
        ample_carta_cm = 6
        alt_carta_cm = 9

        # Convertir la mida de la carta a píxels
        ample_carta_px = ample_carta_cm * 100
        alt_carta_px = alt_carta_cm * 100

        # Definir els punts de destinació
        desti = np.array([[0, 0],  # Cantó superior esquerra
                          [ample_carta_px, 0],  # Cantó superior dreta
                          [0, alt_carta_px],  # Cantó inferior dreta
                          [ample_carta_px, alt_carta_px]])  # Cantó inferior esquerra

        # Definir els punts d'origen
        origen = np.array(coordenades_ordenades)

        # Calcular la matriu Homografia
        H, _ = cv2.findHomography(origen, desti)

        # Aplicar la Homografia

        # Aplicar la homografía a la imatge original
        imatge_final = cv2.warpPerspective(imatge, H, (ample_carta_px, alt_carta_px))

        # Tamany de la foto
        nueva_altura = 600
        nueva_anchura = 400

        # Redimensionar la imatge
        imatge_final_redimensionada = cv2.resize(imatge_final, (nueva_anchura, nueva_altura))

        #cv2.imshow('Homografia final', imatge_final_redimensionada)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        homografies.append(imatge_final)
        cv2.imwrite(nombre_carta, imatge_final)

        if len(homografies) == 3:
            break

    print("NO HI HAN MES HOMOGRAFIES POSSIBLES")
    if len(homografies) == 0:
        return "Null", origen

    return "perf", origen


def HomografiaTaulerCoords(imatge, ample_carta_cm, alt_carta_cm):
    # Cv2 llegeix en BGR
    image_rgb = cv2.cvtColor(imatge, cv2.COLOR_BGR2RGB)

    # Calcular la homografia a partir de les coordenades

    # Convertir la mida de la carta a píxels
    ample_carta_px = ample_carta_cm * 100
    alt_carta_px = alt_carta_cm * 100

    # Definir els punts de destinació
    desti = np.array([[0, 0],  # Cantó superior esquerra
                      [ample_carta_px, 0],  # Cantó superior dreta
                      [0, alt_carta_px],  # Cantó inferior dreta
                      [ample_carta_px, alt_carta_px]])  # Cantó inferior esquerra

    # Definir els punts d'origen
    coordenades_ordenades = [[129.4, 18.5], [551.3, 27.4], [110.4, 452.3], [563.2, 460.3]]
    origen = np.array(coordenades_ordenades)

    # Calcular la matriu Homografia

    H, _ = cv2.findHomography(origen, desti)

    # Aplicar la Homografia

    imagen_original = imatge

    # Aplicar la homografía a la imagen original
    imatge_final = cv2.warpPerspective(imagen_original, H, (ample_carta_px, alt_carta_px))

    """
    cv2.imshow('Homografia final', imatge_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    cv2.imwrite("Imatge_inicial.jpg", imatge_final)
    return imatge_final


def HomografiaCartaPrints(imatge, homografies, nombre_carta):
    altura, amplada, _ = imatge.shape
    x = 100
    y = 100
    amplada_final = amplada - 150
    altura_final = altura - 150

    imatge = imatge[y:y + altura_final, x:x + amplada_final]


    #cv2.imshow('Imagen Original', imatge)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    # Augmento el contrast de la imatge per una millor detecció dels contorns de les cartes

    alpha = 1.5  # Contrast
    beta = 2  # Brillo

    imatge = augmentarContrast(imatge, alpha, beta)


    #cv2.imshow('Mes contrast', imatge)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    # Vull aplicar un filtre gaussià per eliminar el soroll de la imatge
    image_rgb = cv2.cvtColor(imatge, cv2.COLOR_BGR2RGB)

    kernel = (5, 5)
    sigma = 2
    imatge_filtrada = cv2.GaussianBlur(image_rgb, kernel, sigma)


    # Display original and smoothed images
    '''
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb, cmap='gray')
    plt.title('Imatge original')

    plt.subplot(1, 2, 2)
    plt.imshow(imatge_filtrada, cmap='gray')
    plt.title('Filtre gaussià'.format(sigma))

    plt.show()
    '''

    # Bucle principal de l'algorisme, els passos són:
    # 1 - Aplicar un threshold binari i detectar els contorns de la carta amb canny edges
    # 2 - Dibuixar els linees rectes que trobem a la imatge mitjançant les Hough lines
    # 3 - Fer un tractament de les linees detectades, com per exemple aquelles que estan repetides o son molt similars
    # 4 - Trobar els creuaments de les linees que contindrà les cantonades de la carta + altres punts que son soroll
    # 5 - Eliminar tots els punts que no son cantonades de la carta (angles de 90 graus, distancia entre punts)
    # 6 - Si no s'han trobat 4 punts exactament que serien les cantonades de la carta, cambiar threshold i repetir algorisme

    llista_coordeandes = []
    quatre_punts = False
    threshold = 150
    acabar_algorisme = False

    while not acabar_algorisme:

        # Aplicar threshold binari, aquest es dinamic i va augmentant amb cada iteració per trobar el màxim nombre
        # d'homografies
        #print(threshold)
        umbral, imatge_threshold = cv2.threshold(imatge_filtrada, threshold, 255, cv2.THRESH_BINARY)


        #cv2.imshow('Imagen binarizada', imatge_threshold)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


        # Aplicar canny edges per trobar tots els contorns de la imatge
        imatge_bw = cv2.cvtColor(imatge_threshold, cv2.COLOR_RGB2GRAY)
        contorns = cv2.Canny(imatge_bw, 10, 100)

        # Dilatar la imatge per que els contorns es vegin mes clarament i es puguin detectar facilment amb les linees de
        # Hough
        kernel = np.ones((5, 5), np.uint8)
        contorns = cv2.dilate(contorns, kernel, iterations=1)

        '''
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 1, 1)
        plt.imshow(contorns, cmap='gray')
        plt.title('Contorns')
        plt.show()
        '''

        # Trobar les linees rectes de la imatge mitjançant la transformada de Hough
        lineas = cv2.HoughLinesP(contorns, 1, np.pi / 180, threshold=20, minLineLength=120, maxLineGap=10)

        if lineas is None:  # En el cas de no trobar mes linees (threshold > 255) acabem l'algorisme
            acabar_algorisme = True
            break

        # Eliminem les linees repetides , son aquelles que els seus punts d'inici i final son tan propers que bàsicament
        # es la mateixa linea i per tant no ens interessa
        lineasfinal = eliminarLiniesProperes(lineas, 110)
        #print(lineasfinal)

        # trobar els punts de tall de les linees
        llista_coordenades = []
        graus = []

        for i, linea in enumerate(lineasfinal):
            for j, linea1 in enumerate(lineasfinal):
                if i != j:
                    llista_coordenades.append(InterseccioDosRectes(linea, linea1, graus))

        graus = list(set(graus))
        #print(graus)
        #print(lineasfinal)


        # Dibuixar les linies rectes sobre la carta
        imagen_con_lineas = image_rgb.copy()
        if lineas is not None:
            for linea in lineasfinal:
                x1, y1, x2, y2 = linea[0]
                cv2.line(imagen_con_lineas, (x1, y1), (x2, y2), (0, 0, 255), 5)
        '''
        plt.figure(figsize=(10, 5))
        plt.imshow(imagen_con_lineas)
        plt.title('Líneas rectas detectadas')
        plt.show()
        '''

        llista_coordenades_rep = np.round(llista_coordenades).astype(int)  # Arrodonir que sino no va

        # Eliminar els punts que no son cantonades de la carta ja que tallen fora de la imatge i repetits
        llista_coordenades = []

        for coord in llista_coordenades_rep:
            # Convertir a tupla pq amb numpy array no va
            coord_tuple = tuple(coord)
            if (coord[0] > 0 and coord[1] > 0) and (
                    coord[0] < 800 and coord[1] < 800):  # Verificar que les coordenades no son a fora de la imatge
                if coord_tuple not in llista_coordenades:  # Verificar que no es repetit
                    llista_coordenades.append(coord_tuple)

        # Eliminar punts que estan molt propers entre ells ja que nomes ens serveix un d'ells
        llista_eliminar = []

        for i in range(len(llista_coordenades)):
            for j in range(i + 1, len(llista_coordenades)):
                eliminar = EliminarPuntsPropers(llista_coordenades[i], llista_coordenades[j])
                if eliminar:
                    dist_centre1 = distanciaEntrePunts(llista_coordenades[i], [0, 0])
                    dist_centre2 = distanciaEntrePunts(llista_coordenades[j], [0, 0])
                    if dist_centre1 < dist_centre2:  # Elimino el punt que está mes lluny del punt central de la imatge
                        llista_eliminar.append(llista_coordenades[i])
                    else:
                        llista_eliminar.append(llista_coordenades[j])

        llista_coordenades = [elemento for elemento in llista_coordenades if elemento not in llista_eliminar]

        print(llista_coordenades)
        # Eliminar els punts que no estan a distancia de carta amb dos altres punts

        # · --- ·
        # |     | ==> En aquest cas cadascun dels punts esta a una distancia del costat petit i una distancia del costat
        # |     |     gran que ve determinada per la carta, els punts son valids
        # · --- ·

        # · --- ·
        # |     | ==> En aquest els punts no respecten les mides de la carta i pert tant no son vàlids
        # · ------ ·

        llista_final1 = []
        llista_final = []
        # Un punt no pot estar a distancia de carta de mes de dos punts

        rectes_finals = []

        for i, punt1 in enumerate(llista_coordenades):
            for j, punt2 in enumerate(llista_coordenades):
                if i != j:
                    distancia = distanciaEntrePunts(punt1, punt2)
                    if 160 < distancia < 225:  # Rang de distàncies del costat petit de la carta
                        llista_final1.append(punt1)
                        break


        for i, punt1 in enumerate(llista_final1):
            for j, punt2 in enumerate(llista_final1):
                if i != j:
                    distancia = distanciaEntrePunts(punt1, punt2)
                    if 270 < distancia < 340:  # Rang de distàncies del costat gran de la carta
                        llista_final.append(punt1)
                        break

        #print(rectes_finals)

        llista_coordenades = llista_final

        print(llista_coordenades)

        # Augmentem el threshold per una nova iteració del bucle per intentar aconseguir una millor homografia
        threshold = threshold + 10

        # En cas de no trobar exactament quatre punts que serien les cantonades de la carta, saltem la iteració i ho tornem
        # a intentar amb un nou valor del threshold
        if len(llista_coordenades) != 4:
            continue

        imatge_amb_punts = image_rgb.copy()
        for punt in llista_coordenades:
            cv2.circle(imatge_amb_punts, (punt[0], punt[1]), 10, (0, 0, 255), -1)

        plt.figure(figsize=(10, 5))
        plt.imshow(imatge_amb_punts)
        plt.title('Cantonades de la carta')
        plt.show()


        # Ordenar els punts per fer la homografia
        coordenades_ordenadas = sorted(llista_coordenades, key=lambda coord: coord[1])

        distancia_per_h = distanciaEntrePunts(coordenades_ordenadas[0],coordenades_ordenadas[1])
        distancia_per_h1 = distanciaEntrePunts(coordenades_ordenadas[2],coordenades_ordenadas[3])

        if distancia_per_h > 225 and distancia_per_h1 > 225:
            coordenades_ordenadas = sorted(llista_coordenades, key=lambda coord: coord[0])
            coordenades_mitat1 = coordenades_ordenadas[:2]
            coordenades_mitat2 = coordenades_ordenadas[2:]

            coordenades_ordenadas1 = sorted(coordenades_mitat1, key=lambda coord: coord[1])
            coordenades_ordenadas2 = sorted(coordenades_mitat2, key=lambda coord: coord[1])

            coordenades_ordenades = np.concatenate((coordenades_ordenadas2, coordenades_ordenadas1))

        else:
            coordenades_mitat1 = coordenades_ordenadas[:2]
            coordenades_mitat2 = coordenades_ordenadas[2:]

            coordenades_ordenadas1 = sorted(coordenades_mitat1, key=lambda coord: coord[0])
            coordenades_ordenadas2 = sorted(coordenades_mitat2, key=lambda coord: coord[0])

            coordenades_ordenades = np.concatenate((coordenades_ordenadas1, coordenades_ordenadas2))

        # Calcular la homografia a partir de les coordenades

        # Mida de la carta en centímetres
        ample_carta_cm = 6
        alt_carta_cm = 9

        # Convertir la mida de la carta a píxels
        ample_carta_px = ample_carta_cm * 100
        alt_carta_px = alt_carta_cm * 100

        # Definir els punts de destinació
        desti = np.array([[0, 0],  # Cantó superior esquerra
                          [ample_carta_px, 0],  # Cantó superior dreta
                          [0, alt_carta_px],  # Cantó inferior dreta
                          [ample_carta_px, alt_carta_px]])  # Cantó inferior esquerra

        # Definir els punts d'origen
        origen = np.array(coordenades_ordenades)

        # Calcular la matriu Homografia
        H, _ = cv2.findHomography(origen, desti)

        # Aplicar la Homografia

        # Aplicar la homografía a la imatge original
        imatge_final = cv2.warpPerspective(imatge, H, (ample_carta_px, alt_carta_px))

        # Tamany de la foto
        nueva_altura = 600
        nueva_anchura = 400

        # Redimensionar la imatge
        imatge_final_redimensionada = cv2.resize(imatge_final, (nueva_anchura, nueva_altura))

        #cv2.imshow('Homografia final', imatge_final_redimensionada)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        homografies.append(imatge_final)
        cv2.imwrite(nombre_carta, imatge_final)


        if len(homografies) == 3:
            break


    print("NO HI HAN MES HOMOGRAFIES POSSIBLES")
