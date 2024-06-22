
import cv2
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
import re
from sklearn.preprocessing import MinMaxScaler
from statistics import mode


num_pedazos = 8

def hist_img(img):

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Divide la imagen en canales RGB
    r_channel = image_rgb[:,:,0]
    g_channel = image_rgb[:,:,1]
    b_channel = image_rgb[:,:,2]

    # Calcula los histogramas de cada canal
    hist_r = cv2.calcHist([r_channel],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([g_channel],[0],None,[256],[0,256])
    hist_b = cv2.calcHist([b_channel],[0],None,[256],[0,256])

    # Agrupar los valores del histograma en grupos de 8 o 16
    group_size = 8

    # Agrupar los valores del histograma en grupos de 8
    hist_r_grouped = [np.sum(hist_r[i:i + group_size]) for i in range(0, 256, group_size)]
    hist_g_grouped = [np.sum(hist_g[i:i + group_size]) for i in range(0, 256, group_size)]
    hist_b_grouped = [np.sum(hist_b[i:i + group_size]) for i in range(0, 256, group_size)]

    return [hist_r_grouped, hist_g_grouped, hist_b_grouped]

def recortar_imagen(imagen, ancho, alto):
    altura_original, ancho_original, _ = imagen.shape
    if altura_original >= alto and ancho_original >= ancho:
        imagen_recortada = imagen[:alto, :ancho]
    else:
        raise ValueError("La imagen es demasiado pequeña para ser recortada a las dimensiones especificadas.")
    return imagen_recortada
def recortar_imagen_inferior_derecha(imagen, ancho, alto):
    altura_original, ancho_original, _ = imagen.shape

    # Calcular las coordenadas de inicio para recortar la parte inferior derecha
    x_inicio = max(0, ancho_original - ancho)
    y_inicio = max(0, altura_original - alto)

    # Verificar si las dimensiones especificadas son válidas
    if altura_original >= alto and ancho_original >= ancho:
        imagen_recortada = imagen[y_inicio:, x_inicio:]
    else:
        raise ValueError("La imagen es demasiado pequeña para ser recortada a las dimensiones especificadas.")

    return imagen_recortada


def recortar_secuencial(imagen, num_pedazos):
    altura_original, ancho_original, _ = imagen.shape
    pedazos = []

    # Calcular el número de pedazos por fila y columna
    pedazos_por_fila = int(np.ceil(np.sqrt(num_pedazos)))
    pedazos_por_columna = int(np.ceil(num_pedazos / pedazos_por_fila))

    # Calcular el ancho y alto de cada pedazo
    ancho_pedazo = int(ancho_original / pedazos_por_fila)
    alto_pedazo = int(altura_original / pedazos_por_columna)

    # Asegurarse de que haya suficientes pedazos para el número deseado
    num_pedazos = min(num_pedazos, pedazos_por_fila * pedazos_por_columna)

    # Recortar la imagen en pedazos de manera secuencial
    for i in range(pedazos_por_columna):
        for j in range(pedazos_por_fila):
            x_inicio = j * ancho_pedazo
            y_inicio = i * alto_pedazo
            x_fin = min((j + 1) * ancho_pedazo, ancho_original)
            y_fin = min((i + 1) * alto_pedazo, altura_original)
            pedazo = imagen[y_inicio:y_fin, x_inicio:x_fin]
            pedazos.append(pedazo)

            # Detener el bucle si se han obtenido suficientes pedazos
            if len(pedazos) == num_pedazos:
                return pedazos

    return pedazos

def contar_pixeles_oscuros(img_gray, umbral):
    num_pixeles_oscuros = np.sum(img_gray < umbral)
    prop_pixeles_oscuros = num_pixeles_oscuros / (img_gray.shape[0] * img_gray.shape[1])
    return prop_pixeles_oscuros

def extraer_pix_oscuros(img, umbral=40):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prop_pixeles_oscuros = contar_pixeles_oscuros(img_gray, umbral)
    #print(prop_pixeles_oscuros)
    oscuro = False
    if prop_pixeles_oscuros > 0.015:
        oscuro = True
    return oscuro


def predecirImg(img, modelo, min_values, max_values):
    columnas_deseadas = [
        'px_oscuros', 'Hist_R_g1', 'Hist_R_g2', 'Hist_R_g3', 'Hist_R_g4', 'Hist_R_g5',
        'Hist_R_g6', 'Hist_R_g7', 'Hist_R_g8', 'Hist_R_g9', 'Hist_R_g10', 'Hist_R_g11',
        'Hist_R_g12', 'Hist_R_g13', 'Hist_R_g14', 'Hist_R_g15', 'Hist_R_g16', 'Hist_R_g17',
        'Hist_R_g18', 'Hist_R_g19', 'Hist_R_g20', 'Hist_R_g21', 'Hist_R_g22', 'Hist_R_g23',
        'Hist_R_g24', 'Hist_R_g25', 'Hist_R_g26', 'Hist_R_g27', 'Hist_R_g28', 'Hist_R_g29',
        'Hist_R_g30', 'Hist_R_g31', 'Hist_R_g32', 'Hist_G_g1', 'Hist_G_g2', 'Hist_G_g3',
        'Hist_G_g4', 'Hist_G_g5', 'Hist_G_g6', 'Hist_G_g7', 'Hist_G_g8', 'Hist_G_g9',
        'Hist_G_g10', 'Hist_G_g11', 'Hist_G_g12', 'Hist_G_g13', 'Hist_G_g14', 'Hist_G_g15',
        'Hist_G_g16', 'Hist_G_g17', 'Hist_G_g18', 'Hist_G_g19', 'Hist_G_g20', 'Hist_G_g21',
        'Hist_G_g22', 'Hist_G_g23', 'Hist_G_g24', 'Hist_G_g25', 'Hist_G_g26', 'Hist_G_g27',
        'Hist_G_g28', 'Hist_G_g29', 'Hist_G_g30', 'Hist_G_g31', 'Hist_G_g32', 'Hist_B_g1',
        'Hist_B_g2', 'Hist_B_g3', 'Hist_B_g4', 'Hist_B_g5', 'Hist_B_g6', 'Hist_B_g7',
        'Hist_B_g8', 'Hist_B_g9', 'Hist_B_g10', 'Hist_B_g11', 'Hist_B_g12', 'Hist_B_g13',
        'Hist_B_g14', 'Hist_B_g15', 'Hist_B_g16', 'Hist_B_g17', 'Hist_B_g18', 'Hist_B_g19',
        'Hist_B_g20', 'Hist_B_g21', 'Hist_B_g22', 'Hist_B_g23', 'Hist_B_g24', 'Hist_B_g25',
        'Hist_B_g26', 'Hist_B_g27', 'Hist_B_g28', 'Hist_B_g29', 'Hist_B_g30', 'Hist_B_g31',
        'Hist_B_g32']
    """
     'Valor_0', 'Valor_1', 'Valor_2', 'Valor_3', 'Valor_4', 'Valor_5',
        'Valor_6', 'Valor_7', 'Valor_8', 'Valor_9', 'Valor_Color', 'Valor_Draw_2',
        'Valor_Draw_4', 'Valor_Reverse', 'Valor_Skip', 'Acción_Especial', 'Acción_Normal'
    """

    df_pred = pd.DataFrame(columns=columnas_deseadas)

    # Creo una lista para almacenar los histogramas agrupados por canal
    histogramas = []

    # Recortar la parte superior izquierda de la imagen
    img_recortada = recortar_imagen(img, 200, 300)

    pedazos_aleatorios = recortar_secuencial(img_recortada, num_pedazos)

    for pedazo in pedazos_aleatorios:
        valor_de_px_oscuros = extraer_pix_oscuros(img)
        """
        valor_de_color = df.iloc[indice]['Color']
        """

        hist_r, hist_g, hist_b = hist_img(pedazo)
        histogramas.append([valor_de_px_oscuros, hist_r, hist_g, hist_b])

    img_recortada2 = recortar_imagen_inferior_derecha(img, 200, 300)

    pedazos_aleatorios = recortar_secuencial(img_recortada2, num_pedazos)

    for pedazo in pedazos_aleatorios:
        valor_de_px_oscuros = extraer_pix_oscuros(img)
        """
        valor_de_color = df.iloc[indice]['Color']
        """

        hist_r, hist_g, hist_b = hist_img(pedazo)
        histogramas.append([valor_de_px_oscuros, hist_r, hist_g, hist_b])
    # Creo un DataFrame para almacenar los histogramas agrupados por canal
    df_histogramas = pd.DataFrame(histogramas, columns=['px_oscuros', 'Hist_R', 'Hist_G', 'Hist_B'])

    hist_r_columns = [f'Hist_R_g{i + 1}' for i in range(32)]
    hist_g_columns = [f'Hist_G_g{i + 1}' for i in range(32)]
    hist_b_columns = [f'Hist_B_g{i + 1}' for i in range(32)]

    df_histogramas = pd.concat([
        df_histogramas,
        pd.DataFrame(df_histogramas['Hist_R'].tolist(), columns=hist_r_columns),
        pd.DataFrame(df_histogramas['Hist_G'].tolist(), columns=hist_g_columns),
        pd.DataFrame(df_histogramas['Hist_B'].tolist(), columns=hist_b_columns)
    ], axis=1)

    # Eliminar las columnas originales de histogramas agrupados
    df_histogramas.drop(columns=['Hist_R', 'Hist_G', 'Hist_B'], inplace=True)

    df_pred = df_histogramas
    """
    for columna in columnas_deseadas:
        if columna in df_histogramas.columns:
            df_pred[columna] = df_histogramas[columna]
    """

    pd.set_option('display.max_rows', None)  # Mostrar todas las filas
    pd.set_option('display.max_columns', None)  # Mostrar todas las columnas

    # df_pred['px_oscuros'] = extraer_pix_oscuros(imagen_path)

    # print(df_pred.to_string(index=False))

    # mostrar_hist(histogramas[0])

    # print("-----------------------------------------------")

    # Ahora definimos lo que va a ser el target y lo que van a ser los atributos
    from sklearn.model_selection import train_test_split

    # Normalizo los datos que tienen que ver con el RGB de cada imagen
    hist_columns_pred = [col for col in df_pred.columns if col.startswith('Hist_')]
    hist_data_pred = df_pred[hist_columns_pred]

    # Normalizar los datos de histogramas utilizando los valores mínimos y máximos proporcionados
    scaled_hist_data_pred = (hist_data_pred - min_values) / (max_values - min_values)

    df_pred[hist_columns_pred] = scaled_hist_data_pred

    # print(df_pred.to_string(index=False))

    # print("-----------------------------------------------")

    # Verificar el DataFrame actualizado
    # print(df_pred.to_string(index=False))

    df_pred.columns = range(df_pred.shape[1])

    y_pred = modelo.predict(df_pred)

    colores = ['Blue', 'Green', 'Red', 'Wild', 'Yellow']

    # Mostrar el resultado de la predicción

    valor_mas_frecuente = mode(y_pred)

    # print(y_pred)
    # print("Predicción el color es: ", colores[valor_mas_frecuente])

    return colores[valor_mas_frecuente]


from joblib import dump, load

modelo_cargado = load('modelo_entrenado_svm.joblib')
d_min = load('scaler_data_min.joblib')
d_max = load('scaler_data_max.joblib')



