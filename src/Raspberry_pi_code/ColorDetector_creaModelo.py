"""

Hacer un modelo que utilice los histogramas que genero SVM o un KNN

Para que me de un rango.

"""

import cv2
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
import re
from sklearn.preprocessing import MinMaxScaler

# Suprimir la salida de libpng
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

def contar_pixeles_oscuros(img_gray, umbral):
    num_pixeles_oscuros = np.sum(img_gray < umbral)
    prop_pixeles_oscuros = num_pixeles_oscuros / (img_gray.shape[0] * img_gray.shape[1])
    return prop_pixeles_oscuros

def extraer_pix_oscuros(img_path, umbral=40):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prop_pixeles_oscuros = contar_pixeles_oscuros(img_gray, umbral)
    #print(prop_pixeles_oscuros)
    oscuro = False
    if prop_pixeles_oscuros > 0.015:
        oscuro = True
    return oscuro
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

def mostrar_hist(histograma):
    # Crear un rango para el eje x
    x = np.arange(32)

    # Crear una figura y subtramas
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))

    # Plot del histograma del canal rojo
    ax[0].bar(x, histograma[0], color='red', alpha=0.7)
    ax[0].set_title('Histograma canal rojo')

    # Plot del histograma del canal verde
    ax[1].bar(x, histograma[1], color='green', alpha=0.7)
    ax[1].set_title('Histograma canal verde')

    # Plot del histograma del canal azul
    ax[2].bar(x, histograma[2], color='blue', alpha=0.7)
    ax[2].set_title('Histograma canal azul')

    # Ajustar los espacios entre las subtramas
    plt.tight_layout()

    # Mostrar el plot
    plt.show()

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

def metrics(y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred)
    acc = accuracy_score(y_test, y_pred)

    return conf, prec, rec, f1, sup, acc

def mostrar_metricas(conf, prec, rec, f1, sup, acc):
  print (np.array(100*conf.T/conf.sum(axis = 1),dtype = int).T)
  print(f'Accuracy:{round(acc,2)}')
  print(f'Precision:{round(prec[0],2)}')
  print(f'Recall:{round(rec[0],2)}')
  print(f'F1:{round(f1[0],2)}')
  print(f'Sup:{sup}')

def report(y_test, y_test_pred):
  print(classification_report(y_test, y_test_pred))

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


carpeta = './generated_images'
#Obtengo la lista de archivos de la carpeta
lista_archivos = os.listdir(carpeta)
imagenes = [archivo for archivo in lista_archivos if archivo.endswith('.jpg')]

#Creo una columna nombre con la extension de la imagen
df = pd.DataFrame(imagenes, columns=['Nombre'])
# Modificar la columna 'Nombre'
df['Nombre'] = df['Nombre'].apply(lambda x: os.path.splitext(x)[0] + '.jpg')

df['Color'] = df['Nombre'].apply(lambda x: re.search('^(.*?)_', x).group(1))
#df['Valor'] = df['Nombre'].apply(lambda x: re.search('_(.*?).jpg', x).group(1))
#df['Valor'] = df['Valor'].str.split('_').str[:-1].apply(lambda x: '_'.join(x))
# Agregar una nueva columna 'Acción' que indica si se detecta un número en 'Valor'
#df['Acción'] = df['Valor'].apply(lambda x: 'Normal' if re.search(r'\b\d', x) else 'Especial')

df['px_oscuros'] = df['Nombre'].apply(lambda x: extraer_pix_oscuros('./generated_images/' + x))

# Eliminar la columna 'Nombre' del DataFrame
df.drop(columns=['Nombre'], inplace=True)

# Creo una lista para almacenar los histogramas agrupados por canal
histogramas = []

num_pedazos = 12
ancho_pedazo = int(200/num_pedazos)
alto_pedazo = int(300/num_pedazos)


img1 = cv2.imread("./generated_images/Blue_0_1.jpg")
img_recortada = recortar_imagen(img1, 200, 300)
plt.imshow(img_recortada)
plt.title('Imagen Recortada')
plt.axis('off')  # Oculta los ejes
plt.show()
pedazos_aleatorios = recortar_secuencial(img_recortada, num_pedazos)
# Mostrar los pedazos aleatorios
for i, pedazo in enumerate(pedazos_aleatorios):
    plt.subplot(1, num_pedazos, i + 1)  # Crea subtramas para mostrar cada pedazo
    plt.imshow(pedazo)
    plt.title(f'Pedazo {i + 1}')
    plt.axis('off')  # Oculta los ejes
plt.show()

img_recortada2 = recortar_imagen_inferior_derecha(img1, 200,300)
plt.imshow(img_recortada2)
plt.title('Imagen Recortada')
plt.axis('off')  # Oculta los ejes
plt.show()

pedazos_aleatorios = recortar_secuencial(img_recortada2, num_pedazos)
# Mostrar los pedazos aleatorios
for i, pedazo in enumerate(pedazos_aleatorios):
    plt.subplot(1, num_pedazos, i + 1)  # Crea subtramas para mostrar cada pedazo
    plt.imshow(pedazo)
    plt.title(f'Pedazo {i + 1}')
    plt.axis('off')  # Oculta los ejes
plt.show()



for indice, imagen in enumerate(imagenes, start=0):
    ruta_imagen = os.path.join(carpeta, imagen)
    img = cv2.imread(ruta_imagen)

    # Recortar la parte superior izquierda de la imagen
    img_recortada = recortar_imagen(img,200, 300)

    pedazos_aleatorios = recortar_secuencial(img_recortada, num_pedazos)

    for pedazo in pedazos_aleatorios:
        valor_de_px_oscuros = df.iloc[indice]['px_oscuros']
        valor_de_color = df.iloc[indice]['Color']

        hist_r, hist_g, hist_b = hist_img(pedazo)
        histogramas.append([valor_de_color, valor_de_px_oscuros, hist_r, hist_g, hist_b])

    img_recortada2 = recortar_imagen_inferior_derecha(img,200, 300)

    pedazos_aleatorios = recortar_secuencial(img_recortada2, num_pedazos)

    for pedazo in pedazos_aleatorios:
        valor_de_px_oscuros = df.iloc[indice]['px_oscuros']
        valor_de_color = df.iloc[indice]['Color']

        hist_r, hist_g, hist_b = hist_img(pedazo)
        histogramas.append([valor_de_color, valor_de_px_oscuros, hist_r, hist_g, hist_b])

# Creo un DataFrame para almacenar los histogramas agrupados por canal
df_histogramas = pd.DataFrame(histogramas, columns=['Color','px_oscuros','Hist_R', 'Hist_G', 'Hist_B'])

hist_r_columns = [f'Hist_R_g{i+1}' for i in range(32)]
hist_g_columns = [f'Hist_G_g{i+1}' for i in range(32)]
hist_b_columns = [f'Hist_B_g{i+1}' for i in range(32)]

df_histogramas = pd.concat([
    df_histogramas,
    pd.DataFrame(df_histogramas['Hist_R'].tolist(), columns=hist_r_columns),
    pd.DataFrame(df_histogramas['Hist_G'].tolist(), columns=hist_g_columns),
    pd.DataFrame(df_histogramas['Hist_B'].tolist(), columns=hist_b_columns)
], axis=1)

# Eliminar las columnas originales de histogramas agrupados
df_histogramas.drop(columns=['Hist_R', 'Hist_G', 'Hist_B'], inplace=True)

df = df_histogramas

pd.set_option('display.max_rows', None)  # Mostrar todas las filas
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas

#print(df.to_string(index=False))

#mostrar_hist(histogramas[0])

#print("-----------------------------------------------")

#Mirem per sobre el dataset
#print (df.keys())
#print (df.shape)

#print("-----------------------------------------------")

# Ahora definimos lo que va a ser el target y lo que van a ser los atributos
from sklearn.model_selection import train_test_split


# Normalizo los datos que tienen que ver con el RGB de cada imagen
hist_columns = [col for col in df.columns if col.startswith('Hist_')]
hist_data = df[hist_columns]

scaler = MinMaxScaler()
scaled_hist_data = scaler.fit_transform(hist_data)

df[hist_columns] = scaled_hist_data

#print(df.to_string(index=False))

#print("-----------------------------------------------")

#columnas_categoricas = ['Valor','Acción']
# Aplicar one-hot encoding a las columnas categóricas
#df = pd.get_dummies(df, columns=columnas_categoricas)

from sklearn.preprocessing import LabelEncoder

# Inicializar el LabelEncoder
label_encoder = LabelEncoder()

# Aplicar el LabelEncoder a la columna 'Color'
df['Color'] = label_encoder.fit_transform(df['Color'])

# Verificar el DataFrame actualizado
#print(df.to_string(index=False))

#print("-----------------------------------------------")

df['px_oscuros'] = df['px_oscuros'].astype(int)

target = 'Color'
atributos = [k for k in df.keys() if k!=target]

print(atributos)

df = df.sample(frac=1, random_state=42)
# Definimos lo que serán los atributos y el target
X = df[atributos].to_numpy()
y = df[target].to_numpy()

# Hacemos un split separando el train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print("-----------------------------------------------")

#print("---------------------  Logistic Regression --------------------------")
from sklearn.svm import SVC

# Creamos el modelo SVC con los mejores hiperparámetros
mejor_modelo_svm = SVC(kernel='rbf', C=10, coef0=0.0, degree= 2, gamma= 0.1)

# Entrenar el modelo SVM
mejor_modelo_svm.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_svm = mejor_modelo_svm.predict(X_test)

# Calcular métricas de evaluación
conf_svm, prec_svm, rec_svm, f1_svm, sup_svm, acc_svm = metrics(y_test, y_pred_svm)
# Mostrar métricas
#mostrar_metricas(conf_svm, prec_svm, rec_svm, f1_svm, sup_svm, acc_svm)
# Reporte de clasificación
#report(y_test, y_pred_svm)


"""

Predecir uno en concreto

"""

from statistics import mode

print("")

def predecirImg(imagen_path, modelo, min_values, max_values):

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

    img = cv2.imread(imagen_path)

    # Recortar la parte superior izquierda de la imagen
    img_recortada = recortar_imagen(img,200, 300)

    pedazos_aleatorios = recortar_secuencial(img_recortada, num_pedazos)

    for pedazo in pedazos_aleatorios:

        valor_de_px_oscuros = extraer_pix_oscuros(imagen_path)
        """
        valor_de_color = df.iloc[indice]['Color']
        """

        hist_r, hist_g, hist_b = hist_img(pedazo)
        histogramas.append([valor_de_px_oscuros,hist_r, hist_g, hist_b])

    img_recortada2 = recortar_imagen_inferior_derecha(img, 200, 300)

    pedazos_aleatorios = recortar_secuencial(img_recortada2, num_pedazos)

    for pedazo in pedazos_aleatorios:
        valor_de_px_oscuros = extraer_pix_oscuros(imagen_path)
        """
        valor_de_color = df.iloc[indice]['Color']
        """

        hist_r, hist_g, hist_b = hist_img(pedazo)
        histogramas.append([valor_de_px_oscuros, hist_r, hist_g, hist_b])
    # Creo un DataFrame para almacenar los histogramas agrupados por canal
    df_histogramas = pd.DataFrame(histogramas, columns=['px_oscuros','Hist_R', 'Hist_G', 'Hist_B'])

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

    #df_pred['px_oscuros'] = extraer_pix_oscuros(imagen_path)

    #print(df_pred.to_string(index=False))

    #mostrar_hist(histogramas[0])

    #print("-----------------------------------------------")

    # Ahora definimos lo que va a ser el target y lo que van a ser los atributos
    from sklearn.model_selection import train_test_split

    # Normalizo los datos que tienen que ver con el RGB de cada imagen
    hist_columns_pred = [col for col in df_pred.columns if col.startswith('Hist_')]
    hist_data_pred = df_pred[hist_columns_pred]

    # Normalizar los datos de histogramas utilizando los valores mínimos y máximos proporcionados
    scaled_hist_data_pred = (hist_data_pred - min_values) / (max_values - min_values)

    df_pred[hist_columns_pred] = scaled_hist_data_pred

    #print(df_pred.to_string(index=False))

    #print("-----------------------------------------------")

    # Verificar el DataFrame actualizado
    #print(df_pred.to_string(index=False))

    df_pred.columns = range(df_pred.shape[1])

    y_pred = modelo.predict(df_pred)

    colores = ['Blue','Green','Red','Wild','Yellow']

    # Mostrar el resultado de la predicción

    valor_mas_frecuente = mode(y_pred)

    print(y_pred)
    print("Predicción el color es: ", colores[valor_mas_frecuente])


    return colores[valor_mas_frecuente]

"""
roja_path = './cartas/Red_3.jpg'
predecirImg(roja_path, mejor_modelo_svm,scaler.data_min_, scaler.data_max_)

yellow_path = './cartas/Yellow_3.jpg'
predecirImg(yellow_path, mejor_modelo_svm,scaler.data_min_, scaler.data_max_)

green_path = './cartas/Green_3.jpg'
predecirImg(green_path, mejor_modelo_svm,scaler.data_min_, scaler.data_max_)

blue_path = './cartas/Blue_3.jpg'
predecirImg(blue_path, mejor_modelo_svm,scaler.data_min_, scaler.data_max_)

blue_path = './cartas/Wild_Color.jpg'
predecirImg(blue_path, mejor_modelo_svm,scaler.data_min_, scaler.data_max_)

blue_path = './cartas/Wild_Draw_4.jpg'
predecirImg(blue_path, mejor_modelo_svm,scaler.data_min_, scaler.data_max_)
"""

from joblib import dump, load

# Guardar el modelo entrenado
dump(mejor_modelo_svm, 'modelo_entrenado_svm.joblib')
dump(scaler.data_min_, 'scaler_data_min.joblib')
dump(scaler.data_max_, 'scaler_data_max.joblib')


# Cargar el modelo entrenado
#modelo_cargado = load('modelo_entrenado_svm.joblib')
#d_min = load('scaler_data_min.joblib')
#d_max = load('scaler_data_max.joblib')

#print(scaler.data_min_)
#print(scaler.data_max_)

