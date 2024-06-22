from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

def recortarCentro(img):

    altura, ancho, _ = img.shape

    centro_imagen = [altura // 2, ancho // 2]

    return img[(centro_imagen[0] - 250):(centro_imagen[0] + 250),(centro_imagen[1] - 250):(centro_imagen[1] + 250),:]

def pasarGrises(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img_gray

def reducirTamaño(img):   

    img_reduced = cv2.resize(img, (30, 30), interpolation=cv2.INTER_LINEAR)

    return img_reduced

def predecir_imagen(img):

    label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'Draw2': 10, 'Reverse': 11, 'Skip': 12, 'Color': 13, 'Draw4': 14}

    model = load_model("unoNumberRecognition.h5")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = recortarCentro(img)
    # plt.imshow(img)
    # plt.show()
    img = reducirTamaño(img)
    # plt.imshow(img)
    # plt.show()
    img = pasarGrises(img)
    # plt.imshow(img)
    # plt.show()

    img = img.reshape(-1, 30, 30, 1)

    prediction = model.predict(img, verbose=0)
    predicted_label = list(label_map.keys())[np.argmax(prediction)]
    return predicted_label




'''
print("PREDIR CARTA")

print(predecir_imagen('FOTO_CARLES_1.png'))
print(predecir_imagen('FOTO_CARLES_1_r.png'))
print(predecir_imagen('FOTO_CARLES_2.png'))
print(predecir_imagen('FOTO_CARLES_2_r.png'))
print(predecir_imagen('FOTO_CARLES_3.png'))
print(predecir_imagen('FOTO_CARLES_3_r.png'))
print(predecir_imagen('FOTO_CARLES_4.png'))
print(predecir_imagen('FOTO_CARLES_4_r.png'))
print(predecir_imagen('FOTO_CARLES_5.png'))
print(predecir_imagen('FOTO_CARLES_5_r.png'))
print(predecir_imagen('FOTO_CARLES_6.png'))
print(predecir_imagen('FOTO_CARLES_6_r.png'))
print(predecir_imagen('FOTO_CARLES_7.png'))
print(predecir_imagen('FOTO_CARLES_7_r.png'))
print(predecir_imagen('FOTO_CARLES_8.png'))
print(predecir_imagen('FOTO_CARLES_8_r.png'))
print(predecir_imagen('FOTO_CARLES_9.png'))
print(predecir_imagen('FOTO_CARLES_9_r.png'))
print(predecir_imagen('FOTO_CARLES_10.png'))
print(predecir_imagen('FOTO_CARLES_10_r.png'))
print(predecir_imagen('FOTO_CARLES_11.png'))
print(predecir_imagen('FOTO_CARLES_11_r.png'))
print(predecir_imagen('FOTO_CARLES_12.png'))
print(predecir_imagen('FOTO_CARLES_12_r.png'))
'''


