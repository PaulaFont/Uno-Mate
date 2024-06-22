# Uno-Mate <img src="https://github.com/PaulaFont/Uno-Mate/blob/main/miscelanea/Logo_UnoMate_mod_2.png" alt="Logo" style="width: 50px; height: 50px; vertical-align: middle;">

## Contendio del repositorio
- **3D_files**: Directorio que contiene todos los ficheros relativos al diseño 3D del robot
- **Fritzing**: Dentro de este tenemos todo el prototipaje de conexiones hecho en simulación
- **Pressupostos**: En el interior podemos encontrar los ficheros con la lista de componentes y sus respectivos precios
- **Sprints**: Aquí se encuentran todos los documentos relativos a los sprints
- **miscelanea**: Carpeta donde se encuentra archivos de recursos para el proyecto
- **src**: En esta carpeta es donde tenemos el código de nuestro robot

## Descripción del repositorio
### Contenido

1. [Descripcion](#descripcion)
2. [Requisitos](#requisitos)
3. [Librerias](#librerias)
4. [Esquema conexiones](#Esquema-conexiones)
5. [Ficheros-3D](#Ficheros-3D)
6. [Referencias](#referencias)
7. [Autores](#autores)


### Descripcion
<div style="text-align: justify;">

El objetivo del proyecto consiste en hacer el monitoreo de una partida del UNO, una de las cosas principales que hacemos en el proyecto es mirar cada una de las cartas que se van tirando en cada turno y por quién son tiradas, además de controlar si se hacen trampas o no. A más, otros objetivos del robot es que sea posible mezclar, repartir y dar diferentes cartas a los jugadores.

Para hacer el monitoreo de la partida lo que hemos hecho ha sido diferentes algoritmos. Uno que nos permite recortar la carta que está arriba del montón, y otros dos que nos permiten reconocer el número de la carta y el color de la carta.
Para hacer estos algoritmos hemos tenido que entrenar diferentes modelos. Para esto tuvimos que hacer un conjunto de datos customizado, ya que no encontrábamos ningún conjunto de datos que se ajustara a nuestra situación de las cartas del Uno. 

La funcionalidad de mezclar está diseñada para mezclar las cartas que se encuentran en las dos cajas laterales. En la caja de almacenaje de cartas (ubicada en el medio) se juntarán los dos bloques quedando completamente mezclados.
En el caso de repartir cartas, si el robot detecta que solo es necesario repartir una carta, se reparte una, si se detecta una carta de +2 se reparten dos cartas y para el caso del +4  se reparten  cuatro cartas.

Además, tendremos un audio que irá informando al jugador de los movimientos que puede hacer en cada momento.

En el caso de que la partida se acabe, lo que hacemos es guardar en la base de datos Firestore Database el historial de la partida, es decir, las cartas que se han tirado o robado, en qué turno, el día, la hora de la partida y el jugador que ha ganado. Este registro de partida estará disponible para todos los  participantes.

</div>


## Requisitos
### Como jugar una partida

 1. Conectar el robot a la red eléctrica
 2. Conectar el robot a internet
 3. Establecer conexión entre la aplicación y el robot
 4. Empezar la partida desde la aplicación indicando los jugadores
 5. Seguir las instrucciones que el robot proporciona por voz
 6. Una vez finalizada la partida se puede consultar a través de la app


## Librerias

- [Hardware (Fritzing)](https://github.com/PaulaFont/Uno-Mate/tree/main/Fritzing)

- [3D](https://www.tinkercad.com/)

- Detecció de cartes (VC)

- [Speech to text](https://cloud.google.com/speech-to-text/?hl=es&utm_source=google&utm_medium=cpc&utm_campaign=emea-es-all-es-dr-bkws-all-all-trial-e-gcp-1707574&utm_content=text-ad-none-any-DEV_c-CRE_593880918158-ADGP_Hybrid+%7C+BKWS+-+EXA+%7C+Txt+-+AI+And+Machine+Learning+-+Speech+to+Text+-+v1-KWID_43700053288209417-kwd-21425535976-userloc_20270&utm_term=KW_google%20speech%20to%20text-NET_g-PLAC_&&gad_source=1&gclid=CjwKCAjw8diwBhAbEiwA7i_sJRV1cr_KDYNgeVYz4GjR6m7_OZMuziSL3FX58t5i6XlOnxXAtUswKRoCR_wQAvD_BwE&gclsrc=aw.ds)

## Esquema-conexiones

![Image text](https://github.com/PaulaFont/Uno-Mate/blob/main/Fritzing/Esquema_conexiones_UnoMate.png)


## Ficheros-3D

Cajas de madera:
- Cajas de madera arriba : Tenemos los diseños de la plataforma donde jugaremos y las cajas de los laterales, donde irán las cartas.
- Caja de madera medio: En este fichero nos encontramos la caja del medio que será donde caerán las cartas.
- Caja en forma de U: En esta caja será donde pondremos todos los componentes de nuestro robot, además encima irán cada una de las cajas mencionadas anteriormente

Diseños 3D:
- Rampa para cartas 3D: En este fichero lo que tenemos es el modelo de rampa 3D que irá en cada una de las cajas laterales, que permitirá que las cartas salgan más fácilmente



## Referencias
- [Shuffle Inspiration Video](https://www.youtube.com/watch?v=kTARmpW6t8g)
- [Card Detector Github](https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector)
- [Similar Project RLP](https://rlpengineeringschooluab2023.wordpress.com/2023/06/06/slapbot/)


## Autores

- Rubén García Viciana (1634065)
- Marc Artero Pons (1632512)
- Carles Fornés Mas (1633536)
- Paula Font Solà (1633214)
