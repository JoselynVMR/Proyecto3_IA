# Proyecto3_IA

Este es un repositorio para almacenar el proyecto 3 del curso de inteligencia artificial

## Elaborado por
- Esteban Pérez Picado – 2021046572
- Joselyn Montero Rodríguez – 2022136356
- Manuel A. Rodríguez Murillo – 2021028686

## Ejecucion

Este proyecto tiene varias carpetas con diferentes contenidos, pero para el envió del mismo mediante la plataforma y almacenamiento en GitHub, pero estas mismas se explican en la parte de `Organización de carpetas`, lo único necesario para una ejecución inicial es la carpeta de `data` como se explica a continuación.

* Descargar el set de datos en formato .zip de Kaggle `https://www.kaggle.com/datasets/gpiosenka/butterflies-100-image-dataset-classification`.
* Posterior a extro extraer los datos, especificamente el `.csv` y las tres carpetas `test`, `train` y `valid` y agregarlas dentro de la carpeta `data` del proyecto.
* Con esto y los paquetes conrrespondientes de python ya esta todo listo para ejecutar el proyecto desde el archivo Jupyter `proyecto3.ipynb`.

## Organizacion de carpetas

Es importante recalcar que como se mencionó anteriormente que la mayorías de carpetas de este proyecto van a estar vacías, debido a que por ejemplo la carpeta de `checkpoints` llega a pegar casi los 15GB y la de data los 500MB, por lo se borraron todos los logs y solo se mantuvo la información necesaria y para la validación como la carpeta `clusters` con las imágenes del espacio latente, la carpeta `configuration` con todo el manejo de datos mediante Hydra y la carpeta `scripts` que contiene todo el código base.

* La carpeta `checkpoints` almacena toda la informacion importante de la ejecucion de los entrenamientos de los diferentes modelos.
* La carpeta `clusters` almacena todas las imagenes de los modelos DAE y VDAE asi como los mapas del espacio latente tsne_kmeans y tsne_only.
* La carpeta `configuration` almacena todas las configuraciones .ymal para los modelos y los parametros de ejecucion de los mismos.
* La carpeta `data` almacena toda la informacion del dataset como el conjunto de especies seleccionado con 30 especies y tamaños ajustados.
* La carpeta `outputs` almacena los logs de la ejecuciones realizadas mediante Hydra.
* La carpeta `scripts` contiene la parte mas importante del proyecto, con todos los scrips de python necesarios, desde utilidades generales, modelos de analisis, los entrenamientos y los propios modelos.
* La carpeta `wanddb` almacena todos los logs de las ejecuciones realizadas por wandb.
* La carpeta `weights` almacena todos los pesos de los auntoencoders entrenados, especificamente de los AU normales con 10 y 30 porciento de los labels y de los DAE y VDAE, para usarlos posteriormente.

## Notebook

El notebook se divide por las siguientes 3 partes importantes.

### Analisis de datos

* Comprensión de los datos.
* Visualizacion de los datos.
* Seleccion de especies.
* Reduccion de dimension.
* Incorporacion de muestras.

### Experimento 1

* Entrenamiento de autoencoders con 10 y 30 porciento.
* Entrenamiento de clasificadores A, B1 y B2 con 10 y 30 porciento.
* Entrenamiento de clasificadores cuantizados A, B1 y B2 con 10 y 30 porciento.

### Experimento 2

* Entrenamiento del DAE.
* Muesta del espacio latente del DAE
* Entrenamiento del VDAE.
* Muesta del espacio latente del VDAE
