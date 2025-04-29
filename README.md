# Manual de Uso

## Requerimientos

- Python 3.10.16
- pip
- conda

## Instalación

1. Clonar el repositorio
2. Crear un entorno virtual con conda
3. Instalar las dependencias con pip

```bash
conda create -n sad python=3.10.16
conda activate sad
pip install -r requirementsNuestro.txt
```

## Ayuda

```bash
python clasificador.py --help
=== Clasificador ===
usage: clasificador.py [-h] -m MODE -f FILE -a ALGORITHM -p PREDICTION [-e ESTIMATOR] [-c CPU] [-v] [--debug]

Practica de algoritmos de clasificación de datos.

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Modo de ejecución (train, test, prediction)
  -f FILE, --file FILE  Fichero csv (/Path_to_file)
  -a ALGORITHM, --algorithm ALGORITHM
                        Algoritmo a ejecutar (kNN, decision_tree o random_forest)
  -p PREDICTION, --prediction PREDICTION
                        Columna a predecir (Nombre de la columna)
  -e ESTIMATOR, --estimator ESTIMATOR
                        Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
  -c CPU, --cpu CPU     Número de CPUs a utilizar [-1 para usar todos]
  -v, --verbose         Muestra las metricas por la termina
  --debug               Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]
```

## Uso

Basico

```bash
python clasificador.py -m train -a kNN -f iris.csv -p Especie
```

Avanzado

```bash
python clasificador.py -m train -a kNN -f iris.csv -p Especie -e accuracy -c 4 -v --debug
```
'''
Ejemplo para consiguer modelos de naive bayes con el tripadvisor y airbnbreviews

python clasificador.py -m train  -f tripadvisor_hotel_reviews.csv -a naive_bayes -p "Rating" -e f1_weighted -v --debug
python clasificador.py -m train  -f AirBNBReviews.csv -a naive_bayes -p "Positive or Negative" -e f1_weighted -v --debug

Ejemplo para conseguir una prediccion sobre el test
python clasificador.py -m test  -f tripadvisor_hotel_reviews.csv -a naive_bayes -p "Rating" -e f1_weighted -v --debug
python clasificador.py -m test  -f AirBNBReviews.csv -a naive_bayes -p "Positive or Negative" -e f1_weighted -v --debug

Ejemplo para conseguir una prediccion en general
python clasificador.py -m prediction  -f tripadvisor_hotel_reviews.csv -a naive_bayes -p "Rating" -e f1_weighted -v --debug
python clasificador.py -m prediction  -f AirBNBReviews.csv -a naive_bayes -p "Positive or Negative" -e f1_weighted -v --debug

## JSON

```json
{
    "preproceso":{ 
        "Preprocesar?": "si", // Indica si se debe realizar preprocesamiento (sí, no)
        "unique_category_threshold": 51, // Numero de apariciones unicas para considerar una columna como categorica (int)
        "cat_num?": "si", // Indica si se deben convertir variables categóricas en numéricas
        "categorial_features":["no"], // Lista de variables categóricas a procesar (si escribes 'no' se aplica a todas las categorial features)
        "missing_values": "imputar", // Método de manejo de valores faltantes (imputar, drop)
        "impute_strategy": "mode", // Estrategia de imputación (mean, mode, median)
        "cols_imputar": [" "], // Lista de columnas en las que aplicar la imputación  (no hace falta rellenar)
        "cols_outliers": ["no"], // Lista de columnas en las que detectar y manejar outliers (si pones 'no' no se aplica)
        "scaling": "zscore", // Método de escalado de datos (standar, absmaxmin, minmax, zscore)
        "text_process": "tf-idf", // Método de procesamiento de texto (tf-idf, bow)
        "cols_concatenar":["no"], // Lista de columnas a concatenar (ninguna en este caso)
        "cols_eliminar": ["no"], // Lista de columnas a eliminar (ninguna en este caso) (si pones 'no' no se aplica)
        "max_arg_textProcessor": "no", //Numero de palabras maximas que pilla el tf-idf, si es "no" no hay un limite
        "normalize?": "no", // Indica si se debe normalizar la data (si, no)
        "normalize_features": ["v2"], // Lista de variables a normalizar (no hace falta)
        "normalize_vector":["minusculas","acentos","stopwords","caracEsp","lematizar"], // Operaciones de normalización en texto (minúsculas, quitar acentos, stopwords, caracteres especiales, lematizar)
        "sampling": "undersampling" // Método de muestreo para balancear clases (undersampling, oversampling)
    },
    "kNN": {
        "n_neighbors": [3, 5, 7],             // Numero de vecinos (lista de enteros)
        "weights": ["uniform", "distance"],   // Peso de los vecinos (uniform, distance)
        "algorithm": ["auto"],                // Algoritmo para calcular los vecinos (auto, ball_tree, kd_tree, brute)
        "leaf_size": [30],                    // Tamaño de la hoja (lista de enteros)
        "p": [2]                              // Parametro de la distancia (1 para manhattan, 2 para euclidean)
    },
    "decision_tree": {
        "criterion": ["gini","entropy"],                // Criterio para medir la calidad de la particion (gini, entropy)
        "max_depth": [5, 10, 20, 30],         // Profundidad maxima del arbol (lista de enteros)
        "min_samples_split": [2, 5, 10],      // Numero minimo de muestras para dividir un nodo (lista de enteros)
        "min_samples_leaf": [1, 2, 4],        // Numero minimo de muestras para ser una hoja (lista de enteros)
        "max_features": ["sqrt", "log2"],     // Numero maximo de caracteristicas a considerar (sqrt, log2)
        "splitter": ["best"]                  // Estrategia para elegir la particion (best, random)
    },
    "random_forest": {
        "n_estimators": [50],                 // Numero de arboles (lista de enteros)
        "criterion": ["gini"],                // Criterio para medir la calidad de la particion (gini, entropy)
        "max_depth": [5, 10],                 // Profundidad maxima del arbol (lista de enteros)
        "min_samples_split": [2, 5, 10],      // Numero minimo de muestras para dividir un nodo (lista de enteros)
        "min_samples_leaf": [1, 2, 4],        // Numero minimo de muestras para ser una hoja (lista de enteros)
        "max_features": ["sqrt", "log2"],     // Numero maximo de caracteristicas a considerar (sqrt, log2)  
        "bootstrap": [false,true]                  // Si se deben usar muestras bootstrap (true, false)
    }
    ,"naive_bayes": {
        "alpha": [0.00000001,0.0001,0.1, 0.25, 0.5, 0.75, 1.0, 2.0],
        "fit_prior": [true, false]
        }
}
```