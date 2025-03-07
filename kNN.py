 -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo kNN
Recoge los datos de un fichero csv y los clasifica en función de los k vecinos más cercanos
"""

import sys
import sklearn as sk
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.impute import SimpleImputer
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def concatenar_columnas(data, columnas):
    """
    Concatenar varias columnas en un nuevo columna llamada 'concatenada'.
    
    Parámetros:
    - X_train: DataFrame de pandas que contiene las columnas a concatenar.
    - columnas: Lista de nombres de las columnas que deseas concatenar.
    - sep: Separador que se usará para concatenar las columnas (por defecto es ',').
    
    Retorna:
    - DataFrame con la columna 'concatenada' añadida.
    """
    sep=' '
    # Verificar que la lista de columnas tiene al menos dos columnas
    if len(columnas) < 2:
        raise ValueError("La lista de columnas debe contener al menos dos columnas.")
    
    # Asegurarse de que no haya valores NaN antes de concatenar
    for col in columnas:
        data[col] = data[col].fillna('')  # Reemplazar NaN por una cadena vacía
    
    # Usar str.cat() para concatenar las columnas seleccionadas
    data[columnas[0]] = data[columnas[0]].str.cat(data[columnas[1]], sep=sep)
    
    # Si hay más de dos columnas, concatenarlas
    for col in columnas[2:]:
        data[columnas[0]] = data[columnas[0]].str.cat(data[col], sep=sep)
    
    return data

def guardarModelo(clf):
    nombreModel = "nombreParAlmacenar.sav"
    with open(nombreModel, 'wb') as archivo:
        pickle.dump(clf, archivo)
    print("Modelo guardado correctamente empleando Pickle")

def cargarModelo(nombreModel):
    with open(nombreModel, 'rb') as archivo:
        modelo = pickle.load(archivo)
    return modelo
#Preprocesado      
def limpiar_texto(texto):
    """
    Limpia y normaliza texto basado en configuraciones de `miJson`:
    - Convierte a minúsculas
    - Elimina acentos y caracteres especiales
    - Elimina stopwords
    - Aplica lematización
    """

    # Obtener la configuración de preprocesamiento
    opciones = miJson.get('preproceso', {}).get('normalize_vector', [])

    lemmatizer = WordNetLemmatizer()
    stop_words_list = set(stopwords.words("english")) if "stopwords" in opciones else set()

    # Si el texto es NaN, None o vacío, devolverlo sin cambios
    if texto is None or texto == "" or pd.isna(texto):
        return texto

    # 1. Convertir a minúsculas
    if "minusculas" in opciones:
        texto = texto.lower().strip()

    # 2. Quitar acentos
    if "acentos" in opciones:
        texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')

    # 3. Eliminar caracteres especiales y números
    if "caracEsp" in opciones:
        texto = re.sub(r'[^a-z\s]', '', texto)

    # 4. Tokenizar el texto (convertir en lista de palabras)
    palabras = word_tokenize(texto, language="english") if "tokenizar" in opciones else texto.split()

    # 5. Eliminar stopwords y aplicar lematización si está activado
    if "lematizar" in opciones:
        palabras = [lemmatizer.lemmatize(p) for p in palabras if p not in stop_words_list]

    # 6. Unir palabras procesadas en una sola cadena
    texto_limpio = " ".join(palabras)

    return texto_limpio

def cat_numerico(df, columnas):
    """
    Convierte columnas categóricas a numéricas usando pd.factorize().
    
    Parámetros:
    - df: DataFrame de Pandas
    - columnas: Lista de nombres de columnas a convertir (opcional).
                Si es None, convierte todas las categóricas.

    Retorna:
    - DataFrame con las columnas convertidas a valores numéricos.
    """
    if columnas is None:
        # Si no se especifican columnas, convertir todas las categóricas
        columnas = df.select_dtypes(exclude=['number']).columns.tolist()
    else:
        # Filtrar solo las que existen en el DataFrame
        columnas = [col for col in columnas if col in df.columns]

    # Aplicar pd.factorize() a las columnas seleccionadas
    df[columnas] = df[columnas].apply(lambda x: pd.factorize(x)[0])
    
    return df

###############ESCALADO############
def z_score(v):
    #input: it expects a column from the data frame
    #output: the column will be scaled using the z-score technique
    
    # copy the column
    v_norm = v
    # apply the z-score method
    v_norm = (v - v.mean()) / v.std()
    return v_norm
def maximum_absolute_scaling(df):
    #input: a whole dataFrame
    #output: a whole dataFrame where integer type columns with a mean value > 60 will be scaled
    
    print(df.head())
    # copy the dataframe
    df_scaled = df.copy()
    # apply maximum absolute scaling
    for column in df_scaled.columns:
        if df_scaled.dtypes[column] == np.int64 and df_scaled[column].mean()>60:
            df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
    return df_scaled
def min_max_scaling(v):
    #input: it expects a column from the data frame
    #output: the column will be scaled using the min-max technique
    
    # copy the column
    v_norm = v
    # apply min-max scaling
    #for column in df_norm.columns:
    v_norm = (v - v.min()) / (v.max() - v.min())

    return v_norm 
def escaladoEstandar(dt):
    # Escalamos los datos
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    dt= sc.fit_transform(dt)
    return dt


def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    data = pd.read_csv(file, encoding="latin1")

    #data = pd.read_csv(file, encoding="utf-8")
    return data

def calculate_fscore(y_test, y_pred):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    from sklearn.metrics import f1_score
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    return fscore_micro, fscore_macro

def calculate_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm

def kNN(data, k, weights, p):
    """
    Función para implementar el algoritmo kNN
    
    :param data: Datos a clasificar
    :type data: pandas.DataFrame
    :param k: Número de vecinos más cercanos
    :type k: int
    :param weights: Pesos utilizados en la predicción ('uniform' o 'distance')
    :type weights: str
    :param p: Parámetro para la distancia métrica (1 para Manhattan, 2 para Euclídea)
    :type p: int
    :return: Clasificación de los datos
    :rtype: tuple
    """
    # Seleccionamos las características y la clase
    X = data.iloc[:, :-1].values # Todas las columnas menos la última
    y = data.iloc[:, -1].values # Última columna
    
    # Dividimos los datos en entrenamiento y test
    from sklearn.model_selection import train_test_split
    np.random.seed(42)  # Set a random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    print("Dimensiones de X_train:", X_train.shape)
    print("Dimensiones de X_test:", X_test.shape)

    
    ##### Preproceso ######
    if miJson['preproceso']['Preprocesar?']=='si' :

        if isinstance(X_train, np.ndarray):
            print("⚠️ X_train es un numpy.ndarray. Convirtiéndolo a DataFrame...")
            cols = data.columns.tolist()[:-1]            
            X_train = pd.DataFrame(X_train,columns=cols)
            X_test = pd.DataFrame(X_test,columns=cols)
        print("Columnas en X_train:", X_train.columns.tolist())

        # Concatenar con un delimitador, como una coma
        X_train=concatenar_columnas(X_train,miJson['preproceso']['cols_concatenar'])
        X_test=concatenar_columnas(X_test,miJson['preproceso']['cols_concatenar'])

        if miJson['preproceso']['missing_values'] == 'imputar':
            print("Imputación en proceso.")
            # Configurar la estrategia de imputación
            if miJson['preproceso']['impute_strategy'] == 'mean':
                    imputer = SimpleImputer(strategy="mean")
                    print("Metodo es mean")
            elif miJson['preproceso']['impute_strategy'] == 'mode':
                    imputer = SimpleImputer(strategy="most_frequent")
            # Obtener las columnas a imputar desde el JSON
            cols_imputar = miJson['preproceso'].get('cols_imputar', [])
    
            # Verificar si las columnas a imputar existen en X_train
            cols_imputar_validas = [col for col in cols_imputar if col in X_train.columns]
            print(cols_imputar_validas)
            print("Valores NaN en las columnas a imputar:", X_train[cols_imputar_validas].isnull().sum())
            if cols_imputar_validas:  # Solo imputar si hay columnas válidas
                X_train[cols_imputar_validas] = imputer.fit_transform(X_train[cols_imputar_validas])
                X_test[cols_imputar_validas] = imputer.transform(X_test[cols_imputar_validas])
                print("Imputación completada.")
            else:
                print(f"⚠️ Advertencia: No se encontraron las columnas {cols_imputar} en X_train. Verifica los nombres.")    
        print("Valores faltantes en X_train:", X_train.isnull().sum())

        # Pasar de categorial a númerico
        if miJson['preproceso']['cat_num?']=='si' :
            cols_cat_num= miJson['preproceso'].get('categorial_features', [])
            cat_numerico(X_train,cols_cat_num)
            cat_numerico(X_test,cols_cat_num)
            for col in miJson['preproceso']['categorial_features']:
                print(f"Valores únicos en {col}: {X_train[col].unique()}")

        if miJson['preproceso']['normalize?']=='si' :
            cols_simplificar=miJson['preproceso']['normalize_features']
            for col in cols_simplificar:
                X_train[col] = X_train[col].fillna('').apply(lambda x: limpiar_texto(x))
                X_test[col] = X_test[col].fillna('').apply(lambda x: limpiar_texto(x))
            print(X_train.head())  # Esto imprimirá las primeras 5 filas de todas las columnas por defecto
        
        if miJson['preproceso']['scaling']=='standar' :
         # Escalamos los datos de forma standard
         from sklearn.preprocessing import StandardScaler
         sc = StandardScaler()
         X_train = sc.fit_transform(X_train)
         X_test = sc.transform(X_test)
        elif miJson['preproceso']['scaling']=='absmaxmin' :
            # Escalamos los datos de forma absminmax
            X_train=maximum_absolute_scaling(X_train)
            X_test=maximum_absolute_scaling(X_test)
        elif miJson['preproceso']['scaling']=='minmax' :
            # Escalamos los datos de forma minmax
            cols=X_train.columns
            for col in cols :
                min_max_scaling(col)
            cols=X_test.columns
            for col in cols :
                min_max_scaling(col)
        elif miJson['preproceso']['scaling']=='zscore' :
            # Escalamos los datos de forma zscore
            cols=X_train.columns
            for col in cols :
                z_score(col)
            cols=X_test.columns
            for col in cols :
                z_score(col)
        cols_eliminar=miJson['preproceso']['cols_eliminar']
        X_train = X_train.drop(cols_eliminar, axis=1)
        X_test= X_test.drop(cols_eliminar, axis=1)

   


   



    
    # Entrenamos el modelo
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = k, weights = weights, p = p)
    classifier.fit(X_train, y_train)
    
    # Predecimos los resultados
    y_pred = classifier.predict(X_test)
    
    return y_test, y_pred

if __name__ == "__main__":
    with open('config.json', 'r', encoding='utf-8') as file:
        miJson = json.load(file)  
    # Comprobamos que se han introducido los parámetros correctos
    if len(sys.argv) < 3:
        print("Error en los parámetros de entrada")
        print("Uso: kNN.py <fichero*> <k*> <weights> <p>")
        sys.exit(1)
    
    # Cargamos los datos
    data = load_data(sys.argv[1])  # Cargar los datos desde el archivo CSV

    print("Tipo de data:", type(data))  # Verificar si es un DataFrame
    print("Primeras filas de data:\n", data.head())  # Ver las primeras filas
    # Identificar las columnas 'Unnamed' en el DataFrame
    unnamed_cols = [col for col in data.columns if 'Unnamed' in col]

    # Renombrar las columnas 'Unnamed' de forma que tengan nombres más significativos
    new_names = {col: f"New_Column_{i}" for i, col in enumerate(unnamed_cols)}

    # Renombrar las columnas
    data.rename(columns=new_names, inplace=True)

    # Nombre de la columna que quieres mover
    columna_target = miJson["df_Caracteristicas"]["Target"]
    # Reorganizar el DataFrame para mover el target al final
    data = data[[col for col in data.columns if col != columna_target] + [columna_target]]
    
    # Verificar las columnas categóricas y numéricas
    categorical_columns = data.select_dtypes(include=['object']).columns
    numerical_columns = data.select_dtypes(include=['number']).columns
    print("Columnas categóricas:", categorical_columns)
    print("Columnas numéricas:", numerical_columns)

    # Implementamos el algoritmo kNN
    y_test, y_pred = kNN(data, int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'uniform', int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    # Mostramos la matriz de confusión
    print("\nMatriz de confusión:")
    print(calculate_confusion_matrix(y_test, y_pred))
  
  
   
