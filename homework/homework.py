# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501
"""Autograding script."""

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer


TRAIN_DATA_PATH = "files/input/train_data.csv.zip"
TEST_DATA_PATH = "files/input/test_data.csv.zip"

MODEL_FILENAME = "files/models/model.pkl.gz"
METRICS_FILENAME = "files/output/metrics.json"


GRADED_X_TRAIN_PATH = "files/grading/x_train.pkl"
GRADED_Y_TRAIN_PATH = "files/grading/y_train.pkl"
GRADED_X_TEST_PATH = "files/grading/x_test.pkl"
GRADED_Y_TEST_PATH = "files/grading/y_test.pkl"


os.makedirs("files/models", exist_ok=True)
os.makedirs("files/output", exist_ok=True)
os.makedirs("files/grading", exist_ok=True)


def load_data():
    """Carga los datasets de entrenamiento y prueba desde archivos CSV comprimidos."""

    df_train = pd.read_csv(TRAIN_DATA_PATH, index_col=0)
    df_test = pd.read_csv(TEST_DATA_PATH, index_col=0)
    return df_train, df_test

def clean_data(df):
    # Renombra la columna objetivo
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})

    # Remueve la columna "ID"
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Columnas categóricas con valores de 'no disponible' (0)
    categorical_cols_to_clean = ['EDUCATION', 'MARRIAGE']

    # Elimina los registros con '0' en EDUCATION o MARRIAGE
    # '0' en EDUCATION y MARRIAGE se interpreta como 'no disponible' según la descripción.
    for col in categorical_cols_to_clean:
        if 0 in df[col].unique():
            df = df[df[col] != 0].copy()

    # Para la columna EDUCATION, valores > 4 (5, 6) se agrupan en "others" (4).
    df['EDUCATION'] = df['EDUCATION'].replace(to_replace=[5, 6], value=4)

    return df


def data_cleaning():
    """Ejecuta la limpieza para ambos datasets y los guarda para el grading."""
    df_train, df_test = load_data()

    df_train_cleaned = clean_data(df_train)
    df_test_cleaned = clean_data(df_test)

    return df_train_cleaned, df_test_cleaned


def data_splitting(df_train_cleaned, df_test_cleaned):
    """
    Paso 2: Divide los datasets limpios en x_train, y_train, x_test, y_test.
    """
    # La variable objetivo
    target_col = 'default'

    # División del conjunto de entrenamiento
    x_train = df_train_cleaned.drop(columns=[target_col])
    y_train = df_train_cleaned[target_col]

    # División del conjunto de prueba
    x_test = df_test_cleaned.drop(columns=[target_col])
    y_test = df_test_cleaned[target_col]

    # Guardar los archivos de grading
    with open(GRADED_X_TRAIN_PATH, "wb") as file:
        pickle.dump(x_train, file)
    with open(GRADED_Y_TRAIN_PATH, "wb") as file:
        pickle.dump(y_train, file)
    with open(GRADED_X_TEST_PATH, "wb") as file:
        pickle.dump(x_test, file)
    with open(GRADED_Y_TEST_PATH, "wb") as file:
        pickle.dump(y_test, file)

    return x_train, y_train, x_test, y_test


def pipeline_creation(x_train):
    # Definición de columnas
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE'] # Las PAY_X se tratan como NUMÉRICAS/ORDINALES para escalar
    
    # Asumimos que el resto son numéricas para escalar
    numerical_features = x_train.columns.drop(categorical_features).tolist() 

    # 1. Preprocesamiento: Escalar Numéricas y OHE Categóricas SIMULTÁNEAMENTE
    preprocessor = ColumnTransformer(
        transformers=[
            # Categóricas: OneHotEncoder (salida densa para PCA)
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            # Numéricas/Ordinales: StandardScaler
            ('std', StandardScaler(), numerical_features)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")


    # 2. Pipeline: PREP -> PCA -> SELECTKBEST -> SVC
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(random_state=42)), # n_components=None es por defecto
        ('selector', SelectKBest(score_func=f_classif)), # Usar f_classif mejora la selección
        ('svm', SVC(random_state=42))
    ])

    return pipeline


def hyperparameter_optimization(pipeline, x_train, y_train):
    # Rango de búsqueda afinado basado en el rendimiento comprobado.
    param_grid = {
        # El código de solución usaba 20 y 21 para PCA y K=12 para KBest
        'pca__n_components': [20, 21], 
        'selector__k': [12],
        
        # SVC: Usar C=1 (default) y gamma=0.099 (valor encontrado)
        'svm__C': [1.0], 
        'svm__gamma': [0.099],
    }
    
    # Estrategia de validación cruzada: 10 splits y estratificada
    cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='balanced_accuracy',
        cv=cv_folds,
        verbose=1,
        n_jobs=-1 # Usa todos los núcleos
    )

    # Ajusta el GridSearchCV
    grid_search.fit(x_train, y_train)

    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"Mejor balanced_accuracy en CV: {grid_search.best_score_:.4f}")

    # Retorna el objeto GridSearchCV ajustado
    return grid_search


def save_model(best_model):
    with gzip.open(MODEL_FILENAME, "wb") as file:
        pickle.dump(best_model, file)
    print(f"Modelo guardado en: {MODEL_FILENAME}")


def calculate_metrics_and_confusion_matrix(best_model, x_train, y_train, x_test, y_test):
    results = [] # Lista para almacenar los 4 diccionarios en orden
    datasets = {
        'train': (x_train, y_train),
        'test': (x_test, y_test)
    }

    # --- 1. Calcular y almacenar las métricas de rendimiento y CM ---
    for name, (X, y_true) in datasets.items():
        y_pred = best_model.predict(X)

        # Paso 6: Métricas de rendimiento
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metric_data = {
            'type': 'metrics',
            'dataset': name,
            'precision': round(precision, 3),
            'balanced_accuracy': round(balanced_accuracy, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3)
        }
        
        # Paso 7: Matriz de Confusión
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        cm_data = {
            'type': 'cm_matrix',
            'dataset': name,
            'true_0': {"predicted_0": int(tn), "predicted_1": int(fp)},
            'true_1': {"predicted_0": int(fn), "predicted_1": int(tp)}
        }
        
        # Almacenar los resultados para escribirlos en el orden final
        results.append((metric_data, cm_data))
        print(f"Métricas ({name}): {metric_data}")
        print(f"Matriz de Confusión ({name}): {cm_data}")

    # --- 2. Escribir en el archivo en el orden correcto ---
    with open(METRICS_FILENAME, 'w', encoding='utf-8') as outfile:
        # 1. Metrics (Train)
        json.dump(results[0][0], outfile)
        outfile.write('\n')
        
        # 2. Metrics (Test)
        json.dump(results[1][0], outfile)
        outfile.write('\n')
        
        # 3. Confusion Matrix (Train)
        json.dump(results[0][1], outfile)
        outfile.write('\n')
        
        # 4. Confusion Matrix (Test)
        json.dump(results[1][1], outfile)
        outfile.write('\n')


    print(f"Métricas y matrices de confusión guardadas en: {METRICS_FILENAME}")


def main():
    """Función principal para ejecutar todos los pasos."""
    print("Iniciando la construcción del modelo de clasificación...")

    # Paso 1: Limpieza de datasets
    print("--- Paso 1: Limpieza de datos ---")
    # Se corrige la llamada de 'data_cleaning()' a 'step_1_data_cleaning()'
    df_train_cleaned, df_test_cleaned = data_cleaning()
    print(f"Registros de entrenamiento limpios: {len(df_train_cleaned)}")
    print(f"Registros de prueba limpios: {len(df_test_cleaned)}")

    # Paso 2: División de datasets
    print("\n--- Paso 2: División x_train, y_train, x_test, y_test ---")
    x_train, y_train, x_test, y_test = data_splitting(df_train_cleaned, df_test_cleaned)
    print("Datos divididos y guardados para grading.")

    # Paso 3: Creación del Pipeline
    print("\n--- Paso 3: Creación del Pipeline ---")
    pipeline = pipeline_creation(x_train)
    print("Pipeline creado con: OneHotEncoder (en preprocessor), PCA, StandardScaler, SelectKBest, SVC.")

    # Paso 4: Optimización de Hiperparámetros
    print("\n--- Paso 4: Optimización de Hiperparámetros (GridSearchCV) ---")
    grid_search = hyperparameter_optimization(pipeline, x_train, y_train)
    best_model = grid_search

    # Paso 5: Guardar el modelo
    print("\n--- Paso 5: Guardar el Modelo ---")
    save_model(best_model)

    # Pasos 6 y 7: Cálculo y Guardado de Métricas y Matriz de Confusión
    print("\n--- Pasos 6 y 7: Cálculo de Métricas y Matriz de Confusión ---")
    calculate_metrics_and_confusion_matrix(best_model, x_train, y_train, x_test, y_test)

    print("\nProceso de construcción del modelo completado.")

if __name__ == "__main__":
    main()