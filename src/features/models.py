from sklearn.linear_model import BayesianRidge
import joblib
import numpy as np

def cargar_modelo(ruta_modelo="../models/bayesian_model.pkl"):
    """
    Carga un modelo entrenado desde un archivo .pkl
    """
    return joblib.load(ruta_modelo)

def guardar_modelo(modelo, ruta_modelo="../models/bayesian_model.pkl"):
    """
    Guarda el modelo entrenado en un archivo .pkl
    """
    joblib.dump(modelo, ruta_modelo)

def predecir_precio(modelo, X):
    """
    Realiza predicciones usando el modelo cargado.
    X debe estar preprocesado igual que en el entrenamiento.
    Devuelve los precios en escala original.
    """
    predicciones_log = modelo.predict(X)
    predicciones = np.expm1(predicciones_log)
    return predicciones

def entrenar_modelo(X_train, y_train):
    """
    Entrena un modelo BayesianRidge y lo retorna.
    """
    modelo = BayesianRidge()
    modelo.fit(X_train, y_train)
    return modelo



# Ejemplo de uso integrado
if __name__ == "__main__":
    # Aquí deberías importar tus funciones de preprocesamiento y entrenar el modelo
    # from src.preprocessing.preprocesamiento import cargar_datos, igualar_tipos, unir_datasets, limpiar_nulos, codificar_variables, escalar_variables, separar_train_test
    # train, test = cargar_datos("ruta_train.csv", "ruta_test.csv")
    # train, test = igualar_tipos(train, test)
    # full, ntrain = unir_datasets(train, test)
    # full = limpiar_nulos(full)
    # full = codificar_variables(full)
    # full, scaler = escalar_variables(full)
    # X_train, X_test, _ = separar_train_test(full, ntrain)
    # y_train = ... # tu vector objetivo, por ejemplo: np.log1p(train_original["SalePrice"])
    # modelo = entrenar_modelo(X_train, y_train)
    # guardar_modelo(modelo)
    pass