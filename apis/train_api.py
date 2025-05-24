from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import uvicorn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.preprocesamiento import (
    cargar_datos, igualar_tipos, unir_datasets, limpiar_nulos,
    codificar_variables, escalar_variables, separar_train_test
)
from src.features.models import entrenar_modelo, guardar_modelo
import pandas as pd
import numpy as np


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

class TrainRequest(BaseModel):
    train_path: str = "../data/raw/train.csv"
    test_path: str = "../data/raw/test.csv"
    model_path: str = "../models/bayesian_model.pkl"

@app.post("/train")
def train_model(request: TrainRequest):
    try:
        # Preprocesamiento
        train, test = cargar_datos(request.train_path, request.test_path)
        train, test = igualar_tipos(train, test)
        full, ntrain = unir_datasets(train, test)
        full = limpiar_nulos(full)
        full = codificar_variables(full)
        full, scaler = escalar_variables(full)
        X_train, X_test, _ = separar_train_test(full, ntrain)
        y_train = np.log1p(pd.read_csv(request.train_path)["SalePrice"])

        # Entrenar y guardar modelo
        modelo = entrenar_modelo(X_train, y_train)
        guardar_modelo(modelo, request.model_path)

        # Realizar predicción sobre el set de test
        y_pred = modelo.predict(X_test)
        y_pred = np.expm1(y_pred)  # Invertir la transformación log1p

        cantidad_predicciones = len(y_pred)

        return {
            "message": "Modelo entrenado y guardado exitosamente.",
            "model_path": request.model_path,
            "cantidad_predicciones": cantidad_predicciones,
            "predicciones": y_pred.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    # Iniciar servidor
    uvicorn.run(app, host="0.0.0.0", port=8000)