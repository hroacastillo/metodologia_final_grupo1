import pytest
import numpy as np
from sklearn.linear_model import BayesianRidge
from src.features.models import entrenar_modelo, predecir_precio, guardar_modelo, cargar_modelo

@pytest.fixture
def datos_de_prueba():
    """
    Fixture que genera datos sintéticos para entrenamiento y predicción.
    """
    np.random.seed(0)
    X = np.random.rand(50, 3)
    y = np.log1p(np.dot(X, [30000, 20000, 10000]) + 150000)
    return X, y

def test_entrenar_modelo(datos_de_prueba):
    X, y = datos_de_prueba
    modelo = entrenar_modelo(X, y)
    assert isinstance(modelo, BayesianRidge)
    assert hasattr(modelo, 'coef_')

def test_predecir_precio(datos_de_prueba):
    X, y = datos_de_prueba
    modelo = entrenar_modelo(X, y)
    predicciones = predecir_precio(modelo, X)
    assert predicciones.shape == (50,)
    assert (predicciones > 0).all()

def test_guardar_y_cargar_modelo(tmp_path, datos_de_prueba):
    X, y = datos_de_prueba
    modelo = entrenar_modelo(X, y)
    ruta = tmp_path / "modelo.pkl"
    guardar_modelo(modelo, ruta)
    modelo_cargado = cargar_modelo(ruta)
    assert isinstance(modelo_cargado, BayesianRidge)
    np.testing.assert_array_almost_equal(modelo.coef_, modelo_cargado.coef_)