import pandas as pd
import numpy as np
import pytest
from src.preprocessing import preprocesamiento

def test_cargar_datos(tmp_path):
    # Crear archivos CSV temporales
    train = pd.DataFrame({'A': [1, 2], 'SalePrice': [100, 200]})
    test = pd.DataFrame({'A': [3, 4]})
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    t, te = preprocesamiento.cargar_datos(str(train_path), str(test_path))
    assert t.shape == (2, 2)
    assert te.shape == (2, 1)

def test_igualar_tipos():
    train = pd.DataFrame({'BsmtFinSF1': [1], 'GarageArea': [2]})
    test = pd.DataFrame({'BsmtFinSF1': [3], 'GarageArea': [4]})
    t, te = preprocesamiento.igualar_tipos(train, test)
    assert t['BsmtFinSF1'].dtype == float
    assert te['GarageArea'].dtype == float

def test_unir_datasets():
    train = pd.DataFrame({'A': [1], 'SalePrice': [100]})
    test = pd.DataFrame({'A': [2]})
    full, ntrain = preprocesamiento.unir_datasets(train, test)
    assert full.shape[0] == 2
    assert ntrain == 1

def test_limpiar_nulos():
    df = pd.DataFrame({
        'LotFrontage': [np.nan, 80],
        'Neighborhood': ['A', 'A'],
        'Alley': [np.nan, np.nan],
        'MasVnrArea': [np.nan, 1],
        'GarageArea': [np.nan, 2],
        'SomeCat': [np.nan, 'a'],
        'SomeNum': [np.nan, 1]
    })
    df_clean = preprocesamiento.limpiar_nulos(df)
    assert df_clean.isnull().sum().sum() == 0
    assert (df_clean['Alley'] == 'None').all()
    assert (df_clean['MasVnrArea'] == 0).iloc[0]

def test_codificar_variables():
    df = pd.DataFrame({'A': ['x', 'y'], 'B': [1, 2]})
    df_enc = preprocesamiento.codificar_variables(df)
    assert 'A_y' in df_enc.columns or 'A_x' in df_enc.columns

def test_escalar_variables():
    df = pd.DataFrame({'A': [1, 2], 'B': [10, 20]})
    df_scaled, scaler = preprocesamiento.escalar_variables(df)
    assert np.allclose(df_scaled.mean(), 0, atol=1)
    # Test transform with existing scaler
    df2 = pd.DataFrame({'A': [3, 4], 'B': [30, 40]})
    df2_scaled, _ = preprocesamiento.escalar_variables(df2, scaler)
    assert df2_scaled.shape == df2.shape

def test_separar_train_test():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    ntrain = 2
    train, test, _ = preprocesamiento.separar_train_test(df, ntrain)
    assert train.shape[0] == 2
    assert test.shape[0] == 1