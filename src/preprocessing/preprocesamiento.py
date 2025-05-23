from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def cargar_datos(
    train_path: str,
    test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los archivos de entrenamiento y prueba."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def igualar_tipos(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Corrige las diferencias de tipos entre train y test."""
    float_cols = [
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"
    ]
    for col in float_cols:
        if col in train.columns and col in test.columns:
            train[col] = train[col].astype(float)
            test[col] = test[col].astype(float)
    return train, test


"""Creo que debemos podriamos optar por unir los csv [Queda en veremos]"""


def unir_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    """Concatena train y test para preprocesarlos de forma conjunta."""
    ntrain = train.shape[0]
    train = train.drop("SalePrice", axis=1, errors="ignore")
    full = pd.concat([train, test], axis=0, ignore_index=True)
    return full, ntrain


def limpiar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de nulos."""
    # Numéricas con agrupamiento
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = (
            df.groupby("Neighborhood")["LotFrontage"]
            .transform(lambda x: x.fillna(x.median()))
        )
    # Categóricas que significan ausencia
    none_fill = [
        "Alley", "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
        "GarageFinish", "GarageQual", "GarageCond", "PoolQC",
        "Fence", "MiscFeature", "MasVnrType"
    ]
    for col in none_fill:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    # Numéricas a 0 donde None = No existe
    zero_fill = [
        "MasVnrArea", "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath"
    ]
    for col in zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    # Demás categóricas por moda
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    # Numéricas restantes por media
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
    return df


def codificar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding para variables categóricas."""
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def escalar_variables(
    df: pd.DataFrame,
    scaler: Optional[RobustScaler] = None
) -> Tuple[pd.DataFrame, RobustScaler]:
    """Escala variables numéricas con RobustScaler."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if scaler is None:
        scaler = RobustScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler


def separar_train_test(
    df: pd.DataFrame,
    ntrain: int,
    train_labels: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """Separa nuevamente los datasets después de procesar juntos."""
    train = df.iloc[:ntrain, :].copy()
    test = df.iloc[ntrain:, :].copy()
    if train_labels is not None:
        return train, test, train_labels
    else:
        return train, test, None
