from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def cargar_datos(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Carga los archivos de entrenamiento y prueba. Permite que uno sea None."""
    train = pd.read_csv(train_path) if train_path is not None else None
    test = pd.read_csv(test_path) if test_path is not None else None
    return train, test

def cargar_archivo(path: str) -> pd.DataFrame:
    """
    Carga un solo archivo CSV y lo retorna como DataFrame.
    """
    return pd.read_csv(path)


def igualar_tipos(
    train: Optional[pd.DataFrame] = None,
    test: Optional[pd.DataFrame] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Corrige las diferencias de tipos entre train y test, permitiendo que uno sea None."""
    float_cols = [
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"
    ]
    if train is not None:
        for col in float_cols:
            if col in train.columns:
                train[col] = train[col].astype(float)
    if test is not None:
        for col in float_cols:
            if col in test.columns:
                test[col] = test[col].astype(float)
    return train, test


"""Creo que debemos podriamos optar por unir los csv [Queda en veremos]"""


def unir_datasets(
    train: Optional[pd.DataFrame] = None,
    test: Optional[pd.DataFrame] = None
) -> Tuple[Optional[pd.DataFrame], int]:
    """Concatena train y test para preprocesarlos de forma conjunta. Permite que uno sea None."""
    ntrain = 0
    if train is not None and test is not None:
        ntrain = train.shape[0]
        train = train.drop("SalePrice", axis=1, errors="ignore")
        full = pd.concat([train, test], axis=0, ignore_index=True)
        return full, ntrain
    elif train is not None:
        ntrain = train.shape[0]
        train = train.drop("SalePrice", axis=1, errors="ignore")
        return train.copy(), ntrain
    elif test is not None:
        return test.copy(), ntrain
    else:
        return None, ntrain


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
