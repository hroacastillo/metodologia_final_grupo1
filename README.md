# Trabajo Final: Metodología para Data Science
Repositorio para el trabajo final del curso Metodología para Data Science

## Estructura del Proyecto
```
├── data/                  # Datos originales y procesados
│   ├── raw/              # Datos originales
│   └── processed/        # Datos procesados
├── notebooks/            # Jupyter notebooks con el análisis
├── src/                  # Código fuente
│   ├── preprocessing/    # Scripts de limpieza y procesamiento
│   ├── features/         # Feature engineering
│   └── test/             # Test
├── reports/              # Reportes y visualizaciones
├── .github/              # Configuración de GitHub
│   └── workflows/        # GitHub Actions workflows
├── .pre-commit-config.yaml # Configuración de pre-commit hooks
├── tests/                # Tests unitarios y de integración
└── README.md             # Este archivo
```
# 1.Importación de Librerías #


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.pipeline import make_pipeline

```

# 2.Carga de datasets #


```python
#Carga de datasets train y test
train_df = pd.read_csv("../data/raw/train.csv")
test_df = pd.read_csv("../data/raw/test.csv")

#Datasets train y test combinados
combined_df = pd.concat([train_df,test_df], axis=0)
```

# 3.Análisis exploratorio de datos #

## 3.1Dataset train ##


```python
#Listar las 5 primeras filas del dataset train
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
#Forma del dataset train
train_df.shape
```




    (1460, 81)




```python
#Información del dataset train
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     588 non-null    object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB



```python
#Tipo de datos del dataset train
train_dtype = train_df.dtypes
train_dtype.value_counts()
```




    object     43
    int64      35
    float64     3
    Name: count, dtype: int64




```python
#Valores nulos del dataset train
train_df.isnull().sum().sort_values(ascending= False).head(20)
```




    PoolQC          1453
    MiscFeature     1406
    Alley           1369
    Fence           1179
    MasVnrType       872
    FireplaceQu      690
    LotFrontage      259
    GarageQual        81
    GarageFinish      81
    GarageType        81
    GarageYrBlt       81
    GarageCond        81
    BsmtFinType2      38
    BsmtExposure      38
    BsmtCond          37
    BsmtQual          37
    BsmtFinType1      37
    MasVnrArea         8
    Electrical         1
    Condition2         0
    dtype: int64



## 3.2 Dataset test ##



```python
#Listar las 5 primeras filas del dataset test
test_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>120</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>60</td>
      <td>RL</td>
      <td>78.0</td>
      <td>9978</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>5005</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>144</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
#Forma del dataset test
test_df.shape

```




    (1459, 80)




```python
#Información del dataset test
test_df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 80 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1459 non-null   int64  
     1   MSSubClass     1459 non-null   int64  
     2   MSZoning       1455 non-null   object 
     3   LotFrontage    1232 non-null   float64
     4   LotArea        1459 non-null   int64  
     5   Street         1459 non-null   object 
     6   Alley          107 non-null    object 
     7   LotShape       1459 non-null   object 
     8   LandContour    1459 non-null   object 
     9   Utilities      1457 non-null   object 
     10  LotConfig      1459 non-null   object 
     11  LandSlope      1459 non-null   object 
     12  Neighborhood   1459 non-null   object 
     13  Condition1     1459 non-null   object 
     14  Condition2     1459 non-null   object 
     15  BldgType       1459 non-null   object 
     16  HouseStyle     1459 non-null   object 
     17  OverallQual    1459 non-null   int64  
     18  OverallCond    1459 non-null   int64  
     19  YearBuilt      1459 non-null   int64  
     20  YearRemodAdd   1459 non-null   int64  
     21  RoofStyle      1459 non-null   object 
     22  RoofMatl       1459 non-null   object 
     23  Exterior1st    1458 non-null   object 
     24  Exterior2nd    1458 non-null   object 
     25  MasVnrType     565 non-null    object 
     26  MasVnrArea     1444 non-null   float64
     27  ExterQual      1459 non-null   object 
     28  ExterCond      1459 non-null   object 
     29  Foundation     1459 non-null   object 
     30  BsmtQual       1415 non-null   object 
     31  BsmtCond       1414 non-null   object 
     32  BsmtExposure   1415 non-null   object 
     33  BsmtFinType1   1417 non-null   object 
     34  BsmtFinSF1     1458 non-null   float64
     35  BsmtFinType2   1417 non-null   object 
     36  BsmtFinSF2     1458 non-null   float64
     37  BsmtUnfSF      1458 non-null   float64
     38  TotalBsmtSF    1458 non-null   float64
     39  Heating        1459 non-null   object 
     40  HeatingQC      1459 non-null   object 
     41  CentralAir     1459 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1459 non-null   int64  
     44  2ndFlrSF       1459 non-null   int64  
     45  LowQualFinSF   1459 non-null   int64  
     46  GrLivArea      1459 non-null   int64  
     47  BsmtFullBath   1457 non-null   float64
     48  BsmtHalfBath   1457 non-null   float64
     49  FullBath       1459 non-null   int64  
     50  HalfBath       1459 non-null   int64  
     51  BedroomAbvGr   1459 non-null   int64  
     52  KitchenAbvGr   1459 non-null   int64  
     53  KitchenQual    1458 non-null   object 
     54  TotRmsAbvGrd   1459 non-null   int64  
     55  Functional     1457 non-null   object 
     56  Fireplaces     1459 non-null   int64  
     57  FireplaceQu    729 non-null    object 
     58  GarageType     1383 non-null   object 
     59  GarageYrBlt    1381 non-null   float64
     60  GarageFinish   1381 non-null   object 
     61  GarageCars     1458 non-null   float64
     62  GarageArea     1458 non-null   float64
     63  GarageQual     1381 non-null   object 
     64  GarageCond     1381 non-null   object 
     65  PavedDrive     1459 non-null   object 
     66  WoodDeckSF     1459 non-null   int64  
     67  OpenPorchSF    1459 non-null   int64  
     68  EnclosedPorch  1459 non-null   int64  
     69  3SsnPorch      1459 non-null   int64  
     70  ScreenPorch    1459 non-null   int64  
     71  PoolArea       1459 non-null   int64  
     72  PoolQC         3 non-null      object 
     73  Fence          290 non-null    object 
     74  MiscFeature    51 non-null     object 
     75  MiscVal        1459 non-null   int64  
     76  MoSold         1459 non-null   int64  
     77  YrSold         1459 non-null   int64  
     78  SaleType       1458 non-null   object 
     79  SaleCondition  1459 non-null   object 
    dtypes: float64(11), int64(26), object(43)
    memory usage: 912.0+ KB



```python
#Tipo de datos del dataset test
test_dtype = test_df.dtypes
test_dtype.value_counts()

```




    object     43
    int64      26
    float64    11
    Name: count, dtype: int64




```python
#Valores nulos del dataset test
test_df.isnull().sum().sort_values(ascending= False).head(20)
```




    PoolQC          1456
    MiscFeature     1408
    Alley           1352
    Fence           1169
    MasVnrType       894
    FireplaceQu      730
    LotFrontage      227
    GarageYrBlt       78
    GarageCond        78
    GarageFinish      78
    GarageQual        78
    GarageType        76
    BsmtCond          45
    BsmtQual          44
    BsmtExposure      44
    BsmtFinType1      42
    BsmtFinType2      42
    MasVnrArea        15
    MSZoning           4
    BsmtHalfBath       2
    dtype: int64



## 3.3 Comparación de los datasets Train y Test ##

3.3.1 Comparación de tipos de datos 


```python
#Como la columna "SalePrice" no está disponible en el dataset test, lo eliminaremos.
trn_dtype = train_dtype.drop('SalePrice')
trn_dtype.compare(test_dtype)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>self</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BsmtFinSF1</th>
      <td>int64</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>int64</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>int64</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>int64</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>int64</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>int64</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>int64</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>int64</td>
      <td>float64</td>
    </tr>
  </tbody>
</table>
</div>



3.3.2 Comparación de valores nulos de los datasets


```python
null_train = train_df.isnull().sum()
null_test = test_df.isnull().sum()
null_train = null_train.drop('SalePrice')
null_comp_df = null_train.compare(null_test).sort_values(['self'],ascending = [False])
null_comp_df  
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>self</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1453.0</td>
      <td>1456.0</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1406.0</td>
      <td>1408.0</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1369.0</td>
      <td>1352.0</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1179.0</td>
      <td>1169.0</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>872.0</td>
      <td>894.0</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>690.0</td>
      <td>730.0</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>259.0</td>
      <td>227.0</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>81.0</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>81.0</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>81.0</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>81.0</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>81.0</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>38.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>38.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>37.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>37.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>37.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>8.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>MSZoning</th>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Exterior2nd</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Exterior1st</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>KitchenQual</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Functional</th>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>SaleType</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



3.3.3 Comparación de distribución



```python
numerical_features = [col for col in train_df.columns if train_df[col].dtypes != 'O']
discrete_features = [col for col in numerical_features if len(train_df[col].unique()) < 25 and col not in ['Id']]
continuous_features = [feature for feature in numerical_features if feature not in discrete_features+['Id']]
categorical_features = [col for col in train_df.columns if train_df[col].dtype == 'O']

print("Total Number of Numerical Columns : ",len(numerical_features))
print("Number of discrete features : ",len(discrete_features))
print("No of continuous features are : ", len(continuous_features))
print("Number of discrete features : ",len(categorical_features))
```

    Total Number of Numerical Columns :  38
    Number of discrete features :  18
    No of continuous features are :  19
    Number of discrete features :  43


## 3.4 Encontrar el valor adecuado para los valores faltantes - Numérico ##

3.4.1 Valor medio de relleno


```python
#Verificar la distribución normal de las columnas que tienen valores nulos al completarlas con el valor medio
null_features_numerical = [col for col in combined_df.columns if combined_df[col].isnull().sum() > 0 and col not in categorical_features]
plt.figure(figsize=(30,20))
sns.set()

warnings.simplefilter("ignore")
for i,var in enumerate(null_features_numerical):
  plt.subplot(4,3,i+1)
  sns.distplot(combined_df[var],bins=20,kde_kws={'linewidth':3,'color':'red'},label="original")
  sns.distplot(combined_df[var],bins=20,kde_kws={'linewidth':2,'color':'yellow'},label="mean")
```


    
![png](output_27_0.png)
    


3.4.2 Rellenar valor medio


```python
plt.figure(figsize=(30,20))
sns.set()
warnings.simplefilter("ignore")
for i,var in enumerate(null_features_numerical):
  plt.subplot(4,3,i+1)
  sns.distplot(combined_df[var],bins=20,kde_kws={'linewidth':3,'color':'red'},label="original")
  sns.distplot(combined_df[var],bins=20,kde_kws={'linewidth':2,'color':'yellow'},label="median")
```


    
![png](output_29_0.png)
    


## 3.5 Análisis de variables temporales ##


```python
#Variables que contienen información del año
year_feature = [col for col in combined_df.columns if 'Yr' in col or 'Year' in col]
year_feature
```




    ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']




```python
#Verificar si existe una relación entre los campos "Year Sold" y "Sales price"
combined_df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('House Price')
plt.title('House price vs YearSold')
```




    Text(0.5, 1.0, 'House price vs YearSold')




    
![png](output_32_1.png)
    



```python
#Aquí veremos cómo las variables temporales (Year features) afectan al precio de la vivienda.
for fet in year_feature:
  if fet != 'YrSold':
    hs = combined_df.copy()
    hs[fet] = hs['YrSold'] - hs[fet]
    plt.scatter(hs[fet],hs['SalePrice'])
    plt.xlabel(fet)
    plt.ylabel('SalePrice')
    plt.show()
```


    
![png](output_33_0.png)
    



    
![png](output_33_1.png)
    



    
![png](output_33_2.png)
    


## 3.6 Correlación de datos ##


```python
# Seleccionar solo las columnas numéricas
numeric_df = train_df.select_dtypes(include=['number'])

# Calcular la matriz de correlación con método 'spearman'
training_corr = numeric_df.corr(method='spearman')

# Graficar el heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(training_corr, cmap="YlGnBu", linewidths=0.5, annot=True)
plt.title("Spearman Correlation Heatmap (Solo columnas numéricas)")
plt.show()
```


    
![png](output_35_0.png)
    


# 4. Feature Engineering #


## 4.1 Eliminación de columnas ##


```python

drop_columns = ["Id", "Alley", "Fence", "LotFrontage", "FireplaceQu", "PoolArea", "LowQualFinSF", "3SsnPorch", "MiscVal", 'RoofMatl','Street','Condition2','Utilities','Heating','Label']
# Eliminar Columnas
print("Number of columns before dropping : ",len(combined_df.columns))
print("Number of dropping columns : ",len(drop_columns))
combined_df.drop(columns=drop_columns, inplace=True, errors='ignore')
print("Number of columns after dropping : ",len(combined_df.columns))
```

    Number of columns before dropping :  81
    Number of dropping columns :  15
    Number of columns after dropping :  67


## 4.2 Cambio de variable temporal ##


```python
# Variables temporales (Date Time Variables)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:

    combined_df[feature]=combined_df['YrSold']-combined_df[feature]

combined_df[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GarageYrBlt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>5</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31</td>
      <td>31</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>6</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>91</td>
      <td>36</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>8</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



## 4.3 Completar valores faltantes ##


### 4.3.1 Característica numérica ###


```python
for col in null_features_numerical:
  if col not in drop_columns:    
    combined_df[col] = combined_df[col].fillna(0.0)
```

### 4.3.2 Característica categórica ###


```python
null_features_categorical = [col for col in combined_df.columns if combined_df[col].isnull().sum() > 0 and col in categorical_features]
cat_feature_mode = ["SaleType", "Exterior1st", "Exterior2nd", "KitchenQual", "Electrical", "Functional"]

for col in null_features_categorical:
  if col != 'MSZoning' and col not in cat_feature_mode:
    combined_df[col] = combined_df[col].fillna('NA')
  else:
    combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])
```

### 4.4 Convertir característica numérica a categórica ###


```python
convert_list = ['MSSubClass']
for col in convert_list:
  combined_df[col] = combined_df[col].astype('str')
```

## 4.5 Aplicar PowerTransformer a las columnas ##


```python
#Obtener las características excepto los tipo objetos
numeric_feats = combined_df.dtypes[combined_df.dtypes != 'object'].index

#Comprobar la desviación de todas las características numéricas
skewed_feats = combined_df[numeric_feats].apply(lambda x : skew(x.dropna())).sort_values(ascending = False)
print('\n Skew in numberical features: \n')
skewness_df = pd.DataFrame({'Skew' : skewed_feats})
print(skewness_df.head(10))
```

    
     Skew in numberical features: 
    
                        Skew
    LotArea        12.822431
    KitchenAbvGr    4.302254
    BsmtFinSF2      4.146143
    EnclosedPorch   4.003891
    ScreenPorch     3.946694
    BsmtHalfBath    3.931594
    MasVnrArea      2.613592
    OpenPorchSF     2.535114
    WoodDeckSF      1.842433
    1stFlrSF        1.469604



```python
#Aplicar PowerTransformer a las columnas
log_list = ['BsmtUnfSF', 'LotArea', '1stFlrSF', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']

for col in log_list:
    power = PowerTransformer(method='yeo-johnson', standardize=True)
    combined_df[[col]] = power.fit_transform(combined_df[[col]]) 

print('Number of skewed numerical features got transform : ', len(log_list))
```

    Number of skewed numerical features got transform :  6


## 4.6 Reagrupar características ##


```python
#Reagrupar características
regroup_dict = {
    'HeatingQC':['Fa','Po'],
    'GarageQual':['Fa','Po'],
    'GarageCond':['Fa','Po'],
}
 
for col, regroup_value in regroup_dict.items():
    mask = combined_df[col].isin(regroup_value)
    combined_df[col][mask] = 'Other'
```

## 4.7 Codificación de características categóricas ##

### 4.7.1 Get-Dummies ###


```python
# Generar columnas ficticias one-hot
combined_df = pd.get_dummies(combined_df).reset_index(drop=True)
```


```python
new_train_data = combined_df.iloc[:len(train_df), :]
new_test_data = combined_df.iloc[len(train_df):, :]
X_train = new_train_data.drop('SalePrice', axis=1)
y_train = np.log1p(new_train_data['SalePrice'].values.ravel())
X_test = new_test_data.drop('SalePrice', axis=1)
```


```python
pre_precessing_pipeline = make_pipeline(RobustScaler())

X_train = pre_precessing_pipeline.fit_transform(X_train)
X_test = pre_precessing_pipeline.transform(X_test)

print(X_train.shape)
print(X_test.shape)
```

    (1460, 269)
    (1459, 269)
# 📡 API

Este proyecto incluye una API desarrollada con FastAPI para exponer el modelo predictivo como servicio.

### Archivos relacionados

- `src/api/main.py`: punto de entrada principal de la API.

### Ejecución

Para ejecutarla localmente:

```bash
uvicorn src.api.main:app --reload
```

Por defecto estará disponible en [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Endpoints disponibles

- `GET /`: mensaje de bienvenida.
- `POST /predict`: permite enviar un JSON con una lista de características para obtener predicciones.

#### Ejemplo de petición:

```json
{
  "features": [[0.25, 0.4, 0.35]]
}
```

#### Ejemplo de respuesta:

```json
{
  "predicciones": [173452.85]
}
```

## 🧪 Pruebas

El proyecto incluye pruebas unitarias para validar el funcionamiento de los módulos.

### Estructura

Las pruebas están en `tests/unit/`:
- `test_models.py`: validación de entrenamiento, predicción y guardado del modelo.
- `test_preprocesamiento.py`: validación del preprocesamiento de datos.

### Cómo ejecutar

Desde la raíz del proyecto:

```bash
pytest
```

Si ocurre un error `ModuleNotFoundError: No module named 'src'`, usar:

```bash
PYTHONPATH=src pytest
```

O agregar un archivo `pytest.ini` con:

```ini
[pytest]
pythonpath = src
```