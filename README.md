# Trabajo Final: Metodología para Data Science
Repositorio para el trabajo final del curso Metodología para Data Science

# Predicción de Precios de Viviendas en Boston

## 🌟 Objetivo del Proyecto

Desarrollar un sistema de predicción del precio de viviendas en Boston utilizando técnicas de ciencia de datos. El modelo debe ser capaz de aprender patrones a partir de un conjunto de datos estructurado que incluye variables como el número de habitaciones, la tasa de criminalidad, el acceso a servicios, entre otros. Este proyecto es parte del curso de Metodología para Data Science y busca aplicar todas las etapas de un flujo de trabajo real de ciencia de datos.

---

## 📊 Descripción del Problema

En el ámbito inmobiliario, estimar con precisión el valor de una propiedad es esencial para la toma de decisiones de compra, venta o inversión. Contar con una herramienta que automatice esta estimación permite mejorar la eficiencia y transparencia del mercado. El problema planteado es predecir el precio medio de una vivienda por zona en Boston usando un conjunto de datos disponible públicamente.

---

## 💡 Propuesta de Solución

- Realizar un análisis exploratorio detallado del conjunto de datos.
- Aplicar técnicas de preprocesamiento para preparar los datos para el modelado.
- Evaluar distintos modelos de regresión: Lineal, Random Forest, Gradient Boosting.
- Utilizar validación cruzada y ajuste de hiperparámetros.
- Documentar y automatizar las tareas usando `pipelines` y pruebas unitarias.
- Posibilidad de exponer el modelo mediante una API REST (FastAPI).

---

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
## 🤝 Flujo de Trabajo Aplicado

### 1. Ingesta y Análisis Exploratorio (EDA)
- Uso de `pandas`, `matplotlib`, `seaborn` para entender la distribución de las variables.
- Se identificaron outliers y correlaciones fuertes entre algunas variables y el precio.

### 2. Preprocesamiento
- Eliminación de datos faltantes.
- Normalización de variables continuas.
- Codificación de variables categóricas si es necesario.
- Transformaciones (logaritmos, escalado, etc.).

### 3. Modelado
- Se probaron los siguientes modelos:
  - Regresión lineal
  - Random Forest
  - Gradient Boosting
- Métricas utilizadas:
  - MAE (Error absoluto medio)
  - RMSE (Raíz del error cuadrático medio)
  - R² (Coeficiente de determinación)

### 4. Validación
- K-Fold Cross Validation
- Grillas de hiperparámetros con `GridSearchCV`

### 5. Automatización y Pruebas
- Se desarrollaron funciones reutilizables para el procesamiento de datos.
- Las pruebas unitarias se realizaron con `pytest` para funciones de transformación.

---

## 🔄 Control de Versiones (Git)

- Se utilizaron ramas por funcionalidad: `feature/eda`, `feature/preprocessing`, `feature/modeling`, `feature/testing`, etc.
- Commits frecuentes que documentan el progreso de cada etapa.
- Pull Requests (PRs) para revisiones de código entre integrantes.

Ejemplo:
```
[feature/rcs/test] test_preprocesamiento
```

---
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