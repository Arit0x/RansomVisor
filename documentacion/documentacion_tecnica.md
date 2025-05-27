# Documentación Técnica - RansomVisor

Esta documentación está dirigida a desarrolladores que deseen entender, mantener o extender la aplicación RansomVisor.

## Índice

1. [Arquitectura del Sistema](#arquitectura-del-sistema)
2. [Componentes Principales](#componentes-principales)
3. [Flujo de Datos](#flujo-de-datos)
4. [Stack Tecnológico](#stack-tecnológico)
5. [Estructura de Código](#estructura-de-código)
6. [Modelo de Predicción](#modelo-de-predicción)
7. [Extensión del Sistema](#extensión-del-sistema)
8. [Consideraciones de Rendimiento](#consideraciones-de-rendimiento)

## Arquitectura del Sistema

RansomVisor utiliza una arquitectura basada en componentes con Streamlit como framework principal. El sistema se organiza en torno a estos conceptos fundamentales:

- **Aplicación principal** (`app.py`): Punto de entrada que configura la navegación y define la estructura general.
- **Páginas especializadas** (`pages/`): Componentes modulares para cada funcionalidad específica.
- **Lógica de negocio** (`modeling/`, `eda.py`): Implementación de análisis y modelado predictivo.
- **Utilidades compartidas** (`utils.py`, `sidebar.py`): Funciones y componentes reutilizables.

El diseño sigue un patrón similar a MVC (Modelo-Vista-Controlador):
- **Modelo**: Implementado en `utils.py` y el ecosistema modular en `modeling/` para acceso y manipulación de datos.
- **Vista**: Páginas individuales en el directorio `pages/` que manejan la presentación.
- **Controlador**: Lógica de coordinación en `app.py` y controladores específicos de página.

## Componentes Principales

### 1. Sistema de Navegación Multipágina

Implementado en `app.py` utilizando el sistema de navegación de Streamlit. Cada página es un módulo Python independiente con una función principal que renderiza la interfaz.

```python
# app.py (fragmento)
from pages.home import home_app
from pages.modelado_modular import modelado_modular_app
# ...

home_page = st.Page(home_app, title="Visión General", icon="📊")
modelado_page = st.Page(modelado_modular_app, title="Modelado", icon="🤖")
# ...

pg = st.navigation([home_page, modelado_page, ...])
pg.run()
```

### 2. Sistema de Gestión de Datos

Implementado en `utils.py`, maneja la carga, transformación y almacenamiento de datos.

Funciones clave:
- `carga_datos_victimas_por_ano()`: Carga datos históricos de ataques.
- `actualizar_snapshot()`: Actualiza datos desde fuentes externas.
- `load_snapshot()`: Carga snapshots guardados localmente.

### 3. Motor de Predicción Modular

Implementado como un ecosistema de componentes en `modeling/`, con una arquitectura modular que facilita el mantenimiento y extensión.

#### 3.1 Arquitectura Modular

El sistema de predicción se ha refactorizado a una arquitectura modular con los siguientes componentes:

```
modeling/
├── data/                    # Componentes de datos
│   ├── loader.py            # Carga de datos de diferentes fuentes
│   └── preprocessor.py      # Preprocesamiento de datos
├── features/                # Ingeniería de características
│   ├── outliers.py          # Detección y manejo de outliers
│   └── regressors.py        # Generación de regresores externos
├── models/                  # Componentes del modelo
│   ├── prophet_model.py     # Implementación del modelo Prophet
│   ├── evaluator.py         # Evaluación de rendimiento
│   └── calibrator.py        # Calibración de intervalos
├── utils/                   # Utilidades
│   └── visualization.py     # Visualización de resultados
├── ransomware_forecaster.py       # Implementación original (legado)
├── ransomware_forecaster_modular.py # Nueva implementación modular
└── integration.py           # Capa de integración con la UI
```

#### 3.2 Capa de Integración

El módulo `integration.py` actúa como puente entre la implementación modular y la interfaz de usuario, proporcionando:

- Funciones wrapper que gestionan el estado de Streamlit
- Compatibilidad con versiones anteriores
- Manejo de caché y optimización de rendimiento

### 4. Sistema de Visualización

Utiliza principalmente Plotly para gráficos interactivos, con algunos componentes en Matplotlib. Las visualizaciones están distribuidas entre los diferentes módulos de página y centralizado en `utils/visualization.py` para el modelo predictivo.

## Flujo de Datos

1. **Ingesta de datos**:
   - Carga inicial desde archivos JSON/CSV locales en `modeling/victimas_ransomware_mod.json` y `modeling/cve_diarias_regresor_prophet.csv`
   - Actualización opcional desde fuentes externas
   - Preprocesamiento y transformación a través de `DataPreprocessor`

2. **Almacenamiento**:
   - Datos originales guardados como snapshots en formato JSON
   - Estado de la aplicación gestionado a través de `st.session_state`
   - Resultados de modelos almacenados en la sesión para persistencia

3. **Procesamiento**:
   - Análisis exploratorio en `eda.py`
   - Detección de outliers en `features/outliers.py`
   - Generación de regresores en `features/regressors.py`
   - Modelado predictivo en `models/prophet_model.py`

4. **Presentación**:
   - Renderizado en páginas individuales
   - Visualizaciones interactivas con Plotly
   - Componentes de UI con Streamlit
   - Guía de usuario interactiva en `user_guide.py`

## Stack Tecnológico

- **Framework Web**: Streamlit 1.24.0+
- **Análisis de Datos**: Pandas, NumPy
- **Visualización**: Plotly, Matplotlib, Seaborn
- **Modelado Predictivo**: Prophet, Scikit-learn
- **Otras Bibliotecas**:
  - Holidays: Para manejo de días festivos
  - Datetime: Manipulación de fechas y tiempos
  - Statsmodels: Análisis estadístico avanzado
  - OptBayesianOpt: Optimización bayesiana de hiperparámetros

## Estructura de Código

```
web_final/
├── app.py                   # Punto de entrada principal
├── utils.py                 # Funciones de utilidad general
├── eda.py                   # Análisis exploratorio de datos
├── sidebar.py               # Controles de la barra lateral
├── alerts.py                # Sistema de alertas
├── assets/                  # Recursos estáticos
│   ├── css/                 # Hojas de estilo
│   └── img/                 # Imágenes y gráficos
├── data/                    # Datos y snapshots
├── documentacion/           # Documentación del proyecto
├── modeling/                # Sistema de modelado predictivo (detallado anteriormente)
└── pages/                   # Páginas individuales
    ├── __init__.py
    ├── home.py              # Página de inicio/visión general
    ├── tendencias.py        # Análisis de tendencias
    ├── geografia.py         # Distribución geográfica
    ├── alertas.py           # Sistema de alertas
    ├── modelado_modular.py  # Nueva interfaz modular de predicción
    └── sectores.py          # Análisis por sectores industriales
```

## Modelo de Predicción

### Componentes Clave

#### 1. RansomwareForecasterModular

Clase principal que orquesta todos los componentes del sistema predictivo. Implementa una fachada unificada para las operaciones de predicción.

```python
# Ejemplo de uso de RansomwareForecasterModular
forecaster = RansomwareForecasterModular()
df = forecaster.load_data('path/to/data.json', 'path/to/cve.csv')
forecaster.preprocess_data(outlier_method='isolation_forest', outlier_strategy='winsorize')
forecaster.train_model(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)
forecast = forecaster.make_forecast(periods=30)
metrics = forecaster.evaluate_model(cv_periods=30)
```

#### 2. DataLoader y DataPreprocessor

Componentes responsables de cargar y preparar datos para el modelado.

- **DataLoader**: Carga datos de ransomware y CVEs desde diferentes fuentes.
- **DataPreprocessor**: Aplica transformaciones, maneja outliers y prepara datos para Prophet.

#### 3. RansomwareProphetModel

Encapsula la lógica del modelo Prophet para la predicción de ataques ransomware.

Características principales:
- Configuración de estacionalidad (diaria, semanal, anual)
- Manejo de regresores externos (CVEs, eventos)
- Gestión de puntos de cambio (changepoints)
- Optimización de hiperparámetros

#### 4. IntervalCalibrator

Mejora la precisión de los intervalos de predicción mediante técnicas de calibración.

```python
# Ejemplo de calibración de intervalos
calibrator = IntervalCalibrator()
forecast_calibrated = calibrator.calibrate_conformal(
    train_df=df_train,
    forecast_df=forecast,
    target_coverage=0.9
)
```

#### 5. Evaluación y Backtesting

El sistema incluye herramientas robustas para evaluar el rendimiento del modelo:

- **Validación cruzada temporal**: Evalúa el modelo en diferentes períodos históricos.
- **Backtesting configurable**: Simula cómo habría funcionado el modelo en el pasado.
- **Métricas de error**: RMSE, MAE, SMAPE y cobertura de intervalos de predicción.

## Extensión del Sistema

### Añadir Nuevos Componentes

Para extender el sistema con nuevas funcionalidades, seguir estos pasos:

1. Identificar el componente adecuado para la extensión
2. Implementar la nueva funcionalidad como una clase o función en el módulo correspondiente
3. Integrar con el resto del sistema mediante la clase `RansomwareForecasterModular` o `integration.py`

### Ejemplo: Añadir un Nuevo Detector de Outliers

```python
# En features/outliers.py
class OutlierDetector:
    # Código existente...
    
    def detect_isolation_forest(self, data, contamination=0.05):
        """
        Detecta outliers usando Isolation Forest
        
        Args:
            data: DataFrame con datos
            contamination: Proporción esperada de outliers
            
        Returns:
            Índices de outliers detectados
        """
        from sklearn.ensemble import IsolationForest
        
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(data.values.reshape(-1, 1))
        return data.index[predictions == -1].tolist()
```

Luego integrar en `RansomwareForecasterModular`:

```python
# En ransomware_forecaster_modular.py
def preprocess_data(self, outlier_method='isolation_forest', ...):
    # Código existente...
    if outlier_method == 'isolation_forest':
        outliers = self.outlier_detector.detect_isolation_forest(
            self.df_prophet['y'], contamination=0.05
        )
    # Resto del código...
```

## Consideraciones de Rendimiento

### Optimizaciones Implementadas

1. **Caché de Streamlit**: Se utilizan los decoradores `@st.cache_data` y `@st.cache_resource` para optimizar operaciones costosas:
   ```python
   @st.cache_data
   def load_data_cached(_self, ransomware_file, cve_file):
       # Código para cargar datos...
   ```

2. **Validación Cruzada Eficiente**: La implementación de validación cruzada se ha optimizado para evitar bloqueos y uso excesivo de memoria:
   ```python
   # Desactivar paralelismo para mayor estabilidad
   df_cv = cross_validation(model=model, initial=initial, period=period, 
                           horizon=horizon, parallel=None)
   ```

3. **Preparación de Datos Selectiva**: Sólo se realizan transformaciones de datos cuando es necesario:
   ```python
   if self.df_prophet is None or force_reload:
       # Realizar preparación de datos...
   ```

### Recomendaciones

- **Tamaño de Datos**: Para conjuntos de datos grandes (>10,000 filas), considerar submuestreo o agregación.
- **Paralelismo**: Usar con precaución; en sistemas con recursos limitados, desactivar paralelismo (`parallel=None`).
- **Validación Cruzada**: Limitar el horizonte de validación a períodos razonables (14-30 días).
- **Regresores**: Seleccionar regresores relevantes mediante la función de selección automática para evitar sobreajuste.
