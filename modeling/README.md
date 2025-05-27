# Ransomware Forecaster - Arquitectura Modular

Este directorio contiene una implementación modular del sistema de predicción de ataques ransomware, diseñada para mejorar la mantenibilidad, testabilidad y escalabilidad del proyecto.

## Estructura de directorios

```
modeling/
├── data/                     # Manejo de datos
│   ├── loader.py             # Carga y validación
│   └── preprocessor.py       # Preprocesamiento
├── features/                 # Ingeniería de características
│   ├── outliers.py           # Detección y manejo de outliers
│   └── regressors.py         # Generación y selección de regresores
├── models/                   # Modelos predictivos
│   ├── prophet_model.py      # Wrapper para Prophet
│   ├── evaluator.py          # Evaluación de modelos
│   └── calibrator.py         # Calibración de intervalos
├── utils/                    # Utilidades
│   └── visualization.py      # Visualizaciones con Plotly
├── ransomware_forecaster.py  # Implementación original (mantenida para compatibilidad)
├── ransomware_forecaster_modular.py  # Implementación modular con compatibilidad
└── integration.py            # Facilitador para la integración con Streamlit
```

## Scripts auxiliares
```
scripts/
└── train_model.py            # Script para entrenar desde línea de comandos
```

## Características principales

- **Diseño modular**: Cada componente tiene una responsabilidad única y clara
- **Robustez mejorada**: Validación de datos, manejo de errores y logging
- **Rendimiento optimizado**: Parámetros ajustados para producción
- **Mejor cobertura de intervalos**: Calibración agresiva para garantizar ~90%
- **Compatibilidad**: Mantiene interfaz compatible con el código existente

## Cómo utilizar la versión modular

### Opción 1: Usando la clase integradora

La clase `RansomwareForecasterModular` proporciona la misma interfaz que la implementación original:

```python
from modeling.ransomware_forecaster_modular import RansomwareForecasterModular

# Crear instancia
forecaster = RansomwareForecasterModular()

# Cargar datos
df = forecaster.load_data(ransomware_file='modeling/victimas_ransomware_mod.json')

# Preparar datos
forecaster.prepare_data(outlier_method='iqr', use_log_transform=True)

# Añadir regresores
forecaster.add_regressors(enable_regressors=True)

# Optimizar y entrenar
forecaster.optimize_model_step3()
forecaster.train_model(
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.2,
    seasonality_prior_scale=10.0,
    interval_width=0.8
)

# Generar predicción
forecast = forecaster.make_forecast(periods=30)

# Calibrar intervalos
coverage = forecaster.calibrate_intervals_conformal()

# Visualizar resultados
fig = forecaster.plot_forecast()
component_figs = forecaster.plot_components()
```

### Opción 2: Usando los módulos individuales

También puedes usar directamente los módulos individuales para mayor flexibilidad:

```python
from modeling.data.loader import DataLoader
from modeling.data.preprocessor import DataPreprocessor
from modeling.features.outliers import OutlierDetector
from modeling.features.regressors import RegressorGenerator
from modeling.models.prophet_model import RansomwareProphetModel
from modeling.models.calibrator import IntervalCalibrator
from modeling.utils.visualization import RansomwarePlotter

# Cargar y preparar datos
loader = DataLoader()
preprocessor = DataPreprocessor()
outlier_detector = OutlierDetector()

# Cargar datos
df_raw = loader.load_ransomware_data('modeling/victimas_ransomware_mod.json')

# Preparar datos para Prophet
df_prophet = preprocessor.prepare_prophet_data(df_raw)

# Detectar y manejar outliers
outliers = outlier_detector.detect_outliers(df_prophet)
df_prophet = outlier_detector.handle_outliers(df_prophet, outliers)

# Transformación logarítmica
df_prophet = preprocessor.apply_log_transform(df_prophet)

# Entrenar modelo
model = RansomwareProphetModel()
model.fit(df_prophet)

# Generar predicción
forecast = model.predict(periods=30)

# Calibrar intervalos
calibrator = IntervalCalibrator()
forecast = calibrator.calibrate_conformal(df_prophet, forecast)

# Visualizar resultados
plotter = RansomwarePlotter()
fig = plotter.plot_forecast(df_prophet, forecast)
```

### Opción 3: Usando el módulo de integración

El módulo `integration.py` proporciona funciones para facilitar la transición
a la versión modular sin romper la interfaz de Streamlit existente:

```python
import streamlit as st
from modeling.integration import (
    initialize_streamlit_state,
    load_data_wrapper,
    train_model_wrapper,
    generate_forecast_wrapper,
    evaluate_model_wrapper,
    plot_forecast_wrapper,
    plot_components_wrapper
)

# Inicializar estado de Streamlit
initialize_streamlit_state()

# Cargar datos
df = load_data_wrapper()

if st.session_state.data_loaded:
    # Entrenar modelo con parámetros
    train_model_wrapper(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.2,
        seasonality_prior_scale=10.0,
        interval_width=0.8
    )
    
    # Generar predicción
    forecast = generate_forecast_wrapper(periods=30)
    
    # Visualizar resultados
    fig = plot_forecast_wrapper()
    st.plotly_chart(fig)
    
    # Mostrar componentes
    component_figs = plot_components_wrapper()
    for component, fig in component_figs.items():
        st.subheader(f"Componente: {component}")
        st.plotly_chart(fig)
```

## Entrenamiento desde línea de comandos

El script `scripts/train_model.py` permite entrenar y evaluar el modelo desde la línea de comandos:

```
python scripts/train_model.py --data modeling/victimas_ransomware_mod.json --periods 30 --seasonality multiplicative --save-plots
```

## Mejores prácticas

1. **Validación de datos**: Valida siempre la estructura de los archivos de entrada
2. **Manejo de outliers**: Utiliza `iqr` para detectar outliers y `winsorize` para tratarlos
3. **Modo de estacionalidad**: Usa `multiplicative` para este tipo de datos
4. **Calibración de intervalos**: Calibra siempre los intervalos después de la predicción
5. **Logging**: Revisa los logs para obtener información detallada del proceso

## Migración gradual

Se recomienda migrar gradualmente a la nueva arquitectura siguiendo estos pasos:

1. Primero, usa el módulo de `integration.py` en tu aplicación Streamlit
2. Luego, cambia a `RansomwareForecasterModular` directamente
3. Finalmente, si necesitas personalización avanzada, usa los módulos individuales
