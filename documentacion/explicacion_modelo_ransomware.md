# Modelo de Predicción de Ataques Ransomware

## Arquitectura General

El modelo implementado utiliza **Prophet** (desarrollado por Facebook/Meta) como motor principal de predicción. Prophet es especialmente adecuado para series temporales con:

- Tendencias no lineales que cambian con el tiempo
- Múltiples estacionalidades (diaria, semanal, anual)
- Efectos de días festivos y eventos especiales
- Capacidad para incorporar regresores externos

## Arquitectura Modular

El sistema de predicción de ataques ransomware ha sido implementado con una arquitectura modular que divide las responsabilidades en componentes especializados:

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

### Componentes Principales

1. **RansomwareForecasterModular**: Clase principal que orquesta todos los componentes y proporciona una interfaz unificada.

2. **DataLoader**: Responsable de cargar datos de ransomware y vulnerabilidades desde diversas fuentes.

3. **DataPreprocessor**: Realiza la limpieza y preparación de datos para el modelado.

4. **OutlierDetector**: Identifica y maneja valores atípicos en los datos temporales.

5. **RegressorGenerator**: Genera variables externas para incorporar al modelo.

6. **RansomwareProphetModel**: Encapsula la lógica específica del modelo Prophet.

7. **IntervalCalibrator**: Calibra los intervalos de predicción para mayor precisión.

8. **ModelEvaluator**: Proporciona métricas de rendimiento y validación del modelo.

## Flujo de Trabajo del Modelo

### 1. Preparación de Datos

#### 1.1. Carga de Datos

El modelo ingiere dos fuentes principales de datos:

- **Datos de ataques ransomware**: Archivo JSON (`victimas_ransomware_mod.json`) que contiene registros históricos con información como:
  - Fecha del ataque
  - Grupo criminal responsable
  - Sector atacado
  - Región geográfica

- **Datos de vulnerabilidades (CVEs)**: Archivo CSV (`cve_diarias_regresor_prophet.csv`) con:
  - Conteo diario de vulnerabilidades publicadas
  - Métricas de severidad agregadas

#### 1.2. Procesamiento Inicial

1. **Conversión de fechas**: Transforma la columna de fecha a formato datetime para análisis temporal.
2. **Agrupación diaria**: Agrega los ataques por día para crear una serie temporal de frecuencia diaria.
3. **Serie temporal continua**: Genera un DataFrame con todos los días del rango, asignando cero a los días sin ataques registrados.

#### 1.3. Tratamiento de Valores Atípicos (Outliers)

El modelo detecta outliers utilizando varios métodos:

- **IQR (Rango Intercuartílico)**: Método robusto basado en cuartiles.
- **Z-Score**: Basado en desviaciones estándar desde la media.
- **MAD (Median Absolute Deviation)**: Más robusto que el Z-score tradicional en presencia de valores extremos.

Ofrece cuatro estrategias para manejar outliers:

- **Cap**: Limita los valores extremos al percentil 95, conservando la tendencia pero reduciendo la influencia de picos anómalos.
- **Remove**: Elimina outliers y los reemplaza mediante interpolación lineal.
- **Winsorize**: Reemplaza los valores extremos con el valor en el límite del percentil.
- **None**: Mantiene los outliers sin modificación (útil para análisis de eventos inusuales).

#### 1.4. Transformación Logarítmica

Opcionalmente aplica una transformación log(y+1) para:
- Estabilizar la varianza de los datos
- Manejar la heteroscedasticidad (variabilidad no constante)
- Mejorar el rendimiento en series con distribución sesgada

### 2. Construcción del Modelo

#### 2.1. Modelo Base Prophet

La clase `RansomwareProphetModel` configura un modelo Prophet con parámetros optimizados para la predicción de ataques ransomware:

```python
model = Prophet(
    changepoint_prior_scale=changepoint_prior_scale,  # Control de flexibilidad en tendencia
    seasonality_prior_scale=seasonality_prior_scale,  # Control de estacionalidad
    holidays_prior_scale=holidays_prior_scale,        # Impacto de eventos especiales
    seasonality_mode=seasonality_mode,                # 'additive' o 'multiplicative'
    interval_width=interval_width,                    # Ancho del intervalo de confianza
    daily_seasonality=daily_seasonality,              # Estacionalidad diaria
    weekly_seasonality=weekly_seasonality,            # Estacionalidad semanal
    yearly_seasonality=yearly_seasonality             # Estacionalidad anual
)
```

#### 2.2. Detección de Puntos de Cambio

Los puntos de cambio (changepoints) son momentos en los que la tendencia cambia significativamente. El modelo puede:

- **Detectar automáticamente**: Prophet identifica puntos potenciales a lo largo de la serie.
- **Especificar manualmente**: Incorporar conocimiento de eventos importantes que podrían afectar la tendencia.

```python
# Detección automática de puntos de cambio
if use_detected_changepoints:
    potential_changepoints = self._detect_changepoints(df_prophet['y'])
    model.changepoints = pd.DatetimeIndex(potential_changepoints)
```

#### 2.3. Configuración de Estacionalidad

El modelo considera múltiples niveles de estacionalidad:

- **Estacionalidad anual**: Captura patrones que se repiten cada año (ej. incrementos en temporada navideña)
- **Estacionalidad mensual**: Variaciones específicas de cada mes
- **Estacionalidad semanal**: Patrones que varían según el día de la semana
- **Estacionalidad diaria**: Variaciones a lo largo del día (cuando hay datos por hora)

#### 2.4. Incorporación de Regresores Externos

El componente `RegressorGenerator` prepara variables externas que pueden influir en los ataques:

- **Vulnerabilidades (CVEs)**: Añade el conteo diario de vulnerabilidades como indicador de oportunidades para atacantes.
- **Regresores de calendario**: Añade indicadores para días especiales (ej. fin de mes, festivos).
- **Variables de lag**: Incorpora valores pasados de la serie para capturar dependencias temporales.
- **Eventos especiales**: Días como Patch Tuesday de Microsoft que pueden influir en la actividad de ransomware.

```python
# Ejemplo de adición de regresores
if enable_regressors and self.cve_data is not None:
    model.add_regressor('cve_count', standardize=True)
    
# Añadir indicadores de calendario
model.add_regressor('is_weekend', standardize=False)
model.add_regressor('is_month_end', standardize=False)
```

### 3. Optimización del Modelo

#### 3.1. Optimización de Hiperparámetros

El modelo incluye capacidades de optimización automática de hiperparámetros mediante:

- **Grid Search**: Búsqueda exhaustiva en una cuadrícula de parámetros predefinidos.
- **Optimización Bayesiana**: Exploración eficiente del espacio de parámetros mediante BayesianOptimization.

Los principales parámetros optimizados son:
- `changepoint_prior_scale`: Controla la flexibilidad de la tendencia
- `seasonality_prior_scale`: Ajusta la fuerza de los componentes estacionales
- `holidays_prior_scale`: Determina el impacto de eventos especiales
- `seasonality_mode`: Selecciona entre modo aditivo o multiplicativo

#### 3.2. Selección de Regresores

La implementación incluye un proceso automático para seleccionar los regresores más relevantes:

```python
# Selección automática de regresores
if use_optimal_regressors:
    selected_regressors = self._select_optimal_regressors(
        df_prophet, candidate_regressors, 
        correlation_threshold=correlation_threshold,
        vif_threshold=vif_threshold
    )
```

Este proceso:
1. Calcula la correlación entre cada regresor y la variable objetivo
2. Elimina regresores con correlación por debajo del umbral
3. Comprueba multicolinealidad mediante el Factor de Inflación de Varianza (VIF)
4. Selecciona el subconjunto óptimo de regresores

#### 3.3. Calibración de Intervalos

El componente `IntervalCalibrator` permite ajustar los intervalos de predicción para mayor precisión:

```python
calibrator = IntervalCalibrator()
forecast_calibrated = calibrator.calibrate_conformal(
    train_df=df_train,
    forecast_df=forecast,
    target_coverage=0.9
)
```

La calibración utiliza técnicas conformales para garantizar que los intervalos de predicción tengan la cobertura adecuada.

### 4. Evaluación del Modelo

La evaluación se realiza mediante dos enfoques complementarios:

#### 4.1. Validación Cruzada Temporal

El componente `ModelEvaluator` implementa validación cruzada temporal:

```python
from prophet.diagnostics import cross_validation, performance_metrics

# Realizar validación cruzada
df_cv = cross_validation(
    model=model,
    initial=initial,
    period=period,
    horizon=horizon,
    parallel=None  # Desactivar paralelismo para mayor estabilidad
)

# Calcular métricas
metrics = {
    'rmse': np.sqrt(np.mean(df_cv['abs_error'] ** 2)),
    'mae': np.mean(df_cv['abs_error']),
    'smape': np.mean(df_cv['smape']),
    'coverage': ((df_cv['y'] >= df_cv['yhat_lower']) & 
                (df_cv['y'] <= df_cv['yhat_upper'])).mean() * 100
}
```

Este proceso:
1. Divide los datos en múltiples ventanas temporales
2. Entrena el modelo con datos hasta cierto punto
3. Predice para un horizonte específico
4. Compara predicciones con valores reales
5. Calcula métricas de error: RMSE, MAE, SMAPE y cobertura

#### 4.2. Backtesting

El módulo de backtesting permite evaluar el rendimiento retrospectivo del modelo:

1. Seleccionar una fecha de corte en el pasado
2. Entrenar el modelo con datos hasta esa fecha
3. Generar predicciones para el período posterior
4. Comparar con los valores reales observados

Este enfoque simula cómo habría funcionado el modelo en condiciones reales del pasado.

### 5. Generación de Predicciones

Una vez entrenado y validado, el modelo genera predicciones futuras:

```python
# Generar predicciones
future = model.make_future_dataframe(periods=periods, include_history=include_history)

# Añadir regresores al dataframe futuro
if self.selected_regressors and len(self.selected_regressors) > 0:
    future = self._add_future_regressors(future, periods)
    
# Realizar predicción
forecast = model.predict(future)
```

Las predicciones incluyen:
- Valor medio estimado (`yhat`)
- Intervalos de confianza (`yhat_lower` y `yhat_upper`)
- Descomposición en componentes (tendencia, estacionalidad, efectos de regresores)

## Beneficios del Enfoque Modular

La arquitectura modular del sistema de predicción ofrece varias ventajas:

1. **Mantenibilidad**: Cada componente tiene responsabilidades claramente definidas
2. **Extensibilidad**: Facilita añadir nuevas funcionalidades o técnicas
3. **Testabilidad**: Permite probar cada componente de forma aislada
4. **Reutilización**: Los componentes pueden usarse en otros contextos
5. **Evolución gradual**: Posibilidad de mejorar componentes individuales sin afectar al resto

## Consideraciones Prácticas

### Fortalezas del Modelo

1. **Robustez ante outliers**: Manejo sofisticado de valores atípicos
2. **Captura de patrones complejos**: Modelado de múltiples estacionalidades
3. **Incorporación de conocimiento experto** a través de eventos y regresores
4. **Cuantificación de la incertidumbre** con intervalos de confianza
5. **Evaluación rigurosa** mediante validación cruzada y backtesting

### Limitaciones

1. **Dependencia de datos históricos**: Requiere suficientes datos pasados para entrenamiento
2. **Cambios estructurales**: Dificultad para predecir cambios drásticos sin precedentes
3. **Requisitos computacionales**: La validación cruzada puede ser intensiva en recursos
4. **Equilibrio complejidad-generalización**: Riesgo de sobreajuste con modelos muy complejos

La arquitectura modular permite ajustar la complejidad según las necesidades, desde un modelo básico hasta uno que incorpore múltiples fuentes de datos externas.
