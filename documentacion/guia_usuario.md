# Guía de Usuario - RansomVisor

Esta guía proporciona instrucciones detalladas sobre cómo utilizar cada sección de RansomVisor, nuestra aplicación de análisis y predicción de ransomware.

## Índice
1. [Navegación General](#navegación-general)
2. [Visión General](#visión-general)
3. [Tendencias](#tendencias)
4. [Geografía](#geografía)
5. [Alertas](#alertas)
6. [Modelado](#modelado)
7. [Sectores](#sectores)
8. [Preguntas Frecuentes](#preguntas-frecuentes)

## Navegación General

RansomVisor está organizado en múltiples páginas accesibles desde la barra lateral izquierda. Cada página tiene un propósito específico y proporciona diferentes análisis sobre datos de ransomware.

- El botón **Actualizar todo** en la parte superior de la barra lateral actualiza todos los datos desde las fuentes.
- Cada página tiene controles específicos en la barra lateral para filtrar y personalizar las visualizaciones.

## Visión General

La página de inicio proporciona un dashboard general con las métricas más importantes.

### Funcionalidades

- **Resumen de Ataques Recientes**: Muestra los ataques de ransomware más recientes con información sobre víctimas y grupos responsables.
- **Estadísticas Clave**: Proporciona métricas resumidas como total de ataques, víctimas y tendencias.
- **Gráficos de Evolución**: Muestra la evolución temporal de ataques por grupos criminales.

### Consejos de Uso

- Utilice los filtros de la barra lateral para centrarse en períodos de tiempo específicos.
- Pase el cursor sobre los gráficos para ver información detallada.

## Tendencias

Esta sección ofrece análisis detallados de la evolución de ataques ransomware a lo largo del tiempo.

### Funcionalidades

- **Evolución Temporal**: Gráficos que muestran la evolución de ataques por período.
- **Análisis por Grupo**: Comparativa de actividad entre diferentes grupos criminales.
- **Variación Estacional**: Identificación de patrones temporales en los ataques.

### Consejos de Uso

- Ajuste la granularidad temporal (diaria, semanal, mensual) para diferentes perspectivas.
- Utilice los selectores para comparar múltiples grupos criminales.
- Active/desactive series individuales haciendo clic en la leyenda.

## Geografía

Esta página proporciona visualización geográfica de los ataques ransomware.

### Funcionalidades

- **Mapa Global**: Mapa de calor interactivo que muestra la concentración de ataques por región.
- **Distribución por País**: Desglose de ataques por país con gráficos comparativos.
- **Evolución Geográfica**: Análisis de cómo ha cambiado la distribución geográfica con el tiempo.

### Consejos de Uso

- Acerque/aleje el mapa para ver diferentes niveles de detalle.
- Filtre por período temporal para ver cómo cambia la distribución geográfica.
- Haga clic en un país para ver detalles específicos.

## Alertas

El sistema de alertas notifica sobre nuevos ataques y tendencias emergentes.

### Funcionalidades

- **Alertas de Nuevos Ataques**: Notificaciones sobre ataques recientes con detalles importantes.
- **Alertas de Tendencias**: Identificación de cambios significativos en patrones de ataque.
- **Alertas de Grupos Activos**: Notificaciones sobre actividad inusual de grupos criminales específicos.

### Configuración de Alertas

- **Umbral de Severidad**: Ajuste el nivel mínimo de severidad para recibir alertas.
- **Filtros por Sector**: Configure alertas específicas para sectores de interés.
- **Notificaciones**: Seleccione cómo desea recibir las alertas (en la aplicación, por correo, etc.).

## Modelado

La sección de modelado es el núcleo predictivo de RansomVisor, que permite entrenar modelos para predecir futuros ataques ransomware.

### Interfaz de Modelado

La interfaz de modelado está organizada en un flujo de trabajo secuencial:

1. **Carga de Datos**: Selección y configuración de fuentes de datos.
2. **Preparación de Datos**: Opciones para preprocesamiento y transformación.
3. **Entrenamiento del Modelo**: Configuración y entrenamiento del modelo Prophet.
4. **Generación de Predicciones**: Creación y visualización de pronósticos.
5. **Evaluación y Prueba**: Validación del rendimiento del modelo.

### 1️⃣ Carga de Datos

En esta sección, puede cargar y configurar las fuentes de datos para el modelado.

#### Opciones Disponibles:

- **Fuente de Datos**: Seleccione el archivo de datos de ataques ransomware.
- **Datos de Regresores**: Opcionalmente, seleccione archivos de regresores externos (como CVEs).
- **Enfoque de Modelado**: 
  - **Conteo Diario**: Predice el número de ataques por día (recomendado).
  - **Días Entre Ataques**: Predice el intervalo entre ataques consecutivos.
- **Transformación Logarítmica**: Opción para estabilizar series temporales con alta variabilidad.

#### Pasos para Cargar Datos:

1. Haga clic en "Cargar Datos" para iniciar el proceso.
2. Espere a que se complete la carga y procesamiento inicial.
3. Verifique el resumen de datos cargados que se muestra.

### 2️⃣ Preparación de Datos

Esta sección permite configurar cómo se preparan los datos antes del modelado.

#### Opciones de Detección de Outliers:

- **Método de Detección**:
  - **IQR (Rango Intercuartílico)**: Método basado en cuartiles, robusto y general.
  - **Z-Score**: Basado en desviaciones estándar, adecuado para datos normales.
  - **MAD (Desviación Absoluta de la Mediana)**: Más robusto que Z-Score para datos sesgados.
  - **Ninguno**: No aplicar detección de outliers.

- **Estrategia de Manejo**:
  - **Winsorize**: Reemplaza valores extremos con el percentil 95/5.
  - **Cap**: Limita valores extremos sin eliminarlos.
  - **Remove**: Elimina outliers y los reemplaza mediante interpolación.
  - **None**: Mantiene los outliers sin modificación.

- **Umbral de Detección**: Ajusta la sensibilidad de la detección (valores típicos entre 1.5 y 3.0).

#### Opciones Adicionales:

- **Umbral Mínimo de Víctimas**: Define el número mínimo de víctimas para considerar un día como "día de ataque".
- **Transformación Logarítmica**: Aplica log(y+1) para estabilizar la varianza (útil para datos con alta variabilidad).

### 3️⃣ Entrenamiento del Modelo

En esta sección, configure y entrene el modelo Prophet.

#### Parámetros Básicos:

- **Changepoint Prior Scale**: Controla la flexibilidad de la tendencia (valores típicos: 0.05 - 0.5).
- **Seasonality Prior Scale**: Controla la fuerza de los componentes estacionales (valores típicos: 1.0 - 25.0).
- **Seasonality Mode**: Seleccione entre aditivo (variación constante) o multiplicativo (variación proporcional).
- **Interval Width**: Anchura del intervalo de confianza (ej. 0.8 para 80% de confianza).

#### Opciones Avanzadas:

- **Uso de Puntos de Cambio Detectados**: Habilita la detección automática de cambios en la tendencia.
- **Inclusión de Eventos Especiales**: Incorpora días festivos y eventos conocidos al modelo.
- **Activación de Regresores Externos**: Utiliza variables externas como CVEs para mejorar las predicciones.
- **Optimización Bayesiana**: Busca automáticamente los mejores hiperparámetros.
- **Selección Óptima de Regresores**: Identifica y utiliza solo los regresores más relevantes.
- **Calibración de Intervalos**: Ajusta los intervalos de predicción para mayor precisión.

### 4️⃣ Generación de Predicciones

Una vez entrenado el modelo, puede generar y visualizar predicciones.

#### Opciones de Predicción:

- **Período de Predicción**: Número de días en el futuro para predecir (típicamente 30-90 días).
- **Inclusión de Datos Históricos**: Opción para incluir datos de entrenamiento en la visualización.
- **Visualización de Intervalos**: Muestra bandas de confianza alrededor de la predicción.
- **Visualización de Puntos de Cambio**: Resalta momentos donde la tendencia cambia significativamente.

#### Visualizaciones Disponibles:

- **Gráfico Principal**: Muestra datos históricos y predicciones futuras con intervalos de confianza.
- **Componentes del Modelo**: Desglosa la predicción en tendencia y componentes estacionales.
- **Impacto de Regresores**: Visualiza la contribución de variables externas a la predicción.

### 5️⃣ Evaluación y Prueba del Modelo

Esta sección permite evaluar rigurosamente el rendimiento del modelo.

#### Validación Cruzada Temporal:

- **Horizonte de Validación**: Número de días futuros para validar en cada iteración.
- **Métricas Calculadas**:
  - **RMSE**: Error cuadrático medio, sensible a errores grandes.
  - **MAE**: Error absoluto medio, métrica robusta del error promedio.
  - **SMAPE**: Error porcentual absoluto simétrico medio, útil para comparabilidad.
  - **Cobertura**: Porcentaje de valores reales dentro del intervalo de confianza.

#### Backtesting (Prueba Histórica):

- **Fecha de Corte**: Seleccione un punto en el pasado para simular una predicción.
- **Análisis Visual**: Compare predicciones simuladas con datos reales observados.
- **Métricas de Error**: Calcule métricas de rendimiento específicas para el periodo de prueba.

### Guía Integrada del Usuario

Dentro de la sección de modelado, puede acceder a una guía completa del usuario haciendo clic en el botón "Ver Guía de Usuario". Esta guía proporciona:

- Explicaciones detalladas de cada parámetro
- Recomendaciones para diferentes escenarios
- Interpretación de resultados
- Preguntas frecuentes y solución de problemas

## Sectores

Esta página permite analizar el impacto del ransomware por sectores industriales.

### Funcionalidades

- **Visión General de Sectores**: Comparativa de afectación entre diferentes sectores.
- **Análisis de Sector Específico**: Datos detallados al seleccionar un sector particular.
- **Evolución Temporal por Sector**: Análisis de cómo ha evolucionado la amenaza en cada sector.

### Visualizaciones Disponibles

- **Gráfico de Barras por Sector**: Comparativa del número total de víctimas por sector.
- **Mapa de Víctimas por Sector**: Distribución geográfica de víctimas en cada sector.
- **Evolución Mensual**: Tendencia temporal de ataques por sector seleccionado.

### Consejos de Uso

- Utilice los filtros de la barra lateral para centrarse en sectores específicos.
- Compare diferentes sectores para identificar los más vulnerables.
- Analice la evolución temporal para detectar cambios en el enfoque de los atacantes.

## Preguntas Frecuentes

### Preguntas Generales

**¿Con qué frecuencia se actualizan los datos?**  
Los datos se actualizan cuando se presiona el botón "Actualizar Todo" en la barra lateral. Recomendamos actualizarlos al iniciar la aplicación para tener la información más reciente.

**¿Puedo exportar los resultados?**  
Sí, la mayoría de los gráficos y tablas tienen opciones de exportación. Busque el icono de descarga en la esquina superior derecha de cada visualización.

**¿Cómo puedo interpretar los intervalos de confianza en las predicciones?**  
Los intervalos de confianza representan el rango donde se espera que caigan los valores reales con cierta probabilidad (por defecto 80%). Intervalos más amplios indican mayor incertidumbre en la predicción.

### Preguntas sobre el Modelo

**¿Qué significan los hiperparámetros del modelo?**

- **changepoint_prior_scale**: Controla la flexibilidad de la tendencia. Valores más altos permiten cambios más abruptos en la tendencia, mientras que valores más bajos producen tendencias más suaves.

- **seasonality_prior_scale**: Controla la fuerza de los componentes estacionales. Valores más altos permiten 
  patrones estacionales más pronunciados. Si los patrones semanales o anuales parecen subestimados, aumente este valor.

- **holidays_prior_scale**: Controla el impacto de eventos especiales. Aumente este valor si ciertos días o 
  eventos tienen un impacto significativo en los ataques.

**¿Cómo puedo mejorar la precisión del modelo?**

1. **Datos Suficientes**: Asegúrese de tener al menos un año de datos históricos para capturar patrones estacionales.

2. **Regresores Relevantes**: Incluya variables externas que puedan influir en los ataques (como CVEs).

3. **Ajuste de Hiperparámetros**: Utilice la optimización bayesiana para encontrar los mejores parámetros.

4. **Manejo de Outliers**: Pruebe diferentes estrategias para manejar valores atípicos según la naturaleza de los datos.

5. **Validación Rigurosa**: Use validación cruzada y backtesting para evaluar el rendimiento real del modelo.

**¿Qué debo hacer si las predicciones parecen demasiado planas o demasiado volátiles?**

Si sus predicciones son demasiado "planas" comparadas con la volatilidad histórica:

1. Aumente `changepoint_prior_scale` para permitir más flexibilidad en la tendencia
2. Aumente `seasonality_prior_scale` para capturar patrones estacionales más fuertes
3. Considere si el modo de estacionalidad (`additive` vs `multiplicative`) es apropiado

Si sus predicciones son demasiado volátiles o inestables:

1. Reduzca `changepoint_prior_scale` para suavizar la tendencia
2. Pruebe el modo de estacionalidad aditivo para patrones más estables
3. Asegúrese de que el manejo de outliers es apropiado para sus datos

### Preguntas sobre Interpretación

**¿Cómo debo interpretar las métricas de evaluación?**

- **RMSE (Error Cuadrático Medio)**: Indica el error promedio en las mismas unidades que los datos. Valores más bajos son mejores, pero es sensible a errores grandes.

- **MAE (Error Absoluto Medio)**: Representa el error promedio absoluto. Más robusto que RMSE frente a outliers.

- **SMAPE (Error Porcentual Absoluto Simétrico Medio)**: Error expresado como porcentaje, útil para comparar modelos entre diferentes series.

- **Cobertura**: Porcentaje de valores reales que caen dentro del intervalo de confianza. Idealmente debería estar cerca del nivel de confianza configurado (ej. 80%).

**¿Cómo puedo usar las predicciones para la planificación de seguridad?**

1. **Identificar Períodos de Alto Riesgo**: Utilice las predicciones para identificar períodos futuros con mayor probabilidad de ataques.

2. **Planificación de Recursos**: Asigne más recursos de seguridad durante períodos de alto riesgo proyectado.

3. **Actualizaciones Preventivas**: Programe actualizaciones de sistemas y parches durante períodos de menor riesgo.

4. **Concienciación**: Aumente la formación y concienciación del personal durante períodos de mayor riesgo.

5. **Evaluación de Controles**: Utilice los intervalos de confianza para evaluar el nivel de incertidumbre y ajustar los controles en consecuencia.
