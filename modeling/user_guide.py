"""
Guía de Usuario - Modelo de Predicción de Ataques Ransomware

Este módulo proporciona una guía completa para utilizar el sistema de predicción
de ataques ransomware, incluyendo explicaciones sobre cada función, parámetros
y cómo interpretar los resultados.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show_user_guide():
    """
    Muestra la guía de usuario completa para el modelo de predicción de ransomware.
    """
    # Estilos CSS para mejorar la legibilidad en tema oscuro
    st.markdown("""
    <style>
    .guide-section {
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 5px;
        background-color: rgba(70, 70, 70, 0.2);
        border-left: 3px solid rgba(100, 180, 255, 0.8);
    }
    .guide-header {
        color: rgba(100, 180, 255, 0.9);
        margin-bottom: 10px;
    }
    .guide-subheader {
        color: rgba(150, 200, 255, 0.9);
        margin-top: 15px;
        margin-bottom: 8px;
    }
    .guide-code {
        background-color: rgba(50, 50, 50, 0.5);
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        margin: 10px 0;
    }
    .guide-tip {
        background-color: rgba(70, 100, 150, 0.2);
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid rgba(100, 200, 150, 0.8);
        margin: 10px 0;
    }
    .guide-warning {
        background-color: rgba(150, 100, 70, 0.2);
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid rgba(255, 150, 100, 0.8);
        margin: 10px 0;
    }
    
    /* Estilos adicionales para arreglar los encabezados con fondo blanco */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Estilos para los encabezados de sección numerados */
    .block-container div[data-testid="stVerticalBlock"] > div:nth-child(n) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) h2 {
        color: #ffffff !important;
        background-color: rgba(70, 70, 70, 0.5) !important;
        padding: 10px !important;
        border-radius: 5px !important;
        border-left: 3px solid rgba(100, 180, 255, 0.8) !important;
    }
    
    /* Estilos para los contenedores de entrada */
    div[data-baseweb="input"] {
        background-color: rgba(70, 70, 70, 0.3) !important;
    }
    
    /* Estilos para los selectores */
    div[data-baseweb="select"] {
        background-color: rgba(70, 70, 70, 0.3) !important;
    }
    
    /* Estilos para los sliders */
    div[data-testid="stSlider"] {
        background-color: rgba(70, 70, 70, 0.2) !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    
    /* Estilos específicos para los textos en columnas que tienen fondo blanco */
    .row-widget.stHorizontal div {
        background-color: transparent !important;
    }
    
    /* Estilos para los textos en columnas */
    .row-widget.stHorizontal div p {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Estilos para los textos descriptivos */
    .element-container div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Estilos para los textos en columnas específicas */
    div[data-testid="column"] div[data-testid="stMarkdownContainer"] {
        background-color: rgba(70, 70, 70, 0.3) !important;
        padding: 10px !important;
        border-radius: 5px !important;
        margin-bottom: 10px !important;
    }
    
    /* Estilos para los textos en columnas específicas */
    div[data-testid="column"] div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    
    /* Estilos para todos los textos en la aplicación */
    p, li, span {
        color: #ffffff !important;
    }
    
    /* Estilos para los fondos blancos en columnas */
    div[data-testid="column"] {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title(" Guía de Usuario - Modelo de Predicción de Ransomware")
    
    st.markdown("""
    Esta guía te ayudará a entender cómo utilizar el sistema de predicción de ataques ransomware,
    explicando cada función, sus parámetros y cómo interpretar los resultados.
    """)
    
    # Crear un índice de contenidos en el área principal
    st.markdown("## Contenido")
    
    # Crear columnas para mostrar el índice de forma más compacta
    col1, col2 = st.columns(2)
    
    sections = {
        "introduccion": "1️⃣ Introducción",
        "flujo_trabajo": "2️⃣ Flujo de Trabajo",
        "carga_datos": "3️⃣ Carga de Datos",
        "preparacion_datos": "4️⃣ Preparación de Datos",
        "entrenamiento": "5️⃣ Entrenamiento del Modelo",
        "prediccion": "6️⃣ Generación de Predicciones",
        "evaluacion": "7️⃣ Evaluación del Modelo",
        "visualizacion": "8️⃣ Visualización de Resultados",
        "interpretacion": "9️⃣ Interpretación de Resultados",
        "optimizaciones": "🔧 Optimizaciones Avanzadas",
        "faq": "❓ Preguntas Frecuentes"
    }
    
    # Dividir las secciones entre las dos columnas
    items = list(sections.items())
    mid = len(items) // 2
    
    with col1:
        for key, title in items[:mid]:
            st.markdown(f"- [{title}](#{key})")
    
    with col2:
        for key, title in items[mid:]:
            st.markdown(f"- [{title}](#{key})")
    
    # Añadir un separador para mejorar la legibilidad
    st.markdown("---")
    
    # Introducción
    st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
    st.header("1️⃣ Introducción", anchor="introduccion")
    st.markdown("""
    El Sistema de Predicción de Ataques Ransomware es una herramienta avanzada diseñada para pronosticar
    el número esperado de ataques de ransomware en los próximos días o semanas. Utiliza modelos de series
    temporales basados en Prophet con optimizaciones avanzadas para proporcionar predicciones precisas
    y confiables.
    
    ### ¿Qué es el ransomware?
    
    El ransomware es un tipo de malware que cifra los archivos de la víctima y exige un rescate para restaurar
    el acceso. Los ataques de ransomware se han vuelto cada vez más frecuentes y sofisticados, representando
    una amenaza significativa para organizaciones de todos los tamaños.
    
    ### ¿Por qué es importante predecir los ataques?
    
    Predecir tendencias futuras de ataques ransomware permite:
    
    - **Planificación proactiva**: Asignar recursos de seguridad cuando más se necesitan
    - **Gestión de riesgos**: Evaluar la exposición potencial y tomar medidas preventivas
    - **Preparación de respuesta**: Garantizar que los equipos estén listos durante períodos de alto riesgo
    - **Justificación de inversiones**: Proporcionar datos para respaldar decisiones de inversión en seguridad
    
    ### Características principales:
    
    - **Transformaciones óptimas de datos**: Aplicación automática de transformaciones logarítmicas para mejorar la precisión.
    - **Feature Engineering avanzado**: Generación automática de características temporales y externas relevantes.
    - **Optimización bayesiana de hiperparámetros**: Búsqueda eficiente de la mejor configuración del modelo.
    - **Calibración de intervalos de predicción**: Garantiza que los intervalos de confianza sean precisos.
    - **Interfaz visual interactiva**: Visualización clara de predicciones y resultados.
    
    Este sistema está diseñado para ser utilizado por analistas de seguridad, equipos de respuesta a incidentes
    y responsables de la toma de decisiones en ciberseguridad para anticipar y prepararse mejor ante posibles
    oleadas de ataques de ransomware.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Flujo de trabajo
    st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
    st.header("2️⃣ Flujo de Trabajo", anchor="flujo_trabajo")
    st.markdown("""
    El sistema de predicción de ataques ransomware sigue un flujo de trabajo estructurado que guía al usuario
    a través de todo el proceso, desde la carga de datos hasta la interpretación de resultados. A continuación,
    se detalla cada paso del proceso:
    """)
    
    # Crear diagrama de flujo visual
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
        <h3>1️⃣ Carga de Datos</h3>
        <ul>
        <li>Cargar datos históricos de ataques</li>
        <li>Cargar datos de CVE (opcional)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
        <h3>2️⃣ Preparación de Datos</h3>
        <ul>
        <li>Detección de outliers</li>
        <li>Transformación de datos</li>
        <li>Feature engineering</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
        <h3>3️⃣ Entrenamiento</h3>
        <ul>
        <li>Optimización de hiperparámetros</li>
        <li>Entrenamiento del modelo</li>
        <li>Validación cruzada</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
        <h3>4️⃣ Predicción</h3>
        <ul>
        <li>Generación de predicciones</li>
        <li>Visualización de resultados</li>
        <li>Interpretación</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white; margin-top: 20px;">
    <h3>Descripción detallada del flujo de trabajo:</h3>
    
    <p><strong>1. Carga de Datos:</strong></p>
    <ul>
        <li><strong>Selección de enfoque:</strong> Elija entre "Conteo de ataques por día" o "Días entre ataques"</li>
        <li><strong>Carga de datos históricos:</strong> Utilice datos propios o los datos de ejemplo incluidos</li>
        <li><strong>Datos de vulnerabilidades (CVE):</strong> Opcionalmente, incluya datos de vulnerabilidades para mejorar las predicciones</li>
        <li><strong>Visualización inicial:</strong> Examine los datos cargados para entender sus características</li>
    </ul>
    
    <p><strong>2. Preparación de Datos:</strong></p>
    <ul>
        <li><strong>Detección de outliers:</strong> Identifique valores atípicos que podrían afectar la precisión</li>
        <li><strong>Tratamiento de outliers:</strong> Elimine, reemplace o interpole valores atípicos</li>
        <li><strong>Transformaciones:</strong> Aplique transformaciones logarítmicas u otras para estabilizar la varianza</li>
        <li><strong>Feature engineering:</strong> Genere características adicionales como días de la semana, festivos, etc.</li>
    </ul>
    
    <p><strong>3. Entrenamiento del Modelo:</strong></p>
    <ul>
        <li><strong>Configuración de parámetros:</strong> Ajuste los parámetros del modelo según sus necesidades</li>
        <li><strong>Optimización bayesiana:</strong> Encuentre automáticamente los mejores hiperparámetros</li>
        <li><strong>Entrenamiento:</strong> Ajuste el modelo a los datos históricos</li>
        <li><strong>Validación cruzada:</strong> Evalúe el rendimiento del modelo en diferentes períodos de tiempo</li>
    </ul>
    
    <p><strong>4. Generación de Predicciones:</strong></p>
    <ul>
        <li><strong>Horizonte de predicción:</strong> Especifique cuántos días hacia el futuro desea predecir</li>
        <li><strong>Calibración de intervalos:</strong> Ajuste los intervalos de confianza para mayor precisión</li>
        <li><strong>Visualización:</strong> Examine gráficamente las predicciones y sus componentes</li>
        <li><strong>Interpretación:</strong> Analice los resultados y extraiga conclusiones accionables</li>
    </ul>
    
    <p>Es importante seguir este flujo en orden, ya que cada paso depende de los anteriores.
    El sistema guiará al usuario a través de cada etapa con indicadores visuales de progreso.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cerrar la sección de flujo de trabajo
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Carga de datos
    st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
    st.header("3️⃣ Carga de Datos", anchor="carga_datos")
    st.markdown("""
    ### Función: `load_data_wrapper`
    
    Esta función permite cargar los datos históricos de ataques ransomware y, opcionalmente,
    datos de vulnerabilidades (CVE) para mejorar las predicciones.
    
    #### Parámetros:
    
    - **ransomware_file**: Ruta al archivo JSON con datos de ataques ransomware.
      - Formato esperado: Archivo JSON con registros que contengan al menos una columna de fecha ('fecha', 'date' o 'ds').
      - Valor predeterminado: 'modeling/victimas_ransomware_mod.json'
    
    - **cve_file**: Ruta al archivo CSV con datos diarios de CVE.
      - Formato esperado: CSV con columnas 'fecha' y 'cve_count' (o similar).
      - Valor predeterminado: 'modeling/cve_diarias_regresor_prophet.csv'
    
    - **enfoque**: Método para procesar los datos.
      - 'conteo_diario': Cuenta el número de ataques por día.
      - 'dias_entre_ataques': Calcula el tiempo entre ataques consecutivos.
      - Valor predeterminado: 'conteo_diario'
    
    - **use_log_transform**: Si aplicar transformación logarítmica a los datos.
      - Recomendado: Activar para series con valores extremos o varianza no constante.
      - Valor predeterminado: False
    """)
    
    st.markdown('<div class="guide-code">', unsafe_allow_html=True)
    st.markdown("""
    ```python
    # Cargar datos con configuración predeterminada
    df = load_data_wrapper()
    
    # Cargar datos con transformación logarítmica
    df = load_data_wrapper(use_log_transform=True)
    
    # Cargar datos desde archivos personalizados
    df = load_data_wrapper(
        ransomware_file='mis_datos/ataques.json',
        cve_file='mis_datos/vulnerabilidades.csv'
    )
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="guide-tip">', unsafe_allow_html=True)
    st.markdown("""
    #### Consejos:
    
    - Asegúrese de que los datos tengan una frecuencia regular (preferiblemente diaria).
    - La transformación logarítmica es recomendable cuando hay valores extremos o picos en los datos.
    - Incluir datos de CVE puede mejorar significativamente la precisión de las predicciones.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Preparación de Datos
    st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
    st.header("4️⃣ Preparación de Datos", anchor="preparacion_datos")
    st.markdown("""
    La preparación adecuada de los datos es fundamental para obtener predicciones precisas. Esta sección explica
    las técnicas disponibles para limpiar, transformar y enriquecer los datos antes del modelado.
    
    ### Detección y tratamiento de outliers
    
    Los valores atípicos (outliers) pueden distorsionar significativamente las predicciones. El sistema ofrece
    varios métodos para detectar y manejar estos valores:
    
    #### Métodos de detección:
    
    - **Método IQR (Rango Intercuartil)**:
      - Identifica valores que están fuera de 1.5 × IQR desde el primer y tercer cuartil.
      - Efectivo para distribuciones aproximadamente normales.
    
    - **Método Z-Score**:
      - Identifica valores que están a más de N desviaciones estándar de la media.
      - Configurable mediante el parámetro `threshold` (típicamente 3.0).
    
    - **Método de Desviación Absoluta de la Mediana (MAD)**:
      - Más robusto que Z-Score para distribuciones no normales.
      - Menos afectado por los propios outliers durante la detección.
    
    #### Opciones de tratamiento:
    
    - **Eliminación**: Simplemente elimina los valores atípicos del conjunto de datos.
      - Ventaja: Simple y efectivo.
      - Desventaja: Puede crear huecos en los datos temporales.
    
    - **Imputación**: Reemplaza los valores atípicos con valores estimados.
      - Métodos disponibles: media, mediana, interpolación lineal, interpolación LOCF (último valor observado).
      - Recomendado para mantener la continuidad temporal.
    
    - **Winsorización**: Recorta los valores extremos a un percentil específico.
      - Menos drástico que la eliminación completa.
      - Preserva la existencia del punto de datos pero reduce su impacto.
    
    ### Transformaciones de datos
    
    Las transformaciones pueden mejorar significativamente el rendimiento del modelo:
    
    - **Transformación logarítmica**:
      - Ideal para datos con crecimiento exponencial o alta variabilidad.
      - Estabiliza la varianza y normaliza distribuciones sesgadas.
      - Implementada mediante `np.log1p()` para manejar valores cero.
    
    - **Transformación de raíz cuadrada**:
      - Menos agresiva que la logarítmica.
      - Útil para datos moderadamente sesgados.
    
    - **Transformación de Box-Cox**:
      - Familia flexible de transformaciones potenciales.
      - Encuentra automáticamente el mejor parámetro lambda.
      - Requiere valores estrictamente positivos.
    
    ### Feature Engineering
    
    La generación de características adicionales puede mejorar significativamente las predicciones:
    
    - **Características temporales**:
      - Día de la semana: Captura patrones semanales.
      - Mes del año: Captura estacionalidad anual.
      - Día del mes: Captura patrones mensuales.
      - Es fin de semana: Indicador binario para fines de semana.
    
    - **Eventos especiales**:
      - Días festivos: Incorporados mediante la funcionalidad de holidays de Prophet.
      - Eventos de seguridad: Conferencias, divulgación de vulnerabilidades importantes.
      - Períodos especiales: Fin de año fiscal, temporada de compras, etc.
    
    - **Variables externas**:
      - Conteo de CVE: Número de vulnerabilidades publicadas.
      - Indicadores económicos: Pueden correlacionarse con actividad de ransomware.
      - Menciones en medios: Cobertura mediática de ransomware o ciberseguridad.
    
    ### Consejos para la preparación de datos
    
    - **Inspección visual**: Siempre visualice sus datos antes y después de la preparación.
    - **Enfoque iterativo**: Pruebe diferentes métodos de tratamiento y compare resultados.
    - **Documentación**: Mantenga registro de todas las transformaciones aplicadas para poder invertirlas después.
    - **Validación**: Verifique que las transformaciones no introduzcan artefactos o distorsiones.
    - **Conocimiento del dominio**: Utilice su comprensión del ransomware para guiar decisiones de preparación.
    
    ### Ejemplo práctico
    
    ```python
    # Detección de outliers usando el método IQR
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identificar outliers
    outliers = df[(df['y'] < lower_bound) | (df['y'] > upper_bound)]
    
    # Visualizar outliers
    plt.figure(figsize=(10, 6))
    plt.scatter(df['ds'], df['y'], alpha=0.5)
    plt.scatter(outliers['ds'], outliers['y'], color='red')
    plt.title('Detección de Outliers - Método IQR')
    plt.show()
    
    # Imputar outliers con la mediana
    df_cleaned = df.copy()
    df_cleaned.loc[(df_cleaned['y'] < lower_bound) | (df_cleaned['y'] > upper_bound), 'y'] = df['y'].median()
    
    # Aplicar transformación logarítmica
    df_cleaned['y_log'] = np.log1p(df_cleaned['y'])
    
    # Añadir características temporales
    df_cleaned['day_of_week'] = df_cleaned['ds'].dt.dayofweek
    df_cleaned['month'] = df_cleaned['ds'].dt.month
    df_cleaned['is_weekend'] = df_cleaned['day_of_week'].isin([5, 6]).astype(int)
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Entrenamiento del Modelo
    st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
    st.header("5️⃣ Entrenamiento del Modelo", anchor="entrenamiento")
    st.markdown("""
    El entrenamiento del modelo es una fase crítica que determina la calidad de las predicciones futuras.
    Esta sección explica cómo configurar y entrenar efectivamente el modelo Prophet para predicciones de ransomware.
    
    ### Función: `train_model_wrapper`
    
    Esta función configura y entrena un modelo Prophet con los parámetros especificados.
    
    #### Parámetros principales:
    
    - **df**: DataFrame con los datos de entrenamiento.
      - Debe contener al menos las columnas 'ds' (fechas) y 'y' (valores).
    
    - **yearly_seasonality**: Configuración de estacionalidad anual.
      - True: Detecta automáticamente estacionalidad anual.
      - False: No incluye estacionalidad anual.
      - Número entero: Especifica el número de términos de Fourier (complejidad).
      - Valor predeterminado: True
    
    - **weekly_seasonality**: Configuración de estacionalidad semanal.
      - Similar a yearly_seasonality pero para patrones semanales.
      - Valor predeterminado: True
    
    - **daily_seasonality**: Configuración de estacionalidad diaria.
      - Útil solo para datos con múltiples puntos por día.
      - Valor predeterminado: False
    
    - **seasonality_mode**: Modo de estacionalidad.
      - 'additive': Efectos estacionales constantes independientes del nivel.
      - 'multiplicative': Efectos estacionales que escalan con el nivel.
      - Valor predeterminado: 'additive'
    
    - **changepoint_prior_scale**: Flexibilidad de la tendencia.
      - Valores más altos permiten cambios más abruptos en la tendencia.
      - Valor predeterminado: 0.05
    
    - **seasonality_prior_scale**: Fuerza de los componentes estacionales.
      - Valores más altos permiten patrones estacionales más pronunciados.
      - Valor predeterminado: 10.0
    
    - **holidays_prior_scale**: Impacto de eventos especiales.
      - Valores más altos dan más peso a los efectos de días festivos.
      - Valor predeterminado: 10.0
    
    - **changepoint_range**: Proporción de datos donde pueden ocurrir puntos de cambio.
      - 0.8 significa que los puntos de cambio solo pueden ocurrir en el primer 80% de los datos.
      - Valor predeterminado: 0.8
    
    - **interval_width**: Ancho de los intervalos de predicción.
      - 0.95 corresponde a un intervalo de confianza del 95%.
      - Valor predeterminado: 0.8
    
    - **country_holidays**: País para incluir días festivos automáticamente.
      - Ejemplo: 'ES' para España, 'US' para Estados Unidos.
      - Valor predeterminado: None
    
    - **custom_holidays**: DataFrame con eventos personalizados.
      - Debe tener columnas 'holiday', 'ds', 'lower_window', 'upper_window'.
      - Valor predeterminado: None
    
    - **regressors**: Lista de nombres de columnas para usar como regresores.
      - Ejemplo: ['cve_count'] para incluir conteo de CVEs como variable externa.
      - Valor predeterminado: None
    """)
    st.markdown('<div class="guide-code">', unsafe_allow_html=True)
    st.markdown("""
    ```python
    # Entrenamiento básico con parámetros predeterminados
    model = train_model_wrapper(df)
    
    # Entrenamiento personalizado
    model = train_model_wrapper(
        df,
        yearly_seasonality=20,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=15.0,
        holidays_prior_scale=10.0,
        interval_width=0.9,
        country_holidays='US'
    )
    
    # Entrenamiento con regresores externos
    model = train_model_wrapper(
        df,
        regressors=['cve_count', 'media_mentions'],
        changepoint_prior_scale=0.05
    )
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="guide-tip">', unsafe_allow_html=True)
    st.markdown("""
    #### Consejos para el entrenamiento efectivo:
    
    1. **Estacionalidad**:
       - Active yearly_seasonality si tiene al menos 2 años de datos.
       - Active weekly_seasonality si espera patrones diferentes por día de la semana.
       - Ajuste la complejidad (número de términos) según la complejidad observada en los datos.
    
    2. **Modo de estacionalidad**:
       - Use 'additive' (predeterminado) si la amplitud de los patrones estacionales es constante.
       - Use 'multiplicative' si la amplitud de los patrones estacionales aumenta con el nivel general.
    
    3. **Flexibilidad de la tendencia**:
       - Aumente changepoint_prior_scale si la tendencia parece demasiado rígida.
       - Disminuya changepoint_prior_scale si la tendencia es demasiado flexible y sobreajusta.
    
    4. **Días festivos**:
       - Incluya country_holidays para capturar automáticamente efectos de días festivos nacionales.
       - Añada custom_holidays para eventos específicos relevantes para ransomware.
    
    5. **Regresores externos**:
       - Incluya variables que puedan tener relación causal con ataques de ransomware.
       - Asegúrese de que los regresores estén disponibles para todo el período de predicción.
    
    6. **Intervalos de predicción**:
       - Ajuste interval_width según su tolerancia al riesgo (0.8 = 80% de confianza).
       - Valores más altos producen intervalos más amplios pero con mayor cobertura.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generación de Predicciones
    st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
    st.header("6️⃣ Generación de Predicciones", anchor="prediccion")
    st.markdown("""
    Una vez entrenado el modelo, el siguiente paso es generar predicciones para períodos futuros.
    Esta sección explica cómo configurar y ejecutar el proceso de predicción.
    
    ### Función: `generate_forecast_wrapper`
    
    Esta función genera predicciones utilizando un modelo Prophet previamente entrenado.
    
    #### Parámetros principales:
    
    - **model**: Modelo Prophet entrenado.
      - Debe ser un modelo previamente entrenado con `train_model_wrapper`.
    
    - **periods**: Número de períodos futuros a predecir.
      - Para datos diarios, esto equivale al número de días.
      - Valor predeterminado: 30
    
    - **freq**: Frecuencia de las predicciones.
      - 'D' para diario, 'W' para semanal, 'M' para mensual, etc.
      - Valor predeterminado: 'D'
    
    - **include_history**: Si incluir datos históricos en las predicciones.
      - Útil para visualizar la continuidad entre datos históricos y predicciones.
      - Valor predeterminado: True
    
    - **future_regressors**: Diccionario con valores futuros para regresores externos.
      - Clave: nombre del regresor, Valor: lista o array con valores futuros.
      - Ejemplo: {'cve_count': [10, 12, 8, ...]}
      - Valor predeterminado: None
    """)
    
    st.markdown('<div class="guide-code">', unsafe_allow_html=True)
    st.markdown("""
    ```python
    # Predicción básica para los próximos 30 días
    forecast = generate_forecast_wrapper(model, periods=30)
    
    # Predicción semanal para los próximos 3 meses
    forecast = generate_forecast_wrapper(model, periods=12, freq='W')
    
    # Predicción con regresores externos futuros
    future_cve_counts = [10, 12, 15, 8, 9, 11, 14] * 4  # Valores para 28 días
    forecast = generate_forecast_wrapper(
        model,
        periods=28,
        future_regressors={'cve_count': future_cve_counts}
    )
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="guide-tip">', unsafe_allow_html=True)
    st.markdown("""
    #### Componentes de las predicciones:
    
    El DataFrame de predicciones (`forecast`) contiene varias columnas importantes:
    
    1. **ds**: Fechas para las que se generan predicciones.
    2. **yhat**: Valor predicho (la mejor estimación puntual).
    3. **yhat_lower**: Límite inferior del intervalo de predicción.
    4. **yhat_upper**: Límite superior del intervalo de predicción.
    5. **trend**: Componente de tendencia de la predicción.
    6. Componentes estacionales: weekly, yearly, etc. (si están activados).
    7. Componentes de días festivos (si están incluidos).
    8. Componentes de regresores (si se utilizaron).
    
    #### Consejos para predicciones efectivas:
    
    1. **Horizonte de predicción**:
       - Sea conservador con el horizonte de predicción (periods).
       - La precisión disminuye naturalmente a medida que se predice más lejos en el futuro.
       - Para ransomware, horizontes de 30-60 días suelen ser razonables.
    
    2. **Regresores externos**:
       - Si utilizó regresores durante el entrenamiento, debe proporcionar valores futuros.
       - Para variables como CVEs, considere usar promedios históricos o tendencias si no tiene proyecciones.
    
    3. **Interpretación de intervalos**:
       - Recuerde que yhat_lower y yhat_upper definen el rango de valores probables.
       - Para planificación de contingencia, considere el escenario de yhat_upper (peor caso).
    
    4. **Validación de predicciones**:
       - Compare las primeras predicciones con valores reales a medida que estén disponibles.
       - Reentrenar el modelo si se observan desviaciones significativas.
    
    5. **Actualización regular**:
       - Para mayor precisión, actualice las predicciones regularmente con nuevos datos.
       - Considere un proceso automatizado de predicción diaria o semanal.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Evaluación del Modelo
    st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
    st.header("7️⃣ Evaluación del Modelo", anchor="evaluacion")
    st.markdown("""
    La evaluación del modelo es un paso crítico para determinar la precisión y fiabilidad de las predicciones.
    Nuestro sistema ofrece herramientas robustas para evaluar el rendimiento del modelo de predicción de ransomware.
    
    ### Función: `evaluate_model_wrapper`
    
    Esta función evalúa el rendimiento del modelo utilizando validación cruzada temporal y calcula métricas de error.
    
    #### Parámetros:
    
    - **model**: Modelo Prophet entrenado.
      - Debe ser un modelo previamente entrenado con `train_model_wrapper`.
    
    - **df**: DataFrame con los datos originales.
      - Debe contener al menos las columnas 'ds' (fechas) y 'y' (valores).
    
    - **initial**: Proporción inicial de datos para entrenamiento (0-1).
      - Por ejemplo, 0.5 significa usar el primer 50% de los datos para entrenamiento inicial.
      - Valor predeterminado: 0.5
    
    - **period**: Número de puntos de datos entre cada evaluación.
      - Define el paso entre cada iteración de validación.
      - Valor predeterminado: 30
    
    - **horizon**: Número de puntos de datos a predecir en cada iteración.
      - Define cuántos puntos futuros se predicen en cada paso de validación.
      - Valor predeterminado: 30
    
    - **parallel**: Si ejecutar la validación cruzada en paralelo.
      - Puede acelerar significativamente el proceso en sistemas multicore.
      - Valor predeterminado: False
    """)
    
    st.markdown('<div class="guide-code">', unsafe_allow_html=True)
    st.markdown("""
    ```python
    # Evaluación básica con parámetros predeterminados
    cv_results = evaluate_model_wrapper(model, df)
    
    # Evaluación personalizada
    cv_results = evaluate_model_wrapper(
        model,
        df,
        initial=0.6,  # Usar el primer 60% para entrenamiento inicial
        period=14,    # Evaluar cada 14 días
        horizon=30    # Predecir 30 días en cada iteración
    )
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="guide-tip">', unsafe_allow_html=True)
    st.markdown("""
    #### Cómo funciona la validación cruzada temporal:
    
    A diferencia de la validación cruzada tradicional, la validación cruzada temporal respeta el orden cronológico de los datos:
    
    1. **Proceso secuencial**:
       - Se entrena con datos hasta un punto de corte inicial (definido por `initial`)
       - Se predice el horizonte futuro (definido por `horizon`)
       - Se comparan las predicciones con los valores reales
       - Se avanza el punto de corte (según `period`) y se repite
    
    2. **Ventajas**:
       - Simula escenarios de predicción reales
       - Evita "mirar hacia el futuro" durante el entrenamiento
       - Proporciona una evaluación más realista del rendimiento
    
    #### Métricas de evaluación:
    
    El sistema calcula varias métricas para evaluar diferentes aspectos del rendimiento del modelo:
    
    1. **SMAPE (Symmetric Mean Absolute Percentage Error)**:
       - Mide el error porcentual absoluto de forma simétrica
       - Fórmula: 2 * |actual - predicción| / (|actual| + |predicción|) * 100
       - Ventaja: Funciona bien incluso con valores cercanos a cero
       - Interpretación: Valores más bajos indican mejor rendimiento (0% es perfecto)
       - Rango: 0% a 200%
    
    2. **MAE (Mean Absolute Error)**:
       - Mide el error absoluto promedio
       - Fórmula: promedio(|actual - predicción|)
       - Ventaja: Fácil de interpretar, en las mismas unidades que los datos
       - Interpretación: Cuánto se desvía en promedio la predicción del valor real
    
    3. **RMSE (Root Mean Square Error)**:
       - Mide el error cuadrático medio
       - Fórmula: raíz(promedio((actual - predicción)²))
       - Ventaja: Penaliza errores grandes más que errores pequeños
       - Interpretación: Similar a MAE pero con mayor peso a errores grandes
    
    4. **MSE (Mean Square Error)**:
       - Mide el error cuadrático medio sin la raíz cuadrada
       - Fórmula: promedio((actual - predicción)²)
       - Ventaja: Útil para comparaciones matemáticas
       - Interpretación: Valores más bajos indican mejor rendimiento
    
    5. **Coverage (Cobertura de intervalos)**:
       - Porcentaje de valores reales que caen dentro del intervalo de predicción
       - Fórmula: (valores dentro del intervalo) / (total de valores) * 100
       - Interpretación: Debería ser cercano al nivel de confianza especificado (ej. 80% o 95%)
       - Valores muy bajos indican intervalos demasiado estrechos
       - Valores muy altos indican intervalos demasiado amplios
    
    #### Interpretación de los resultados de evaluación:
    
    1. **Análisis de métricas**:
       - **SMAPE < 10%**: Excelente precisión
       - **SMAPE 10-20%**: Buena precisión
       - **SMAPE 20-50%**: Precisión moderada
       - **SMAPE > 50%**: Baja precisión, considere ajustar el modelo
    
    2. **Análisis de horizonte**:
       - Examine cómo cambian las métricas a medida que aumenta el horizonte
       - Es normal que el error aumente con horizontes más largos
       - Si el error aumenta drásticamente, considere limitar el horizonte de predicción
    
    3. **Análisis de cobertura**:
       - Idealmente, la cobertura debería ser cercana al nivel de confianza del intervalo
       - Cobertura significativamente menor: intervalos demasiado estrechos
       - Cobertura significativamente mayor: intervalos demasiado amplios
    
    #### Backtesting (Prueba retrospectiva):
    
    Nuestro sistema también ofrece capacidades de backtesting para simular cómo habría funcionado el modelo en el pasado:
    
    1. **Selección de fecha de corte**:
       - Elija una fecha histórica como punto de corte
       - El modelo se entrena con datos hasta esa fecha
       - Se generan predicciones a partir de esa fecha
    
    2. **Comparación visual**:
       - Los valores reales se muestran junto con las predicciones
       - Permite evaluar visualmente la precisión del modelo
    
    3. **Métricas específicas**:
       - Se calculan métricas de error solo para el período de prueba
       - Proporciona una evaluación realista del rendimiento esperado
    
    #### Consejos para la evaluación del modelo:
    
    - **Balance entre ajuste y generalización**:
      - Un modelo con errores muy bajos en datos de entrenamiento pero altos en validación está sobreajustado
      - Busque un balance que generalice bien a datos nuevos
    
    - **Ajuste basado en métricas**:
      - Si SMAPE es alto, considere ajustar parámetros como `changepoint_prior_scale`
      - Si la cobertura es baja, aumente `interval_width`
    
    - **Evaluación contextual**:
      - Compare el rendimiento con la variabilidad inherente de los datos
      - Para series muy volátiles, incluso un SMAPE del 30% puede ser aceptable
    
    - **Múltiples horizontes**:
      - Evalúe el modelo con diferentes horizontes de predicción
      - Determine hasta qué punto futuro las predicciones son confiables
    
    - **Validación externa**:
      - Cuando sea posible, valide con datos completamente nuevos
      - Especialmente importante para decisiones críticas de seguridad
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Visualización de Resultados
    st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
    st.header("8️⃣ Visualización de Resultados", anchor="visualizacion")
    st.markdown("""
    La visualización de resultados es fundamental para comprender e interpretar las predicciones generadas por el modelo.
    Prophet ofrece potentes herramientas de visualización que permiten examinar tanto las predicciones como los componentes
    subyacentes del modelo.
    
    ### Función: `visualize_forecast_wrapper`
    
    Esta función genera visualizaciones interactivas de las predicciones y los componentes del modelo.
    
    #### Parámetros:
    
    - **forecast**: DataFrame con las predicciones generadas por `generate_forecast_wrapper`.
      - Debe contener al menos las columnas 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
    
    - **df_original**: DataFrame con los datos originales.
      - Utilizado para mostrar los valores históricos junto con las predicciones.
    
    - **components**: Si visualizar los componentes del modelo (tendencia, estacionalidad).
      - Permite descomponer la predicción en sus componentes básicos.
      - Valor predeterminado: True
    
    - **plot_cap**: Si mostrar la capacidad máxima en el gráfico (si se definió).
      - Relevante solo si se utilizó un límite superior en el modelo.
      - Valor predeterminado: False
    
    - **changepoints**: Si mostrar los puntos de cambio detectados.
      - Muestra dónde el modelo identificó cambios significativos en la tendencia.
      - Valor predeterminado: True
    """)
    
    st.markdown('<div class="guide-code">', unsafe_allow_html=True)
    st.markdown("""
    ```python
    # Visualizar predicciones con configuración predeterminada
    fig = visualize_forecast_wrapper(forecast, df_original)
    
    # Visualizar sin componentes
    fig = visualize_forecast_wrapper(forecast, df_original, components=False)
    
    # Visualizar con puntos de cambio destacados
    fig = visualize_forecast_wrapper(
        forecast,
        df_original,
        changepoints=True
    )
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="guide-tip">', unsafe_allow_html=True)
    st.markdown("""
    #### Cómo funciona la visualización en Prophet:
    
    Prophet utiliza un enfoque de descomposición para modelar series temporales, lo que facilita visualizar cada componente por separado:
    
    1. **Descomposición aditiva o multiplicativa**:
       - Aditiva: y(t) = tendencia(t) + estacionalidad(t) + vacaciones(t) + error(t)
       - Multiplicativa: y(t) = tendencia(t) × estacionalidad(t) × vacaciones(t) × error(t)
    
    2. **Componentes visualizados**:
       - **Tendencia**: Representa la dirección general a largo plazo
       - **Estacionalidad**: Patrones cíclicos predecibles (diarios, semanales, anuales)
       - **Efectos de días festivos**: Impacto de eventos especiales
       - **Regresores externos**: Influencia de variables adicionales (como datos de CVE)
    
    #### Elementos clave en las visualizaciones:
    
    - **Gráfico principal de predicción**:
      - **Puntos negros**: Datos históricos reales
      - **Línea azul**: Predicción (yhat)
      - **Área sombreada azul**: Intervalo de confianza (yhat_lower a yhat_upper)
      - **Líneas verticales rojas** (opcional): Puntos de cambio detectados
    
    - **Gráficos de componentes**:
      - **Tendencia**: Muestra la dirección general sin componentes estacionales
      - **Estacionalidad anual**: Patrones que se repiten cada año
      - **Estacionalidad semanal**: Patrones que se repiten cada semana
      - **Estacionalidad diaria**: Patrones que se repiten cada día
    
    #### Visualizaciones interactivas en la interfaz:
    
    Nuestro sistema mejora las visualizaciones básicas de Prophet con características interactivas:
    
    1. **Zoom y desplazamiento**: Puede acercar áreas específicas de interés
    2. **Información al pasar el cursor**: Muestra valores exactos al pasar el cursor sobre el gráfico
    3. **Leyenda interactiva**: Permite mostrar/ocultar componentes específicos
    4. **Exportación**: Puede descargar gráficos como imágenes PNG
    5. **Personalización**: Ajuste de colores y estilos para mejor visibilidad
    
    #### Pasos para visualizar resultados en la interfaz:
    
    1. **Genere predicciones primero**:
       - Asegúrese de haber completado los pasos de carga de datos, entrenamiento y predicción
    
    2. **Explore las visualizaciones generadas automáticamente**:
       - Gráfico principal de predicción con datos históricos y proyecciones futuras
       - Gráficos de componentes que muestran tendencia y patrones estacionales
    
    3. **Interactúe con los gráficos**:
       - Use las herramientas de zoom para examinar períodos específicos
       - Pase el cursor sobre los puntos para ver valores exactos
       - Utilice la leyenda para mostrar/ocultar componentes
    
    4. **Analice los componentes por separado**:
       - Examine la tendencia para entender la dirección general
       - Observe patrones estacionales para identificar ciclos recurrentes
       - Identifique puntos de cambio donde ocurrieron cambios significativos
    
    #### Consejos para la visualización:
    
    - **Intervalos de confianza**: Preste atención a la amplitud de los intervalos; más amplios indican mayor incertidumbre
    - **Puntos de cambio**: Investigue qué eventos podrían haber causado cambios significativos en la tendencia
    - **Patrones estacionales**: Identifique días de la semana, meses o períodos con mayor actividad de ransomware
    - **Anomalías**: Busque puntos históricos que se desvíen significativamente de la predicción
    - **Exportación**: Guarde visualizaciones importantes para incluirlas en informes o presentaciones
    
    #### Interpretación avanzada de visualizaciones:
    
    - **Tendencia creciente/decreciente**: Indica aumento/disminución general en la actividad de ransomware
    - **Estacionalidad fuerte**: Sugiere patrones predecibles que pueden ayudar en la planificación
    - **Puntos de cambio frecuentes**: Indican una serie temporal volátil con cambios frecuentes en comportamiento
    - **Intervalos amplios**: Sugieren alta incertidumbre, posiblemente debido a datos limitados o alta variabilidad
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Interpretación de Resultados
    st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
    st.header("9️⃣ Interpretación de Resultados", anchor="interpretacion")
    st.markdown("""
    La interpretación correcta de los resultados es crucial para tomar decisiones informadas basadas en las predicciones
    del modelo. Esta sección proporciona una guía detallada sobre cómo interpretar las predicciones de ransomware y
    sus componentes.
    
    ### Componentes principales de las predicciones
    
    Las predicciones generadas por Prophet contienen varios elementos clave que deben interpretarse correctamente:
    
    #### 1. Predicción principal (yhat)
    
    - **Definición**: Es el valor esperado de la predicción en cada punto temporal.
    - **Interpretación**: Representa el número estimado de ataques de ransomware para cada fecha futura.
    - **Uso práctico**: Utilice estos valores para planificar recursos y estrategias de mitigación.
    
    #### 2. Intervalos de confianza (yhat_lower, yhat_upper)
    
    - **Definición**: Definen el rango dentro del cual se espera que caiga el valor real con cierta probabilidad.
    - **Interpretación**: 
      - **yhat_lower**: Límite inferior del intervalo de confianza (escenario optimista).
      - **yhat_upper**: Límite superior del intervalo de confianza (escenario pesimista).
    - **Uso práctico**: 
      - Planificación de contingencia basada en el peor escenario (yhat_upper).
      - Evaluación de la incertidumbre de la predicción (amplitud del intervalo).
    
    #### 3. Componentes de la predicción
    
    - **Tendencia**: 
      - Dirección general a largo plazo de los ataques de ransomware.
      - Un aumento indica crecimiento en la actividad criminal, una disminución sugiere reducción.
    
    - **Estacionalidad**: 
      - Patrones cíclicos predecibles en diferentes escalas temporales.
      - Identifica períodos de mayor o menor riesgo (días de la semana, meses, etc.).
    
    - **Días festivos/Eventos especiales**: 
      - Impacto de eventos específicos en la actividad de ransomware.
      - Útil para identificar períodos de vulnerabilidad especial.
    
    ### Guía para la interpretación práctica
    
    #### Análisis de tendencias
    
    1. **Tendencia creciente**:
       - **Interpretación**: Aumento sostenido en la actividad de ransomware.
       - **Acción recomendada**: Incrementar recursos de seguridad, actualizar defensas y concientizar al personal.
    
    2. **Tendencia decreciente**:
       - **Interpretación**: Disminución en la actividad de ransomware.
       - **Acción recomendada**: Mantener vigilancia, pero posible oportunidad para reasignar algunos recursos.
    
    3. **Tendencia estable**:
       - **Interpretación**: Nivel constante de actividad de ransomware.
       - **Acción recomendada**: Mantener estrategias actuales de defensa y monitoreo.
    
    4. **Cambios abruptos en la tendencia**:
       - **Interpretación**: Posible cambio en tácticas de atacantes o nueva vulnerabilidad.
       - **Acción recomendada**: Investigar causas subyacentes y ajustar defensas según sea necesario.
    
    #### Análisis de estacionalidad
    
    1. **Patrones semanales**:
       - **Ejemplo**: Mayor actividad en días laborables vs. fines de semana.
       - **Acción recomendada**: Ajustar niveles de personal de seguridad según el día de la semana.
    
    2. **Patrones mensuales/trimestrales**:
       - **Ejemplo**: Aumento de actividad al final del trimestre fiscal.
       - **Acción recomendada**: Incrementar vigilancia durante períodos de alto riesgo identificados.
    
    3. **Patrones anuales**:
       - **Ejemplo**: Mayor actividad durante temporadas de vacaciones.
       - **Acción recomendada**: Planificar recursos adicionales para períodos anuales de alto riesgo.
    
    #### Análisis de incertidumbre
    
    1. **Intervalos estrechos**:
       - **Interpretación**: Alta confianza en la predicción.
       - **Acción recomendada**: Planificación más precisa basada en valores previstos.
    
    2. **Intervalos amplios**:
       - **Interpretación**: Alta incertidumbre en la predicción.
       - **Acción recomendada**: Preparar múltiples escenarios y mantener flexibilidad en la respuesta.
    
    3. **Ampliación de intervalos con el tiempo**:
       - **Interpretación**: Aumento de incertidumbre a medida que se predice más lejos en el futuro.
       - **Acción recomendada**: Mayor cautela con predicciones a largo plazo, actualizar frecuentemente.
    
    ### Interpretación avanzada
    
    #### Correlación con factores externos
    
    1. **Regresores adicionales**:
       - Si se incluyeron variables externas (como datos de CVE), analice su impacto.
       - Identifique qué factores tienen mayor influencia en las predicciones.
    
    2. **Eventos no modelados**:
       - Considere factores que el modelo no incorpora (nuevas tecnologías, cambios regulatorios).
       - Ajuste interpretaciones según conocimiento experto no capturado por el modelo.
    
    #### Análisis de anomalías
    
    1. **Valores atípicos históricos**:
       - Identifique puntos donde los datos reales se desviaron significativamente de las predicciones.
       - Investigue causas subyacentes para mejorar predicciones futuras.
    
    2. **Predicciones extremas**:
       - Evalúe críticamente predicciones inusualmente altas o bajas.
       - Considere factores contextuales que podrían explicar o contradecir estos valores.
    
    ### Comunicación de resultados
    
    1. **Audiencia técnica**:
       - Proporcione métricas detalladas, componentes descompuestos y análisis estadístico.
       - Discuta limitaciones del modelo y fuentes de incertidumbre.
    
    2. **Audiencia no técnica**:
       - Enfóquese en tendencias generales y patrones claros.
       - Traduzca predicciones a recomendaciones accionables.
       - Utilice visualizaciones intuitivas con explicaciones claras.
    
    3. **Toma de decisiones**:
       - Presente múltiples escenarios (mejor caso, caso esperado, peor caso).
       - Vincule predicciones con acciones concretas de mitigación.
       - Enfatice el nivel de confianza en diferentes aspectos de la predicción.
    
    ### Limitaciones y consideraciones
    
    1. **Cambios estructurales**:
       - El modelo asume que los patrones pasados continuarán en el futuro.
       - Esté atento a cambios fundamentales en el panorama de amenazas.
    
    2. **Eventos sin precedentes**:
       - Las predicciones pueden ser menos confiables durante eventos disruptivos (como pandemias).
       - Complemente el modelo con análisis cualitativo en situaciones sin precedentes.
    
    3. **Horizonte de predicción**:
       - La confiabilidad disminuye a medida que se predice más lejos en el futuro.
       - Considere horizontes más cortos para decisiones críticas.
    
    4. **Causalidad vs. correlación**:
       - El modelo identifica patrones, no necesariamente relaciones causales.
       - Use conocimiento del dominio para interpretar relaciones identificadas.
    
    ### Ejemplo práctico de interpretación
    
    Supongamos que tenemos las siguientes predicciones para la próxima semana:
    
    | Fecha       | yhat | yhat_lower | yhat_upper |
    |-------------|------|------------|------------|
    | 2023-06-01  | 25   | 18         | 32         |
    | 2023-06-02  | 30   | 22         | 38         |
    | 2023-06-03  | 15   | 8          | 22         |
    | 2023-06-04  | 12   | 5          | 19         |
    | 2023-06-05  | 28   | 20         | 36         |
    
    **Interpretación**:
    
    1. **Patrón semanal**: Se observa menor actividad en fin de semana (03-04) comparado con días laborables.
    2. **Nivel de alerta**: Planificar mayor vigilancia el viernes (02) cuando se espera el pico de actividad.
    3. **Planificación de recursos**: 
       - Escenario esperado: Prepararse para hasta 30 incidentes el viernes.
       - Peor escenario: Tener capacidad para manejar hasta 38 incidentes.
    4. **Incertidumbre**: El rango entre yhat_lower y yhat_upper es de aproximadamente ±8 incidentes, lo que indica un nivel moderado de incertidumbre.
    
    **Acciones recomendadas**:
    
    1. Asignar personal adicional para los días con predicciones más altas (viernes y lunes).
    2. Implementar monitoreo intensificado durante estos días de alto riesgo.
    3. Programar actualizaciones de seguridad críticas para el domingo, cuando se espera menor actividad.
    4. Preparar un plan de contingencia para manejar hasta 38 incidentes el viernes (peor escenario).
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Optimizaciones Avanzadas
    st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
    st.header("🔧 Optimizaciones Avanzadas", anchor="optimizaciones")
    st.markdown("""
    Para obtener el máximo rendimiento del modelo de predicción de ransomware, es posible aplicar diversas
    optimizaciones avanzadas. Esta sección detalla las técnicas disponibles para mejorar la precisión y
    fiabilidad de las predicciones.
    
    ### Función: `optimize_hyperparameters_wrapper`
    
    Esta función realiza una búsqueda sistemática de los mejores hiperparámetros para el modelo Prophet.
    
    #### Parámetros:
    
    - **df**: DataFrame con los datos de entrenamiento.
      - Debe contener al menos las columnas 'ds' (fechas) y 'y' (valores).
    
    - **param_grid**: Diccionario con los rangos de hiperparámetros a probar.
      - Define el espacio de búsqueda para la optimización.
      - Ejemplo: `{'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5], 'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]}`
    
    - **cv_horizon**: Horizonte para validación cruzada en días.
      - Define cuántos días hacia el futuro se evaluará en cada iteración.
      - Valor predeterminado: 30
    
    - **cv_period**: Período entre cortes de validación cruzada.
      - Define la distancia entre puntos de corte consecutivos.
      - Valor predeterminado: 30
    
    - **metric**: Métrica a optimizar ('rmse', 'mae', 'smape', 'mape', 'coverage').
      - Define qué métrica se utilizará para seleccionar los mejores hiperparámetros.
      - Valor predeterminado: 'rmse'
    
    - **parallel**: Si ejecutar la optimización en paralelo.
      - Puede acelerar significativamente el proceso en sistemas multicore.
      - Valor predeterminado: False
    """)
    
    st.markdown('<div class="guide-code">', unsafe_allow_html=True)
    st.markdown("""
    ```python
    # Optimización básica con parámetros predeterminados
    best_params = optimize_hyperparameters_wrapper(df)
    
    # Optimización personalizada
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    best_params = optimize_hyperparameters_wrapper(
        df,
        param_grid=param_grid,
        cv_horizon=60,
        metric='smape'
    )
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="guide-tip">', unsafe_allow_html=True)
    st.markdown("""
    #### Hiperparámetros clave de Prophet
    
    Prophet tiene varios hiperparámetros que pueden ajustarse para mejorar el rendimiento del modelo:
    
    1. **changepoint_prior_scale**:
       - **Función**: Controla la flexibilidad de la tendencia.
       - **Valores más altos**: Permiten cambios más abruptos en la tendencia.
       - **Valores más bajos**: Producen una tendencia más suave y estable.
       - **Rango típico**: 0.001 - 0.5
       - **Impacto**: Afecta principalmente a la capacidad del modelo para capturar cambios en la tendencia.
    
    2. **seasonality_prior_scale**:
       - **Función**: Controla la flexibilidad de los componentes estacionales.
       - **Valores más altos**: Permiten patrones estacionales más fuertes.
       - **Valores más bajos**: Suavizan los patrones estacionales.
       - **Rango típico**: 0.01 - 10.0
       - **Impacto**: Afecta a la magnitud de los patrones estacionales detectados.
    
    3. **holidays_prior_scale**:
       - **Función**: Controla el impacto de los días festivos y eventos especiales.
       - **Valores más altos**: Dan más peso a los efectos de días festivos.
       - **Valores más bajos**: Reducen la influencia de los días festivos.
       - **Rango típico**: 0.01 - 10.0
       - **Impacto**: Crucial cuando los días festivos tienen un efecto significativo.
    
    4. **seasonality_mode**:
       - **Función**: Define cómo se combinan los componentes estacionales con la tendencia.
       - **'additive'**: Los efectos estacionales son constantes en magnitud.
       - **'multiplicative'**: Los efectos estacionales escalan con la tendencia.
       - **Cuándo usar**: 'multiplicative' es mejor cuando la estacionalidad aumenta con el nivel de la serie.
    
    5. **interval_width**:
       - **Función**: Define el ancho de los intervalos de predicción.
       - **Valor típico**: 0.8 (80%) o 0.95 (95%)
       - **Impacto**: Afecta a la cobertura de los intervalos de confianza.
    
    6. **n_changepoints**:
       - **Función**: Número de puntos de cambio potenciales en la tendencia.
       - **Valores más altos**: Permiten más flexibilidad en la tendencia.
       - **Valores más bajos**: Producen una tendencia más estable.
       - **Valor predeterminado**: 25 para series de más de un año.
    
    #### Estrategias de optimización
    
    1. **Búsqueda en cuadrícula (Grid Search)**:
       - **Descripción**: Evalúa todas las combinaciones posibles de hiperparámetros.
       - **Ventajas**: Exhaustiva, garantiza encontrar el mejor conjunto dentro del espacio definido.
       - **Desventajas**: Computacionalmente costosa para espacios grandes.
       - **Cuándo usar**: Cuando el espacio de búsqueda es relativamente pequeño.
    
    2. **Búsqueda aleatoria (Random Search)**:
       - **Descripción**: Evalúa combinaciones aleatorias dentro del espacio de hiperparámetros.
       - **Ventajas**: Más eficiente que la búsqueda en cuadrícula para espacios grandes.
       - **Desventajas**: No garantiza encontrar el óptimo global.
       - **Cuándo usar**: Cuando el espacio de búsqueda es grande.
    
    3. **Optimización bayesiana**:
       - **Descripción**: Utiliza modelos probabilísticos para dirigir la búsqueda.
       - **Ventajas**: Muy eficiente, aprende de evaluaciones anteriores.
       - **Desventajas**: Más compleja de implementar.
       - **Cuándo usar**: Para optimizaciones muy costosas computacionalmente.
    
    #### Técnicas avanzadas de modelado
    
    1. **Regresores externos**:
       - **Descripción**: Incorporar variables adicionales que puedan influir en la actividad de ransomware.
       - **Ejemplos**:
         - Número de vulnerabilidades (CVE) publicadas
         - Eventos de seguridad importantes
         - Indicadores económicos
       - **Implementación**:
         ```python
         # Añadir regresor al modelo
         model = Prophet()
         model.add_regressor('cve_count')
         ```
    
    2. **Detección y tratamiento de outliers**:
       - **Descripción**: Identificar y manejar valores atípicos que puedan distorsionar el modelo.
       - **Técnicas**:
         - Filtrado basado en desviaciones estándar
         - Winsorización (recorte de valores extremos)
         - Imputación de valores
       - **Implementación**:
         ```python
         # Detectar outliers usando el método IQR
         Q1 = df['y'].quantile(0.25)
         Q3 = df['y'].quantile(0.75)
         IQR = Q3 - Q1
         df_filtered = df[(df['y'] >= Q1 - 1.5 * IQR) & (df['y'] <= Q3 + 1.5 * IQR)]
         ```
    
    3. **Transformaciones de datos**:
       - **Descripción**: Aplicar transformaciones matemáticas para mejorar las propiedades de los datos.
       - **Técnicas comunes**:
         - Transformación logarítmica: para series con crecimiento exponencial
         - Transformación de Box-Cox: para estabilizar la varianza
         - Diferenciación: para hacer la serie estacionaria
       - **Implementación**:
         ```python
         # Transformación logarítmica (añadiendo 1 para manejar ceros)
         df['y'] = np.log1p(df['y'])
         
         # Recordar invertir la transformación después de la predicción
         forecast['yhat'] = np.expm1(forecast['yhat'])
         ```
    
    4. **Modelado jerárquico**:
       - **Descripción**: Combinar predicciones de múltiples modelos a diferentes niveles de agregación.
       - **Ejemplo**: Modelar ataques por tipo de ransomware y luego agregar.
       - **Ventajas**: Puede capturar patrones específicos de cada categoría.
    
    5. **Ensamblado de modelos**:
       - **Descripción**: Combinar predicciones de múltiples modelos para mejorar la precisión.
       - **Técnicas**:
         - Promedio simple de predicciones
         - Promedio ponderado basado en rendimiento histórico
         - Stacking (usar un modelo para combinar predicciones de otros)
    
    #### Optimización del flujo de trabajo
    
    1. **Actualización continua del modelo**:
       - **Descripción**: Reentrenar regularmente el modelo con nuevos datos.
       - **Frecuencia recomendada**: Diaria o semanal, dependiendo de la volatilidad.
       - **Ventajas**: Mantiene el modelo actualizado con los patrones más recientes.
    
    2. **Monitoreo de rendimiento**:
       - **Descripción**: Evaluar continuamente la precisión del modelo en nuevos datos.
       - **Métricas clave**: SMAPE, MAE, cobertura de intervalos.
       - **Acción**: Reoptimizar hiperparámetros si el rendimiento se degrada.
    
    3. **Validación con expertos**:
       - **Descripción**: Contrastar predicciones con conocimiento experto en seguridad.
       - **Beneficio**: Identificar predicciones contraintuitivas que puedan indicar problemas.
    
    #### Consideraciones computacionales
    
    1. **Paralelización**:
       - La optimización de hiperparámetros puede ser computacionalmente intensiva.
       - Utilice el parámetro `parallel=True` para acelerar el proceso en sistemas multicore.
    
    2. **Muestreo para pruebas rápidas**:
       - Durante la fase de desarrollo, considere usar un subconjunto de datos.
       - Una vez identificados los mejores hiperparámetros, reentrenar con todos los datos.
    
    3. **Almacenamiento en caché**:
       - Nuestro sistema utiliza el caché de Streamlit para evitar recálculos innecesarios.
       - Los resultados de optimización se almacenan para referencia futura.
    
    #### Ejemplo de flujo de trabajo optimizado
    
    1. **Preparación de datos**:
       - Detectar y tratar outliers
       - Aplicar transformaciones apropiadas
    
    2. **Optimización inicial**:
       - Realizar una búsqueda amplia de hiperparámetros
       - Identificar rangos prometedores
    
    3. **Refinamiento**:
       - Realizar una búsqueda más detallada en los rangos prometedores
       - Seleccionar los mejores hiperparámetros
    
    4. **Entrenamiento final**:
       - Entrenar el modelo con los mejores hiperparámetros y todos los datos
       - Incluir regresores externos relevantes
    
    5. **Validación**:
       - Realizar backtesting para verificar rendimiento
       - Contrastar con conocimiento experto
    
    6. **Despliegue y monitoreo**:
       - Implementar el modelo optimizado
       - Establecer monitoreo continuo
       - Reentrenar regularmente con nuevos datos
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Preguntas Frecuentes
st.markdown(f'<div class="guide-section">', unsafe_allow_html=True)
st.header("❓ Preguntas Frecuentes", anchor="faq")
st.markdown("""
### Preguntas Generales

#### ¿Qué es exactamente este modelo de predicción de ransomware?

Este modelo utiliza Prophet, una biblioteca de pronóstico de series temporales desarrollada por Facebook (Meta), 
para predecir la actividad futura de ataques de ransomware basándose en datos históricos. El modelo identifica 
patrones, tendencias y estacionalidad en los datos para generar predicciones con intervalos de confianza.

#### ¿Qué tan precisas son las predicciones?

La precisión depende de varios factores, incluyendo la calidad y cantidad de datos históricos, la volatilidad 
inherente de los ataques de ransomware, y la configuración del modelo. Típicamente, el modelo puede alcanzar 
un SMAPE (error porcentual absoluto simétrico medio) entre 15-30% para horizontes de predicción a corto plazo. 
La precisión disminuye naturalmente a medida que el horizonte de predicción se extiende más hacia el futuro.

#### ¿Puedo confiar en estas predicciones para tomar decisiones críticas de seguridad?

Las predicciones deben considerarse como una herramienta de apoyo a la toma de decisiones, no como una verdad 
absoluta. Recomendamos complementar estas predicciones con conocimiento experto en seguridad, inteligencia de 
amenazas actualizada y buenas prácticas de ciberseguridad. Los intervalos de confianza proporcionan una 
indicación de la incertidumbre asociada con cada predicción.

#### ¿Con qué frecuencia debo actualizar el modelo?

Recomendamos reentrenar el modelo al menos semanalmente con los datos más recientes. En entornos de alta 
volatilidad o cuando surgen nuevas amenazas significativas, puede ser beneficioso actualizar el modelo con 
mayor frecuencia. El sistema está diseñado para facilitar actualizaciones regulares con mínimo esfuerzo.

### Datos y Preparación

#### ¿Qué formato deben tener mis datos para usar este modelo?

Los datos deben estar en formato tabular con al menos dos columnas:
1. Una columna de fecha/tiempo (llamada 'ds', 'fecha' o 'date')
2. Una columna numérica con el recuento de ataques o incidentes (llamada 'y' o se renombrará automáticamente)

Los formatos aceptados incluyen CSV, Excel y JSON. La frecuencia de los datos puede ser diaria, semanal o mensual, 
pero debe ser consistente.

#### ¿Qué hago si tengo datos faltantes o valores atípicos?

El sistema incluye herramientas para detectar y tratar valores atípicos. Para datos faltantes, Prophet puede 
manejarlos naturalmente mediante interpolación. Sin embargo, para mejores resultados, considere:

- Para períodos cortos de datos faltantes: Utilice la funcionalidad de detección e imputación incluida.
- Para valores atípicos: Use las herramientas de detección de outliers y considere si representan eventos reales 
  (que deberían mantenerse) o errores (que podrían filtrarse).
- Para grandes brechas de datos: Considere si es apropiado modelar los períodos antes y después de la brecha por separado.

#### ¿Cuántos datos históricos necesito para obtener predicciones confiables?

Como regla general:

- Mínimo: Al menos 2-3 veces la longitud del ciclo estacional más largo que espera capturar. Por ejemplo, si 
  espera patrones anuales, idealmente necesitaría 2-3 años de datos.
- Óptimo: Para predicciones diarias con patrones semanales y anuales, 1-2 años de datos diarios proporcionan 
  un buen equilibrio.
- Consideración: Más datos no siempre son mejores si ha habido cambios fundamentales en los patrones de ataque. 
  En ese caso, considere usar solo datos posteriores al cambio significativo.

### Entrenamiento y Predicción

#### ¿Cómo elijo el horizonte de predicción adecuado?

El horizonte de predicción debe basarse en:

1. **Necesidades operativas**: ¿Para qué plazo necesita planificar?
2. **Calidad de datos**: Horizontes más largos requieren más datos históricos de calidad.
3. **Estabilidad del dominio**: En ciberseguridad, donde las tácticas evolucionan rápidamente, horizontes más 
   cortos (1-3 meses) suelen ser más confiables que predicciones a largo plazo.
4. **Resultados de validación**: Examine cómo se degradan las métricas de error a medida que aumenta el horizonte 
   en sus validaciones cruzadas.

Como punto de partida, recomendamos un horizonte de 30-60 días para predicciones de ransomware, ajustando según 
los resultados de validación.

#### ¿Qué significan todos estos hiperparámetros y cómo los ajusto?

Los hiperparámetros principales controlan diferentes aspectos del modelo:

- **changepoint_prior_scale**: Controla la flexibilidad de la tendencia. Valores más altos permiten cambios más 
  abruptos. Si ve que el modelo no captura bien cambios importantes en la tendencia, aumente este valor.

- **seasonality_prior_scale**: Controla la fuerza de los componentes estacionales. Valores más altos permiten 
  patrones estacionales más pronunciados. Si los patrones semanales o anuales parecen subestimados, aumente este valor.

- **holidays_prior_scale**: Controla el impacto de eventos especiales. Aumente este valor si ciertos días o 
  eventos tienen un impacto significativo en los ataques.

- **seasonality_mode**: Use 'multiplicative' si la magnitud de los patrones estacionales aumenta con el nivel 
  general de ataques, de lo contrario use 'additive'.

La función `optimize_hyperparameters_wrapper` puede ayudarle a encontrar automáticamente los mejores valores 
para estos parámetros.

#### ¿Por qué mis intervalos de predicción son tan amplios?

Los intervalos amplios indican alta incertidumbre, que puede deberse a:

1. **Alta variabilidad en los datos históricos**: Series con grandes fluctuaciones naturalmente producen 
   intervalos más amplios.
2. **Datos limitados**: Menos datos históricos generalmente resultan en mayor incertidumbre.
3. **Cambios estructurales recientes**: Si ha habido cambios fundamentales recientes, el modelo puede tener 
   dificultades para hacer predicciones confiables.
4. **Horizonte largo**: La incertidumbre aumenta naturalmente con horizontes de predicción más largos.

Para reducir la amplitud de los intervalos:
- Considere ajustar `interval_width` a un valor menor (por defecto es 0.8 o 80%)
- Optimice los hiperparámetros del modelo
- Incluya regresores externos relevantes si están disponibles
- Considere transformaciones de datos para estabilizar la varianza

### Evaluación e Interpretación

#### ¿Cómo sé si mi modelo está funcionando bien?

Evalúe su modelo utilizando múltiples enfoques:

1. **Métricas cuantitativas**:
   - SMAPE < 20% generalmente indica buen rendimiento para predicciones de ransomware
   - Cobertura de intervalos cercana al nivel de confianza especificado (ej. 80% para interval_width=0.8)

2. **Validación visual**:
   - Las predicciones siguen patrones históricos conocidos
   - Los componentes descompuestos (tendencia, estacionalidad) tienen sentido intuitivo

3. **Backtesting**:
   - El modelo predice con precisión períodos históricos conocidos cuando se entrena con datos hasta cierto punto

4. **Validación de expertos**:
   - Las predicciones se alinean con la intuición de expertos en seguridad
   - Las anomalías detectadas corresponden a eventos reales conocidos

#### ¿Cómo interpreto los componentes del modelo?

Los componentes principales que puede analizar son:

1. **Tendencia**: Representa la dirección general a largo plazo. Un aumento sostenido indica crecimiento en 
   la actividad de ransomware, mientras que una disminución sugiere reducción.

2. **Estacionalidad**: Muestra patrones cíclicos predecibles:
   - Estacionalidad semanal: Identifica días de la semana con mayor/menor actividad
   - Estacionalidad anual: Revela meses o temporadas con patrones distintivos

3. **Días festivos/Eventos**: Muestra el impacto de eventos específicos en la actividad de ransomware.

4. **Puntos de cambio**: Identifica momentos donde la tendencia cambió significativamente, que pueden 
   corresponder a nuevas variantes de ransomware, técnicas de ataque o contramedidas.

#### ¿Qué métricas de evaluación debo priorizar?

Dependiendo de su caso de uso:

- **SMAPE (Error Porcentual Absoluto Simétrico Medio)**: Buena métrica general que funciona bien incluso con 
  valores cercanos a cero. Priorice esta métrica para uso general.

- **MAE (Error Absoluto Medio)**: Útil cuando le interesa el error absoluto en número de ataques, independientemente 
  de la magnitud base.

- **Cobertura de intervalos**: Crítica si está utilizando los intervalos de confianza para planificación de 
  contingencia. Una cobertura cercana al nivel de confianza especificado indica intervalos bien calibrados.

- **RMSE (Error Cuadrático Medio)**: Útil cuando errores grandes son particularmente problemáticos, ya que 
  penaliza más los errores grandes que los pequeños.

- **MSE (Error Cuadrático Medio)**: Útil para comparaciones matemáticas.

### Problemas Comunes

#### El modelo no captura bien los picos extremos en mis datos

Prophet está diseñado para capturar tendencias generales y patrones estacionales, no necesariamente eventos 
extremos únicos. Para mejorar el manejo de picos:

1. Considere añadir eventos especiales usando la funcionalidad de días festivos de Prophet
2. Experimente con valores más altos de `changepoint_prior_scale` para permitir cambios más abruptos
3. Para eventos verdaderamente únicos, considere modelarlos por separado o marcarlos como outliers
4. Pruebe el modo de estacionalidad multiplicativa si los picos escalan con el nivel general

#### Las predicciones parecen demasiado suavizadas y no reflejan la volatilidad real

Si sus predicciones son demasiado "planas" comparadas con la volatilidad histórica:

1. Aumente `changepoint_prior_scale` para permitir más flexibilidad en la tendencia
2. Aumente `seasonality_prior_scale` para capturar patrones estacionales más fuertes
3. Considere si el modo de estacionalidad (`additive` vs `multiplicative`) es apropiado
4. Verifique si hay regresores externos que podrían explicar parte de la volatilidad

#### El rendimiento del modelo se degrada con el tiempo

Si nota que las predicciones recientes son menos precisas que las anteriores:

1. Reentrenar regularmente el modelo con datos nuevos
2. Considere si ha habido cambios fundamentales en el panorama de amenazas
3. Reevalúe y reoptimice los hiperparámetros periódicamente
4. Considere dar más peso a datos recientes o usar una ventana móvil de entrenamiento

#### ¿Cómo puedo incorporar información sobre nuevas vulnerabilidades o amenazas?

Para incorporar información sobre nuevas amenazas:

1. **Regresores externos**: Añada datos sobre conteo de CVEs, menciones en medios, u otros indicadores como regresores
2. **Eventos especiales**: Marque fechas de divulgación de vulnerabilidades importantes como "días festivos"
3. **Actualización frecuente**: Reentrenar el modelo cuando surjan nuevas amenazas significativas
4. **Ajuste manual**: Para amenazas muy recientes, considere ajustar manualmente las predicciones basándose en conocimiento experto

### Integración y Flujo de Trabajo

#### ¿Cómo puedo integrar estas predicciones en mi flujo de trabajo de seguridad?

Las predicciones pueden integrarse de varias maneras:

1. **Planificación de recursos**: Utilice las predicciones para ajustar la asignación de personal de seguridad
2. **Programación de actualizaciones**: Programe actualizaciones críticas durante períodos previstos de baja actividad
3. **Alertas proactivas**: Configure umbrales de alerta basados en predicciones para prepararse para períodos de alto riesgo
4. **Informes de riesgo**: Incluya predicciones en informes periódicos de riesgo para la dirección
5. **Automatización**: Ajuste automáticamente niveles de monitoreo basados en predicciones

#### ¿Puedo exportar los resultados para usarlos en otras herramientas?

Sí, los resultados pueden exportarse en varios formatos:

1. **CSV**: Para análisis en Excel, R, Python u otras herramientas
2. **JSON**: Para integración con aplicaciones web o dashboards
3. **Imágenes**: Los gráficos pueden guardarse como PNG para informes
4. **API**: Para usuarios avanzados, es posible configurar una API para acceder a predicciones en tiempo real

Utilice los botones de descarga en la interfaz para exportar datos y visualizaciones.

#### ¿Cómo puedo automatizar todo este proceso?

Para automatizar el flujo de trabajo:

1. **Scripts programados**: Cree scripts Python que ejecuten todo el proceso (carga, entrenamiento, predicción, evaluación)
2. **Tareas programadas**: Configure tareas programadas (cron jobs en Linux, Task Scheduler en Windows) para ejecutar los scripts
3. **Pipelines de datos**: Implemente pipelines que actualicen automáticamente los datos de entrada
4. **Notificaciones**: Configure alertas automáticas basadas en predicciones o cambios significativos
5. **Almacenamiento de resultados**: Guarde automáticamente predicciones y evaluaciones en una base de datos para seguimiento histórico

Para usuarios avanzados, considere implementar el modelo en un entorno de producción con CI/CD para actualizaciones continuas.
""")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


def main():
    show_user_guide()

if __name__ == "__main__":
    main()
