"""
Página de modelado avanzado de predicción de ransomware utilizando la versión modular.

Esta página implementa la nueva arquitectura modular del sistema de predicción
de ransomware a través del módulo de integración, manteniendo una interfaz
consistente con la implementación original.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import traceback  # Importación explícita de traceback para gestión de errores
import os
import sys
import json
import time
import datetime
from pathlib import Path

# Asegurar que podemos importar módulos desde el directorio principal
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

# Importar el módulo de integración
from modeling import integration

def modelado_modular_app():
    """
    Aplicación principal para el modelado predictivo de ataques ransomware.
    """
    # Estilos CSS personalizados
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem;
        flex: 1;
        min-width: 200px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    .full-width {
        width: 100%;
    }
    .centered {
        text-align: center;
    }
    
    /* Estilos para ocultar/mostrar la barra lateral */
    .hide-sidebar [data-testid="stSidebar"] {
        display: none !important;
        width: 0px !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !important;
    }
    .show-sidebar [data-testid="stSidebar"] {
        display: flex !important;
        width: 260px !important;
    }
    
    /* Estilos responsivos */
    @media (max-width: 768px) {
        .metric-container {
            padding: 8px;
        }
        .metric-value {
            font-size: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Añadir CSS personalizado para mejorar la visualización en tema oscuro
    st.markdown("""
    <style>
    /* Estilos generales para mejorar la legibilidad en tema oscuro */
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
    
    /* Estilos para los textos en columnas del flujo de trabajo */
    .element-container div.stMarkdown p {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Estilos específicos para los textos descriptivos en las secciones numeradas */
    div[data-testid="stVerticalBlock"] div.stMarkdown p {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Estilos para los textos en las secciones del flujo de trabajo */
    div.stMarkdown p {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Estilos para los textos en las columnas */
    div[data-testid="column"] div.stMarkdown p {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Estilos para todos los textos */
    .css-nahz7x p, .css-nahz7x li, .css-nahz7x span {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Estilos para los textos en las secciones numeradas */
    div[data-testid="stExpander"] div.stMarkdown p {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Título e introducción
    st.title("Modelado Predictivo de Ataques Ransomware")
    
    # Información básica visible directamente
    st.markdown("""
    <div class="info-box" style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
        <p>Sistema de predicción de tendencias futuras de ataques ransomware basado en datos históricos.</p>
    </div>
    """, unsafe_allow_html=True)
    
     # Menú como pestañas: Modelado vs Guía
    tab_modelado, tab_guide = st.tabs(["🛠 Modelado", "📖 Guía de Usuario"])

    # ── PESTAÑA: Guía de Usuario ───────────────────────────────
    with tab_guide:
        from modeling.user_guide import show_user_guide
        show_user_guide()

    # ── PESTAÑA: Modelado ───────────────────────────────────────
    with tab_modelado:
    
        # Si llegamos aquí, estamos en la vista de modelado
        # Inicializar variables de estado si no existen
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False

        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False

        if 'forecast_generated' not in st.session_state:
            st.session_state.forecast_generated = False
            
        if 'metrics_calculated' not in st.session_state:
            st.session_state.metrics_calculated = False
        
        # Enfoque y períodos
        if 'enfoque_actual' not in st.session_state:
            st.session_state.enfoque_actual = "conteo_diario"
        
        if 'periods' not in st.session_state:
            st.session_state.periods = 90  # Valor predeterminado
        
        # Inicializar parámetros del modelo si no existen
        if 'params' not in st.session_state:
            st.session_state.params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'interval_width': 0.95,
                'use_detected_changepoints': True,
                'use_optimal_regressors': True,
                'use_bayesian_optimization': True,
                'use_interval_calibration': True,
                'optimization_trials': 20,
                'correlation_threshold': 0.12,
                'vif_threshold': 5.0
            }
        
        # Cargar optimizaciones guardadas
        if 'optimizations' not in st.session_state:
            st.session_state.optimizations = {}
        
        # Visualización del flujo de trabajo (más clara y compacta)
        st.markdown('<div class="section-header"><h2>Estado del Proceso</h2></div>', unsafe_allow_html=True)
        
        # Contenedor de workflow con actualización según el estado
        workflow_status = {
            "data": "active" if not st.session_state.data_loaded else "completed",
            "train": "active" if st.session_state.data_loaded and not st.session_state.model_trained else ("completed" if st.session_state.model_trained else ""),
            "predict": "active" if st.session_state.model_trained and not st.session_state.forecast_generated else ("completed" if st.session_state.forecast_generated else ""),
            "evaluate": "active" if st.session_state.forecast_generated else ""
        }
        
        # Visualización más compacta y clara del flujo de trabajo
        st.markdown("""
        <style>
        .workflow-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding: 1rem;
            background-color: var(--background-light);
            border-radius: 0.5rem;
        }
        .workflow-step-container {
            display: flex;
            align-items: center;
        }
        .workflow-icon {
            margin-right: 10px;
            font-size: 1.5rem;
        }
        .workflow-status {
            margin-left: auto;
            font-weight: bold;
            border-radius: 1rem;
            padding: 0.3rem 0.7rem;
        }
        .status-pending {
            background-color: #f0f0f0;
            color: #777;
        }
        .status-active {
            background-color: #ffc107;
            color: #000;
        }
        .status-completed {
            background-color: #28a745;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Paso 1: Datos
        status_class = "status-completed" if workflow_status["data"] == "completed" else "status-active" if workflow_status["data"] == "active" else "status-pending"
        icon = "" if workflow_status["data"] == "completed" else "" if workflow_status["data"] == "active" else ""
        st.markdown(f"""
        <div class="workflow-row">
            <div class="workflow-step-container">
                <div class="workflow-icon">{icon}</div>
                <div><strong>1. Datos</strong> - Cargar y procesar datos históricos</div>
            </div>
            <div class="workflow-status {status_class}">
                {workflow_status["data"].upper() if workflow_status["data"] else "PENDIENTE"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Paso 2: Entrenamiento
        status_class = "status-completed" if workflow_status["train"] == "completed" else "status-active" if workflow_status["train"] == "active" else "status-pending"
        icon = "" if workflow_status["train"] == "completed" else "" if workflow_status["train"] == "active" else ""
        st.markdown(f"""
        <div class="workflow-row">
            <div class="workflow-step-container">
                <div class="workflow-icon">{icon}</div>
                <div><strong>2. Entrenamiento</strong> - Entrenar modelo con datos</div>
            </div>
            <div class="workflow-status {status_class}">
                {workflow_status["train"].upper() if workflow_status["train"] else "PENDIENTE"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Paso 3: Predicción
        status_class = "status-completed" if workflow_status["predict"] == "completed" else "status-active" if workflow_status["predict"] == "active" else "status-pending"
        icon = "" if workflow_status["predict"] == "completed" else "" if workflow_status["predict"] == "active" else ""
        st.markdown(f"""
        <div class="workflow-row">
            <div class="workflow-step-container">
                <div class="workflow-icon">{icon}</div>
                <div><strong>3. Predicción</strong> - Generar predicciones futuras</div>
            </div>
            <div class="workflow-status {status_class}">
                {workflow_status["predict"].upper() if workflow_status["predict"] else "PENDIENTE"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Paso 4: Evaluación
        status_class = "status-completed" if workflow_status["evaluate"] == "completed" else "status-active" if workflow_status["evaluate"] == "active" else "status-pending"
        icon = "" if workflow_status["evaluate"] == "completed" else "" if workflow_status["evaluate"] == "active" else ""
        st.markdown(f"""
        <div class="workflow-row">
            <div class="workflow-step-container">
                <div class="workflow-icon">{icon}</div>
                <div><strong>4. Evaluación</strong> - Evaluar precisión del modelo</div>
            </div>
            <div class="workflow-status {status_class}">
                {workflow_status["evaluate"].upper() if workflow_status["evaluate"] else "PENDIENTE"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Línea de separación clara
        st.markdown("<hr style='margin: 2rem 0; border-color: var(--border);'>", unsafe_allow_html=True)

        # Sidebar: Configuración avanzada (más limpia y organizada)
        with st.sidebar:
            st.title("Configuración")
            
            # Sección de enfoque de modelado con explicación clara
            st.markdown("### Enfoque de Modelado")
            
            enfoque_modelado = st.radio(
                "Enfoque de modelado:",
                options=["Conteo de ataques por día", "Días entre ataques"],
                index=0 if st.session_state.enfoque_actual == "conteo_diario" else 1,
                help="Conteo diario suma ataques por día. Días entre ataques mide tiempo entre eventos consecutivos."
            )
            
            # Convertir selección a formato interno
            enfoque = "conteo_diario" if enfoque_modelado == "Conteo de ataques por día" else "dias_entre_ataques"
            
            # Actualizar enfoque si cambió
            if enfoque != st.session_state.enfoque_actual:
                st.session_state.enfoque_actual = enfoque
                st.session_state.data_loaded = False
                st.session_state.model_trained = False
                st.session_state.forecast_generated = False
                st.session_state.metrics_calculated = False
            
            # Opciones de preprocesamiento en una sección propia
            st.markdown("### Preprocesamiento")
            
            use_log_transform = st.checkbox(
                "Transformación logarítmica",
                value=True,  # Activado por defecto para mejorar rendimiento con series de ransomware
                help="Aplica logaritmo natural a los valores para estabilizar series con alta variabilidad"
            )
            
            outlier_method = st.selectbox(
                "Método de detección de outliers:",
                options=["std", "iqr", "none"],
                index=1,
                help="IQR usa rango intercuartil. STD usa desviaciones estándar. None desactiva detección."
            )
            
            outlier_strategy = st.selectbox(
                "Estrategia para outliers:",
                options=["remove", "cap", "none"],
                index=1,
                help="Cap limita valores extremos. Remove elimina outliers. None los mantiene."
            )
            
            # Configuración avanzada en un expander con título claro
            with st.expander("Parámetros Avanzados del Modelo", expanded=False):
                st.markdown("""
                <div style="background-color: var(--background-light); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                    <p style="margin: 0; font-size: 0.9rem;">Estos parámetros controlan el comportamiento del modelo Prophet. 
                    Los valores por defecto funcionan bien en la mayoría de casos.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Sección de optimizaciones avanzadas
                st.markdown("#### Optimizaciones Avanzadas")
                
                # Organizar en columnas para ahorrar espacio
                opt_col1, opt_col2 = st.columns(2)
                
                with opt_col1:
                    use_optimal_regressors = st.checkbox(
                        "Selección óptima de regresores",
                        value=True,  # Activado por defecto
                        help="Selecciona automáticamente los mejores regresores basados en correlación y VIF"
                    )
                    
                    use_bayesian_optimization = st.checkbox(
                        "Optimización bayesiana",
                        value=True,  # Activado por defecto
                        help="Busca los mejores hiperparámetros automáticamente usando Bayesian Optimization"
                    )
                
                with opt_col2:
                    use_interval_calibration = st.checkbox(
                        "Calibración de intervalos",
                        value=True,  # Activado por defecto
                        help="Calibra automáticamente los intervalos de predicción para mayor precisión"
                    )
                    
                    optimization_trials = st.slider(
                        "Pruebas de optimización",
                        min_value=10,
                        max_value=50,
                        value=25,  # Aumentado para mejor optimización
                        step=5,
                        help="Número de pruebas para la optimización bayesiana. Más pruebas = mejor resultado pero más tiempo"
                    )
                
                # Parámetros adicionales de optimización
                if use_optimal_regressors:
                    st.markdown("##### Parámetros de selección de regresores")
                    reg_col1, reg_col2 = st.columns(2)
                    
                    with reg_col1:
                        correlation_threshold = st.slider(
                            "Umbral de correlación",
                            min_value=0.05,
                            max_value=0.30,
                            value=0.15,  # Ajustado para equilibrio entre sensibilidad y robustez
                            step=0.01,
                            format="%.2f",
                            help="Correlación mínima requerida para incluir un regresor"
                        )
                    
                    with reg_col2:
                        vif_threshold = st.slider(
                            "Umbral de VIF",
                            min_value=2.0,
                            max_value=10.0,
                            value=4.0,  # Ajustado para equilibrio entre multicolinealidad y complejidad
                            step=0.5,
                            format="%.1f",
                            help="Valor máximo de Factor de Inflación de Varianza permitido (para evitar multicolinealidad)"
                        )
                
                st.markdown("---")
                
                # Parámetros del modelo con descripciones más claras
                st.markdown("#### Parámetros del Modelo Prophet")
                
                # Organizar controles en columnas para mejor espacio
                prophet_col1, prophet_col2 = st.columns(2)
                
                with prophet_col1:
                    changepoint_prior_scale = st.slider(
                        "Flexibilidad de tendencia",
                        min_value=0.001,
                        max_value=0.5,
                        value=0.2,  # Más flexible que el valor por defecto (0.05)
                        step=0.001,
                        format="%.3f",
                        help="Controla la flexibilidad de la tendencia. Mayor valor = más flexible"
                    )
                    
                    seasonality_prior_scale = st.slider(
                        "Fuerza de estacionalidad",
                        min_value=0.1,
                        max_value=20.0,
                        value=5.0,  # Reducido para evitar sobreajuste (original 10.0)
                        step=0.1,
                        format="%.1f",
                        help="Controla la fuerza de la estacionalidad. Mayor valor = más fuerte"
                    )
                
                with prophet_col2:
                    seasonality_mode = st.selectbox(
                        "Modo de estacionalidad",
                        options=["additive", "multiplicative"],
                        index=0,  # Cambiado a additive para series con muchos ceros
                        help="Aditivo es mejor para series con variación constante. Multiplicativo para variación proporcional a la tendencia."
                    )
                    
                    n_changepoints = st.slider(
                        "Número de changepoints",
                        min_value=10,
                        max_value=100,
                        value=80,  # Aumentado para capturar mejor los cambios en tendencia (original 25)
                        step=5,
                        help="Número de puntos donde la tendencia puede cambiar. Mayor = más flexible"
                    )
                
                interval_width = st.slider(
                    "Intervalo de confianza (%)",
                    min_value=50,
                    max_value=99,
                    value=95,  # Valor estándar en estadística
                    step=1,
                    format="%d",
                    help="Ancho del intervalo de confianza. Mayor valor = más confianza pero menos precisión"
                ) / 100.0  # Convertir de porcentaje a proporción
                
                use_detected_changepoints = st.checkbox(
                    "Usar puntos de cambio detectados",
                    value=True,  # Activado por defecto
                    help="Permite al modelo adaptarse a cambios históricos importantes"
                )
                
                # Actualizar parámetros en sesión
                st.session_state.params = {
                    'changepoint_prior_scale': changepoint_prior_scale,
                    'seasonality_prior_scale': seasonality_prior_scale,
                    'seasonality_mode': seasonality_mode,
                    'interval_width': interval_width,
                    'use_detected_changepoints': use_detected_changepoints,
                    'use_optimal_regressors': use_optimal_regressors,
                    'use_bayesian_optimization': use_bayesian_optimization, 
                    'use_interval_calibration': use_interval_calibration,
                    'optimization_trials': optimization_trials,
                    'correlation_threshold': correlation_threshold if use_optimal_regressors else 0.12,
                    'vif_threshold': vif_threshold if use_optimal_regressors else 5.0
                }
            
            # Horizonte de predicción más visible
            st.markdown("### Horizonte de Predicción")
            
            periods = st.slider(
                "Días a predecir",
                min_value=7,
                max_value=365,
                value=st.session_state.periods,
                step=1,
                help="Periodo futuro para el que se generarán predicciones"
            )
            
            # Actualizar horizonte en sesión
            st.session_state.periods = periods
            
            # Enlace a la guía de usuario
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; margin-top: 10px;">
                <p>¿Necesitas ayuda? Consulta la <a href="#" onclick="document.querySelector('div[data-testid=stRadio] div:nth-child(2) label').click(); return false;">Guía de Usuario</a></p>
            </div>
            """, unsafe_allow_html=True)
        
        # 1. Sección de carga de datos con botón a ancho completo
        st.markdown('<div class="section-header"><h2>1. Carga y Exploración de Datos</h2></div>', unsafe_allow_html=True)
        
        # Botón de carga de datos (ancho completo y destacado)
        load_data = st.button(
            " CARGAR DATOS",
            help="Carga los datos históricos de ataques ransomware según el enfoque seleccionado",
            use_container_width=True,
            type="primary"  # Hacer el botón más destacado
        )
        
        # Cargar datos si se presiona el botón
        if load_data or st.session_state.data_loaded:
            try:
                with st.spinner("Cargando datos..."):
                    # Usar el wrapper de carga de datos
                    df_prophet = integration.load_data_wrapper(
                        enfoque=st.session_state.enfoque_actual,
                        use_log_transform=st.session_state.params.get('use_log_transform', False)
                    )
                    
                    if df_prophet is not None and not df_prophet.empty:
                        # Guardar en el estado de la sesión
                        st.session_state.df_prophet = df_prophet
                        st.session_state.data_loaded = True
                        
                        # Mostrar éxito
                        st.success(" Datos cargados correctamente")
                        
                        # Mostrar visualización de datos
                        st.markdown('<div class="section-header"><h3>Visualización de Datos</h3></div>', unsafe_allow_html=True)
                        
                        # Crear tabs para diferentes vistas de datos
                        data_tabs = st.tabs(["Gráfico", "Estadísticas", "Datos Crudos"])
                        
                        with data_tabs[0]:
                            # Gráfico de datos
                            fig = px.line(
                                df_prophet, 
                                x='ds', 
                                y='y', 
                                title=f"Datos Históricos ({st.session_state.enfoque_actual})",
                                labels={'ds': 'Fecha', 'y': 'Valor'}
                            )
                            
                            # Ajustar diseño para modo oscuro
                            is_dark_mode = st.get_option("theme.base") == "dark"
                            text_color = "white" if is_dark_mode else "black"
                            
                            fig.update_layout(
                                xaxis_title="Fecha",
                                yaxis_title="Número de Ataques" if st.session_state.enfoque_actual == "conteo_diario" else "Días entre Ataques",
                                height=400,
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color=text_color)
                            )
                            
                            # Mostrar gráfico
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with data_tabs[1]:
                            # Estadísticas básicas
                            st.markdown("#### Estadísticas Básicas")
                            
                            # Columnas para métricas
                            stats_cols = st.columns(4)
                            
                            with stats_cols[0]:
                                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                st.markdown(f'<div class="metric-value">{len(df_prophet)}</div>', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Total Registros</div>', unsafe_allow_html=True)
                                st.markdown('<div class="caption">Total de registros en el conjunto de datos</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with stats_cols[1]:
                                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                st.markdown(f'<div class="metric-value">{df_prophet["y"].mean():.2f}</div>', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Media</div>', unsafe_allow_html=True)
                                st.markdown('<div class="caption">Media de los valores en el conjunto de datos</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with stats_cols[2]:
                                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                st.markdown(f'<div class="metric-value">{df_prophet["y"].max():.2f}</div>', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Máximo</div>', unsafe_allow_html=True)
                                st.markdown('<div class="caption">Valor máximo en el conjunto de datos</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with stats_cols[3]:
                                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                st.markdown(f'<div class="metric-value">{df_prophet["y"].std():.2f}</div>', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Desv. Estándar</div>', unsafe_allow_html=True)
                                st.markdown('<div class="caption">Desviación estándar de los valores en el conjunto de datos</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        with data_tabs[2]:
                            # Ver datos crudos
                            st.markdown("#### Datos Crudos")
                            st.dataframe(
                                df_prophet.rename(columns={'ds': 'Fecha', 'y': 'Valor'}),
                                use_container_width=True
                            )
                            
                            # Opción para descargar los datos
                            csv = df_prophet.to_csv(index=False)
                            st.download_button(
                                label=" Descargar Datos (CSV)",
                                data=csv,
                                file_name="datos_ransomware.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    else:
                        st.error(" Error al cargar los datos")
                        st.session_state.data_loaded = False
                        
            except Exception as e:
                st.error(f"Error en carga de datos: {str(e)}")
                st.code(traceback.format_exc())
                st.session_state.data_loaded = False
        
        # Línea de separación clara
        st.markdown("<hr style='margin: 2rem 0; border-color: var(--border);'>", unsafe_allow_html=True)
        
        # 2. Sección de entrenamiento del modelo
        st.markdown('<div class="section-header"><h2>2. Entrenamiento del Modelo</h2></div>', unsafe_allow_html=True)
        
        # Botón de entrenamiento a ancho completo y destacado
        train_model = st.button(
            " ENTRENAR MODELO",
            help="Entrena el modelo Prophet con los datos cargados y los parámetros especificados",
            use_container_width=True,
            type="primary" if not st.session_state.model_trained else "secondary",
            disabled=not st.session_state.data_loaded
        )
        
        # Entrenamiento del modelo
        if train_model or st.session_state.model_trained:
            if not st.session_state.data_loaded:
                st.error(" Primero debe cargar los datos")
            else:
                try:
                    # Reiniciar SOLO el modelo Prophet si ya existe, 
                    # pero mantener todos los demás componentes necesarios para predicción y evaluación
                    if 'prophet_model' in st.session_state:
                        st.info("Reiniciando modelo anterior para un nuevo entrenamiento...")
                        del st.session_state.prophet_model
                    
                    # Marcar como no entrenado para forzar nuevo entrenamiento
                    if 'model_trained' in st.session_state:
                        st.session_state.model_trained = False
                    
                    # Mantener intactos: forecaster, df, datos de evaluación y otros componentes necesarios
                    
                    with st.spinner("Entrenando modelo..."):
                        # Forzar parámetros específicos para el modelo con todas las optimizaciones activadas
                        st.session_state.params['use_log_transform'] = True
                        st.session_state.params['use_optimal_regressors'] = True
                        st.session_state.params['use_bayesian_optimization'] = True
                        st.session_state.params['use_interval_calibration'] = True
                        
                        # Mostrar los parámetros que se están utilizando
                        st.info(f"""
                        **Optimizaciones Aplicadas**
                        - log_transform: {st.session_state.params.get('use_log_transform', True)}
                        - optimal_regressors: {st.session_state.params.get('use_optimal_regressors', True)}
                        - bayesian_optimization: {st.session_state.params.get('use_bayesian_optimization', True)}
                        - interval_calibration: {st.session_state.params.get('use_interval_calibration', True)}
                        """)
                        
                        # Entrenar usando el wrapper con los parámetros forzados
                        result = integration.train_model_wrapper(
                            changepoint_prior_scale=st.session_state.params.get('changepoint_prior_scale', 0.05),
                            seasonality_prior_scale=st.session_state.params.get('seasonality_prior_scale', 10.0),
                            seasonality_mode=st.session_state.params.get('seasonality_mode', 'additive'),
                            interval_width=st.session_state.params.get('interval_width', 0.95),
                            use_detected_changepoints=st.session_state.params.get('use_detected_changepoints', True),
                            use_optimal_regressors=st.session_state.params.get('use_optimal_regressors', True),
                            use_bayesian_optimization=st.session_state.params.get('use_bayesian_optimization', True),
                            use_interval_calibration=st.session_state.params.get('use_interval_calibration', True),
                            optimization_trials=st.session_state.params.get('optimization_trials', 20),
                            correlation_threshold=st.session_state.params.get('correlation_threshold', 0.12),
                            vif_threshold=st.session_state.params.get('vif_threshold', 5.0)
                        )
                        
                        if result:
                            st.session_state.model_trained = True
                            st.success(" Modelo entrenado correctamente")
                            
                            # Mostrar componentes del modelo
                            st.markdown('<div class="section-header"><h3>Componentes del Modelo</h3></div>', unsafe_allow_html=True)
                            
                            # Obtener componentes del modelo directamente
                            try:
                                # Asegurar que tenemos acceso al modelo entrenado
                                model = None
                                
                                # Intentar obtener el modelo de diferentes fuentes
                                if 'model' in st.session_state:
                                    model = st.session_state.model
                                elif 'prophet_model' in st.session_state:
                                    model = st.session_state.prophet_model
                                elif 'forecaster' in st.session_state and hasattr(st.session_state.forecaster, 'model') and st.session_state.forecaster.model is not None:
                                    model = st.session_state.forecaster.model
                                    
                                if model is None:
                                    st.error("No se pudo acceder al modelo entrenado")
                                    raise ValueError("Modelo no disponible")
                                
                                # Obtener forecast para los componentes
                                forecast = None
                                
                                # Intentar obtener el forecast de diferentes fuentes
                                if 'forecast' in st.session_state:
                                    forecast = st.session_state.forecast
                                elif 'forecaster' in st.session_state and hasattr(st.session_state.forecaster, 'forecast') and st.session_state.forecaster.forecast is not None:
                                    forecast = st.session_state.forecaster.forecast
                                
                                # Si no hay forecast, generar uno temporal
                                if forecast is None:
                                    try:
                                        # Usar un enfoque más seguro para crear el forecast
                                        # Evitando el uso de make_future_dataframe que puede dar problemas con fechas
                                        if hasattr(model, 'model') and hasattr(model.model, 'predict'):
                                            # Caso 1: Modelo con estructura moderna (model.model)
                                            # Crear manualmente dataframe futuro sin manipulación de fechas
                                            today = pd.Timestamp.now().normalize()
                                            future_dates = pd.date_range(
                                                start=today, 
                                                periods=30, 
                                                freq='D'
                                            )
                                            future = pd.DataFrame({'ds': future_dates})
                                            forecast = model.model.predict(future)
                                            
                                            # Añadir atributo necesario para plot_components
                                            if not hasattr(model, 'uncertainty_samples'):
                                                model.uncertainty_samples = 1000
                                        elif hasattr(model, 'predict'):
                                            # Caso 2: Modelo con método predict directo
                                            # Crear fechas futuras de manera segura
                                            today = pd.Timestamp.now().normalize()
                                            future = pd.DataFrame({'ds': pd.date_range(start=today, periods=30, freq='D')})
                                            forecast = model.predict(future)
                                            
                                            # Añadir atributo necesario para plot_components
                                            if not hasattr(model, 'uncertainty_samples'):
                                                model.uncertainty_samples = 1000
                                        else:
                                            st.warning("No se pueden mostrar los componentes: modelo con estructura no compatible")
                                            # Crear un dataframe vacío para evitar errores
                                            forecast = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
                                    except Exception as e:
                                        st.warning(f"No se pueden generar predicciones para componentes: {str(e)}")
                                        # Crear un dataframe vacío para evitar errores
                                        forecast = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
                                
                                # Alternativa a plot_components_plotly que causa problemas
                                try:
                                    # Verificar si tenemos suficientes datos
                                    if not forecast.empty and 'ds' in forecast.columns and len(forecast) > 5:
                                        st.subheader("Tendencia")
                                        if 'trend' in forecast.columns:
                                            import plotly.graph_objects as go
                                            
                                            # Graficar tendencia
                                            trend_fig = go.Figure()
                                            trend_fig.add_trace(go.Scatter(
                                                x=forecast['ds'], 
                                                y=forecast['trend'],
                                                mode='lines',
                                                name='Tendencia',
                                                line=dict(color='#0072B2')
                                            ))
                                            trend_fig.update_layout(
                                                title='Tendencia a largo plazo',
                                                xaxis_title='Fecha',
                                                yaxis_title='Impacto',
                                                height=300
                                            )
                                            st.plotly_chart(trend_fig, use_container_width=True)
                                        
                                        # Graficar estacionalidad semanal si existe
                                        st.subheader("Estacionalidad semanal")
                                        if 'weekly' in forecast.columns:
                                            import plotly.graph_objects as go
                                            
                                            # Convertir a día de la semana
                                            días = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                                            días_semana = [días[d.weekday()] for d in forecast['ds']]
                                            
                                            weekly_fig = go.Figure()
                                            weekly_fig.add_trace(go.Box(
                                                y=forecast['weekly'],
                                                x=días_semana,
                                                name='Efecto semanal',
                                                marker_color='#E69F00'
                                            ))
                                            weekly_fig.update_layout(
                                                title='Patrón semanal',
                                                xaxis_title='Día',
                                                yaxis_title='Impacto',
                                                height=300
                                            )
                                            st.plotly_chart(weekly_fig, use_container_width=True)
                                        
                                        # Graficar estacionalidad anual si existe
                                        st.subheader("Estacionalidad anual")
                                        if 'yearly' in forecast.columns:
                                            import plotly.graph_objects as go
                                            
                                            # Ordenar por mes
                                            meses = forecast['ds'].dt.month
                                            nombres_meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
                                            
                                            yearly_fig = go.Figure()
                                            yearly_fig.add_trace(go.Box(
                                                y=forecast['yearly'],
                                                x=[nombres_meses[m-1] for m in meses],
                                                name='Efecto anual',
                                                marker_color='#CC79A7'
                                            ))
                                            yearly_fig.update_layout(
                                                title='Patrón anual',
                                                xaxis_title='Mes',
                                                yaxis_title='Impacto',
                                                height=300,
                                                xaxis={'categoryorder':'array', 'categoryarray':nombres_meses}
                                            )
                                            st.plotly_chart(yearly_fig, use_container_width=True)
                                    else:
                                        st.warning("No hay suficientes datos para visualizar los componentes del modelo")
                                except Exception as e:
                                    st.error(f"Error al visualizar componentes: {str(e)}")
                                    
                                    # Si falla el enfoque personalizado, intentar usar directamente plot_components
                                    try:
                                        # Añadir atributos necesarios al modelo
                                        if hasattr(model, 'model') and hasattr(model.model, 'predict'):
                                            # Si es una estructura anidada, añadir atributos al modelo interno
                                            if not hasattr(model.model, 'uncertainty_samples'):
                                                model.model.uncertainty_samples = 1000
                                                
                                            # Usar prophet.plot para componentes directamente del modelo
                                            from prophet.plot import plot_components
                                            import matplotlib.pyplot as plt
                                            
                                            fig = plt.figure(figsize=(12, 10))
                                            plot_components(model.model, forecast)
                                            st.pyplot(fig)
                                        else:
                                            st.warning("No se pueden mostrar los componentes con el método alternativo")
                                    except Exception as e2:
                                        st.error(f"También falló el método alternativo: {str(e2)}")
                            except Exception as e:
                                st.error(f"Error al mostrar componentes del modelo: {str(e)}")
                        else:
                            st.error(" Error al entrenar el modelo")
                except Exception as e:
                    st.error(f"Error en entrenamiento: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Línea de separación clara
        st.markdown("<hr style='margin: 2rem 0; border-color: var(--border);'>", unsafe_allow_html=True)
        
        # 3. Sección de predicción
        st.markdown('<div class="section-header"><h2>3. Generación de Predicciones</h2></div>', unsafe_allow_html=True)
        
        # Información del horizonte de predicción seleccionado
        st.info(f" Horizonte de predicción actual: **{st.session_state.periods} días**. Puede modificarlo en la barra lateral.")
        
        # Botón de predicción a ancho completo y destacado
        generate_forecast = st.button(
            " GENERAR PREDICCIÓN",
            help="Genera predicciones para el periodo futuro especificado",
            use_container_width=True,
            type="primary" if not st.session_state.forecast_generated else "secondary",
            disabled=not st.session_state.model_trained
        )
        
        # Generación de predicciones
        if generate_forecast or st.session_state.forecast_generated:
            if not st.session_state.model_trained:
                st.error(" Primero debe entrenar el modelo")
            else:
                try:
                    with st.spinner("Generando predicción..."):
                        # Asegurar que las optimizaciones permanezcan activadas durante la predicción
                        st.session_state.params['use_log_transform'] = True
                        st.session_state.params['use_optimal_regressors'] = True
                        st.session_state.params['use_bayesian_optimization'] = True
                        st.session_state.params['use_interval_calibration'] = True
                        
                        # Actualizar también las variables usadas en el resumen final
                        st.session_state.use_log_transform = True
                        st.session_state.use_optimal_regressors = True 
                        st.session_state.use_bayesian_optimization = True
                        st.session_state.use_interval_calibration = True
                        st.session_state.selected_regressors = st.session_state.get('selected_regressors', [])
                        st.session_state.optimized_params = st.session_state.get('optimized_params', {})
                        
                        # Mostrar estado de las optimizaciones durante la predicción
                        st.info(f"""
                        **Optimizaciones Aplicadas en Predicción**
                        - log_transform: {st.session_state.params.get('use_log_transform', True)}
                        - optimal_regressors: {st.session_state.params.get('use_optimal_regressors', True)}
                        - bayesian_optimization: {st.session_state.params.get('use_bayesian_optimization', True)}
                        - interval_calibration: {st.session_state.params.get('use_interval_calibration', True)}
                        """)
                        
                        # Asegurar que el modelo esté disponible en la clave correcta de session_state
                        if 'prophet_model' in st.session_state and 'model' not in st.session_state:
                            st.session_state.model = st.session_state.prophet_model
                        
                        # Asegurar que el forecaster tenga el modelo asignado
                        if 'forecaster' in st.session_state and hasattr(st.session_state.forecaster, 'model'):
                            if st.session_state.forecaster.model is None and 'model' in st.session_state:
                                st.session_state.forecaster.model = st.session_state.model
                        
                        # Asegurar que df_prophet esté disponible para la predicción
                        if 'df_prophet' in st.session_state and hasattr(st.session_state.forecaster, 'df_prophet'):
                            if st.session_state.forecaster.df_prophet is None:
                                st.session_state.forecaster.df_prophet = st.session_state.df_prophet
                        
                        # CORRECCIÓN CRÍTICA: Asegurar que df esté disponible para make_forecast_wrapper
                        if 'df_prophet' in st.session_state and 'df' not in st.session_state:
                            st.session_state.df = st.session_state.df_prophet
                        
                        # Generar predicción usando el wrapper
                        forecast = integration.make_forecast_wrapper(None, periods=st.session_state.periods)
                        
                        if forecast is not None:
                            st.session_state.forecast_generated = True
                            st.success(" Predicción generada correctamente")
                            
                            # Mostrar visualización de predicciones
                            st.markdown('<div class="section-header"><h3>Visualización de Predicciones</h3></div>', unsafe_allow_html=True)
                            
                            # Obtener figura de predicción
                            fig = integration.plot_forecast_wrapper()
                            
                            if fig:
                                # Personalizar la figura
                                is_dark_mode = st.get_option("theme.base") == "dark"
                                text_color = "white" if is_dark_mode else "black"
                                
                                fig.update_layout(
                                    title="Predicción de Ataques de Ransomware",
                                    hovermode="x unified",
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    ),
                                    height=500,
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color=text_color)
                                )
                                
                                # Eje Y según el enfoque
                                y_title = "Número de Ataques" if st.session_state.enfoque_actual == "conteo_diario" else "Días entre Ataques"
                                fig.update_yaxes(title=y_title)
                                
                                # Mostrar gráfico
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Organizar resultados en tarjetas expandibles
                                st.markdown("### Resultados de Predicción")
                                
                                # Tarjeta 1: Exportación de Datos
                                with st.expander("  Exportación de Resultados y Opciones de Descarga", expanded=True):
                                    st.markdown('<div class="card card-predict">', unsafe_allow_html=True)
                                    st.markdown('<div class="card-header">Opciones de Descarga</div>', unsafe_allow_html=True)
                                    
                                    # Columnas para opciones de exportación
                                    export_cols = st.columns(2)
                                    
                                    with export_cols[0]:
                                        # Nota sobre exportación de imágenes
                                        st.info(" Para guardar el gráfico, usa la opción de descarga integrada de Plotly (ícono de cámara) en la esquina superior derecha del gráfico.")
                                    
                                    with export_cols[1]:
                                        # Exportar datos
                                        csv = forecast.to_csv(index=False)
                                        st.download_button(
                                            label=" Descargar Datos (CSV)",
                                            data=csv,
                                            file_name="prediccion_ransomware.csv",
                                            mime="text/csv",
                                            use_container_width=True
                                        )
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Tarjeta 2: Tabla de Predicciones
                                with st.expander("  Tabla de Predicciones Detallada y Estadísticas", expanded=True):
                                    st.markdown('<div class="card card-predict">', unsafe_allow_html=True)
                                    st.markdown('<div class="card-header">Valores Numéricos de Predicción</div>', unsafe_allow_html=True)
                                    
                                    # Mostrar solo predicciones futuras (no datos históricos)
                                    future_forecast = forecast[forecast['ds'] > pd.Timestamp.today()]
                                    
                                    # Redondear valores para mejor visualización
                                    display_forecast = future_forecast.copy()
                                    numeric_cols = ['yhat', 'yhat_lower', 'yhat_upper']
                                    display_forecast[numeric_cols] = display_forecast[numeric_cols].round(2)
                                    
                                    # Renombrar columnas para mejor comprensión - Guardar el DataFrame renombrado
                                    display_forecast = display_forecast.rename(columns={
                                        'ds': 'Fecha',
                                        'yhat': 'Predicción',
                                        'yhat_lower': 'Límite Inferior',
                                        'yhat_upper': 'Límite Superior'
                                    })
                                    
                                    # Mostrar la tabla con las columnas renombradas
                                    st.dataframe(
                                        display_forecast,
                                        use_container_width=True
                                    )
                                    
                                    # Mostrar estadísticas en columnas
                                    st.markdown("#### Resumen Estadístico", unsafe_allow_html=True)
                                    
                                    # Calcular algunas estadísticas básicas
                                    avg_prediction = display_forecast['Predicción'].mean()
                                    max_prediction = display_forecast['Predicción'].max()
                                    min_prediction = display_forecast['Predicción'].min()
                                    
                                    stat_cols = st.columns(3)
                                    with stat_cols[0]:
                                        st.metric("Predicción Media", f"{avg_prediction:.2f}")
                                    with stat_cols[1]:
                                        st.metric("Valor Máximo", f"{max_prediction:.2f}")
                                    with stat_cols[2]:
                                        st.metric("Valor Mínimo", f"{min_prediction:.2f}")
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Tarjeta 3: Interpretación de Resultados
                                with st.expander("  Guía de Interpretación de Resultados", expanded=False):
                                    st.markdown('<div class="card card-predict">', unsafe_allow_html=True)
                                    st.markdown('<div class="card-header">Guía de Interpretación</div>', unsafe_allow_html=True)
                                    
                                    st.markdown("""
                                    <div class="info-box" style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
                                        <h4>Cómo interpretar estos resultados:</h4>
                                        <ul>
                                            <li><b>Tendencia general:</b> Observe la dirección general de la línea de predicción para entender si los ataques aumentarán o disminuirán.</li>
                                            <li><b>Intervalos de confianza:</b> El área sombreada muestra el rango donde se espera que ocurran los valores reales. Un intervalo más amplio indica mayor incertidumbre.</li>
                                            <li><b>Puntos de cambio:</b> Busque cambios bruscos en la pendiente que pueden indicar cambios importantes en la tendencia de ataques.</li>
                                            <li><b>Patrones estacionales:</b> Identifique patrones recurrentes (semanales, mensuales) que pueden ayudar a planificar medidas preventivas.</li>
                                        </ul>
                                        <p>Recuerde que estas predicciones son estimaciones basadas en patrones históricos y pueden cambiar si surgen nuevos factores.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Tarjeta 4: Acciones Recomendadas
                                with st.expander("  Recomendaciones y Estrategias de Mitigación", expanded=False):
                                    st.markdown('<div class="card card-predict">', unsafe_allow_html=True)
                                    st.markdown('<div class="card-header">Estrategias de Mitigación</div>', unsafe_allow_html=True)
                                    
                                    st.markdown("""
                                    <div class="info-box" style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
                                        <h4>Basado en estas predicciones, considere:</h4>
                                        <ul>
                                            <li>Reforzar medidas de seguridad durante periodos de alto riesgo proyectado</li>
                                            <li>Programar actualizaciones de sistemas y copias de seguridad adicionales</li>
                                            <li>Informar al equipo de seguridad sobre las tendencias previstas</li>
                                            <li>Revisar y actualizar planes de respuesta a incidentes</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.warning("No se pudo generar el gráfico de predicción")
                        else:
                            st.error(" Error al generar la predicción")
                except Exception as e:
                    st.error(f"Error en predicción: {str(e)}")
                    st.code(traceback.format_exc())
        
        # 4. Sección de evaluación y prueba de predicciones
        st.markdown('<div class="section-header"><h2>4. Evaluación y Prueba del Modelo</h2></div>', unsafe_allow_html=True)
        
        # Crear pestañas para separar los dos tipos de evaluación
        eval_tabs = st.tabs(["Backtesting (Recomendado)", "Validación Cruzada"])
        
        # Tab 1: Backtesting (Ahora principal)
        with eval_tabs[0]:
            # Explicación de backtesting
            st.markdown("""
            <div class="card card-evaluate">
                <div class="card-header">Backtesting Múltiple (Prueba Histórica)</div>
                <div class="info-box" style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
                    <p>Este método evalúa el modelo usando múltiples puntos de corte en datos históricos, 
                    simulando cómo habría funcionado en el pasado.</p>
                    <p>Es más eficiente que la validación cruzada tradicional y proporciona resultados 
                    comparables con un rendimiento significativamente mejor.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Configuración de backtesting
            bt_col1, bt_col2 = st.columns([3, 1])
            
            with bt_col1:
                bt_periods = st.slider(
                    "Horizonte de predicción (días)",
                    min_value=14,
                    max_value=90,
                    value=30,
                    step=1,
                    help="Número de días para predecir en cada iteración de backtesting"
                )
                
                fast_mode = st.checkbox(
                    "Modo rápido (menos iteraciones)", 
                    value=True, 
                    help="Usar solo 2 iteraciones para una evaluación más rápida"
                )
            
            with bt_col2:
                # Botón para ejecutar backtesting
                run_backtest = st.button(
                    " Evaluar Modelo", 
                    use_container_width=True, 
                    disabled=not st.session_state.model_trained,
                    key="run_backtest_button"  # Clave única
                )
            
            if run_backtest:
                if not st.session_state.model_trained:
                    st.error(" Primero debe entrenar el modelo")
                else:
                    try:
                        # Evaluar usando el wrapper
                        metrics = integration.evaluate_model_wrapper(cv_periods=bt_periods)
                        
                        if metrics is not None:
                            st.session_state.metrics_calculated = True
                            st.success(" Evaluación completada")
                            
                            # Verificar que los datos necesarios existen antes de intentar visualizar
                            if ('valid_df' in st.session_state and 
                                'forecast_valid' in st.session_state and 
                                st.session_state.valid_df is not None and 
                                st.session_state.forecast_valid is not None):
                                # Visualizar los resultados de evaluación con soporte para tema oscuro
                                integration.plot_evaluation_results()
                            else:
                                st.warning("No se pueden visualizar los resultados detallados. Solo se mostrarán métricas resumidas.")
                            
                            # Mostrar resultados en cards expandibles
                            with st.expander("  Métricas de Rendimiento del Modelo", expanded=True):
                                st.markdown('<div class="card card-evaluate">', unsafe_allow_html=True)
                                st.markdown('<div class="card-header">Precisión del Modelo</div>', unsafe_allow_html=True)
                                
                                # Descripción de métricas
                                st.markdown("""
                                <div class="info-box" style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
                                    <p>Estas métricas indican la precisión del modelo promediada sobre <b>{iterations}</b> iteraciones de evaluación. 
                                    <b>Valores más bajos indican mejor rendimiento</b> (excepto para la Cobertura, donde un valor más alto es mejor).</p>
                                </div>
                                """.format(iterations=metrics.get('iterations', 3)), unsafe_allow_html=True)
                                
                                # Mostrar métricas en cards
                                metric_cols = st.columns(4)
                                
                                with metric_cols[0]:
                                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                    st.markdown(f'<div class="metric-value">{metrics["rmse"]:.2f}</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-label">RMSE</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="caption">Error cuadrático medio</div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with metric_cols[1]:
                                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                    st.markdown(f'<div class="metric-value">{metrics["mae"]:.2f}</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-label">MAE</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="caption">Error absoluto medio</div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with metric_cols[2]:
                                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                    # Verificar si SMAPE ya está en formato porcentaje (>1.0)
                                    smape_value = metrics["smape"]
                                    if smape_value <= 1.0:
                                        smape_value = smape_value * 100
                                    st.markdown(f'<div class="metric-value">{smape_value:.2f}%</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-label">SMAPE</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="caption">Error porcentual absoluto simétrico medio</div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with metric_cols[3]:
                                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                    # Verificar si coverage ya está en formato porcentaje (>1.0)
                                    coverage_value = metrics["coverage"]
                                    if coverage_value <= 1.0:
                                        coverage_value = coverage_value * 100
                                    st.markdown(f'<div class="metric-value">{coverage_value:.1f}%</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-label">Cobertura</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="caption">% de valores reales dentro del intervalo de predicción</div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Mostrar detalles de cada iteración si están disponibles
                                if 'evaluation_details' in st.session_state:
                                    st.markdown("<hr>", unsafe_allow_html=True)
                                    st.markdown("### Detalles por Iteración")
                                    
                                    # Crear dataframe para visualizar detalles
                                    details_df = pd.DataFrame(st.session_state.evaluation_details)
                                    
                                    # Agregar columna de iteración
                                    details_df.insert(0, 'Iteración', range(1, len(details_df) + 1))
                                    
                                    # Formatear métricas
                                    details_df['rmse'] = details_df['rmse'].round(2)
                                    details_df['mae'] = details_df['mae'].round(2)
                                    details_df['smape'] = details_df['smape'].round(2).astype(str) + '%'
                                    details_df['coverage'] = details_df['coverage'].round(1).astype(str) + '%'
                                    
                                    # Renombrar columnas para mejor visualización
                                    details_df = details_df.rename(columns={
                                        'rmse': 'RMSE',
                                        'mae': 'MAE',
                                        'smape': 'SMAPE',
                                        'coverage': 'Cobertura',
                                        'cutoff': 'Fecha de Corte',
                                        'num_points': 'Puntos Evaluados'
                                    })
                                    
                                    st.dataframe(
                                        details_df[['Iteración', 'Fecha de Corte', 'RMSE', 'MAE', 'SMAPE', 'Cobertura', 'Puntos Evaluados']], 
                                        use_container_width=True
                                    )
                        else:
                            st.error(" Error al realizar la evaluación")
                    except Exception as e:
                        st.error(f"Error en evaluación: {str(e)}")
                        st.code(traceback.format_exc())
        
        # Tab 2: Validación Cruzada (ahora secundaria)
        with eval_tabs[1]:
            # Advertencia sobre rendimiento
            st.warning("⚠️ La validación cruzada tradicional puede ser muy lenta. Se recomienda usar Backtesting para evaluación rutinaria.")
            
            # Explicación de evaluación con validación cruzada
            st.markdown("""
            <div class="card card-evaluate">
                <div class="card-header">Validación Cruzada Temporal</div>
                <div class="info-box" style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
                    <p>La validación cruzada temporal evalúa la precisión del modelo dividiendo los datos en múltiples segmentos de entrenamiento/prueba.</p>
                    <p>Se calculan métricas de error como RMSE, MAE, SMAPE y cobertura para evaluar el rendimiento.</p>
                    <p><b>Nota:</b> Este proceso puede tardar varios minutos o incluso horas dependiendo del tamaño de los datos.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Configuración de evaluación
            cv_col1, cv_col2 = st.columns([3, 1])
            
            with cv_col1:
                cv_periods = st.slider(
                    "Horizonte de validación (días)",
                    min_value=14,
                    max_value=90,
                    value=30,
                    step=1,
                    help="Número de días futuros para validar en cada iteración"
                )
            
            with cv_col2:
                # Botón de evaluación
                evaluate_model = st.button(
                    " Evaluar Modelo", 
                    use_container_width=True, 
                    disabled=not st.session_state.model_trained,
                    key="evaluate_model_button"  # Clave única
                )
            
            if evaluate_model:
                if not st.session_state.model_trained:
                    st.error(" Primero debe entrenar el modelo")
                else:
                    try:
                        with st.spinner("Evaluando modelo con validación cruzada..."):
                            # Evaluar usando el wrapper
                            metrics = integration.evaluate_model_wrapper(cv_periods=cv_periods)
                            
                            if metrics is not None:
                                st.session_state.metrics_calculated = True
                                st.success(" Validación cruzada completada")
                                
                                # Visualizar los resultados de evaluación con soporte para tema oscuro
                                integration.plot_evaluation_results()
                                
                                # Mostrar resultados en cards expandibles
                                with st.expander("  Métricas de Rendimiento del Modelo", expanded=True):
                                    st.markdown('<div class="card card-evaluate">', unsafe_allow_html=True)
                                    st.markdown('<div class="card-header">Precisión del Modelo</div>', unsafe_allow_html=True)
                                    
                                    # Descripción de métricas
                                    st.markdown("""
                                    <div class="info-box" style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
                                        <p>Estas métricas indican la precisión del modelo. <b>Valores más bajos indican mejor rendimiento</b> 
                                        (excepto para la Cobertura, donde un valor más alto es mejor).</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Mostrar métricas en cards
                                    metric_cols = st.columns(4)
                                    
                                    with metric_cols[0]:
                                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                        st.markdown(f'<div class="metric-value">{metrics["rmse"]:.2f}</div>', unsafe_allow_html=True)
                                        st.markdown('<div class="metric-label">RMSE</div>', unsafe_allow_html=True)
                                        st.markdown('<div class="caption">Error cuadrático medio</div>', unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    with metric_cols[1]:
                                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                        st.markdown(f'<div class="metric-value">{metrics["mae"]:.2f}</div>', unsafe_allow_html=True)
                                        st.markdown('<div class="metric-label">MAE</div>', unsafe_allow_html=True)
                                        st.markdown('<div class="caption">Error absoluto medio</div>', unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    with metric_cols[2]:
                                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                        # Verificar si SMAPE ya está en formato porcentaje (>1.0)
                                        smape_value = metrics["smape"]
                                        if smape_value <= 1.0:
                                            smape_value = smape_value * 100
                                        st.markdown(f'<div class="metric-value">{smape_value:.1f}%</div>', unsafe_allow_html=True)
                                        st.markdown('<div class="metric-label">SMAPE</div>', unsafe_allow_html=True)
                                        st.markdown('<div class="caption">Error porcentual</div>', unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    with metric_cols[3]:
                                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                        # Verificar si coverage ya está en formato porcentaje (>1.0)
                                        coverage_value = metrics["coverage"]
                                        if coverage_value <= 1.0:
                                            coverage_value = coverage_value * 100
                                        st.markdown(f'<div class="metric-value">{coverage_value:.1f}%</div>', unsafe_allow_html=True)
                                        st.markdown('<div class="metric-label">Cobertura</div>', unsafe_allow_html=True)
                                        st.markdown('<div class="caption">% de valores reales dentro del intervalo de predicción</div>', unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Interpretación de resultados
                                    with st.expander("  Guía de Interpretación de Métricas", expanded=False):
                                        st.markdown('<div class="card card-evaluate">', unsafe_allow_html=True)
                                        st.markdown('<div class="card-header">Guía de Interpretación</div>', unsafe_allow_html=True)
                                        
                                        st.markdown("""
                                        <div class="info-box" style="background-color: rgba(70, 70, 70, 0.3); padding: 10px; border-radius: 5px; color: white;">
                                            <h4>Cómo interpretar las métricas:</h4>
                                            <ul>
                                                <li><b>RMSE (Error Cuadrático Medio)</b>: Penaliza errores grandes. Valores más bajos son mejores.</li>
                                                <li><b>MAE (Error Absoluto Medio)</b>: Representa el error promedio en las mismas unidades que los datos. Valores más bajos son mejores.</li>
                                                <li><b>SMAPE (Error Porcentual Absoluto Simétrico Medio)</b>: Error expresado como porcentaje. Valores más bajos son mejores.</li>
                                                <li><b>Cobertura</b>: Porcentaje de valores reales que caen dentro del intervalo de predicción. Idealmente debería ser cercano al nivel de confianza establecido (ej. 95%).</li>
                                            </ul>
                                            <p>Un buen modelo tendrá un equilibrio entre precisión (errores bajos) y calibración adecuada de intervalos (cobertura cercana al nivel de confianza).</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.error(" Error al realizar la validación cruzada")
                    except Exception as e:
                        st.error(f"Error en evaluación: {str(e)}")
                        st.code(traceback.format_exc())
        
        # ─── FOOTER ───────────────────────────────────────────────────────────────────
        st.markdown(
            """
            <hr style="margin-top:3rem; border:none; border-top:1px solid #37474F;">
            <div style="text-align:center; font-size:0.8rem; color:#78909C;">
                Dataset &copy; <a href="https://www.ransomware.live" target="_blank" rel="noopener">
                Ransomware.live</a> &mdash; licencia no comercial
            </div>
            """,
            unsafe_allow_html=True
        )
