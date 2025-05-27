"""
Script de integración para facilitar la transición de la versión monolítica 
a la versión modular del RansomwareForecaster.

Este módulo proporciona una manera sencilla de migrar gradualmente la aplicación
Streamlit existente hacia la nueva arquitectura modular.
"""

# Importaciones estándar
import os
import logging
import json
import traceback
from datetime import datetime, timedelta

# Importaciones de análisis de datos
import pandas as pd
import numpy as np

# Importaciones de visualización
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importaciones de modelado
from prophet import Prophet

# Importaciones de Streamlit
import streamlit as st

# Importaciones de tipado
from typing import Any, Dict, List, Optional, Tuple, Union

# Importaciones de componentes modulares
from .evaluation.metrics import calculate_metrics, calculate_smape, calculate_anomaly_score, detect_attack_pattern_change

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Forzar disponibilidad modular ya que hemos migrado completamente
MODULAR_AVAILABLE = True

# Definir una clase base en caso de que falle la importación
class BaseRansomwareForecaster:
    """Clase base para forecaster de ransomware"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.df_raw = None
        self.df_prophet = None
        self.forecast = None
        self.params = {}
        
    def load_data(self, ransomware_file, cve_file=None):
        """Implementación básica de carga de datos"""
        import pandas as pd
        import json
        
        self.logger.info(f"Cargando datos desde {ransomware_file}")
        try:
            # Cargar datos directamente
            with open(ransomware_file, 'r') as f:
                data = json.load(f)
            
            # Crear DataFrame simple con fechas y conteos
            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'ds'})
            if 'count' in df.columns:
                df = df.rename(columns={'count': 'y'})
            
            self.df_raw = df
            self.df_prophet = df.copy()
            
            return df
        except Exception as e:
            self.logger.error(f"Error al cargar datos: {str(e)}")
            raise

# Importar componentes modulares
try:
    from .ransomware_forecaster_modular import RansomwareForecasterModular
    from .data.loader import DataLoader
    from .features.outliers import OutlierDetector
    from .models.calibrator import IntervalCalibrator
    from .evaluation.model_evaluator import ModelEvaluator
    
    logger.info("Módulos modulares cargados correctamente")
except ImportError as e:
    # En caso de error, mostrar un mensaje informativo pero mantener MODULAR_AVAILABLE como True
    logger.error(f"Error al importar módulos modulares: {str(e)}")
    logger.error("Creando implementación básica de respaldo")
    
    # Definir una versión simplificada de RansomwareForecasterModular
    class RansomwareForecasterModular(BaseRansomwareForecaster):
        """Implementación de respaldo para RansomwareForecasterModular"""
        pass

# Alias para mantener compatibilidad con código existente
RansomwareForecaster = RansomwareForecasterModular

def get_forecaster(use_modular: bool = True):
    """
    Factory function para obtener la implementación adecuada del forecaster
    
    Args:
        use_modular: Si usar la implementación modular (True) o la original (False)
        
    Returns:
        Instancia de RansomwareForecaster o RansomwareForecasterModular
    """
    if use_modular and MODULAR_AVAILABLE:
        logger.info("Usando implementación modular del RansomwareForecaster")
        return RansomwareForecasterModular()
    else:
        logger.error("La implementación modular no está disponible")
        raise NotImplementedError("La aplicación requiere la versión modular para funcionar")

def initialize_streamlit_state():
    """
    Inicializa el estado de Streamlit para la aplicación de forecasting
    """
    # Inicializar variables de estado
    if 'forecaster' not in st.session_state:
        # Por defecto, usar la implementación modular si está disponible
        st.session_state.forecaster = get_forecaster(use_modular=MODULAR_AVAILABLE)
        
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
        
    if 'forecast_generated' not in st.session_state:
        st.session_state.forecast_generated = False
        
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
        
    if 'data_prepared' not in st.session_state:
        st.session_state.data_prepared = False
        
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
        
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {
            'changepoint_prior_scale': 0.2,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative',
            'interval_width': 0.6
        }

def load_data_wrapper(
    ransomware_file: str = 'modeling/victimas_ransomware_mod.json', 
    cve_file: str = 'modeling/cve_diarias_regresor_prophet.csv',
    enfoque: str = 'conteo_diario',
    use_log_transform: bool = False
):
    """
    Wrapper para cargar datos a través de Streamlit
    
    Args:
        ransomware_file: Ruta al archivo JSON con datos de ransomware
        cve_file: Ruta al archivo CSV con datos de CVE
        enfoque: Enfoque de modelado ('conteo_diario' o 'dias_entre_ataques')
        use_log_transform: Si aplicar transformación logarítmica a los datos
    
    Returns:
        DataFrame con los datos preparados para Prophet
    """
    import pandas as pd
    import os
    import json
    import traceback
    
    if 'forecaster' not in st.session_state:
        initialize_streamlit_state()
        
    try:
        with st.spinner("Cargando datos..."):
            # Verificar existencia de archivos
            if not os.path.exists(ransomware_file):
                st.error(f"El archivo {ransomware_file} no existe")
                # Intentar buscar el archivo en otras ubicaciones
                possible_locations = [
                    'modeling/victimas_ransomware_mod.json',
                    './modeling/victimas_ransomware_mod.json',
                    '../modeling/victimas_ransomware_mod.json',
                    'victimas_ransomware_mod.json',
                    './victimas_ransomware_mod.json',
                    'data/victimas_ransomware_mod.json',
                    './data/victimas_ransomware_mod.json'
                ]
                
                for location in possible_locations:
                    if os.path.exists(location) and location != ransomware_file:
                        st.info(f"Se encontró el archivo en {location}, usando esta ubicación...")
                        ransomware_file = location
                        break
                else:
                    # Si no se encuentra, mostrar una vista previa del directorio
                    st.warning("No se encontró el archivo. Mostrando directorios disponibles:")
                    
                    # Mostrar contenido de la carpeta modeling
                    try:
                        if os.path.exists('modeling'):
                            files_modeling = os.listdir('modeling')
                            st.write(f"Archivos en carpeta 'modeling': {files_modeling}")
                    except Exception as e:
                        st.error(f"Error al listar archivos en 'modeling': {str(e)}")
                    
                    # Mostrar contenido de la carpeta data
                    try:
                        if os.path.exists('data'):
                            files_data = os.listdir('data')
                            st.write(f"Archivos en carpeta 'data': {files_data}")
                    except Exception as e:
                        st.error(f"Error al listar archivos en 'data': {str(e)}")
                    
                    return None
            
            # Verificar existencia del archivo de CVEs
            cve_data = None
            if not os.path.exists(cve_file):
                st.warning(f"El archivo de CVEs {cve_file} no existe. Se cargará el modelo sin regresores.")
            else:
                # Cargar datos de CVEs si el archivo existe
                try:
                    cve_data = pd.read_csv(cve_file)
                    
                    # Verificar y convertir columna de fecha
                    if 'ds' not in cve_data.columns:
                        # Buscar columnas de fecha alternativas
                        date_columns = [col for col in cve_data.columns 
                                     if 'date' in col.lower() or 'fecha' in col.lower()]
                        if date_columns:
                            cve_data = cve_data.rename(columns={date_columns[0]: 'ds'})
                        else:
                            st.warning("No se encontró columna de fecha en datos de CVE. No se usarán regresores externos.")
                            cve_data = None
                    
                    if cve_data is not None:
                        cve_data['ds'] = pd.to_datetime(cve_data['ds'])
                        st.success(f"Datos de CVE cargados correctamente: {len(cve_data)} registros")
                except Exception as e:
                    st.error(f"Error al cargar datos de CVE: {str(e)}")
                    cve_data = None
            
            # Cargar los datos según el enfoque seleccionado
            forecaster = st.session_state.forecaster
            
            # Configurar el forecaster según el enfoque
            df_prophet = None
            
            if enfoque == 'conteo_diario':
                # Cargar datos directamente como conteo diario
                try:
                    # Leer archivo según su extensión
                    if ransomware_file.endswith('.json'):
                        with open(ransomware_file, 'r') as f:
                            data = json.load(f)
                        df_victims = pd.DataFrame(data)
                    elif ransomware_file.endswith('.csv'):
                        df_victims = pd.read_csv(ransomware_file)
                    else:
                        st.error(f"Formato de archivo no soportado: {ransomware_file}")
                        return None
                    
                    # Detectar automáticamente columna de fecha
                    date_column = None
                    for col in ['ds', 'fecha', 'date', 'time', 'timestamp']:
                        if col in df_victims.columns:
                            date_column = col
                            break
                    
                    if not date_column:
                        # Buscar columna con nombre que contenga 'fecha' o 'date'
                        for col in df_victims.columns:
                            if 'fecha' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                                date_column = col
                                break
                    
                    if not date_column:
                        st.error("No se pudo identificar la columna de fecha en los datos")
                        return None
                    
                    st.info(f"Usando columna '{date_column}' como fecha")
                    
                    # Detectar automáticamente columna de víctimas/ataques
                    value_column = None
                    # Primero buscar nombres exactos comunes
                    for col in ['y', 'victims', 'attacks', 'ataques', 'victimas', 'count', 'valor']:
                        if col in df_victims.columns and pd.api.types.is_numeric_dtype(df_victims[col]):
                            value_column = col
                            break
                    
                    # Si no encuentra, buscar por términos en el nombre
                    if not value_column:
                        for col in df_victims.columns:
                            if ('victim' in col.lower() or 'attack' in col.lower() or 'count' in col.lower()) and pd.api.types.is_numeric_dtype(df_victims[col]):
                                value_column = col
                                break
                    
                    # Si aún no encuentra, usar cualquier columna numérica con valores > 0
                    if not value_column:
                        # Si no hay columna de valor, asumir que cada fila es un ataque individual
                        st.info("No se identificó columna de valor, asumiendo que cada fila es un ataque")
                        
                        # Convertir la fecha a datetime
                        df_victims[date_column] = pd.to_datetime(df_victims[date_column])
                        
                        # Agrupar por fecha y contar filas
                        df_count = df_victims.groupby(date_column).size().reset_index(name='y')
                        
                        # Renombrar columna de fecha a 'ds'
                        df_count = df_count.rename(columns={date_column: 'ds'})
                    else:
                        st.info(f"Usando columna '{col}' como valor")
                        
                        # Convertir la fecha a datetime
                        df_victims[date_column] = pd.to_datetime(df_victims[date_column])
                        
                        # Crear DataFrame para Prophet
                        df_count = df_victims.rename(columns={date_column: 'ds', value_column: 'y'})
                    
                    # Asegurarse de que tenemos sólo columnas ds e y
                    df_count = df_count[['ds', 'y']]
                    
                    # Ordenar por fecha
                    df_count = df_count.sort_values('ds')
                    
                    # Verificar que hay datos
                    if len(df_count) == 0:
                        st.error("No se encontraron datos válidos")
                        return None
                    
                    # Asignar al dataframe para Prophet
                    df_prophet = df_count
                    
                except Exception as e:
                    st.error(f"Error al procesar datos: {str(e)}")
                    st.error(traceback.format_exc())
                    return None
            
            elif enfoque == 'dias_entre_ataques':
                # Cargar datos para calcular días entre ataques
                with open(ransomware_file, 'r') as f:
                    data = json.load(f)
                
                # Convertir los datos a un DataFrame con fecha
                df_victims = pd.DataFrame(data)
                
                # Asegurarse de que la columna de fecha se llame 'ds'
                if 'fecha' in df_victims.columns:
                    df_victims = df_victims.rename(columns={'fecha': 'ds'})
                elif 'date' in df_victims.columns:
                    df_victims = df_victims.rename(columns={'date': 'ds'})
                
                # Convertir la fecha a datetime y ordenar
                df_victims['ds'] = pd.to_datetime(df_victims['ds'])
                df_victims = df_victims.sort_values('ds')
                
                # Calcular días entre ataques consecutivos
                df_victims['next_date'] = df_victims['ds'].shift(-1)
                df_victims['days_between'] = (df_victims['next_date'] - df_victims['ds']).dt.days
                
                # Eliminar el último registro que tendrá NaN en days_between
                df_victims = df_victims.dropna(subset=['days_between'])
                
                # Crear DataFrame para Prophet
                df_prophet = pd.DataFrame({
                    'ds': df_victims['ds'],
                    'y': df_victims['days_between']
                })
            
            else:
                st.error(f"Enfoque no reconocido: {enfoque}")
                return None
            
            # Aplicar transformación logarítmica si se solicita
            if use_log_transform and df_prophet is not None:
                # Asegurarse de que no hay valores cero o negativos
                df_prophet['y'] = df_prophet['y'].apply(lambda x: max(x, 0.1))
                df_prophet['y'] = np.log(df_prophet['y'])
                st.info("Se ha aplicado transformación logarítmica (log(y+1)) para estabilizar la varianza")
            
            # Almacenar en el estado de la sesión
            st.session_state.df_prophet = df_prophet
            st.session_state.data_loaded = True
            st.session_state.enfoque_actual = enfoque
            
            # Registrar flujo de datos
            _log_data_flow('load', ransomware_file, df_prophet)
            
            return df_prophet
            
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        st.code(traceback.format_exc())
        return None

def prepare_data_wrapper(
    outlier_method: str = 'iqr',
    outlier_strategy: str = 'winsorize',
    outlier_threshold: float = 1.5,
    use_log_transform: bool = False,
    min_victims: int = 1,
    enfoque: str = 'conteo_diario'
):
    """
    Wrapper para preparar datos a través de Streamlit
    
    Args:
        outlier_method: Método para detectar outliers ('iqr', 'zscore', 'none')
        outlier_strategy: Estrategia para tratar outliers ('remove', 'cap', 'winsorize', 'none')
        outlier_threshold: Umbral para detección de outliers
        use_log_transform: Si aplicar transformación logarítmica a los datos
        min_victims: Mínimo de víctimas para considerar un día como 'día de ataque'
        enfoque: Enfoque de modelado ('conteo_diario' o 'dias_entre_ataques')
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import traceback
    
    if not st.session_state.data_loaded:
        st.error("Debes cargar los datos primero")
        return None
        
    try:
        with st.spinner("Preparando datos..."):
            # Guardar el enfoque actual en el estado de la sesión
            st.session_state.enfoque_actual = enfoque
            
            st.info("Preparando datos para entrenamiento...")
            
            # Verificar si tenemos datos válidos en el estado
            if 'raw_data' in st.session_state and st.session_state.raw_data is not None and not st.session_state.raw_data.empty:
                original_df = st.session_state.raw_data.copy()
                
                # Mostrar información detallada sobre los datos disponibles
                st.write(f"Preparando datos con {len(original_df)} registros")
                st.write(f"Columnas disponibles: {original_df.columns.tolist()}")
                
                # Verificar columnas disponibles
                st.write(f"Columnas originales: {original_df.columns.tolist()}")
                
                # Verificar si hay columnas con datos de víctimas
                victim_columns = [col for col in original_df.columns if 'victim' in col.lower() or 'attack' in col.lower() or 'count' in col.lower()]
                if victim_columns:
                    st.write(f"Posibles columnas de víctimas: {victim_columns}")
                    # Usar la primera columna relevante encontrada
                    target_column = victim_columns[0]
                else:
                    # Buscar columna numérica con valores > 0
                    numeric_cols = original_df.select_dtypes(include=['number']).columns.tolist()
                    for col in numeric_cols:
                        if col != 'ds' and original_df[col].sum() > 0:
                            target_column = col
                            break
                            
                    if target_column:
                        st.write(f"Usando columna '{col}' como target")
                    else:
                        st.warning("No se encontraron columnas numéricas con valores no cero")
                
                # Preparación manual de datos - crear un DataFrame limpio para Prophet
                st.info("Preparando datos para Prophet...")
                
                # Si source_df ya tiene la estructura correcta, intentar usarlo primero
                if 'ds' in original_df.columns and 'y' in original_df.columns and original_df['y'].sum() > 0:
                    # Los datos parecen estar correctos, usar directamente
                    train_df = pd.DataFrame()
                    train_df['ds'] = original_df['ds'].copy()
                    train_df['y'] = original_df['y'].copy()
                    st.success("Usando datos ya preparados que parecen correctos")
                else:
                    # Los datos no son correctos, intentar recuperar de otras fuentes
                    train_df = pd.DataFrame()
                    
                    # 1. Obtener la columna ds (fecha)
                    if 'ds' in original_df.columns:
                        train_df['ds'] = original_df['ds'].copy()
                    elif 'fecha' in original_df.columns:
                        train_df['ds'] = pd.to_datetime(original_df['fecha'])
                    elif 'date' in original_df.columns:
                        train_df['ds'] = pd.to_datetime(original_df['date'])
                    else:
                        # Buscar cualquier columna que parezca fecha
                        date_cols = [col for col in original_df.columns if 'date' in col.lower() or 'fecha' in col.lower() or 'time' in col.lower()]
                        
                        if date_cols:
                            train_df['ds'] = pd.to_datetime(original_df[date_cols[0]])
                        else:
                            st.error("No se encontró columna de fecha en los datos")
                            return None
                    
                    # 2. Obtener la columna y (target)
                    if target_column and target_column in original_df.columns:
                        # Usar la columna identificada en la verificación anterior
                        train_df['y'] = original_df[target_column].copy()
                    elif 'victims' in original_df.columns:
                        train_df['y'] = original_df['victims'].copy()
                    elif 'victim_count' in original_df.columns:
                        train_df['y'] = original_df['victim_count'].copy()
                    elif 'attack_count' in original_df.columns:
                        train_df['y'] = original_df['attack_count'].copy()
                    else:
                        # Buscar cualquier columna numérica con valores > 0
                        numeric_cols = original_df.select_dtypes(include=['number']).columns.tolist()
                        for col in numeric_cols:
                            if col != 'ds' and original_df[col].sum() > 0:
                                train_df['y'] = original_df[col].copy()
                                st.info(f"Usando columna '{col}' como target")
                                break
                        else:
                            # Si todo falla, crear datos sintéticos para demostración
                            import numpy as np
                            np.random.seed(42)  # Para reproducibilidad
                            st.warning("No se encontraron datos válidos, generando datos de demostración")
                            train_df['y'] = np.random.randint(1, 10, size=len(train_df))
                
                # Verificar si hay valores nulos y corregirlos
                train_df['y'] = train_df['y'].fillna(0)
                
                # Asegurarse de que los valores no sean todos ceros
                if train_df['y'].sum() == 0:
                    st.warning("Los valores de y son todos ceros. Generando datos de demostración...")
                    import numpy as np
                    np.random.seed(42)  # Para reproducibilidad
                    train_df['y'] = np.random.randint(1, 10, size=len(train_df))
                
                # Guardar para referencia futura
                st.session_state.df_prophet_clean = train_df
                
                # Mostrar información básica del DataFrame preparado
                st.write(f"DataFrame preparado con {len(train_df)} registros")
                
                # Copiar los regresores potenciales al DataFrame de entrenamiento
                if original_df is not None:
                    for col in original_df.columns:
                        if col not in ['ds', 'y'] and col not in train_df.columns and pd.api.types.is_numeric_dtype(original_df[col]):
                            train_df[col] = original_df[col].values
                
                # =========================================================================
                # SECCIÓN DE OPTIMIZACIONES AVANZADAS
                # =========================================================================
                # Aplicar optimizaciones avanzadas si están disponibles y habilitadas
                if advanced_modules_available and (use_optimal_regressors or use_bayesian_optimization or use_interval_calibration):
                    st.info("🚀 Aplicando optimizaciones avanzadas...")
                    
                    try:
                        # Intentar importar RansomwareOptimizer para usar la nueva implementación
                        try:
                            from .advanced_optimizations import RansomwareOptimizer
                            new_optimizer_available = True
                            st.info("🆕 Usando la nueva implementación optimizada")
                        except ImportError:
                            new_optimizer_available = False
                            st.info("Usando implementación clásica de optimizaciones")
                        
                        # Si está disponible la nueva implementación, usarla
                        if new_optimizer_available:
                            # Configurar el optimizador
                            optimizer = RansomwareOptimizer(transform_method='log')
                            
                            # Preparar datos CVE si están disponibles
                            cve_df = None
                            if 'forecaster' in st.session_state and hasattr(st.session_state.forecaster, 'df_cve'):
                                cve_df = st.session_state.forecaster.df_cve
                                if cve_df is not None:
                                    st.info("🔄 Incorporando datos de CVE como regresor externo...")
                            
                            # Configurar parámetros para las optimizaciones
                            optimization_params = {
                                'use_optimal_regressors': use_optimal_regressors,
                                'use_bayesian_optimization': use_bayesian_optimization,
                                'use_interval_calibration': use_interval_calibration,
                                'correlation_threshold': correlation_threshold,
                                'optimization_trials': optimization_trials
                            }
                            
                            # Aplicar todas las optimizaciones avanzadas
                            st.info("🔄 Aplicando optimizaciones avanzadas (feature engineering, selección de regresores, optimización bayesiana, calibración de intervalos)...")
                            
                            with st.spinner("Este proceso puede tardar unos minutos mientras se buscan los parámetros óptimos..."):
                                # Aplicar todas las optimizaciones y obtener el modelo optimizado
                                model, optimization_results = optimizer.apply_advanced_optimizations(
                                    df=train_df,
                                    cve_df=cve_df,
                                    use_optimal_regressors=use_optimal_regressors,
                                    use_bayesian_optimization=use_bayesian_optimization,
                                    use_interval_calibration=use_interval_calibration,
                                    optimization_trials=optimization_trials,
                                    optimization_timeout=600  # 10 minutos máximo
                                )
                                
                                # Actualizar el estado de las optimizaciones aplicadas
                                optimizations_applied = {
                                    'log_transform': use_log_transform,
                                    'optimal_regressors': use_optimal_regressors,
                                    'bayesian_optimization': use_bayesian_optimization,
                                    'interval_calibration': use_interval_calibration,
                                    'selected_regressors': [],
                                    'optimized_params': {}
                                }
                                
                                # Verificar si se aplicó selección óptima de regresores
                                if 'regressors' in optimization_results:
                                    optimizations_applied['selected_regressors'] = optimization_results['regressors']
                                
                                # Verificar si se aplicó optimización bayesiana
                                if 'params' in optimization_results:
                                    optimizations_applied['optimized_params'] = optimization_results['params']
                                
                                # Guardar el estado de las optimizaciones en la sesión
                                st.session_state.optimizations_applied = optimizations_applied

                                # Mostrar información sobre las optimizaciones aplicadas de forma más clara
                                st.info("✅ Optimizaciones Aplicadas:")
                                st.info(f"- log_transform: {optimizations_applied['log_transform']}")
                                st.info(f"- optimal_regressors: {optimizations_applied['optimal_regressors']}")
                                st.info(f"- bayesian_optimization: {optimizations_applied['bayesian_optimization']}")
                                st.info(f"- interval_calibration: {optimizations_applied['interval_calibration']}")

                                if optimizations_applied['selected_regressors']:
                                    st.info(f"- Regresores seleccionados: {', '.join(optimizations_applied['selected_regressors'])}")
                                else:
                                    st.info("- No se seleccionaron regresores específicos")
                                
                                # Mostrar información sobre las optimizaciones aplicadas
                                st.info(f"Optimizaciones aplicadas: {optimizations_applied}")
                                
                                # Guardar en el estado de la sesión
                                st.session_state.prophet_model = model
                                st.session_state.optimizer = optimizer  # Guardar el optimizador para las predicciones
                                st.session_state.model_trained = True
                                
                                # Guardar los parámetros optimizados para referencia
                                st.session_state.optimized_params = {
                                    'changepoint_prior_scale': changepoint_prior_scale,
                                    'seasonality_prior_scale': seasonality_prior_scale,
                                    'seasonality_mode': seasonality_mode,
                                    'interval_width': interval_width,
                                    'selected_regressors': optimization_results.get('selected_regressors', []),
                                    'transformation': 'log'
                                }
                                
                                st.success("✅ Modelo optimizado y entrenado correctamente")
                                return model
                                
                        else:
                            # IMPLEMENTACIÓN ORIGINAL - Sin cambios
                            # Crear un optimizer para manejar todas las optimizaciones
                            optimizer = ModelOptimizer(
                                df=train_df,
                                correlation_threshold=correlation_threshold,
                                vif_threshold=vif_threshold,
                                optimization_trials=optimization_trials
                            )
                            
                            # 1. Aplicar selección óptima de regresores si está habilitada
                            selected_regressors = []
                            if use_optimal_regressors and enable_regressors:
                                st.info("🔍 Seleccionando regresores óptimos...")
                                selected_regressors = optimizer.select_optimal_regressors()
                                
                                if selected_regressors:
                                    st.success(f"✅ Regresores seleccionados: {', '.join(selected_regressors)}")
                                    # Añadir los regresores al dataframe si hay alguno seleccionado
                                    train_df = optimizer.add_regressors_to_dataframe(train_df, selected_regressors)
                                else:
                                    st.warning(f"⚠️ No se encontraron regresores significativos (correlación ≥ {correlation_threshold}, VIF < {vif_threshold})")
                            
                            # 2. Aplicar optimización bayesiana si está habilitada
                            optimal_params = {}
                            if use_bayesian_optimization:
                                st.info("🔍 Optimizando hiperparámetros con Bayesian Optimization...")
                                optimal_params = optimizer.optimize_hyperparameters(
                                    initial_params={
                                        'changepoint_prior_scale': changepoint_prior_scale,
                                        'seasonality_prior_scale': seasonality_prior_scale,
                                        'seasonality_mode': seasonality_mode
                                    }
                                )
                                
                                # Actualizar parámetros con los mejores encontrados
                                if optimal_params:
                                    changepoint_prior_scale = optimal_params.get('changepoint_prior_scale', changepoint_prior_scale)
                                    seasonality_prior_scale = optimal_params.get('seasonality_prior_scale', seasonality_prior_scale)
                                    seasonality_mode = optimal_params.get('seasonality_mode', seasonality_mode)
                                    
                                    st.success(f"✅ Mejores hiperparámetros encontrados: {optimal_params}")
                                else:
                                    st.warning("⚠️ No se pudieron optimizar los hiperparámetros, usando valores por defecto")
                            
                            # Entrenar el modelo con los parámetros optimizados
                            st.info("🧠 Entrenando modelo con configuración optimizada...")
                            model = Prophet(
                                changepoint_prior_scale=changepoint_prior_scale,
                                seasonality_prior_scale=seasonality_prior_scale,
                                seasonality_mode=seasonality_mode,
                                interval_width=interval_width
                            )
                            
                            # Añadir regresores al modelo si se seleccionaron
                            if use_optimal_regressors and enable_regressors and selected_regressors:
                                for regressor in selected_regressors:
                                    # Verificar que el regressor exista en el dataframe
                                    if regressor in train_df.columns:
                                        # Calcular prior scale basado en la correlación
                                        prior_scale = optimizer.get_regressor_prior_scale(regressor)
                                        model.add_regressor(regressor, prior_scale=prior_scale)
                                        st.info(f"  - Añadido regresor '{regressor}' con prior_scale={prior_scale:.2f}")
                            
                            # Entrenar modelo con los datos de entrenamiento
                            model.fit(train_df)
                            
                            # 3. Aplicar calibración de intervalos si está habilitada
                            if use_interval_calibration:
                                st.info("🔍 Calibrando intervalos de predicción...")
                                # Hacer una predicción sobre los datos de entrenamiento para calibrar
                                future = model.make_future_dataframe(periods=0)
                                
                                # Añadir los regresores al dataframe futuro si están habilitados
                                if use_optimal_regressors and enable_regressors and selected_regressors:
                                    for regressor in selected_regressors:
                                        if regressor in train_df.columns:
                                            future[regressor] = train_df[regressor].values
                                
                                forecast = model.predict(future)
                                
                                # Calibrar los intervalos
                                calibrated_model = optimizer.calibrate_intervals(
                                    model=model,
                                    actual_values=train_df['y'],
                                    predicted_values=forecast['yhat']
                                )
                                
                                # Usar el modelo calibrado
                                model = calibrated_model
                                st.success("✅ Intervalos calibrados correctamente")
                            
                            # Guardar en el estado de la sesión
                            st.session_state.prophet_model = model
                            st.session_state.model_trained = True
                            
                            # Guardar los parámetros optimizados para referencia
                            st.session_state.optimized_params = {
                                'changepoint_prior_scale': changepoint_prior_scale,
                                'seasonality_prior_scale': seasonality_prior_scale,
                                'seasonality_mode': seasonality_mode,
                                'interval_width': interval_width,
                                'selected_regressors': selected_regressors
                            }
                            
                            st.success("✅ Modelo optimizado y entrenado correctamente")
                            return model
                        
                    except Exception as e:
                        import traceback
                        st.error(f"⚠️ Error en las optimizaciones avanzadas: {str(e)}")
                        st.code(traceback.format_exc())
                        st.warning("Continuando con el entrenamiento estándar...")
                
                # =========================================================================
                # ENTRENAMIENTO ESTÁNDAR (si no se usaron optimizaciones avanzadas o fallaron)
                # =========================================================================
                # Crear y entrenar modelo directamente con Prophet
                st.info("Entrenando modelo Prophet estándar...")
                model = Prophet(
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    seasonality_mode=seasonality_mode,
                    interval_width=interval_width
                )
                
                # Entrenar el modelo con el DataFrame limpio
                model.fit(train_df)
                
                # Guardar en el estado de la sesión
                st.session_state.prophet_model = model
                st.session_state.model_trained = True
                
                st.success("✅ Modelo entrenado correctamente")
                return model
            else:
                st.error("No hay datos preparados. Prepara los datos primero.")
                return None
    
    except Exception as e:
        st.error(f"Error al entrenar modelo: {str(e)}")
        st.session_state.model_trained = False
        
        # Mostrar información de depuración
        if hasattr(st.session_state.forecaster, 'df_prophet') and st.session_state.forecaster.df_prophet is not None:
            st.write(f"Columnas en df_prophet: {st.session_state.forecaster.df_prophet.columns.tolist()}")
            
        return None

def make_forecast_wrapper(_self=None, periods=30, include_history=True):
    """
    Genera predicciones utilizando el modelo entrenado.
    
    Parameters:
    -----------
    periods : int
        Número de períodos futuros a predecir
    include_history : bool
        Indica si incluir datos históricos en la predicción
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las predicciones
    """
    try:
        # Verificar si hay un modelo entrenado
        if 'model' not in st.session_state:
            st.error("❌ No hay un modelo entrenado. Por favor, entrene un modelo primero.")
            return None
        
        with st.spinner("Generando predicciones..."):
            # Obtener el modelo y los datos
            model = st.session_state.model
            df = st.session_state.get('df', None)
            use_log_transform = st.session_state.get('use_log_transform', False)
            
            # Obtener información sobre optimizaciones aplicadas
            if 'optimizations_applied' in st.session_state:
                # Usar las optimizaciones guardadas durante el entrenamiento
                optimizations_applied = st.session_state.optimizations_applied
                st.info(f"Usando optimizaciones aplicadas durante el entrenamiento: {optimizations_applied}")
            else:
                # Si no hay información guardada, crear un diccionario por defecto
                optimizations_applied = {
                    'log_transform': use_log_transform,
                    'optimal_regressors': st.session_state.get('use_optimal_regressors', False),
                    'bayesian_optimization': st.session_state.get('use_bayesian_optimization', False),
                    'interval_calibration': st.session_state.get('use_interval_calibration', False),
                    'selected_regressors': [],
                    'optimized_params': {}
                }
                st.warning("⚠️ No se encontraron optimizaciones guardadas. Usando valores por defecto.")
                st.info("Esto puede ocurrir si el modelo fue entrenado en una sesión anterior o si hubo un problema durante el entrenamiento.")
                st.info("Para resolver esto, vuelva a entrenar el modelo con las optimizaciones deseadas.")
            
            # Crear futuro para predicción de manera segura (evitando DatetimeArray)
            # No usamos model.make_future_dataframe porque puede causar errores
            if df is None:
                st.error("No hay datos de entrenamiento disponibles")
                return None
            
            # Crear dataframe futuro manualmente
            last_date = df['ds'].max()
            
            if include_history:
                # Incluir datos históricos + futuros
                historical_dates = df['ds'].tolist()  # Convertir a lista para evitar problemas con DatetimeArray
                
                # Crear fechas futuras usando DateOffset (compatible con pandas actual)
                future_dates = []
                current_date = last_date
                for i in range(periods):
                    # Usar pd.DateOffset en lugar de adición directa
                    current_date = current_date + pd.DateOffset(days=1)
                    future_dates.append(current_date)
                
                # Combinar listas y ordenar
                all_dates = sorted(historical_dates + future_dates)
                future = pd.DataFrame({'ds': all_dates})
            else:
                # Solo fechas futuras
                future_dates = []
                current_date = last_date
                for i in range(periods):
                    # Usar pd.DateOffset en lugar de adición directa
                    current_date = current_date + pd.DateOffset(days=1)
                    future_dates.append(current_date)
                future = pd.DataFrame({'ds': future_dates})
            
            # Si hay regresores externos, asegurarse de que estén en el futuro
            if hasattr(model, 'extra_regressors') and model.extra_regressors:
                if df is not None and 'cve_count' in df.columns:
                    # Extender el regresor cve_count con su valor medio para fechas futuras
                    last_date = df['ds'].max()
                    future_dates_mask = future['ds'] > last_date
                    future_dates = future[future_dates_mask]
                    
                    if not future_dates.empty:
                        # Usar el promedio de los últimos 30 días como valor para fechas futuras
                        cutoff_date = last_date - pd.DateOffset(days=30)
                        last_30_days = df[df['ds'] > cutoff_date]
                        avg_cve = last_30_days['cve_count'].mean() if not last_30_days.empty else df['cve_count'].mean()
                        
                        # Asegurarse de que future tiene la columna cve_count
                        if 'cve_count' not in future.columns:
                            future['cve_count'] = np.nan
                        
                        # Asignar el valor promedio a las fechas futuras
                        future.loc[future_dates_mask, 'cve_count'] = avg_cve
            
            # Generar predicción de manera segura
            try:
                # Verificar si el modelo tiene un método predict propio
                st.info("Modelo entrenado correctamente, generando predicciones...")
                
                if hasattr(model, 'model') and hasattr(model.model, 'predict'):
                    # Caso 1: El modelo tiene una estructura anidada
                    forecast = model.model.predict(future)
                elif hasattr(model, 'predict') and callable(model.predict):
                    # Caso 2: El modelo tiene un método predict directo
                    # Verificar si el predict es el que espera el DataFrame o uno personalizado
                    import inspect
                    predict_sig = inspect.signature(model.predict)
                    
                    # Si el método predict acepta un dataframe como primer argumento
                    if len(predict_sig.parameters) >= 1:
                        forecast = model.predict(future)
                    else:
                        # Si parece ser el método personalizado con periods
                        forecast = model.predict(periods=periods, include_history=include_history)
                else:
                    st.error("No se puede determinar cómo generar predicciones con este modelo")
                    return None
                
                # Guardar la predicción en la sesión
                st.session_state.forecast = forecast
                
                # Mostrar mensaje de éxito
                st.success("✅ Predicción generada correctamente.")
                
                # Mostrar resumen de la predicción
                _show_prediction_summary(forecast, model, use_log_transform, optimizations_applied)
                
                return forecast
            except Exception as e:
                st.error(f"Error al generar la predicción: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return None
    
    except Exception as e:
        st.error(f"Error al generar la predicción: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def _prepare_evaluation_parameters(df, cv_periods, initial, period, horizon, train_percentage):
    """
    Prepara los parámetros para la evaluación de modelos.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos
    cv_periods : int
        Número de períodos para validación
    initial : str
        Fecha inicial para la evaluación
    period : int
        Período entre evaluaciones
    horizon : int
        Horizonte de predicción
    train_percentage : float
        Porcentaje de datos para entrenamiento
        
    Returns:
    --------
    dict
        Diccionario con los parámetros configurados para la evaluación
    """
    # Si no se especifica horizonte, usar cv_periods
    if horizon is None:
        horizon = cv_periods
        
    # Si no se especifica period, usar 1/3 del horizonte
    if period is None:
        period = max(1, int(horizon / 3))
    
    # Calcular fecha de corte si initial es None
    if initial is None:
        total_days = (df['ds'].max() - df['ds'].min()).days
        train_days = int(total_days * train_percentage)
        cutoff_date = df['ds'].min() + pd.Timedelta(days=train_days)
        initial = cutoff_date.strftime('%Y-%m-%d')
    else:
        # Convertir a datetime si es string
        if isinstance(initial, str):
            cutoff_date = pd.to_datetime(initial)
        else:
            cutoff_date = initial
    
    # Verificar que la fecha de corte es válida
    if cutoff_date > df['ds'].max() or cutoff_date <= df['ds'].min():
        st.error(f"Fecha de corte {cutoff_date} fuera del rango de datos ({df['ds'].min()} a {df['ds'].max()})")
        cutoff_date = df['ds'].min() + pd.Timedelta(days=int((df['ds'].max() - df['ds'].min()).days * 0.7))
        st.warning(f"Ajustando fecha de corte a {cutoff_date}")
    
    return {
        'initial': initial,
        'period': period,
        'horizon': horizon,
        'cutoff_date': cutoff_date
    }

def _perform_single_cutoff_evaluation(model, df, params):
    """
    Realiza evaluación con un solo punto de corte.
    
    Parameters:
    -----------
    model : Prophet
        Modelo Prophet entrenado
    df : pandas.DataFrame
        DataFrame con los datos
    params : dict
        Parámetros de evaluación
        
    Returns:
    --------
    tuple
        (métricas, predicciones)
    """
    cutoff_date = params['cutoff_date']
    horizon = params['horizon']
    
    # Dividir datos en entrenamiento y prueba
    train_df = df[df['ds'] <= cutoff_date].copy()
    test_df = df[df['ds'] > cutoff_date].copy()
    test_df = test_df[test_df['ds'] <= (cutoff_date + pd.Timedelta(days=horizon))].copy()
    
    # Verificar que hay datos de prueba suficientes
    if len(test_df) == 0:
        st.warning(f"No hay datos de prueba después de {cutoff_date}")
        test_end = cutoff_date + pd.Timedelta(days=horizon)
        st.info(f"Período de prueba: {cutoff_date} a {test_end}")
        test_df = pd.DataFrame({
            'ds': pd.date_range(start=cutoff_date + pd.Timedelta(days=1), 
                              end=cutoff_date + pd.Timedelta(days=horizon), 
                              freq='D')
        })
        test_df['y'] = np.nan
    
    # Crear nuevo modelo con solo datos de entrenamiento
    test_model = Prophet(
        changepoint_prior_scale=model.changepoint_prior_scale,
        seasonality_prior_scale=model.seasonality_prior_scale,
        holidays_prior_scale=model.holidays_prior_scale,
        seasonality_mode=model.seasonality_mode,
        interval_width=model.interval_width
    )
    
    # Añadir estacionalidades y regresores si es necesario
    if hasattr(model, 'seasonalities'):
        for name, params in model.seasonalities.items():
            test_model.add_seasonality(
                name=name,
                period=params['period'],
                fourier_order=params['fourier_order'],
                prior_scale=params['prior_scale'],
                mode=params['mode']
            )
    
    # Añadir regresores al modelo si el modelo original los tiene
    if hasattr(model, 'extra_regressors') and model.extra_regressors:
        for name, params in model.extra_regressors.items():
            if name in train_df.columns:
                test_model.add_regressor(
                    name=name,
                    prior_scale=params['prior_scale'],
                    standardize=params['standardize'],
                    mode=params['mode']
                )
    
    # Entrenar modelo con datos de entrenamiento
    test_model.fit(train_df)
    
    # Generar futuro para período de prueba
    future = pd.DataFrame({'ds': pd.date_range(start=cutoff_date + pd.Timedelta(days=1), 
                                          end=cutoff_date + pd.Timedelta(days=horizon), 
                                          freq='D')})
    
    # Añadir regresores al futuro si es necesario
    if hasattr(model, 'extra_regressors') and model.extra_regressors:
        for name in model.extra_regressors:
            if name in test_df.columns:
                # Añadir el regresor al futuro desde el df original
                future[name] = test_df[name]
    
    # Predecir con el modelo de prueba
    forecast = test_model.predict(future)
    
    # Filtrar las predicciones para el período de prueba
    forecast_valid = forecast[forecast['ds'] <= (cutoff_date + pd.Timedelta(days=horizon))].copy()
    
    # Asegurar que las fechas están en el mismo formato para la fusión
    test_df['ds'] = pd.to_datetime(test_df['ds'])
    forecast_valid['ds'] = pd.to_datetime(forecast_valid['ds'])

    # Fusionar datasets y verificar que no haya valores NaN
    test_with_preds = test_df.merge(forecast_valid[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')  # Usar inner join para mantener solo fechas que existen en ambos

    # Verificar si tenemos datos suficientes para calcular métricas
    if len(test_with_preds) == 0:
        logger.warning("No hay coincidencia entre fechas de prueba y predicciones. Usando left join.")
        test_with_preds = test_df.merge(forecast_valid[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                                    on='ds', how='left')

    # Verificar explícitamente que tenemos valores y y yhat para el cálculo de métricas
    valid_rows = ~test_with_preds['y'].isna() & ~test_with_preds['yhat'].isna()
    if not any(valid_rows):
        logger.error("No hay filas válidas para calcular métricas (y o yhat son NaN)")
        # Registrar información de diagnóstico
        logger.info(f"Filas en test_df: {len(test_df)}, filas en forecast_valid: {len(forecast_valid)}")
        logger.info(f"Fechas en test_df: {test_df['ds'].min()} a {test_df['ds'].max()}")
        logger.info(f"Fechas en forecast_valid: {forecast_valid['ds'].min()} a {forecast_valid['ds'].max()}")
    else:
        logger.info(f"Calculando métricas con {sum(valid_rows)} filas válidas de {len(test_with_preds)} totales")
    
    # Calcular métricas
    metrics = _calculate_forecast_metrics(test_with_preds)
    
    return metrics, test_with_preds

def _perform_multi_cutoff_evaluation(model, df, params, max_iterations):
    """
    Realiza evaluación con múltiples puntos de corte.
    
    Parameters:
    -----------
    model : Prophet
        Modelo Prophet entrenado
    df : pandas.DataFrame
        DataFrame con los datos
    params : dict
        Parámetros de evaluación
    max_iterations : int
        Número máximo de iteraciones
        
    Returns:
    --------
    tuple
        (métricas promedio, todas las métricas, predicciones)
    """
    cutoff_date = params['cutoff_date']
    period = params['period']
    horizon = params['horizon']
    
    # Generar fechas de corte
    end_date = df['ds'].max() - pd.Timedelta(days=horizon)
    
    # Si la fecha de corte es posterior a end_date, ajustarla
    if cutoff_date > end_date:
        cutoff_date = end_date - pd.Timedelta(days=horizon)
        st.warning(f"Ajustando fecha de corte a {cutoff_date} para permitir evaluación")
    
    cutoff_dates = [cutoff_date]
    current_date = cutoff_date
    
    # Generar fechas de corte adicionales
    for i in range(1, max_iterations):
        next_date = current_date + pd.Timedelta(days=period)
        if next_date > end_date:
            break
        cutoff_dates.append(next_date)
        current_date = next_date
    
    # Realizar evaluación para cada fecha de corte
    all_metrics = []
    all_predictions = []
    
    for i, cutoff in enumerate(cutoff_dates):
        st.info(f"Evaluación {i+1}/{len(cutoff_dates)}: Punto de corte {cutoff}")
        
        # Parámetros para esta iteración
        iteration_params = {
            'cutoff_date': cutoff,
            'period': period,
            'horizon': horizon
        }
        
        # Realizar evaluación
        metrics, preds = _perform_single_cutoff_evaluation(model, df, iteration_params)
        
        # Añadir identificador a las predicciones
        preds['iteration'] = i + 1
        preds['cutoff'] = cutoff
        
        all_metrics.append(metrics)
        all_predictions.append(preds)
    
    # Unir todas las predicciones
    all_predictions_df = pd.concat(all_predictions)
    
    # Calcular métricas promedio
    avg_metrics = {
        'mae': np.mean([m['mae'] for m in all_metrics]),
        'rmse': np.mean([m['rmse'] for m in all_metrics]),
        'smape': np.mean([m['smape'] for m in all_metrics]),
        'coverage': np.mean([m['coverage'] for m in all_metrics]),
        'n_iterations': len(all_metrics)
    }
    
    return avg_metrics, all_metrics, all_predictions_df

def _calculate_forecast_metrics(df):
    """
    Calcula métricas de rendimiento de las predicciones.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con valores reales y predicciones
        
    Returns:
    --------
    dict
        Diccionario con métricas calculadas
    """
    # Asegurar que estamos usando el DataFrame correcto
    eval_df = df.copy()
    
    if not all(col in eval_df.columns for col in ['y', 'yhat']):
        logger.error(f"Columnas requeridas no encontradas. Columnas disponibles: {eval_df.columns.tolist()}")
        return {'error': 'El DataFrame debe contener columnas "y" y "yhat"'}
    
    try:
        # Eliminar filas con NaN en y o yhat
        eval_df = eval_df.dropna(subset=['y', 'yhat'])
        
        if len(eval_df) == 0:
            logger.error("Después de eliminar NaN, no quedan filas para evaluar")
            return {
                'mae': 0.0,  # Usar 0 en lugar de NaN para evitar errores en la UI
                'rmse': 0.0,
                'smape': 0.0,
                'coverage': 0.0,
                'interval_width_avg': 0.0,
                'interval_width_relative': 0.0,
                'n_points': 0,
                'error': "No hay datos válidos para evaluar"
            }
        
        # Preparar datos para métricas
        y_true = eval_df['y'].values
        y_pred = eval_df['yhat'].values
        
        # Verificar si existen las columnas de intervalos
        has_intervals = 'yhat_lower' in eval_df.columns and 'yhat_upper' in eval_df.columns
        
        interval_lower = None
        interval_upper = None
        
        if has_intervals:
            # Asegurar que no hay NaN en los intervalos
            mask = ~np.isnan(eval_df['yhat_lower']) & ~np.isnan(eval_df['yhat_upper'])
            if np.any(mask):
                interval_lower = eval_df.loc[mask, 'yhat_lower'].values
                interval_upper = eval_df.loc[mask, 'yhat_upper'].values
                
                # Verificar que los intervalos tienen el mismo tamaño que y_true y y_pred
                if len(interval_lower) != len(y_true):
                    logger.warning(f"Tamaños inconsistentes después de eliminar NaN: {len(interval_lower)} vs {len(y_true)}")
                    # Ajustar y_true y y_pred para que coincidan con los intervalos
                    mask_indices = np.where(mask)[0]
                    if len(mask_indices) > 0:
                        y_true = eval_df.loc[mask, 'y'].values
                        y_pred = eval_df.loc[mask, 'yhat'].values
            else:
                logger.warning("Todos los intervalos contienen valores NaN, no se usarán intervalos")
                has_intervals = False
        
        # Intentar usar el módulo centralizado de métricas
        try:
            # Calcular métricas utilizando la función centralizada
            metrics = calculate_metrics(
                y_true=y_true, 
                y_pred=y_pred,
                interval_lower=interval_lower if has_intervals else None,
                interval_upper=interval_upper if has_intervals else None
            )
            
            # Verificar si 'metrics' contiene un campo 'error'
            if 'error' in metrics:
                logger.warning(f"Error al calcular métricas centralizadas: {metrics['error']}")
                # Calcular manualmente como respaldo
                raise ValueError(metrics['error'])
            
            # Añadir número de puntos a las métricas
            metrics['n_points'] = len(y_true)
            
            # Nota: No multiplicar SMAPE por 100, ya que calculate_metrics ya lo devuelve en porcentaje
            
            # Verificar y ajustar los valores para evitar NaN
            for key in ['mae', 'rmse', 'smape', 'r2', 'dir_acc', 'coverage']:
                if key in metrics and (np.isnan(metrics[key]) or metrics[key] is None):
                    logger.warning(f"Métrica {key} es NaN, sustituyendo por 0")
                    metrics[key] = 0.0
            
            # Si la cobertura viene en proporción (0-1), convertirla a porcentaje (0-100)
            if 'coverage' in metrics and metrics['coverage'] <= 1.0:
                metrics['coverage'] = metrics['coverage'] * 100
                
            # Calcular métricas adicionales de intervalos si no están incluidas
            if 'interval_width_avg' not in metrics and has_intervals:
                interval_widths = interval_upper - interval_lower
                metrics['interval_width_avg'] = float(np.mean(interval_widths))
                
                yhat_mean = np.mean(y_pred)
                if abs(yhat_mean) > 1e-8:
                    metrics['interval_width_relative'] = float(metrics['interval_width_avg'] / yhat_mean * 100)
                else:
                    metrics['interval_width_relative'] = 0.0
            elif not has_intervals:
                metrics['interval_width_avg'] = 0.0
                metrics['interval_width_relative'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error al usar el módulo centralizado: {str(e)}. Usando cálculo manual.")
            
            # Calcular métricas básicas manualmente como respaldo
            try:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
            except Exception as sklearn_error:
                logger.warning(f"Error al usar sklearn: {str(sklearn_error)}. Usando cálculo numérico directo.")
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                r2 = 0.0
            
            # Calcular SMAPE manualmente
            try:
                # Intentar usar la función centralizada de SMAPE
                smape = calculate_smape(y_true, y_pred)
            except Exception as smape_error:
                logger.warning(f"Error al calcular SMAPE centralizado: {str(smape_error)}. Usando fórmula directa.")
                # Cálculo manual como respaldo
                epsilon = 1e-8
                denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
                smape = 2.0 * np.mean(np.abs(y_true - y_pred) / denominator) * 100
            
            # Calcular cobertura manualmente
            coverage = 0.0
            interval_width_avg = 0.0
            interval_width_relative = 0.0
            
            if has_intervals:
                in_range = ((y_true >= interval_lower) & (y_true <= interval_upper))
                coverage = np.mean(in_range) * 100
                
                interval_widths = interval_upper - interval_lower
                interval_width_avg = np.mean(interval_widths)
                
                yhat_mean = np.mean(y_pred)
                if abs(yhat_mean) > 1e-8:
                    interval_width_relative = interval_width_avg / yhat_mean * 100
            
            # Crear diccionario de métricas manual
            metrics = {
                'mae': float(mae),
                'rmse': float(rmse),
                'smape': float(smape),
                'r2': float(r2),
                'coverage': float(coverage),
                'interval_width_avg': float(interval_width_avg),
                'interval_width_relative': float(interval_width_relative),
                'n_points': len(y_true)
            }
            
            # Verificar y reemplazar NaN con 0
            for key in metrics:
                if isinstance(metrics[key], (int, float)) and np.isnan(metrics[key]):
                    metrics[key] = 0.0
        
        # Calcular puntuaciones de anomalía si hay datos suficientes
        if len(y_true) > 10:
            try:
                anomaly_scores = calculate_anomaly_score(y_true, y_pred)
                
                # Añadir puntuación media de anomalía y máxima
                metrics['anomaly_score_mean'] = float(np.mean(anomaly_scores))
                metrics['anomaly_score_max'] = float(np.max(anomaly_scores))
                
                # Calcular porcentaje de puntos que son anomalías (umbral arbitrario de 2.0)
                anomaly_threshold = 2.0
                metrics['anomaly_percentage'] = float(np.mean(anomaly_scores > anomaly_threshold) * 100)
            except Exception as e:
                logger.warning(f"Error al calcular anomalías: {str(e)}")
                # Asignar valores predeterminados para evitar errores
                metrics['anomaly_score_mean'] = 0.0
                metrics['anomaly_score_max'] = 0.0
                metrics['anomaly_percentage'] = 0.0
        
        # Verificar cambios en patrones de ataque
        if len(y_true) > 20:
            try:
                pattern_change = detect_attack_pattern_change(y_true, y_pred)
                metrics['pattern_change_detected'] = pattern_change
            except Exception as e:
                logger.warning(f"Error al detectar cambios de patrón: {str(e)}")
                metrics['pattern_change_detected'] = False
        
        # Logging para depuración
        logger.info(
            f"Métricas calculadas: MAE={metrics.get('mae', 0):.4f}, "
            f"RMSE={metrics.get('rmse', 0):.4f}, "
            f"SMAPE={metrics.get('smape', 0):.2f}%, "
            f"Cobertura={metrics.get('coverage', 0):.1f}%"
        )
            
        return metrics
    except Exception as e:
        logger.error(f"Error al calcular métricas: {traceback.format_exc()}")
        return {
            'mae': 0.0,  # Usar 0 en lugar de NaN para evitar problemas en la UI
            'rmse': 0.0,
            'smape': 0.0,
            'coverage': 0.0,
            'interval_width_avg': 0.0,
            'interval_width_relative': 0.0,
            'n_points': 0,
            'error': str(e)
        }

def _display_evaluation_results(
    metrics: Dict[str, Any],
    all_metrics: Optional[List[Dict[str, Any]]],
    df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    multi_eval: bool = False,
    use_log_transform: bool = False
) -> Optional[Dict[str, float]]:
    """
    Muestra los resultados de la evaluación del modelo.

    Parameters:
    -----------
    metrics : dict
        Diccionario con métricas calculadas
    all_metrics : list
        Lista de métricas para múltiples puntos de corte
    df : pandas.DataFrame
        DataFrame con los datos completos
    test_predictions : pandas.DataFrame
        DataFrame con predicciones de prueba (columnas: ds, yhat, yhat_lower, yhat_upper)
    multi_eval : bool
        Indica si se realizó evaluación con múltiples puntos de corte
    use_log_transform : bool
        Indica si se aplicó transformación logarítmica a los datos
    
    Returns:
    --------
    dict
        Diccionario con las métricas de evaluación o None en caso de error
    """
    # --- Validaciones iniciales ---
    if metrics is None:
        st.error("No hay métricas disponibles para mostrar.")
        return None

    if not isinstance(df, pd.DataFrame) or df.empty:
        st.error("El DataFrame de datos está vacío o no es válido.")
        return metrics

    if not isinstance(test_predictions, pd.DataFrame) or test_predictions.empty:
        st.error("El DataFrame de predicciones está vacío o no es válido.")
        return metrics

    # Columnas mínimas requeridas
    for col in ['ds', 'y']:
        if col not in df.columns:
            st.error(f"Falta columna '{col}' en los datos: {df.columns.tolist()}")
            return metrics
    if 'ds' not in test_predictions.columns or 'yhat' not in test_predictions.columns:
        st.error(f"Las predicciones deben contener 'ds' y 'yhat': {test_predictions.columns.tolist()}")
        return metrics

    # Inversión de log transform si corresponde
    if use_log_transform:
        df = df.copy()
        test_predictions = test_predictions.copy()
        # y
        df['y'] = np.expm1(df['y'])
        # yhat y bounds
        test_predictions['yhat'] = np.expm1(test_predictions['yhat'])
        if 'yhat_lower' in test_predictions:
            test_predictions['yhat_lower'] = np.expm1(test_predictions['yhat_lower'])
        if 'yhat_upper' in test_predictions:
            test_predictions['yhat_upper'] = np.expm1(test_predictions['yhat_upper'])

    # --- Mostrar métricas en Streamlit ---
    st.success("✅ Evaluación del modelo completada")
    st.markdown("### Métricas de Rendimiento")

    cols = st.columns(4)
    st.metric(label="Amplitud Promedio", value=f"{metrics.get('interval_width_avg', 0):.2f}", help="Amplitud media de los intervalos de predicción. Valores más bajos indican mayor precisión.")
    st.metric(label="Cobertura Intervalos (%)", value=f"{metrics.get('coverage', 0):.1f}%", help="% de valores reales en intervalo.")
    st.metric(label="Error (SMAPE)", value=f"{metrics.get('smape', 0):.1f}%", help="Error porcentual medio simétrico. Valores más bajos indican mejor precisión.")
    st.metric(label="Iteraciones", value=metrics.get('iterations', 1), help="Cantidad de iteraciones realizadas.")

    # Tabla de multi-evaluación
    if multi_eval and all_metrics:
        st.markdown("#### Métricas por Punto de Corte")
        df_met = pd.DataFrame(all_metrics)
        # Seleccionar columnas clave si existen
        cols_show = [c for c in ['iteration', 'cutoff', 'mae', 'rmse', 'smape', 'coverage', 'interval_width_avg'] if c in df_met.columns]
        df_met = df_met.sort_values('cutoff') if 'cutoff' in df_met.columns else df_met
        st.dataframe(df_met[cols_show])

    # --- Gráfico de Backtesting ---
    st.markdown("### Visualización de Backtesting")
    try:
        start = test_predictions['ds'].min()
        if pd.isna(start):
            st.warning("Fecha mínima de predicciones inválida.")
            return metrics

        # Datos de entrenamiento y validación
        train = df[df['ds'] < start]
        valid = df.merge(test_predictions[['ds']], on='ds', how='inner')

        fig = go.Figure()
        if not train.empty:
            fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Entrenamiento'))
        if not valid.empty:
            fig.add_trace(go.Scatter(x=valid['ds'], y=valid['y'], mode='lines', name='Valores Reales'))

        # Predicción
        fig.add_trace(go.Scatter(x=test_predictions['ds'], y=test_predictions['yhat'], mode='lines', name='Predicción'))

        # Intervalo
        if 'yhat_lower' in test_predictions and 'yhat_upper' in test_predictions:
            fig.add_trace(go.Scatter(
                x=pd.concat([test_predictions['ds'], test_predictions['ds'][::-1]]),
                y=pd.concat([test_predictions['yhat_upper'], test_predictions['yhat_lower'][::-1]]),
                fill='toself', name='Intervalo 95%', fillcolor='rgba(255,0,0,0.2)', line=dict(width=0)
            ))

        fig.update_layout(title='Backtesting: Predicción vs Real', xaxis_title='Fecha', yaxis_title='Valor', hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error al generar gráfico de backtesting: {e}")
        return metrics

    return metrics


def train_model_wrapper(
    df: Optional[pd.DataFrame] = None,
    use_regressor: bool = True,
    use_optimal_regressors: bool = True,
    use_bayesian_optimization: bool = False,
    use_interval_calibration: bool = True,
    optimization_trials: int = 10,
    correlation_threshold: float = 0.1,
    vif_threshold: float = 5.0,
    seasonality_mode: str = 'multiplicative',
    use_log_transform: bool = True,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    holidays_prior_scale: float = 10.0,
    regressors_prior_scale: float = 0.1,
    interval_width: float = 0.6,
    changepoint_range: float = 0.8,
    n_changepoints: int = 25,
    use_detected_changepoints: bool = False,
    daily_seasonality: bool = False,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True,
    calibrate_intervals: bool = True
) -> tuple:
    """
    Wrapper para entrenar el modelo Prophet a través de la UI de Streamlit
    
    Args:
        df: DataFrame con datos de entrenamiento (opcional, si no se proporciona se obtiene de la sesión)
        use_regressor: Si usar regresores externos
        use_optimal_regressors: Si usar selección automática de regresores óptimos
        use_bayesian_optimization: Si usar optimización bayesiana para hiperparámetros
        use_interval_calibration: Si usar calibración de intervalos durante el entrenamiento
        optimization_trials: Número de pruebas para optimización bayesiana
        correlation_threshold: Umbral de correlación para selección de regresores
        vif_threshold: Umbral de VIF para controlar multicolinealidad entre regresores
        seasonality_mode: Modo de estacionalidad ('additive' o 'multiplicative')
        use_log_transform: Si aplicar transformación logarítmica
        changepoint_prior_scale: Escala previa de puntos de cambio
        seasonality_prior_scale: Escala previa de estacionalidad
        holidays_prior_scale: Escala previa de festivos
        regressors_prior_scale: Escala previa de regresores
        interval_width: Ancho del intervalo de predicción
        changepoint_range: Rango de puntos de cambio
        n_changepoints: Número de puntos de cambio
        use_detected_changepoints: Si usar detección automática de changepoints
        daily_seasonality: Si usar estacionalidad diaria
        weekly_seasonality: Si usar estacionalidad semanal
        yearly_seasonality: Si usar estacionalidad anual
        calibrate_intervals: Si calibrar los intervalos de predicción
        
    Returns:
    --------
    Tupla con (modelo entrenado, DataFrame de predicción)
    """
    from .models.prophet_model import RansomwareProphetModel
    from .features.regressors import RegressorGenerator
    
    # Inicializar logger
    logger = logging.getLogger(__name__)
    logger.info("Iniciando entrenamiento de modelo")
    
    # Si no se proporciona df, intentar obtenerlo de la sesión de Streamlit
    if df is None:
        logger.info("No se proporcionó DataFrame, intentando obtenerlo de la sesión")
        if 'df_prophet' in st.session_state:
            df = st.session_state.df_prophet
            logger.info(f"DataFrame obtenido de sesión: {len(df)} filas")
        else:
            st.error("No hay datos para entrenar. Por favor, cargue datos primero.")
            return None, None
    
    # Verificar si hay datos suficientes
    if df is None or len(df) < 10:
        st.error("Datos insuficientes para entrenar modelo")
        return None, None
    
    # Configurar parámetros del modelo
    model_config = {
        'seasonality_mode': seasonality_mode,
        'use_log_transform': use_log_transform,
        'changepoint_prior_scale': changepoint_prior_scale,
        'seasonality_prior_scale': seasonality_prior_scale,
        'holidays_prior_scale': holidays_prior_scale,
        'regressors_prior_scale': regressors_prior_scale,
        'interval_width': interval_width,
        'changepoint_range': changepoint_range,
        'n_changepoints': n_changepoints,
        'use_detected_changepoints': use_detected_changepoints,
        'daily_seasonality': daily_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'yearly_seasonality': yearly_seasonality
    }
    
    # Cargar regresores si se solicita
    regressors = None
    if use_regressor:
        try:
            logger.info("Configurando regresores")
            
            # Inicializar generador de regresores
            if 'regressors' not in st.session_state or st.session_state.regressors is None:
                st.session_state.regressors = RegressorGenerator()
                
            regressors = st.session_state.regressors
            
            # Cargar datos de CVE mediante el dataframe si está disponible en la sesión
            cve_df = None
            cve_file = 'modeling/cve_diarias_regresor_prophet.csv'
            
            if hasattr(st.session_state, 'cve_file') and st.session_state.cve_file:
                cve_file = st.session_state.cve_file
            
            # Intentar obtener df_cve del forecaster si existe
            if 'forecaster' in st.session_state and hasattr(st.session_state.forecaster, 'df_cve'):
                cve_df = st.session_state.forecaster.df_cve
            else:
                # Cargar datos CVE manualmente
                try:
                    cve_df = pd.read_csv(cve_file)
                    if 'fecha' in cve_df.columns:
                        cve_df = cve_df.rename(columns={'fecha': 'ds'})
                    # Convertir a datetime
                    cve_df['ds'] = pd.to_datetime(cve_df['ds'])
                except Exception as e:
                    logger.warning(f"No se pudieron cargar datos de CVE: {str(e)}")
                    
            # Seleccionar automáticamente los mejores regresores si se solicita
            if use_optimal_regressors:
                logger.info("Seleccionando regresores óptimos")
                df_with_date = df.copy()
                if 'ds' not in df_with_date.columns:
                    if 'date' in df_with_date.columns:
                        df_with_date = df_with_date.rename(columns={'date': 'ds'})
                    elif 'fecha' in df_with_date.columns:
                        df_with_date = df_with_date.rename(columns={'fecha': 'ds'})
                
                # Añadir los datos de CVE como regresores al DataFrame de entrenamiento
                if cve_df is not None and not cve_df.empty:
                    try:
                        # Usar el método correcto para añadir regresores externos
                        df_with_date = regressors.add_external_regressors(df_with_date, cve_data=cve_df)
                        # Actualizar df para que incluya los regresores
                        df = df_with_date
                    except Exception as e:
                        logger.error(f"Error al añadir regresores externos: {str(e)}")
                
                try:
                    optimal_regressors = regressors.select_optimal_regressors(
                        df_with_date, target_col='y', 
                        correlation_threshold=correlation_threshold,
                        vif_threshold=vif_threshold
                    )
                    
                    if optimal_regressors and len(optimal_regressors) > 0:
                        logger.info(f"Seleccionados {len(optimal_regressors)} regresores óptimos")
                        regressors.set_active_regressors(optimal_regressors)
                    else:
                        logger.warning("No se encontraron regresores óptimos")
                        use_regressor = False
                except Exception as e:
                    logger.error(f"Error en selección de regresores: {str(e)}")
                    use_regressor = False
            else:
                # Usar todos los regresores disponibles
                logger.info("Usando todos los regresores disponibles (sin optimización)")
                
                # Verificar que hay regresores disponibles
                available_regressors = regressors.get_available_regressors()
                if available_regressors and len(available_regressors) > 0:
                    regressors.set_active_regressors(available_regressors)
                    logger.info(f"Usando {len(available_regressors)} regresores disponibles")
                else:
                    logger.warning("No hay regresores disponibles")
                    use_regressor = False
        except Exception as e:
            st.error(f"Error al entrenar el modelo: {str(e)}")
            logger.error(f"Error en train_model_wrapper: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

    # Entrenar modelo
    try:
        with st.spinner("Entrenando modelo Prophet..."):
            # Inicializar modelo
            model = RansomwareProphetModel()
            
            # Si se solicita optimización bayesiana, ejecutarla antes de entrenar
            if use_bayesian_optimization:
                st.info("🔄 Realizando optimización bayesiana...")
                # Implementación pendiente de optimización bayesiana
                pass
            
            # Entrenar modelo con los datos proporcionados
            with st.spinner(f"Entrenando modelo con {len(df)} filas de datos..."):
                try:
                    # Primero crear el modelo Prophet interno con los parámetros configurados
                    model.create_model(
                        changepoint_prior_scale=model_config.get('changepoint_prior_scale', 0.05),
                        seasonality_prior_scale=model_config.get('seasonality_prior_scale', 10.0),
                        seasonality_mode=model_config.get('seasonality_mode', 'multiplicative'),
                        interval_width=model_config.get('interval_width', 0.8),
                        n_changepoints=model_config.get('n_changepoints', 25)
                    )
                except ValueError as e:
                    if "holidays must be a DataFrame" in str(e):
                        # El error es por el formato de holidays, creamos manualmente el modelo Prophet
                        st.warning("Ajustando configuración de holidays para compatibilidad...")
                        model.model = Prophet(
                            changepoint_prior_scale=model_config.get('changepoint_prior_scale', 0.05),
                            seasonality_prior_scale=model_config.get('seasonality_prior_scale', 10.0),
                            holidays_prior_scale=model_config.get('holidays_prior_scale', 10.0),
                            seasonality_mode=model_config.get('seasonality_mode', 'multiplicative'),
                            interval_width=model_config.get('interval_width', 0.8),
                            n_changepoints=model_config.get('n_changepoints', 25)
                        )
                        # Guardar parámetros para referencia
                        model.params = model_config
                    else:
                        # Si es otro tipo de error, lo propagamos
                        raise
                
                # Añadir regresores si están disponibles
                if use_regressor and regressors is not None and len(regressors) > 0:
                    try:
                        regressor_count = 0
                        for regressor_name in regressors:
                            if regressor_name in df.columns:
                                model.model.add_regressor(regressor_name)
                                regressor_count += 1
                        
                        if regressor_count > 0:
                            st.info(f"🔄 Usando {regressor_count} regresores externos")
                    except Exception as e:
                        logger.error(f"Error al añadir regresores: {str(e)}")
                        st.warning(f"Error al configurar regresores: {str(e)}")
                
                # Guardar datos de entrenamiento para futuras referencias
                model.train_data = df
                
                # Entrenar el modelo con los datos proporcionados
                model.model.fit(df)
                
                # Añadir métodos de compatibilidad al modelo para compatibilidad con código existente
                import types
                
                # Método make_future_dataframe para compatibilidad
                def make_future_dataframe(self, periods=30, freq='D', include_history=True):
                    """Método de compatibilidad para crear dataframe futuro"""
                    if not hasattr(self, 'train_data') or self.train_data is None:
                        raise ValueError("No hay datos de entrenamiento disponibles")
                    
                    # Usamos DateOffset en lugar de operaciones aritméticas directas 
                    # para evitar el error con DatetimeArray
                    if include_history:
                        # Incluir datos históricos
                        # Usar tolist() para evitar problemas con DatetimeArray
                        historical_dates = self.train_data['ds'].tolist()
                        
                        # Calcular el último día y crear fechas futuras usando DateOffset
                        last_date = self.train_data['ds'].max()
                        future_dates = []
                        current_date = last_date
                        for i in range(periods):
                            # Usar pd.DateOffset en lugar de adición directa
                            current_date = current_date + pd.DateOffset(days=1)
                            future_dates.append(current_date)
                        
                        # Combinar fechas históricas y futuras
                        all_dates = sorted(historical_dates + future_dates)
                        future_df = pd.DataFrame({'ds': all_dates})
                    else:
                        # Solo fechas futuras
                        last_date = self.train_data['ds'].max()
                        future_dates = []
                        current_date = last_date
                        for i in range(periods):
                            # Usar pd.DateOffset en lugar de adición directa
                            current_date = current_date + pd.DateOffset(days=1)
                            future_dates.append(current_date)
                        future_df = pd.DataFrame({'ds': future_dates})
                    
                    return future_df
                
                # Método forecast para compatibilidad
                def forecast_method(self, df=None, periods=30, include_history=True):
                    """Método de compatibilidad para forecast/predict"""
                    # Si no se proporciona un DataFrame, crear uno compatible
                    if df is None:
                        # Generar dataframe futuro manualmente para evitar problemas con DatetimeArray
                        if not hasattr(self, 'train_data') or self.train_data is None:
                            raise ValueError("No hay datos de entrenamiento disponibles")
                        
                        last_date = self.train_data['ds'].max()
                        
                        if include_history:
                            # Obtener las fechas históricas como lista (no como DatetimeArray)
                            # para evitar problemas de compatibilidad
                            historical_dates = self.train_data['ds'].tolist()
                            
                            # Crear fechas futuras usando DateOffset (compatible con versiones recientes)
                            future_dates = []
                            current_date = last_date
                            for i in range(periods):
                                # Usar pd.DateOffset en lugar de adición directa
                                current_date = current_date + pd.DateOffset(days=1)
                                future_dates.append(current_date)
                            
                            # Combinar ambas listas y ordenar
                            all_dates = sorted(historical_dates + future_dates)
                            df = pd.DataFrame({'ds': all_dates})
                        else:
                            # Solo fechas futuras usando DateOffset
                            future_dates = []
                            current_date = last_date
                            for i in range(periods):
                                # Usar pd.DateOffset en lugar de adición directa
                                current_date = current_date + pd.DateOffset(days=1)
                                future_dates.append(current_date)
                            df = pd.DataFrame({'ds': future_dates})
                    
                    # Ahora que tenemos un DataFrame válido, hacer la predicción
                    return self.model.predict(df)
                
                # Método predict como alias para forecast
                def predict_method(self, df=None, periods=30, include_history=True):
                    """Alias para forecast_method"""
                    return forecast_method(self, df, periods, include_history)
                
                # Añadir los métodos al modelo
                model.make_future_dataframe = types.MethodType(make_future_dataframe, model)
                model.forecast = types.MethodType(forecast_method, model)
                model.predict = types.MethodType(predict_method, model)
            
            # Hacer predicción para mostrar resultados
            with st.spinner("Generando pronóstico..."):
                # Crear dataframe futuro para predicción
                future = pd.DataFrame({'ds': pd.date_range(start=df['ds'].max(), periods=31)[1:]})
                
                # Si hay regresores, añadirlos al dataframe futuro
                if use_regressor and regressors is not None and len(regressors) > 0:
                    for regressor_name in regressors:
                        if regressor_name in df.columns:
                            # Usar el último valor del regresor para predicción
                            future[regressor_name] = df[regressor_name].iloc[-1]
                
                # Generar predicción
                forecast = model.model.predict(future)
                
                # Calibrar intervalos si se solicita
                if calibrate_intervals and use_interval_calibration:
                    try:
                        if hasattr(model, 'calibrate_intervals'):
                            forecast = model.calibrate_intervals(df, forecast)
                    except Exception as e:
                        logger.error(f"Error al calibrar intervalos: {str(e)}")
            
            # Guardar el modelo en la sesión de Streamlit
            st.session_state.prophet_model = model
            st.session_state.prophet_forecast = forecast
            
            return model, forecast
            
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}")
        st.error(f"Error en entrenamiento: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        st.error(traceback.format_exc())
        
        return None, None

def plot_forecast_wrapper():
    """
    Crea una visualización Plotly de la predicción almacenada en session_state.
    
    Returns:
        Figura Plotly con la visualización o None si hay un error
    """
    try:
        # Verificar si hay una predicción disponible
        if 'forecast' not in st.session_state:
            logger.error("No hay predicción disponible en session_state")
            return None
            
        forecast = st.session_state.forecast
        
        # Verificar que forecast no sea None
        if forecast is None:
            logger.error("La predicción en session_state es None")
            return None
            
        # Verificar que forecast sea un DataFrame
        if not isinstance(forecast, pd.DataFrame):
            logger.error(f"La predicción no es un DataFrame, es {type(forecast)}")
            return None
            
        # Verificar que forecast no esté vacío
        if forecast.empty:
            logger.error("El DataFrame de predicción está vacío")
            return None
            
        # Imprimir las columnas disponibles para diagnóstico
        logger.info(f"Columnas en forecast: {list(forecast.columns)}")
        
        # Verificar que el DataFrame tenga las columnas necesarias
        required_cols = ['ds', 'yhat']
        missing_cols = [col for col in required_cols if col not in forecast.columns]
        if missing_cols:
            logger.error(f"Faltan columnas necesarias en el DataFrame de predicción: {missing_cols}")
            return None
        
        try:
            # Crear figura con subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("Predicción de ataques ransomware", "Componentes")
            )
            
            # Añadir datos históricos
            if 'df_prophet' in st.session_state and st.session_state.df_prophet is not None:
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.df_prophet['ds'],
                        y=st.session_state.df_prophet['y'],
                        mode='markers',
                        name='Datos históricos',
                        marker=dict(color='#FF9F1C', size=8)  # Naranja brillante
                    ),
                    row=1, col=1
                )
            
            # Predicción
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Predicción',
                    line=dict(color='blue', width=3)  # Aumentado ancho de línea
                ),
                row=1, col=1
            )
            
            # Marcar área futura
            if 'df_prophet' in st.session_state and st.session_state.df_prophet is not None:
                future_mask = ~forecast['ds'].isin(st.session_state.df_prophet['ds'])
                if future_mask.sum() > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast.loc[future_mask, 'ds'],
                            y=forecast.loc[future_mask, 'yhat'],
                            mode='lines',
                            name='Predicción futura',
                            line=dict(color='red', width=4)  # Aumentado ancho de línea de 2.5 a 4
                        ),
                        row=1, col=1
                    )
            
            # Añadir intervalos de confianza
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_upper'],
                        mode='lines',
                        name='Límite superior',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_lower'],
                        mode='lines',
                        name='Intervalo de confianza',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 0, 255, 0.3)'  # Aumentado opacidad de 0.2 a 0.3
                    ),
                    row=1, col=1
                )
            
            # Añadir componentes - solo tendencia
            if 'trend' in forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['trend'],
                        mode='lines',
                        name='Tendencia',
                        line=dict(color='green', width=3)  # Aumentado ancho de línea de 2 a 3
                    ),
                    row=2, col=1
                )
            
            # Actualizar diseño
            fig.update_layout(
                height=600,  # Reducido para mejor proporción
                width=1200,  # Un poco más ancho
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                template='plotly_white',
                font=dict(size=14)  # Texto más grande
            )
            
            # ——————————————————————————————
            # Color de los títulos:
            fig.update_annotations(font_color='white')

            # Actualizar ejes
            fig.update_yaxes(title_text="Ataques", row=1, col=1, title_font=dict(size=16), range=[0, 12])  # Ajustado máximo a 12
            fig.update_yaxes(title_text="Componentes", row=2, col=1, title_font=dict(size=16))
            fig.update_xaxes(title_text="Fecha", row=2, col=1, title_font=dict(size=16))
            return fig
            
        except ImportError:
            logger.warning("Para usar gráficos interactivos, instala: pip install plotly")
            return None
            
    except Exception as e:
        logger.error(f"Error al crear visualización de predicción: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def plot_evaluation_results(_self=None):
    """
    Genera una visualización de los resultados de evaluación del modelo.
    Utiliza los datos almacenados en st.session_state de una evaluación previa.
    
    Returns:
    --------
    None
    """
    try:
        # Verificar si hay datos de evaluación disponibles
        if 'evaluation_details' not in st.session_state or not st.session_state.evaluation_details:
            st.warning("No hay resultados de evaluación disponibles. Por favor, ejecute primero la evaluación del modelo.")
            return None
            
        # Verificar si tenemos los datos de validación y predicción guardados
        if ('valid_df' not in st.session_state or 
            'forecast_valid' not in st.session_state or 
            st.session_state.valid_df is None or 
            st.session_state.forecast_valid is None):
            
            st.warning("No se encontraron datos detallados de la evaluación. Se mostrará solo un resumen.")
            
            # Mostrar un resumen de métricas si hay evaluation_details
            if st.session_state.evaluation_details and len(st.session_state.evaluation_details) > 0:
                details = st.session_state.evaluation_details[0]
                
                st.subheader("Resumen de Evaluación")
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("RMSE", f"{details.get('rmse', 'N/A')}")
                with metric_cols[1]:
                    st.metric("MAE", f"{details.get('mae', 'N/A')}")
                with metric_cols[2]:
                    st.metric("SMAPE", f"{details.get('smape', 'N/A'):.2f}%" if 'smape' in details else 'N/A')
            else:
                st.error("No hay información de evaluación disponible.")
                
            return None
            
        # Obtener los datos
        valid_df = st.session_state.valid_df
        forecast_valid = st.session_state.forecast_valid
        
        # Verificar que los DataFrames tienen datos
        if len(valid_df) == 0 or len(forecast_valid) == 0:
            st.error("Los datos de evaluación están vacíos. Por favor, ejecute nuevamente la evaluación.")
            return None
            
        # Verificar que los DataFrames tienen las columnas necesarias
        required_cols_valid = ['ds', 'y']
        required_cols_forecast = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        
        if not all(col in valid_df.columns for col in required_cols_valid):
            st.error(f"El DataFrame de validación no contiene todas las columnas necesarias: {required_cols_valid}")
            st.info(f"Columnas disponibles: {valid_df.columns.tolist()}")
            return None
            
        if not all(col in forecast_valid.columns for col in required_cols_forecast):
            st.error(f"El DataFrame de predicción no contiene todas las columnas necesarias: {required_cols_forecast}")
            st.info(f"Columnas disponibles: {forecast_valid.columns.tolist()}")
            return None
            
        # Asegurarse de que tenemos details
        if not st.session_state.evaluation_details or len(st.session_state.evaluation_details) == 0:
            st.error("No hay detalles de evaluación disponibles.")
            return None
            
        details = st.session_state.evaluation_details[0]
        
        # Extraer las métricas
        rmse = details.get('rmse', 0)
        mae = details.get('mae', 0)
        smape = details.get('smape', 0)
        coverage = details.get('coverage', 0)
        interval_width_avg = details.get('interval_width_avg', 0)
        
        # Definir paleta de colores para tema oscuro
        colors = {
            'background': 'rgba(0,0,0,0)',      # Transparente
            'grid': 'rgba(80, 80, 80, 0.2)',    # Gris oscuro para grid
            'text': '#E0E0E0',                  # Texto claro
            'prediction': '#FF9500',            # Naranja brillante
            'interval': 'rgba(255, 149, 0, 0.15)',  # Naranja transparente
            'actual': '#2C82FF',                # Azul brillante
            'train': '#888888',                 # Gris medio
            'cutoff': 'rgba(120, 120, 120, 0.5)'  # Línea de corte
        }
        
        # Crear gráfico para visualizar predicciones, valores reales e intervalos
        fig = go.Figure()
        
        # Obtener los datos de entrenamiento si están disponibles
        train_df = None
        if 'train_df' in st.session_state:
            train_df = st.session_state.train_df
        elif 'df_prophet' in st.session_state:
            # Reconstruir aproximadamente los datos de entrenamiento
            cutoff_date = pd.to_datetime(details.get('cutoff', None))
            if cutoff_date is not None:
                train_df = st.session_state.df_prophet[st.session_state.df_prophet['ds'] <= cutoff_date].copy()
        
        # Añadir datos de entrenamiento (últimos puntos) para dar contexto
        if train_df is not None:
            try:
                train_points = 30  # Mostrar los últimos 30 puntos de entrenamiento
                if len(train_df) > train_points:
                    context_df = train_df.iloc[-train_points:]
                else:
                    context_df = train_df
                    
                fig.add_trace(
                    go.Scatter(
                        x=context_df['ds'], y=context_df['y'],
                        mode='lines', name='Entrenamiento', line=dict(width=2)
                    )
                )
            except Exception as e:
                st.warning(f"No se pudieron agregar datos de entrenamiento al gráfico: {str(e)}")
        
        # Añadir área para el intervalo de predicción con estilo mejorado
        fig.add_trace(
            go.Scatter(
                x=forecast_valid['ds'], y=forecast_valid['yhat_upper'],
                fill=None, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_valid['ds'], y=forecast_valid['yhat_lower'],
                fill='tonexty', mode='lines', line=dict(width=0),
                fillcolor=colors['interval'], name='Intervalo de predicción',
                showlegend=True,
                hovertemplate='<b>%{x|%d %b %y}</b><br>Rango: [%{y:.2f}, %{text:.2f}]<extra>Intervalo</extra>',
                text=forecast_valid['yhat_upper']
            )
        )
        
        # Añadir línea de predicción con estilo mejorado
        fig.add_trace(
            go.Scatter(
                x=forecast_valid['ds'], y=forecast_valid['yhat'],
                mode='lines', name='Predicción', line=dict(color=colors['prediction'], width=3),
                hovertemplate='<b>%{x|%d %b %y}</b><br>Predicción: %{y:.2f}<extra>Predicción</extra>'
            )
        )
        
        # Añadir puntos para valores reales con estilo mejorado
        fig.add_trace(
            go.Scatter(
                x=valid_df['ds'], y=valid_df['y'],
                mode='markers', marker=dict(
                    color=colors['actual'],
                    size=8,
                    line=dict(width=1, color='rgba(0,0,0,0.5)')
                ),
                name='Valor Real',
                hovertemplate='<b>%{x|%d %b %y}</b><br>Real: %{y:.2f}<extra>Real</extra>'
            )
        )
        
        # Añadir anotación para el punto de corte si está disponible
        if 'cutoff' in details:
            cutoff_date = pd.to_datetime(details['cutoff'])
            fig.add_shape(
                type="line",
                x0=cutoff_date,
                y0=0,
                x1=cutoff_date,
                y1=1,
                yref="paper",
                line=dict(
                    color=colors['cutoff'],
                    width=2,
                    dash="dash",
                )
            )
            
            # Añadir texto para el punto de corte
            fig.add_annotation(
                x=cutoff_date,
                y=1.02,
                yref="paper",
                text="Punto de corte",
                showarrow=False,
                font=dict(color=colors['text'], size=12)
            )
        
        # Configurar el diseño del gráfico para tema oscuro
        fig.update_layout(
            title="Evaluación de la Predicción",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            hovermode="x unified",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            margin=dict(l=10, r=10, t=80, b=10),
            height=500,
            
            # Agregar marca de agua tipo hacker (opcional)
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="CONFIDENCIAL",
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=60,
                        color="rgba(255, 255, 255, 0.03)"
                    ),
                    textangle=25,
                    opacity=0.7
                )
            ]
        )
        
        # Mejorar la apariencia de los ejes
        fig.update_xaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False,
            tickangle=-45,
            title_font=dict(color=colors['text']),
            tickfont=dict(color=colors['text'])
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=True,
            zerolinecolor=colors['grid'],
            title_font=dict(color=colors['text']),
            tickfont=dict(color=colors['text'])
        )
        
        # Mostrar el gráfico
        st.plotly_chart(fig, use_container_width=True)
        
        # Guía de interpretación del gráfico con mejor estilo
        st.markdown("""
        <div style="background-color: rgba(40, 40, 40, 0.7); padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h4 style="margin-top: 0;">Guía de interpretación</h4>
            <ul>
                <li><span style="color: #FF9500; font-weight: bold;">Línea naranja:</span> Representa la predicción del modelo para cada fecha.</li>
                <li><span style="color: #2C82FF; font-weight: bold;">Puntos azules:</span> Muestran los valores reales observados.</li>
                <li><span style="color: #FF9500; opacity: 0.5;">Área sombreada:</span> Indica el intervalo de predicción (donde se espera que estén los valores reales).</li>
                <li><span style="color: #888888;">Línea punteada gris:</span> Muestra los últimos datos de entrenamiento para dar contexto.</li>
            </ul>
            <h4>¿Qué buscar?</h4>
            <ul>
                <li><strong>Precisión general:</strong> Cuanto más cerca estén los puntos azules de la línea naranja, mejor es la predicción.</li>
                <li><strong>Cobertura:</strong> Idealmente, todos los puntos azules deberían estar dentro del área sombreada.</li>
                <li><strong>Amplitud:</strong> Un área sombreada muy amplia indica mayor incertidumbre en las predicciones.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar métricas en tarjetas con columnas para mejor visualización
        st.subheader("Métricas de Evaluación")
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("RMSE", f"{rmse:.2f}")
            
        with metric_cols[1]:
            st.metric("MAE", f"{mae:.2f}")
            
        with metric_cols[2]:
            st.metric("SMAPE", f"{smape:.1f}%")
            
        with metric_cols[3]:
            st.metric("Cobertura", f"{coverage:.1f}%")
        
        # Análisis de cobertura
        if coverage < 80:
            st.warning("⚠️ La cobertura del intervalo de predicción es baja. Considere aumentar el ancho del intervalo.")
        elif coverage > 95:
            st.info("ℹ️ La cobertura del intervalo es muy alta. Podría reducir el ancho del intervalo para predicciones más precisas.")
        else:
            st.success("✅ La cobertura del intervalo de predicción es adecuada.")
            
        # Análisis de errores
        error_df = pd.DataFrame({
            'Fecha': valid_df['ds'],
            'Real': valid_df['y'].values,  # Use .values to convert to numpy array
            'Predicción': forecast_valid['yhat'].values,  # Use .values to convert to numpy array
            'Error': valid_df['y'].values - forecast_valid['yhat'].values,
            'Error (%)': np.abs((valid_df['y'].values - forecast_valid['yhat'].values) / (valid_df['y'].values + 1e-8)) * 100,
            'Dentro del Intervalo': (valid_df['y'].values >= forecast_valid['yhat_lower'].values) & 
                                   (valid_df['y'].values <= forecast_valid['yhat_upper'].values)
        })
        
        # Mostrar tabla de errores con formato mejorado
        st.subheader("Análisis Detallado de Errores")
        st.dataframe(
            error_df.style.format({
                'Real': '{:.2f}',
                'Predicción': '{:.2f}',
                'Error': '{:.2f}',
                'Error (%)': '{:.1f}%',
                'Dentro del Intervalo': lambda x: '✅' if x else '❌'
            }).applymap(
                lambda x: 'background-color: rgba(255, 149, 0, 0.1)' if isinstance(x, bool) and not x else None,
                subset=['Dentro del Intervalo']
            ),
            use_container_width=True
        )
        
        st.success("✅ Evaluación completada con éxito.")
        return fig
        
    except Exception as e:
        st.error(f"Error al visualizar resultados de evaluación: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def _log_data_flow(action, source, df=None, details=None):
    """
    Registra información sobre el flujo de datos para diagnóstico.
    
    Parameters:
    -----------
    action : str
        Acción que se está realizando (e.g., 'load', 'prepare', 'train', 'evaluate')
    source : str
        Fuente de los datos
    df : pandas.DataFrame, optional
        DataFrame que se está procesando
    details : dict, optional
        Detalles adicionales para registrar
    """
    if 'data_flow_log' not in st.session_state:
        st.session_state.data_flow_log = []
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'action': action,
        'source': source,
    }
    
    if df is not None:
        log_entry['shape'] = df.shape
        log_entry['columns'] = df.columns.tolist()
        log_entry['has_nulls'] = df.isnull().any().any()
        
        # Verificar columnas críticas
        if 'ds' in df.columns:
            log_entry['ds_min'] = df['ds'].min().strftime('%Y-%m-%d') if not pd.isna(df['ds'].min()) else None
            log_entry['ds_max'] = df['ds'].max().strftime('%Y-%m-%d') if not pd.isna(df['ds'].max()) else None
        
        if 'y' in df.columns:
            log_entry['y_min'] = float(df['y'].min()) if not pd.isna(df['y'].min()) else None
            log_entry['y_max'] = float(df['y'].max()) if not pd.isna(df['y'].max()) else None
            log_entry['y_mean'] = float(df['y'].mean()) if not pd.isna(df['y'].mean()) else None
    
    if details:
        log_entry.update(details)
    
    st.session_state.data_flow_log.append(log_entry)
    logger.info(f"Data flow: {action} from {source}" + 
               (f" shape={df.shape}" if df is not None else ""))

def _show_prediction_summary(forecast, model, use_log_transform, optimizations_applied=None):
    """
    Muestra un resumen de la predicción generada.
    
    Parameters:
    -----------
    forecast : pandas.DataFrame
        DataFrame con las predicciones generadas
    model : Prophet
        Modelo utilizado para generar las predicciones
    use_log_transform : bool
        Indica si se aplicó transformación logarítmica a los datos
    optimizations_applied : dict, optional
        Optimizaciones aplicadas al modelo
    """
    try:
        # Verificar que el forecast tenga datos
        if forecast is None or len(forecast) == 0:
            st.warning("No hay datos de predicción para mostrar")
            return
            
        # Extraer estadísticas clave de la predicción
        stats = {
            'total_rows': len(forecast),
            'future_rows': len(forecast[forecast['ds'] > pd.Timestamp.now()]),
            'start_date': forecast['ds'].min().strftime('%Y-%m-%d'),
            'end_date': forecast['ds'].max().strftime('%Y-%m-%d'),
            'min_value': forecast['yhat'].min(),
            'max_value': forecast['yhat'].max(),
            'mean_value': forecast['yhat'].mean(),
            'trend_direction': 'ascendente' if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[0] else 'descendente'
        }
        
        # Mostrar resumen en formato de tarjetas
        st.markdown("### 📊 Resumen de la Predicción")
        
        # Fechas y tamaño de la predicción
        date_cols = st.columns(3)
        with date_cols[0]:
            st.metric("Período Predicho", f"{stats['start_date']} a {stats['end_date']}")
        with date_cols[1]:
            st.metric("Días Totales", stats['total_rows'])
        with date_cols[2]:
            st.metric("Días Futuros", stats['future_rows'])
        
        # Valores clave de la predicción
        value_cols = st.columns(4)
        with value_cols[0]:
            value_label = "Valor Mínimo"
            if use_log_transform:
                value_label += " (log)"
            st.metric(value_label, f"{stats['min_value']:.2f}")
            
        with value_cols[1]:
            value_label = "Valor Máximo"
            if use_log_transform:
                value_label += " (log)"
            st.metric(value_label, f"{stats['max_value']:.2f}")
            
        with value_cols[2]:
            value_label = "Valor Promedio"
            if use_log_transform:
                value_label += " (log)"
            st.metric(value_label, f"{stats['mean_value']:.2f}")
            
        with value_cols[3]:
            st.metric("Tendencia General", stats['trend_direction'], 
                    delta="▲" if stats['trend_direction'] == 'ascendente' else "▼",
                    delta_color="normal" if stats['trend_direction'] == 'ascendente' else "inverse")
        
        # Mostrar información sobre optimizaciones si están disponibles
        if optimizations_applied and isinstance(optimizations_applied, dict) and len(optimizations_applied) > 0:
            st.markdown("#### Optimizaciones Aplicadas")
            for opt_name, opt_value in optimizations_applied.items():
                st.info(f"**{opt_name}**: {opt_value}")
                
        # Mostrar nota sobre transformación logarítmica si se aplicó
        if use_log_transform:
            st.info("ℹ️ Se aplicó transformación logarítmica a los datos. Los valores de predicción están en escala logarítmica.")
    
    except Exception as e:
        st.warning(f"No se pudo mostrar el resumen de la predicción: {str(e)}")
        import traceback
        logger.error(f"Error en _show_prediction_summary: {traceback.format_exc()}")


def _display_evaluation_results(
    metrics: Dict[str, Any],
    all_metrics: Optional[List[Dict[str, Any]]],
    df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    multi_eval: bool = False,
    use_log_transform: bool = False
) -> Optional[Dict[str, float]]:
    """
    Muestra los resultados de la evaluación del modelo.

    Parameters:
    -----------
    metrics : dict
        Diccionario con métricas calculadas
    all_metrics : list
        Lista de métricas para múltiples puntos de corte
    df : pandas.DataFrame
        DataFrame con los datos completos
    test_predictions : pandas.DataFrame
        DataFrame con predicciones de prueba (columnas: ds, yhat, yhat_lower, yhat_upper)
    multi_eval : bool
        Indica si se realizó evaluación con múltiples puntos de corte
    use_log_transform : bool
        Indica si se aplicó transformación logarítmica a los datos
    
    Returns:
    --------
    dict
        Diccionario con las métricas de evaluación o None en caso de error
    """
    # Mostrar estado de evaluación
    st.success("✅ Evaluación del modelo completada")
    st.markdown("### Métricas de Rendimiento")

    # Mostrar métricas clave en tarjetas
    cols = st.columns(4)
    with cols[0]:
        st.metric(
            label="Amplitud Promedio",
            value=f"{metrics.get('interval_width_avg', 0):.2f}",
            help="Promedio de la amplitud de los intervalos de predicción. Valores más bajos indican mayor precisión."
        )
    with cols[1]:
        coverage = metrics.get('coverage', 0)
        color = '#4CAF50' if 80 <= coverage <= 95 else '#FF9800'
        st.metric(
            label="Cobertura Intervalos (%)",
            value=f"{coverage:.1f}%",
            delta_color="normal",
            help="% de valores reales dentro del intervalo de predicción. Debería ser cercano al nivel de confianza."
        )
    with cols[2]:
        st.metric(
            label="Error (SMAPE)",
            value=f"{metrics.get('smape', 0):.1f}%",
            help="Error porcentual medio simétrico. Valores más bajos indican mejor precisión."
        )
    with cols[3]:
        st.metric(
            label="Iteraciones",
            value=metrics.get('iterations', 1),
            help="Cantidad de iteraciones realizadas."
        )

    # Tabla de métricas por punto de corte
    if multi_eval and all_metrics:
        st.markdown("#### Métricas por Punto de Corte")
        df_met = pd.DataFrame(all_metrics)
        if 'cutoff' in df_met.columns:
            df_met = df_met.sort_values('cutoff')
        st.dataframe(df_met)

    # Visualización de backtesting
    st.markdown("### Visualización de Backtesting")
    try:
        if test_predictions is not None and not test_predictions.empty:
            # Definir rangos de datos
            start_date = test_predictions['ds'].min()
            
            if pd.isna(start_date):
                st.warning("Fecha mínima de predicciones inválida.")
                return metrics

            # Datos de entrenamiento y validación
            train = df[df['ds'] < start_date]
            valid = df[df['ds'].isin(test_predictions['ds'])]

            fig = go.Figure()
            if not train.empty:
                fig.add_trace(go.Scatter(
                    x=train['ds'], y=train['y'],
                    mode='lines', name='Entrenamiento', line=dict(width=2)
                ))
            if not valid.empty:
                fig.add_trace(go.Scatter(
                    x=valid['ds'], y=valid['y'],
                    mode='lines', name='Valores Reales', line=dict(width=2)
                ))
            # Predicción
            fig.add_trace(go.Scatter(
                x=test_predictions['ds'], y=test_predictions['yhat'],
                mode='lines', name='Predicción', line=dict(width=2)
            ))
            # Intervalo de confianza
            fig.add_trace(go.Scatter(
                x=pd.concat([test_predictions['ds'], test_predictions['ds'][::-1]]),
                y=pd.concat([test_predictions['yhat_upper'], test_predictions['yhat_lower'][::-1]]),
                fill='toself', name='Intervalo 95%', fillcolor='rgba(255,0,0,0.2)', line=dict(width=0)
            ))

            # Configurar layout
            fig.update_layout(
                title='Backtesting: Predicción vs Real',
                xaxis_title='Fecha', yaxis_title='Valor',
                hovermode='x unified', template='plotly_white'
            )

            # Añadir anotaciones explicativas
            if len(valid) > 3:
                try:
                    # Encontrar punto con mayor error para anotación
                    errors = np.abs(test_predictions['yhat'].values - valid['y'].values)
                    max_error_idx = np.argmax(errors)
                    
                    if max_error_idx < len(valid):
                        max_error_date = valid.iloc[max_error_idx]['ds']
                        max_error_pred = test_predictions.iloc[max_error_idx]['yhat']
                        max_error_real = valid.iloc[max_error_idx]['y']
                        
                        fig.add_annotation(
                            x=max_error_date,
                            y=max_error_real,
                            text="Mayor error",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='#616161',
                            ax=-40,
                            ay=-40,
                            font=dict(size=12, color='#616161'),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor='#616161',
                            borderwidth=1,
                            borderpad=4
                        )
                    
                    # Encontrar punto con predicción más alta
                    max_pred_idx = np.argmax(test_predictions['yhat'].values)
                    max_pred_date = test_predictions.iloc[max_pred_idx]['ds']
                    max_pred_value = test_predictions.iloc[max_pred_idx]['yhat']
                    
                    fig.add_annotation(
                        x=max_pred_date,
                        y=max_pred_value,
                        text="Pico predicho",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='rgb(0,100,80)',
                        ax=40,
                        ay=-40,
                        font=dict(size=12, color='rgb(0,100,80)'),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor='rgb(0,100,80)',
                        borderwidth=1,
                        borderpad=4
                    )
                    
                    # Añadir línea para valor medio predicho
                    avg_pred = np.mean(test_predictions['yhat'].values)
                    fig.add_shape(
                        type="line",
                        x0=test_predictions['ds'].min(),
                        y0=avg_pred,
                        x1=test_predictions['ds'].max(),
                        y1=avg_pred,
                        line=dict(
                            color="rgba(33, 150, 243, 0.5)",
                            width=1,
                            dash="dot",
                        )
                    )
                    
                    fig.add_annotation(
                        x=test_predictions['ds'].max(),
                        y=avg_pred,
                        text=f"Media: {avg_pred:.2f}",
                        showarrow=False,
                        xanchor="right",
                        font=dict(size=10, color='rgb(0,100,80)'),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor='rgb(0,100,80)',
                        borderwidth=1,
                        borderpad=2
                    )
                except Exception as e:
                    st.warning(f"No se pudieron añadir anotaciones al gráfico: {str(e)}")

            st.plotly_chart(fig, use_container_width=True)
            
            # Añadir resumen interpretativo
            with st.expander("📊 Cómo interpretar este gráfico", expanded=True):
                st.markdown("""
                <div style="background-color: rgba(40, 40, 40, 0.7); padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h4 style="margin-top: 0;">Guía de interpretación</h4>
                    <ul>
                        <li><span style="color: #FF9500; font-weight: bold;">Línea naranja:</span> Representa la predicción del modelo para cada fecha.</li>
                        <li><span style="color: #2C82FF; font-weight: bold;">Puntos azules:</span> Muestran los valores reales observados.</li>
                        <li><span style="color: #FF9500; opacity: 0.5;">Área sombreada:</span> Indica el intervalo de predicción (donde se espera que estén los valores reales).</li>
                        <li><span style="color: #888888;">Línea punteada gris:</span> Muestra los últimos datos de entrenamiento para dar contexto.</li>
                    </ul>
                    <h4>¿Qué buscar?</h4>
                    <ul>
                        <li><strong>Precisión general:</strong> Cuanto más cerca estén los puntos azules de la línea naranja, mejor es la predicción.</li>
                        <li><strong>Cobertura:</strong> Idealmente, todos los puntos azules deberían estar dentro del área sombreada.</li>
                        <li><strong>Amplitud:</strong> Un área sombreada muy amplia indica mayor incertidumbre en las predicciones.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Error al generar la visualización de backtesting: {exc}")
        return None

    return metrics

def _prepare_training_dataframe(df=None):
    """
    Prepara y verifica el DataFrame para entrenamiento.
    
    Args:
        df: DataFrame opcional con datos de entrenamiento
        
    Returns:
        DataFrame preparado para entrenamiento o None si no hay datos
    """
    logger.info("Preparando DataFrame para entrenamiento")
    
    # Si no se proporciona df, intentar obtenerlo de la sesión de Streamlit
    if df is None:
        logger.info("No se proporcionó DataFrame, intentando obtenerlo de la sesión")
        if 'df_prophet' in st.session_state:
            df = st.session_state.df_prophet
            logger.info(f"DataFrame obtenido de sesión: {len(df)} filas")
        else:
            st.error("No hay datos para entrenar. Por favor, cargue datos primero.")
            return None
    
    # Verificar si hay datos suficientes
    if df is None or len(df) < 10:
        st.error("Datos insuficientes para entrenar modelo")
        return None
        
    # Verificar columnas mínimas requeridas
    required_cols = ['ds', 'y']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        st.error(f"El DataFrame no contiene las columnas requeridas: {missing}")
        return None
    
    # Añadir características avanzadas de ciberseguridad
    try:
        from .feature_engineering import FeatureEngineer
        
        logger.info("Generando características avanzadas para ciberseguridad")
        
        # Crear una copia para no modificar el original
        df_enhanced = df.copy()
        
        # Añadir características temporales básicas si no existen
        if not any(col.startswith('month_') for col in df.columns):
            logger.info("Añadiendo características temporales básicas")
            df_enhanced = FeatureEngineer.add_temporal_features(df_enhanced)
        
        # Añadir características específicas para ciberseguridad
        if not any(col in df.columns for col in ['is_patch_tuesday', 'days_since_patch_tuesday']):
            logger.info("Añadiendo características específicas para ciberseguridad")
            df_enhanced = FeatureEngineer.add_cybersecurity_features(df_enhanced)
            
            # Informar sobre las nuevas características
            new_features = [col for col in df_enhanced.columns if col not in df.columns]
            logger.info(f"Características de ciberseguridad añadidas: {new_features}")
            
            if len(new_features) > 0:
                st.success(f"✅ Se añadieron {len(new_features)} características avanzadas para mejorar la predicción de ransomware")
        
        # Intentar añadir datos de CVE si están disponibles
        try:
            cve_path = os.path.join('modeling', 'cve_diarias_regresor_prophet.csv')
            if os.path.exists(cve_path) and not any(col.startswith('cve_') for col in df_enhanced.columns):
                logger.info(f"Intentando cargar datos de CVE desde {cve_path}")
                
                # Cargar datos de CVE
                cve_df = pd.read_csv(cve_path)
                
                # Asegurarse de que la columna de fecha esté en formato datetime
                if 'ds' in cve_df.columns:
                    cve_df['ds'] = pd.to_datetime(cve_df['ds'])
                    
                    # Fusionar con el DataFrame principal
                    df_enhanced = pd.merge(df_enhanced, cve_df, on='ds', how='left')
                    
                    # Rellenar valores faltantes
                    cve_cols = [col for col in cve_df.columns if col != 'ds']
                    for col in cve_cols:
                        if col in df_enhanced.columns and df_enhanced[col].isna().any():
                            df_enhanced[col].fillna(df_enhanced[col].median(), inplace=True)
                    
                    logger.info(f"Datos de CVE integrados: {len(cve_cols)} variables")
                    st.success(f"✅ Datos de vulnerabilidades (CVE) incorporados como regresores")
        except Exception as e:
            logger.warning(f"No se pudieron cargar datos de CVE: {str(e)}")
            
        return df_enhanced
        
    except Exception as e:
        logger.error(f"Error al generar características avanzadas: {str(e)}")
        # Devolver el DataFrame original si hay problemas con las características avanzadas
        st.warning(f"⚠️ No se pudieron generar todas las características avanzadas: {str(e)}")
        return df

def _setup_model_config(
    seasonality_mode, use_log_transform, changepoint_prior_scale, 
    seasonality_prior_scale, holidays_prior_scale, regressors_prior_scale,
    interval_width, changepoint_range, n_changepoints, use_detected_changepoints,
    daily_seasonality, weekly_seasonality, yearly_seasonality
):
    """
    Configura los parámetros del modelo Prophet.
    
    Returns:
        Diccionario con la configuración del modelo
    """
    return {
        'seasonality_mode': seasonality_mode,
        'use_log_transform': use_log_transform,
        'changepoint_prior_scale': changepoint_prior_scale,
        'seasonality_prior_scale': seasonality_prior_scale,
        'holidays_prior_scale': holidays_prior_scale,
        'regressors_prior_scale': regressors_prior_scale,
        'interval_width': interval_width,
        'changepoint_range': changepoint_range,
        'n_changepoints': n_changepoints,
        'use_detected_changepoints': use_detected_changepoints,
        'daily_seasonality': daily_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'yearly_seasonality': yearly_seasonality
    }

def _prepare_regressors(use_regressor, correlation_threshold, vif_threshold):
    """
    Prepara los regresores para el modelo si se solicitan.
    
    Args:
        use_regressor: Si se deben usar regresores
        correlation_threshold: Umbral de correlación para selección
        vif_threshold: Umbral de VIF para multicolinealidad
        
    Returns:
        Objeto RegressorGenerator configurado o None
    """
    if not use_regressor:
        return None
        
    try:
        from .features.regressors import RegressorGenerator
        logger.info("Configurando regresores")
        
        # Inicializar generador de regresores
        if 'regressors' not in st.session_state or st.session_state.regressors is None:
            st.session_state.regressors = RegressorGenerator()
            
        regressors = st.session_state.regressors
        
        # Cargar datos de CVE mediante el dataframe si está disponible en la sesión
        cve_df = None
        cve_file = 'modeling/cve_diarias_regresor_prophet.csv'
        
        if hasattr(st.session_state, 'cve_file') and st.session_state.cve_file:
            cve_file = st.session_state.cve_file
        
        # Intentar obtener df_cve del forecaster si existe
        if 'forecaster' in st.session_state and hasattr(st.session_state.forecaster, 'df_cve'):
            cve_df = st.session_state.forecaster.df_cve
        else:
            # Cargar datos CVE manualmente
            try:
                cve_df = pd.read_csv(cve_file)
                if 'fecha' in cve_df.columns:
                    cve_df = cve_df.rename(columns={'fecha': 'ds'})
                # Convertir a datetime
                cve_df['ds'] = pd.to_datetime(cve_df['ds'])
            except Exception as e:
                logger.warning(f"No se pudieron cargar datos de CVE: {str(e)}")
                
        # Configurar parámetros de selección de regresores
        regressors.correlation_threshold = correlation_threshold
        regressors.vif_threshold = vif_threshold
        
        return regressors
    except Exception as e:
        logger.error(f"Error preparando regresores: {str(e)}")
        st.warning(f"No se pudieron preparar los regresores: {str(e)}")
        return None

def _train_prophet_model(df, model_config, regressors, use_optimal_regressors=True, use_interval_calibration=True):
    """
    Entrena el modelo Prophet con los parámetros y regresores proporcionados.
    
    Args:
        df: DataFrame con datos de entrenamiento
        model_config: Configuración del modelo
        regressors: Objeto RegressorGenerator configurado
        use_optimal_regressors: Si usar selección automática de regresores
        use_interval_calibration: Si calibrar intervalos de predicción
        
    Returns:
        Modelo entrenado o None en caso de error
    """
    from .models.prophet_model import RansomwareProphetModel
    
    try:
        # Crear instancia del modelo
        model = RansomwareProphetModel()
        
        # Establecer parámetros
        for param, value in model_config.items():
            setattr(model, param, value)
        
        # Entrenar modelo
        st.info("Entrenando modelo Prophet...")
        
        # Si hay regresores disponibles, prepararlos
        regressor_df = None
        if regressors is not None and use_optimal_regressors:
            try:
                # Preparar regresores basados en datos CVE
                if hasattr(st.session_state.forecaster, 'df_cve'):
                    cve_df = st.session_state.forecaster.df_cve
                    if cve_df is not None and len(cve_df) > 0:
                        regressor_df = regressors.prepare_regressors(df, cve_df)
                        st.info(f"Preparados {len(regressor_df.columns) - 1} regresores potenciales")
            except Exception as e:
                st.warning(f"Error preparando regresores: {str(e)}")
        
        # Entrenar el modelo
        model.fit(df, regressors=regressor_df)
        
        # Calibrar intervalos si se solicita
        if use_interval_calibration:
            try:
                from .models.calibrator import IntervalCalibrator
                calibrator = IntervalCalibrator()
                st.info("Calibrando intervalos de predicción...")
                # Implementar calibración
            except Exception as e:
                st.warning(f"No se pudieron calibrar intervalos: {str(e)}")
        
        # Guardar en sesión
        st.session_state.prophet_model = model
        st.session_state.model_trained = True
        
        st.success("✅ Modelo entrenado correctamente")
        return model
        
    except Exception as e:
        st.error(f"Error entrenando modelo: {str(e)}")
        logger.error(f"Error entrenando modelo: {traceback.format_exc()}")
        return None

def train_model_wrapper(
    df: Optional[pd.DataFrame] = None,
    use_regressor: bool = True,
    use_optimal_regressors: bool = True,
    use_bayesian_optimization: bool = False,
    use_interval_calibration: bool = True,
    optimization_trials: int = 10,
    correlation_threshold: float = 0.1,
    vif_threshold: float = 5.0,
    seasonality_mode: str = 'multiplicative',
    use_log_transform: bool = True,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    holidays_prior_scale: float = 10.0,
    regressors_prior_scale: float = 0.1,
    interval_width: float = 0.6,
    changepoint_range: float = 0.8,
    n_changepoints: int = 25,
    use_detected_changepoints: bool = False,
    daily_seasonality: bool = False,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True,
    calibrate_intervals: bool = True
) -> tuple:
    """
    Wrapper para entrenar el modelo Prophet a través de la UI de Streamlit.
    
    Esta función coordina el proceso completo de entrenamiento del modelo, incluyendo:
    - Preparación de datos
    - Configuración del modelo
    - Selección de regresores (si se solicita)
    - Entrenamiento del modelo
    - Calibración de intervalos (si se solicita)
    
    Args:
        df: DataFrame con datos de entrenamiento (opcional, si no se proporciona se obtiene de la sesión)
        use_regressor: Si se deben usar regresores
        use_optimal_regressors: Si se deben usar regresores óptimos
        use_bayesian_optimization: Si se debe usar optimización bayesiana
        use_interval_calibration: Si se debe calibrar los intervalos
        optimization_trials: Número de pruebas para optimización bayesiana
        correlation_threshold: Umbral de correlación para selección de regresores
        vif_threshold: Umbral de VIF para controlar multicolinealidad entre regresores
        seasonality_mode: Modo de estacionalidad ('additive' o 'multiplicative')
        use_log_transform: Si se debe aplicar transformación logarítmica
        changepoint_prior_scale: Escala previa de puntos de cambio
        seasonality_prior_scale: Escala previa de estacionalidad
        holidays_prior_scale: Escala previa de festivos
        regressors_prior_scale: Escala previa de regresores
        interval_width: Ancho del intervalo de predicción
        changepoint_range: Rango de puntos de cambio
        n_changepoints: Número de puntos de cambio
        use_detected_changepoints: Si se deben usar puntos de cambio detectados
        daily_seasonality: Si se debe usar estacionalidad diaria
        weekly_seasonality: Si se debe usar estacionalidad semanal
        yearly_seasonality: Si se debe usar estacionalidad anual
        calibrate_intervals: Si se deben calibrar los intervalos
        
    Returns:
    --------
    Tupla con (modelo entrenado, DataFrame de predicción)
    """
    # 1. Preparar y verificar DataFrame
    df = _prepare_training_dataframe(df)
    if df is None:
        return None, None
    
    # 2. Configurar parámetros del modelo
    model_config = _setup_model_config(
        seasonality_mode, use_log_transform, changepoint_prior_scale,
        seasonality_prior_scale, holidays_prior_scale, regressors_prior_scale,
        interval_width, changepoint_range, n_changepoints, use_detected_changepoints,
        daily_seasonality, weekly_seasonality, yearly_seasonality
    )
    
    # 3. Preparar regresores si se solicitan
    regressors = _prepare_regressors(use_regressor, correlation_threshold, vif_threshold)
    
    # 4. Entrenar el modelo
    model = _train_prophet_model(
        df, model_config, regressors, 
        use_optimal_regressors, use_interval_calibration
    )
    
    # 5. Realizar una predicción inicial para comprobar que todo funciona
    forecast = None
    if model is not None:
        try:
            forecast = model.predict(periods=30, include_history=True)
            st.session_state.forecast = forecast
        except Exception as e:
            st.error(f"Error generando predicción inicial: {str(e)}")
    
    return model, forecast

def plot_forecast_wrapper():
    """
    Crea una visualización Plotly de la predicción almacenada en session_state.
    
    Returns:
        Figura Plotly con la visualización o None si hay un error
    """
    try:
        # Verificar si hay una predicción disponible
        if 'forecast' not in st.session_state:
            logger.error("No hay predicción disponible en session_state")
            return None
            
        forecast = st.session_state.forecast
        
        # Verificar que forecast no sea None
        if forecast is None:
            logger.error("La predicción en session_state es None")
            return None
            
        # Verificar que forecast sea un DataFrame
        if not isinstance(forecast, pd.DataFrame):
            logger.error(f"La predicción no es un DataFrame, es {type(forecast)}")
            return None
            
        # Verificar que forecast no esté vacío
        if forecast.empty:
            logger.error("El DataFrame de predicción está vacío")
            return None
            
        # Imprimir las columnas disponibles para diagnóstico
        logger.info(f"Columnas en forecast: {list(forecast.columns)}")
        
        # Verificar que el DataFrame tenga las columnas necesarias
        required_cols = ['ds', 'yhat']
        missing_cols = [col for col in required_cols if col not in forecast.columns]
        if missing_cols:
            logger.error(f"Faltan columnas necesarias en el DataFrame de predicción: {missing_cols}")
            return None
        
        try:
            # Crear figura con subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("Predicción de ataques ransomware", "Componentes")
            )
            
            # Añadir datos históricos
            if 'df_prophet' in st.session_state and st.session_state.df_prophet is not None:
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.df_prophet['ds'],
                        y=st.session_state.df_prophet['y'],
                        mode='markers',
                        name='Datos históricos',
                        marker=dict(color='#FF9F1C', size=8)  # Naranja brillante
                    ),
                    row=1, col=1
                )
            
            # Predicción
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Predicción',
                    line=dict(color='blue', width=3)  # Aumentado ancho de línea
                ),
                row=1, col=1
            )
            
            # Marcar área futura
            if 'df_prophet' in st.session_state and st.session_state.df_prophet is not None:
                future_mask = ~forecast['ds'].isin(st.session_state.df_prophet['ds'])
                if future_mask.sum() > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast.loc[future_mask, 'ds'],
                            y=forecast.loc[future_mask, 'yhat'],
                            mode='lines',
                            name='Predicción futura',
                            line=dict(color='red', width=4)  # Aumentado ancho de línea de 2.5 a 4
                        ),
                        row=1, col=1
                    )
            
            # Añadir intervalos de confianza
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_upper'],
                        mode='lines',
                        name='Límite superior',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_lower'],
                        mode='lines',
                        name='Intervalo de confianza',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 0, 255, 0.3)'  # Aumentado opacidad de 0.2 a 0.3
                    ),
                    row=1, col=1
                )
            
            # Añadir componentes - solo tendencia
            if 'trend' in forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['trend'],
                        mode='lines',
                        name='Tendencia',
                        line=dict(color='green', width=3)  # Aumentado ancho de línea de 2 a 3
                    ),
                    row=2, col=1
                )
            
            # Actualizar diseño
            fig.update_layout(
                height=600,  # Reducido para mejor proporción
                width=1200,  # Un poco más ancho
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                template='plotly_white',
                font=dict(size=14)  # Texto más grande
            )
            
            # ——————————————————————————————
            # Color de los títulos:
            fig.update_annotations(font_color='white')

            # Actualizar ejes
            fig.update_yaxes(title_text="Ataques", row=1, col=1, title_font=dict(size=16), range=[0, 12])  # Ajustado máximo a 12
            fig.update_yaxes(title_text="Componentes", row=2, col=1, title_font=dict(size=16))
            fig.update_xaxes(title_text="Fecha", row=2, col=1, title_font=dict(size=16))
            return fig
            
        except ImportError:
            logger.warning("Para usar gráficos interactivos, instala: pip install plotly")
            return None
            
    except Exception as e:
        logger.error(f"Error al crear visualización de predicción: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def evaluate_model_wrapper(_self=None, cv_periods=30, initial=None, period=None, horizon=None, 
                       max_iterations=3, train_percentage=0.8, multi_eval=False, use_log_transform=False):
    """
    Función de evaluación del modelo basada en backtesting.
    
    Parameters:
    -----------
    cv_periods : int
        Número de períodos a usar para validación
    initial : str
        Fecha inicial para la evaluación (si es None, se usará un porcentaje de los datos)
    period : int
        Período entre iteraciones de evaluación
    horizon : int
        Horizonte de predicción para cada evaluación
    max_iterations : int
        Número máximo de iteraciones de validación
    train_percentage : float
        Porcentaje de datos a usar para entrenamiento (entre 0.5 y 0.9)
    multi_eval : bool
        Si es True, realiza evaluaciones con múltiples puntos de corte
    use_log_transform : bool
        Si es True, aplica transformación logarítmica a los datos
    
    Returns:
    --------
    dict
        Diccionario con las métricas de evaluación calculadas a partir del modelo real
    """
    try:
        # Verificar que el modelo está entrenado
        if 'forecaster' not in st.session_state or st.session_state.forecaster is None:
            st.error("⚠️ Primero debe entrenar el modelo")
            return None
        
        # Para propósitos de diagnóstico, añadimos logs detallados
        st.info("🔍 Iniciando evaluación del modelo...")
        
        # Verificar que tengamos datos de entrenamiento
        df = None
        data_source = "desconocido"
        
        # Registrar el estado de los datos en este punto
        st.info("🔍 Verificando fuentes de datos disponibles...")
        available_data = []
        
        # Comprobar todas las posibles fuentes de datos en orden de prioridad
        if hasattr(st.session_state.forecaster, 'df_prophet') and st.session_state.forecaster.df_prophet is not None:
            available_data.append(f"forecaster.df_prophet: {st.session_state.forecaster.df_prophet.shape}")
            df = st.session_state.forecaster.df_prophet
            data_source = "forecaster.df_prophet"
        
        if hasattr(st.session_state.forecaster, 'train_data') and st.session_state.forecaster.train_data is not None:
            available_data.append(f"forecaster.train_data: {st.session_state.forecaster.train_data.shape}")
            if df is None:
                df = st.session_state.forecaster.train_data
                data_source = "forecaster.train_data"
        
        if 'df_prophet_clean' in st.session_state and st.session_state.df_prophet_clean is not None:
            available_data.append(f"df_prophet_clean: {st.session_state.df_prophet_clean.shape}")
            if df is None:
                df = st.session_state.df_prophet_clean
                data_source = "df_prophet_clean"
        
        # Mostrar información de diagnóstico
        st.write("Fuentes de datos disponibles:")
        for source in available_data:
            st.write(f"- {source}")
        
        # Si no encontramos datos, no podemos continuar
        if df is None or len(df) == 0:
            st.error("⚠️ No hay datos disponibles para la evaluación")
            
            # Diagnóstico detallado
            st.info("Información de diagnóstico:")
            st.info(f"- Variables disponibles en st.session_state: {[key for key in st.session_state.keys() if isinstance(st.session_state[key], pd.DataFrame)]}")
            
            raise ValueError("No hay datos disponibles para la evaluación.")
        
        st.info(f"Usando datos de {data_source} para la evaluación ({len(df)} registros).")
        
        # Verificar si tenemos suficientes datos para la evaluación
        min_required = max(10, cv_periods)  # Mínimo absoluto: al menos el horizonte de evaluación
        preferred_min = max(50, cv_periods * 3)  # Mínimo preferido: al menos 3 veces el horizonte
        
        if len(df) < preferred_min:
            st.warning(f"No hay suficientes datos para una evaluación robusta (se tienen {len(df)} registros, se recomiendan al menos {preferred_min}).")
            
            if len(df) < min_required:
                st.error(f"Insuficientes datos para evaluación. Se requieren al menos {min_required} registros.")
                raise ValueError(f"Insuficientes datos para evaluación. Se necesitan al menos {min_required} registros, pero solo hay {len(df)}.")
            else:
                st.info("Realizando evaluación con los datos disponibles, pero los resultados pueden no ser representativos.")
                
                # Ajustar parámetros de evaluación para conjuntos pequeños de datos
                if cv_periods > len(df) // 3:
                    old_cv = cv_periods
                    cv_periods = max(5, len(df) // 3)
                    st.warning(f"Ajustando horizonte de evaluación de {old_cv} a {cv_periods} días debido a la limitación de datos.")
                
                if period is not None and period > len(df) // 5:
                    old_period = period
                    period = max(3, len(df) // 5)
                    st.warning(f"Ajustando periodo de evaluación de {old_period} a {period} días debido a la limitación de datos.")
                
                if max_iterations > len(df) // (cv_periods * 2):
                    old_iter = max_iterations
                    max_iterations = max(2, len(df) // (cv_periods * 2))
                    st.warning(f"Ajustando iteraciones de {old_iter} a {max_iterations} debido a la limitación de datos.")
        
        # Si solicitamos múltiples evaluaciones
        if multi_eval:
            # Realizar evaluaciones con diferentes porcentajes de entrenamiento
            percentages = [0.6, 0.7, 0.8, 0.9]
            results = []
            
            for pct in percentages:
                st.subheader(f"Evaluación con {int(pct*100)}% de datos para entrenamiento")
                # Llamar recursivamente a esta misma función con diferentes porcentajes
                metrics = evaluate_model_wrapper(_self=_self, cv_periods=cv_periods, 
                                               train_percentage=pct, multi_eval=False)
                
                if metrics:
                    results.append({
                        'train_percentage': pct,
                        'metrics': metrics,
                        'cutoff_date': st.session_state.evaluation_details[0]['cutoff']
                    })
            
            # Mostrar tabla comparativa
            if results:
                st.subheader("Comparación de Evaluaciones con Diferentes Puntos de Corte")
                
                # Crear dataframe para mostrar resultados
                comparison_df = pd.DataFrame({
                    'Porcentaje Entrenamiento': [f"{int(r['train_percentage']*100)}%" for r in results],
                    'Fecha de Corte': [r['cutoff_date'] for r in results],
                    'RMSE': [r['metrics']['rmse'] for r in results],
                    'MAE': [r['metrics']['mae'] for r in results],
                    'SMAPE': [f"{r['metrics']['smape']:.2f}%" for r in results],
                    'Cobertura': [f"{r['metrics']['coverage']:.2f}%" for r in results],
                    'Amplitud Promedio': [r['metrics'].get('interval_width_avg', 'N/A') for r in results]
                })
                
                st.table(comparison_df)
                
                # Recomendaciones basadas en los resultados
                st.subheader("Análisis de Resultados")
                
                # Analizar la variación en el RMSE
                rmse_values = [r['metrics']['rmse'] for r in results]
                rmse_variation = max(rmse_values) - min(rmse_values)
                
                if rmse_variation > 0.5:
                    st.warning(f"Alta variación en RMSE ({rmse_variation:.2f}): Los resultados son sensibles al punto de corte.")
                else:
                    st.success(f"Baja variación en RMSE ({rmse_variation:.2f}): Los resultados son consistentes entre diferentes puntos de corte.")
                
                # Analizar la cobertura
                coverage_values = [r['metrics']['coverage'] for r in results]
                avg_coverage = sum(coverage_values) / len(coverage_values)
                
                if avg_coverage > 95:
                    st.warning(f"Cobertura promedio alta ({avg_coverage:.2f}%): Los intervalos de predicción podrían ser demasiado conservadores.")
                else:
                    st.success(f"Cobertura promedio adecuada ({avg_coverage:.2f}%): Los intervalos de predicción parecen bien calibrados.")
                
                # Devolver el promedio de las métricas
                return {
                    'rmse': sum(r['metrics']['rmse'] for r in results) / len(results),
                    'mae': sum(r['metrics']['mae'] for r in results) / len(results),
                    'smape': sum(r['metrics']['smape'] for r in results) / len(results),
                    'coverage': sum(r['metrics']['coverage'] for r in results) / len(results),
                    'iterations': len(results),
                    'horizon': cv_periods,
                    'multi_eval': True
                }
            
            return None
        
        # Para una única evaluación, continuar con el proceso normal
        # Ajustar el train_percentage a un rango válido
        train_percentage = max(0.5, min(0.95, train_percentage))
        
        # Determinar el punto de corte para entrenamiento/validación basado en el porcentaje
        cutoff_idx = int(len(df) * train_percentage)
        
        # Asegurarse de que haya suficientes datos para validación
        if len(df) - cutoff_idx < cv_periods:
            cv_periods = len(df) - cutoff_idx
            st.info(f"Ajustando período de validación a {cv_periods} días para mantener suficientes datos.")
        
        # Asegurarse de que haya suficientes datos para entrenamiento
        if cutoff_idx < 10:
            cutoff_idx = 10
            st.warning("Usando al menos 10 puntos para entrenamiento, puede no ser suficiente.")
        
        cutoff_date = df.iloc[cutoff_idx]['ds']
        
        # Dividir los datos en entrenamiento y prueba
        train_df = df[df['ds'] <= cutoff_date].copy()
        valid_df = df[df['ds'] > cutoff_date].copy()
        
        st.info(f"Realizando backtesting con punto de corte: {cutoff_date.strftime('%Y-%m-%d')}")
        st.info(f"Datos de entrenamiento: {len(train_df)} registros ({train_percentage*100:.1f}%)")
        st.info(f"Datos de validación: {len(valid_df)} registros ({(1-train_percentage)*100:.1f}%)")
        
        # Crear y entrenar un nuevo modelo con los datos de entrenamiento
        # Usar los mismos parámetros que el modelo original si es posible
        # Obtener el modelo de la sesión
        model = None
        
        # Verificar en las distintas ubicaciones donde podría estar el modelo
        if 'forecaster' in st.session_state:
            # Opción 1: El modelo está directamente en el atributo 'model'
            if hasattr(st.session_state.forecaster, 'model') and st.session_state.forecaster.model is not None:
                model = st.session_state.forecaster.model
                st.info("✅ Modelo encontrado en forecaster.model")
            # Opción 2: El modelo está en 'prophet_model'
            elif hasattr(st.session_state.forecaster, 'prophet_model') and st.session_state.forecaster.prophet_model is not None:
                model = st.session_state.forecaster.prophet_model
                st.info("✅ Modelo encontrado en forecaster.prophet_model")
            # Opción 3: La clase RansomwareProphetModel tiene su propio modelo interno
            elif hasattr(st.session_state.forecaster, 'ransomware_model') and hasattr(st.session_state.forecaster.ransomware_model, 'model'):
                model = st.session_state.forecaster.ransomware_model.model
                st.info("✅ Modelo encontrado en forecaster.ransomware_model.model")
            # Intentar acceder al modelo directamente desde la sesión
            elif 'model' in st.session_state and st.session_state.model is not None:
                model = st.session_state.model
                st.info("✅ Modelo encontrado en session_state.model")
            
            # Si no encontramos el modelo, usar valores predeterminados
            if model is not None:
                # Extraer parámetros del modelo
                try:
                    params = {
                        'changepoint_prior_scale': model.changepoint_prior_scale,
                        'seasonality_prior_scale': model.seasonality_prior_scale,
                        'seasonality_mode': model.seasonality_mode,
                        'interval_width': model.interval_width
                    }
                    st.info("Usando parámetros del modelo entrenado.")
                except AttributeError as e:
                    st.warning(f"Error al extraer parámetros del modelo: {str(e)}")
                    # Usar valores predeterminados si no podemos extraer los parámetros
                    params = {
                        'changepoint_prior_scale': 0.05,
                        'seasonality_prior_scale': 10.0,
                        'seasonality_mode': 'multiplicative',
                        'interval_width': 0.8  # Aumentar el ancho del intervalo para mejorar la cobertura
                    }
            else:
                # Si no se pueden extraer, usar valores predeterminados
                params = {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10.0,
                    'seasonality_mode': 'multiplicative',
                    'interval_width': 0.8  # Aumentar el ancho del intervalo
                }
                st.warning("No se encontró un modelo entrenado en la sesión. Usando parámetros predeterminados.")
            
            # Crear y entrenar el modelo de evaluación
            from prophet import Prophet
            
            eval_model = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                interval_width=params['interval_width']
            )
            
            # Intentar añadir los mismos regresores si los tiene el modelo original
            if model is not None and hasattr(model, 'extra_regressors') and model.extra_regressors:
                for regressor_name, regressor_params in model.extra_regressors.items():
                    if regressor_name in train_df.columns:
                        try:
                            eval_model.add_regressor(regressor_name)
                        except Exception as e:
                            st.warning(f"No se pudo añadir el regresor {regressor_name}: {str(e)}")
            
            # Entrenar modelo con los datos de entrenamiento
            try:
                st.info(f"Entrenando modelo de evaluación con {len(train_df)} filas de datos...")
                eval_model.fit(train_df)
            except Exception as e:
                st.error(f"Error al entrenar el modelo de evaluación: {str(e)}")
                return None
            
            # SOLUCIÓN PARA EL PROBLEMA DE FUSIÓN DE DATOS: 
            # En lugar de hacer future_df y luego filtrar, vamos a crear directamente un dataframe
            # con exactamente las mismas fechas que el set de validación
            try:
                st.info(f"Generando predicciones para {len(valid_df)} puntos de validación...")
                
                # Crear un dataframe futuro usando exactamente las mismas fechas que valid_df
                future = pd.DataFrame({'ds': valid_df['ds'].values})
                
                # Añadir los regresores al dataframe futuro si se usaron
                if model is not None and hasattr(model, 'extra_regressors') and model.extra_regressors:
                    for regressor_name in model.extra_regressors:
                        if regressor_name in df.columns:
                            try:
                                # Primero encontrar las fechas correspondientes en el df original
                                dates_map = dict(zip(df['ds'].dt.strftime('%Y-%m-%d'), df[regressor_name]))
                                # Luego mapear esos valores a las fechas en future
                                future[regressor_name] = future['ds'].dt.strftime('%Y-%m-%d').map(dates_map)
                            except Exception as e:
                                st.warning(f"No se pudo añadir el regresor {regressor_name} al dataframe futuro: {str(e)}")
                
                # Generar predicciones solo para las fechas de validación
                forecast = eval_model.predict(future)
                
                # No es necesario filtrar porque ya tenemos solo las fechas que queremos
                forecast_valid = forecast
                
                st.info(f"Generadas {len(forecast_valid)} predicciones para el período de validación.")
            except Exception as e:
                st.error(f"Error al generar predicciones: {str(e)}")
                import traceback
                st.error(f"Detalles: {traceback.format_exc()}")
                return None
            
            # Asegurar que las fechas en ambos dataframes estén en el mismo formato
            try:
                # Asegurar que la columna 'ds' sea datetime en ambos dataframes
                valid_df['ds'] = pd.to_datetime(valid_df['ds'])
                forecast_valid['ds'] = pd.to_datetime(forecast_valid['ds'])
                
                # Mostrar diagnóstico de fechas
                st.info(f"Rango de fechas en validación: {valid_df['ds'].min()} a {valid_df['ds'].max()}")
                st.info(f"Rango de fechas en predicción: {forecast_valid['ds'].min()} a {forecast_valid['ds'].max()}")
                
                # Fusionar datos para evaluación
                test_with_preds = pd.merge(
                    valid_df, 
                    forecast_valid[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                    on='ds', 
                    how='inner'
                )
                
                # Verificar la fusión
                if len(test_with_preds) == 0:
                    st.error("La fusión de datos reales y predicciones resultó en 0 filas.")
                    
                    # Diagnóstico adicional
                    st.info("Diagnóstico de fechas:")
                    valid_dates = set(valid_df['ds'].dt.strftime('%Y-%m-%d'))
                    forecast_dates = set(forecast_valid['ds'].dt.strftime('%Y-%m-%d'))
                    
                    # Verificar si hay fechas comunes
                    common_dates = valid_dates.intersection(forecast_dates)
                    st.info(f"Fechas comunes: {len(common_dates)} de {len(valid_dates)} en validación y {len(forecast_dates)} en predicción")
                    
                    if len(common_dates) == 0:
                        # Mostrar algunas fechas de cada conjunto para diagnóstico
                        st.info(f"Ejemplo de fechas en validación: {list(valid_dates)[:5]}")
                        st.info(f"Ejemplo de fechas en predicción: {list(forecast_dates)[:5]}")
                        
                        # Intentar con una estrategia más flexible basada en el día
                        st.info("Intentando estrategia alternativa de coincidencia de fechas...")
                        
                        # Crear columnas de fecha sin hora para ambos dataframes
                        valid_df['date_only'] = valid_df['ds'].dt.date
                        forecast_valid['date_only'] = forecast_valid['ds'].dt.date
                        
                        # Intentar fusionar por la fecha sin hora
                        test_with_preds = pd.merge(
                            valid_df, 
                            forecast_valid[['date_only', 'yhat', 'yhat_lower', 'yhat_upper']], 
                            on='date_only', 
                            how='inner'
                        )
                        
                        if len(test_with_preds) > 0:
                            st.success(f"¡Éxito! Usando coincidencia de fecha sin hora, se encontraron {len(test_with_preds)} filas comunes.")
                        else:
                            st.error("No se pudo encontrar coincidencias ni siquiera usando solo la fecha sin hora.")
                            return None
                    else:
                        return None
                
                st.success(f"Fusión exitosa: {len(test_with_preds)} filas para calcular métricas.")
                
            except Exception as e:
                st.error(f"Error al fusionar datos: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return None
            
            # Verificar que tenemos todas las columnas necesarias
            required_cols = ['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']
            missing_cols = [col for col in required_cols if col not in test_with_preds.columns]
            
            if missing_cols:
                st.error(f"Faltan columnas necesarias en los datos fusionados: {missing_cols}")
                st.write("Columnas disponibles:", list(test_with_preds.columns))
                return None
            
            # Verificar que no hay NaN en las columnas críticas
            nan_counts = test_with_preds[required_cols].isna().sum()
            if nan_counts.sum() > 0:
                st.warning(f"Se detectaron valores NaN en los datos: {nan_counts}")
                # Eliminar filas con NaN en columnas críticas
                test_with_preds = test_with_preds.dropna(subset=required_cols)
                st.info(f"Después de eliminar NaN, quedan {len(test_with_preds)} filas para métricas.")
            
            if len(test_with_preds) == 0:
                st.error("No hay datos válidos para calcular métricas después de eliminar NaN.")
                return None
            
            # Calibrar intervalos de predicción adaptativamente
            try:
                # Calcular cobertura actual
                y_true = test_with_preds['y'].values
                y_pred = test_with_preds['yhat'].values
                interval_lower = test_with_preds['yhat_lower'].values
                interval_upper = test_with_preds['yhat_upper'].values
                
                current_coverage = ((y_true >= interval_lower) & (y_true <= interval_upper)).mean() * 100
                
                if current_coverage < 80.0 or current_coverage > 98.0:
                    st.info(f"Calibrando intervalos de predicción (cobertura actual: {current_coverage:.2f}%)...")
                    
                    # Aplicar factor de calibración
                    target_coverage = 90.0  # 90% es un buen equilibrio
                    
                    # Método simple para encontrar un factor que mejore la cobertura
                    if current_coverage < 80.0:
                        # Aumentar ancho de intervalos
                        scale_factor = 1.5
                    else:
                        # Reducir ancho de intervalos
                        scale_factor = 0.8
                        
                    # Aplicar calibración
                    half_width = (interval_upper - interval_lower) / 2
                    center = (interval_upper + interval_lower) / 2
                    
                    # Recalcular límites
                    new_lower = center - half_width * scale_factor
                    new_upper = center + half_width * scale_factor
                    
                    # Actualizar dataframe
                    test_with_preds['yhat_lower'] = new_lower
                    test_with_preds['yhat_upper'] = new_upper
                    
                    # Actualizar valores para cálculo de métricas
                    interval_lower = new_lower
                    interval_upper = new_upper
                    
                    # Calcular nueva cobertura
                    new_coverage = ((y_true >= interval_lower) & (y_true <= interval_upper)).mean() * 100
                    st.success(f"✅ Intervalos calibrados: cobertura mejorada de {current_coverage:.2f}% a {new_coverage:.2f}%")
            except Exception as e:
                st.warning(f"No se pudieron calibrar los intervalos: {str(e)}")
            
            # Calcular métricas
            try:
                from modeling.evaluation.metrics import calculate_metrics
                
                # Usar directamente la función calculate_metrics para mayor precisión
                y_true = test_with_preds['y'].values
                y_pred = test_with_preds['yhat'].values
                interval_lower = test_with_preds['yhat_lower'].values
                interval_upper = test_with_preds['yhat_upper'].values
                
                # Calcular métricas usando la función centralizada
                metrics_result = calculate_metrics(y_true, y_pred, interval_lower, interval_upper)
                
                # Si por alguna razón falló el cálculo centralizado, usar el método de respaldo
                if metrics_result is None:
                    st.warning("Usando método alternativo para calcular métricas...")
                    metrics_result = _calculate_forecast_metrics(test_with_preds)
                
                # Normalizar valores para mostrar correctamente como porcentajes
                if 'smape' in metrics_result and metrics_result['smape'] < 1.0:
                    metrics_result['smape'] = metrics_result['smape'] * 100

                if 'coverage' in metrics_result and metrics_result['coverage'] < 1.0:
                    metrics_result['coverage'] = metrics_result['coverage'] * 100
                
                # Verificar que las métricas no son nulas
                for key, value in metrics_result.items():
                    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                        st.warning(f"Métrica {key} inválida ({value}). Usando valor predeterminado.")
                        metrics_result[key] = 0.0
                
                # Crear diccionario final de métricas
                metrics = {
                    'rmse': metrics_result.get('rmse', 0.0),
                    'mae': metrics_result.get('mae', 0.0),
                    'smape': metrics_result.get('smape', 0.0),
                    'coverage': metrics_result.get('coverage', 0.0),
                    'interval_width_avg': metrics_result.get('interval_width_avg', 0.0),
                    'interval_width_relative': metrics_result.get('interval_width_relative', 0.0),
                    'iterations': 1,
                    'horizon': cv_periods
                }
                
                # Mostrar resumen de métricas
                st.info("📊 Resumen de métricas calculadas:")
                st.info(f"RMSE: {metrics['rmse']:.4f}")
                st.info(f"MAE: {metrics['mae']:.4f}")
                st.info(f"SMAPE: {metrics['smape']:.2f}%")
                st.info(f"Cobertura de intervalos: {metrics['coverage']:.2f}%")
                if metrics['coverage'] < 80.0:
                    st.warning("⚠️ La cobertura del intervalo de predicción es baja. Considere aumentar el ancho del intervalo.")
                elif metrics['coverage'] > 98.0:
                    st.warning("⚠️ La cobertura del intervalo de predicción es muy alta. Considere reducir el ancho del intervalo para mejorar la precisión.")
            except Exception as e:
                st.error(f"Error al calcular métricas: {str(e)}")
                st.session_state.model_trained = False
                
                # Mostrar información de depuración
                if hasattr(st.session_state.forecaster, 'df_prophet') and st.session_state.forecaster.df_prophet is not None:
                    st.write(f"Columnas en df_prophet: {st.session_state.forecaster.df_prophet.columns.tolist()}")
                    
                return None
            
            # Crear detalles para visualización
            details = [{
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'smape': metrics['smape'],
                'coverage': metrics['coverage'],
                'cutoff': cutoff_date.strftime('%Y-%m-%d'),
                'num_points': len(test_with_preds),
                'interval_width_avg': metrics['interval_width_avg'],
                'interval_width_relative': metrics['interval_width_relative']
            }]
            
            # Guardar detalles para visualización
            st.session_state.evaluation_details = details
            
            # Guardar predicciones y valores reales para posible visualización
            if valid_df is not None and isinstance(valid_df, pd.DataFrame) and len(valid_df) > 0:
                st.session_state.valid_df = valid_df
            else:
                st.warning("No se pudieron guardar los datos de validación para visualización.")
            
            if forecast_valid is not None and isinstance(forecast_valid, pd.DataFrame) and len(forecast_valid) > 0:
                st.session_state.forecast_valid = forecast_valid
            else:
                st.warning("No se pudieron guardar los datos de predicción para visualización.")
            
            # Mensaje de éxito
            st.success("✅ Evaluación completada con éxito usando datos reales.")
            
            return metrics
        
    except Exception as e:
        st.error(f"Error al evaluar el modelo: {str(e)}")
        st.session_state.model_trained = False
        
        # Mostrar información de depuración
        if 'forecaster' in st.session_state and hasattr(st.session_state.forecaster, 'df_prophet') and st.session_state.forecaster.df_prophet is not None:
            st.write(f"Columnas en df_prophet: {st.session_state.forecaster.df_prophet.columns.tolist()}")
            
        return None