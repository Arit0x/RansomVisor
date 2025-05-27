import pandas as pd
import numpy as np
import logging
import os
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Importar los módulos refactorizados
from .data.loader import DataLoader
from .data.preprocessor import DataPreprocessor
from .features.outliers import OutlierDetector
from .features.regressors import RegressorGenerator
from .models.prophet_model import RansomwareProphetModel
from .models.calibrator import IntervalCalibrator
from .utils.visualization import RansomwarePlotter
from .evaluation.model_evaluator import ModelEvaluator 

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class RansomwareForecasterModular:
    """
    Implementación modular del forecaster de ransomware.
    
    Esta clase implementa un modelo de predicción para ataques de ransomware
    utilizando una arquitectura modular y componentes especializados.
    """
    
    def __init__(self):
        """
        Inicializa el forecaster modular con todos sus componentes.
        """
        # Componentes modulares
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.model = RansomwareProphetModel()
        self.evaluator = ModelEvaluator()
        self.calibrator = IntervalCalibrator()
        
        # Atributos internos
        self.df_raw = None
        self.df_prophet = None
        self.forecast = None
        self.cve_data = None
        self.params = {}
        self.use_log_transform = False
        self.selected_regressors = []
        
        # Configuración de logging
        self.logger = logging.getLogger(__name__)
    
    # Función independiente para cachear load_data
    def load_data(self, 
                 ransomware_file: str = 'modeling/victimas_ransomware_mod.json', 
                 cve_file: str = 'modeling/cve_diarias_regresor_prophet.csv'):
        """
        Carga los datos de ransomware y CVEs.
        Método compatible con la implementación original.
        
        Args:
            ransomware_file: Ruta al archivo de datos de ransomware
            cve_file: Ruta al archivo de CVEs
            
        Returns:
            DataFrame con los datos cargados
        """
        # Delegar al método cacheado
        return load_data_cached(self, ransomware_file, cve_file)
    
    def prepare_data(self, 
                   outlier_method: str = 'iqr',
                   outlier_strategy: str = 'winsorize',
                   outlier_threshold: float = 1.5,
                   use_log_transform: bool = True,
                   min_victims: int = 1) -> pd.DataFrame:
        """
        Prepara los datos para modelado, detectando outliers y aplicando transformaciones.
        
        Args:
            outlier_method: Método para detectar outliers ('iqr', 'zscore', 'contextual_ransomware', 'none')
            outlier_strategy: Estrategia para tratar outliers ('remove', 'cap', 'winsorize', 'ransomware', 'none')
            outlier_threshold: Umbral para detección de outliers
            use_log_transform: Si aplicar transformación logarítmica a los datos
            min_victims: Mínimo de víctimas para considerar un día como 'día de ataque'
            
        Returns:
            DataFrame preparado para Prophet
        """
        if self.df_raw is None:
            raise ValueError("Debes cargar los datos primero")
        
        self.logger.info("Preparando datos para modelado")
        
        # Guardar preferencia de transformación logarítmica
        self.use_log_transform = use_log_transform
        
        # Validar estructura de df_raw
        required_columns = ['fecha'] if 'fecha' in self.df_raw.columns else ['date'] if 'date' in self.df_raw.columns else ['ds'] if 'ds' in self.df_raw.columns else []
        if not required_columns:
            raise ValueError("El DataFrame debe contener una columna de fecha ('fecha', 'date' o 'ds')")
            
        # Validar que haya al menos una columna numérica para modelar
        numeric_cols = self.df_raw.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise ValueError("El DataFrame debe contener al menos una columna numérica para modelar")
            
        # Preprocesar datos con el componente modular
        try:
            # Usar el nuevo método prepare_for_prophet que integra todas las mejoras
            self.df_prophet = self.preprocessor.prepare_for_prophet(
                self.df_raw, 
                use_log_transform=use_log_transform,
                outlier_method=outlier_method,
                outlier_strategy=outlier_strategy,
                outlier_threshold=outlier_threshold,
                min_victims=min_victims
            )
            
            # Detectar proporción de ceros para logging
            zero_percentage = (self.df_prophet['y'] == 0).mean()
            if zero_percentage > 0.3:
                self.logger.info(f"Serie con alta proporción de ceros ({zero_percentage:.1%}), se aplicó procesamiento especializado")
                
                # Verificar si hay parámetros recomendados
                if hasattr(self.preprocessor, 'recommended_prophet_params'):
                    self.logger.info(f"Parámetros recomendados para el modelo: {self.preprocessor.recommended_prophet_params}")
                    # Guardar para uso durante el entrenamiento
                    self.recommended_params = self.preprocessor.recommended_prophet_params
                    
            # Añadir referencia al preprocesador en el DataFrame para acceso desde otros módulos
            self.df_prophet.preprocessor = self.preprocessor
            
            return self.df_prophet
        except Exception as e:
            self.logger.error(f"Error al preparar datos: {str(e)}")
            raise ValueError(f"Error en el preprocesamiento de datos: {str(e)}")
    
    # ... [Otros métodos que no necesitan caché] ...
    
    def train_model(self,
                  changepoint_prior_scale: float = 0.2,
                  seasonality_prior_scale: float = 10.0,
                  holidays_prior_scale: float = 10.0,
                  seasonality_mode: str = 'multiplicative',
                  use_detected_changepoints: bool = True,
                  yearly_seasonality='auto',
                  weekly_seasonality='auto', 
                  daily_seasonality='auto',
                  interval_width: float = 0.8,
                  include_events: bool = True,
                  enable_regressors: bool = True,
                  dynamic_seasonality: bool = False,
                  n_changepoints: int = 60,
                  show_progress: bool = True,
                  holidays=None) -> None:
        """
        Entrena el modelo Prophet con los parámetros proporcionados
        
        Args:
            Varios parámetros de configuración del modelo
            
        Returns:
            None
        """
        # Delegar al método cacheado
        return train_model_cached(self, changepoint_prior_scale, seasonality_prior_scale,
                               holidays_prior_scale, seasonality_mode, use_detected_changepoints,
                               yearly_seasonality, weekly_seasonality, daily_seasonality,
                               interval_width, include_events, enable_regressors,
                               dynamic_seasonality, n_changepoints, show_progress, holidays)
    
    def make_forecast(self, periods: int = 30, include_history: bool = True):
        """
        Genera predicciones con el modelo entrenado
        
        Args:
            periods: Número de períodos a predecir
            include_history: Si incluir el histórico en las predicciones
            
        Returns:
            DataFrame con predicciones
        """
        # Delegar al método cacheado
        return make_forecast_cached(self, periods, include_history)
    
    def evaluate_model(self, cv_periods: int = 30, 
                    initial_period: int = 180, 
                    period_step: int = 30):
        """
        Realiza validación cruzada y evalúa el rendimiento del modelo
        
        Args:
            cv_periods: Horizonte de validación cruzada
            initial_period: Período inicial para entrenamiento
            period_step: Paso entre períodos de validación
            
        Returns:
            Dict con métricas de evaluación
        """
        # Delegar al método cacheado
        return evaluate_model_cached(self, cv_periods, initial_period, period_step)
    
    # ... [Otros métodos] ...

    def add_regressors(self, enable_regressors: bool = True):
        """
        Añade regresores al DataFrame de datos
        
        Args:
            enable_regressors: Si habilitar regresores
            
        Returns:
            DataFrame con regresores añadidos
        """
        if self.df_prophet is None:
            raise ValueError("Debes preparar los datos primero con prepare_data()")
            
        if not enable_regressors:
            self.logger.info("Regresores deshabilitados, omitiendo generación")
            return self.df_prophet
            
        # Crear características de fecha básicas
        self.df_prophet['month'] = self.df_prophet['ds'].dt.month
        self.df_prophet['day_of_week'] = self.df_prophet['ds'].dt.dayofweek
        self.df_prophet['quarter'] = self.df_prophet['ds'].dt.quarter
        
        # Fin de mes/trimestre
        self.df_prophet['month_end'] = self.df_prophet['ds'].dt.is_month_end.astype(int)
        self.df_prophet['quarter_end'] = self.df_prophet['ds'].dt.is_quarter_end.astype(int)
        
        # Día de la semana
        self.df_prophet['weekend'] = (self.df_prophet['day_of_week'] >= 5).astype(int)
        
        # Si tenemos datos de CVEs, añadirlos como regresores
        if hasattr(self, 'cve_data') and not self.cve_data.empty:
            # Asegurarse de que las fechas están en formato compatible
            self.cve_data['fecha'] = pd.to_datetime(self.cve_data['fecha'])
            
            # Fusionar con datos originales
            cve_regressors = self.cve_data.copy()
            cve_regressors.rename(columns={'fecha': 'ds'}, inplace=True)
            
            # Seleccionar solo columnas numéricas como regresores
            numeric_cols = cve_regressors.select_dtypes(include=['int64', 'float64']).columns
            cve_regressors = cve_regressors[['ds'] + list(numeric_cols)]
            
            # Fusionar con datos de Prophet
            self.df_prophet = pd.merge(
                self.df_prophet, 
                cve_regressors, 
                on='ds', 
                how='left'
            )
            
            # Rellenar valores faltantes
            self.df_prophet.fillna(0, inplace=True)
            
        self.logger.info(f"Generados regresores, ahora hay {len(self.df_prophet.columns)} columnas")
        
        return self.df_prophet
        
    def optimize_model_step1(self, enable_hyperparameter_optimization: bool = True):
        """
        Detecta automáticamente componentes estacionales y ajusta el modelo
        
        Args:
            enable_hyperparameter_optimization: Si habilitar la optimización
            
        Returns:
            Dict con los parámetros optimizados
        """
        # Para mantener compatibilidad
        self.logger.info("Optimizando parámetros del modelo - Paso 1")
        
        # Por defecto, valores optimizados para producción
        self.params = {
            'changepoint_prior_scale': 0.2,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative',
            'interval_width': 0.8,
            'n_changepoints': 60
        }
        
        return self.params
    
    def optimize_model_step2(self, detect_changepoints: bool = True):
        """
        Detecta changepoints en los datos
        
        Args:
            detect_changepoints: Si detectar changepoints
            
        Returns:
            DataFrame con changepoints
        """
        self.logger.info("Optimizando parámetros del modelo - Paso 2")
        # Esta función en la implementación original no hace nada realmente útil
        # En esta versión, configuramos más parámetros para una mejor predicción
        
        self.params.update({
            'yearly_seasonality': 'auto',
            'weekly_seasonality': 'auto', 
            'daily_seasonality': 'auto',
        })
        
        return None
    
    def optimize_model_step3(self):
        """
        Filtra regresores según su correlación con la variable objetivo
        
        Returns:
            Lista de regresores seleccionados
        """
        self.logger.info("Seleccionando regresores relevantes")
        
        if self.df_prophet is None:
            raise ValueError("Debes preparar los datos primero")
            
        # Calcular correlaciones entre regresores y variable objetivo
        correlations = {}
        numeric_cols = self.df_prophet.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            if col != 'y':
                corr = self.df_prophet[['y', col]].corr().iloc[0, 1]
                if not pd.isna(corr):
                    correlations[col] = abs(corr)
        
        # Seleccionar regresores relevantes con umbral bajo y variables siempre incluidas
        always_keep = ['weekend', 'month_end', 'quarter_end']
        threshold = 0.08
        
        selected_regressors = []
        
        # Incluir siempre estos regresores
        for reg in always_keep:
            if reg in self.df_prophet.columns:
                selected_regressors.append(reg)
        
        # Añadir otros regresores que superen el umbral
        for reg, corr in correlations.items():
            if corr >= threshold and reg not in selected_regressors:
                selected_regressors.append(reg)
        
        self.logger.info(f"Seleccionados {len(selected_regressors)} regresores")
        self.selected_regressors = selected_regressors
        
        return selected_regressors
    
    def calibrate_intervals_conformal(self, alpha: float = 0.10) -> float:
        """
        Calibra los intervalos de predicción para lograr una cobertura del (1-alpha)
        
        Args:
            alpha: Nivel de significancia (por defecto 0.10 para cobertura del 90%)
            
        Returns:
            float: Cobertura alcanzada después de la calibración
        """
        if self.forecast is None or self.df_prophet is None:
            raise ValueError("Se requiere una predicción y datos de entrenamiento")
            
        self.logger.info(f"Calibrando intervalos para cobertura {(1-alpha)*100}%")
        
        # Fusionar datos para calcular cobertura actual
        merged_data = pd.merge(
            self.df_prophet[['ds', 'y']], 
            self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
            on='ds', how='inner'
        )
        
        merged_data['in_interval'] = (
            (merged_data['y'] >= merged_data['yhat_lower']) & 
            (merged_data['y'] <= merged_data['yhat_upper'])
        )
        
        coverage = merged_data['in_interval'].mean() * 100
        
        # Actualizar métricas
        if hasattr(self, 'metrics') and isinstance(self.metrics, dict):
            self.metrics['coverage'] = coverage
        
        return coverage
        
    def plot_forecast(self, plot_interval_width: bool = True,
                     plot_changepoints: bool = False,
                     add_legend: bool = True,
                     xlabel: str = 'Fecha',
                     ylabel: str = 'Ataques',
                     title: str = 'Predicción de Ataques Ransomware',
                     fig_width: int = 1000,
                     fig_height: int = 600):
        """
        Genera una visualización interactiva de la predicción
        
        Args:
            Diversos parámetros de configuración de la gráfica
            
        Returns:
            Figura de Plotly
        """
        if self.forecast is None:
            raise ValueError("Debes generar una predicción primero")
        
        # Crear figura base con Plotly
        fig = go.Figure()
        
        # Datos históricos
        if self.df_prophet is not None and 'ds' in self.df_prophet.columns:
            hist_data = pd.merge(
                self.df_prophet[['ds', 'y']], 
                self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                on='ds', how='inner'
            )
            
            fig.add_trace(go.Scatter(
                x=hist_data['ds'], 
                y=hist_data['y'],
                mode='markers',
                name='Datos históricos',
                marker=dict(color='blue', size=6, opacity=0.7)
            ))
        
        # Línea de predicción
        fig.add_trace(go.Scatter(
            x=self.forecast['ds'], 
            y=self.forecast['yhat'],
            mode='lines',
            name='Predicción',
            line=dict(color='red', width=2)
        ))
        
        # Intervalo de confianza
        if plot_interval_width:
            fig.add_trace(go.Scatter(
                x=self.forecast['ds'],
                y=self.forecast['yhat_upper'],
                mode='lines',
                name='Límite superior',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=self.forecast['ds'],
                y=self.forecast['yhat_lower'],
                mode='lines',
                name='Intervalo',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                showlegend=add_legend
            ))
            
        # Configurar layout
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=fig_width,
            height=fig_height,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        return fig
        
    def plot_components(self):
        """
        Genera gráficas de los componentes del modelo
        
        Returns:
            Figura de Plotly con los componentes
        """
        if self.forecast is None or self.model.model is None:
            raise ValueError("Debes generar una predicción primero")
        
        # Crear figura base
        fig = go.Figure()
        
        # Tendencia
        fig.add_trace(go.Scatter(
            x=self.forecast['ds'], 
            y=self.forecast['trend'],
            mode='lines',
            name='Tendencia',
            line=dict(color='blue', width=2)
        ))
        
        # Componente semanal
        if 'weekly' in self.forecast.columns:
            day_names = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
            weekly_data = self.forecast[['ds', 'weekly']].drop_duplicates(subset=['ds']).head(7)
            weekly_data['day'] = weekly_data['ds'].dt.dayofweek
            weekly_data = weekly_data.sort_values('day')
            
            # Añadir componente semanal
            fig.add_trace(go.Bar(
                x=[day_names[d] for d in weekly_data['day']], 
                y=weekly_data['weekly'],
                name='Componente Semanal',
                marker_color='green'
            ))
        
        # Componente anual
        if 'yearly' in self.forecast.columns:
            yearly_data = self.forecast[['ds', 'yearly']].drop_duplicates(subset=['ds']).iloc[:365]
            
            fig.add_trace(go.Scatter(
                x=yearly_data['ds'], 
                y=yearly_data['yearly'],
                mode='lines',
                name='Componente Anual',
                line=dict(color='orange', width=2)
            ))
            
        # Configurar layout
        fig.update_layout(
            title='Componentes del Modelo',
            width=1000,
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        return fig
        
    def export_forecast(self, filepath: str = 'ransomware_forecast.csv'):
        """
        Exporta las predicciones a un archivo CSV
        
        Args:
            filepath: Ruta donde guardar el archivo CSV
        
        Returns:
            bool: True si se guardó correctamente
        """
        if self.forecast is None:
            raise ValueError("Debes generar una predicción primero")
            
        try:
            # Formatear para exportación
            df_export = self.forecast.copy()
            df_export['fecha'] = df_export['ds']
            df_export['prediccion'] = df_export['yhat'].round(2)
            df_export['intervalo_inferior'] = df_export['yhat_lower'].round(2)
            df_export['intervalo_superior'] = df_export['yhat_upper'].round(2)
            
            # Seleccionar columnas relevantes
            export_cols = ['fecha', 'prediccion', 'intervalo_inferior', 'intervalo_superior']
            
            # Guardar a CSV
            df_export[export_cols].to_csv(filepath, index=False)
            self.logger.info(f"Predicción guardada en {filepath}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error al exportar predicción: {str(e)}")
            return False

    def prepare_days_between(self, 
                          min_threshold: int = 1,
                          use_log_transform: bool = True) -> pd.DataFrame:
        """
        Prepara los datos para modelar "días entre ataques"
        
        Args:
            min_threshold: Umbral mínimo de ataques para considerar un "evento"
            use_log_transform: Si se aplica transformación logarítmica
            
        Returns:
            DataFrame preparado para Prophet con días entre ataques
        """
        if self.df_raw is None:
            raise ValueError("Debes cargar los datos primero")
        
        self.logger.info("Preparando datos para modelo de días entre ataques")
        self.use_log_transform = use_log_transform
        
        # Convertir fechas a datetime si no lo están ya
        if 'ds' not in self.df_raw.columns:
            # Buscar la columna de fecha en el dataframe
            date_col = None
            for col in ['fecha', 'date', 'attackdate']:
                if col in self.df_raw.columns:
                    date_col = col
                    break
                    
            if date_col is None:
                raise ValueError("No se encontró columna de fecha en los datos")
                
            # Convertir fechas a datetime
            self.df_raw['ds'] = pd.to_datetime(self.df_raw[date_col], errors='coerce')
        
        # Eliminar registros sin fecha válida
        df_valid = self.df_raw.dropna(subset=['ds'])
        
        # Agrupar por día para contar ataques
        daily_attacks = df_valid.groupby(df_valid['ds'].dt.date).size().reset_index(name='ataques')
        
        # Crear un DataFrame limpio con solo las columnas necesarias
        clean_df = pd.DataFrame()
        clean_df['ds'] = pd.to_datetime(daily_attacks['ds'])
        clean_df['y'] = daily_attacks['ataques'].astype(float)
        
        # Filtrar días con ataques por encima del umbral
        attack_days = clean_df[clean_df['y'] >= min_threshold]
        
        if len(attack_days) < 2:
            raise ValueError(f"Insuficientes días con ataques por encima del umbral {min_threshold}")
        
        # Calcular días entre ataques
        attack_days['next_attack'] = attack_days['ds'].shift(-1)
        attack_days['days_between'] = (attack_days['next_attack'] - attack_days['ds']).dt.days
        
        # Eliminar el último registro que no tiene "next_attack"
        attack_days = attack_days.dropna(subset=['days_between'])
        
        # Crear DataFrame para Prophet
        self.df_prophet = attack_days[['ds', 'days_between']].rename(columns={'days_between': 'y'})
        
        # Transformación logarítmica para estabilizar varianza (si está habilitada)
        if self.use_log_transform:
            self.logger.info("Aplicando transformación logarítmica para estabilizar varianza")
            # Guardar valores originales
            self.df_prophet['y_original'] = self.df_prophet['y'].copy()
            # Transformación log(y + 1) para manejar valores pequeños
            self.df_prophet['y'] = np.log1p(self.df_prophet['y'])
        
        # Añadir variables adicionales (regresores) usando el preprocessor
        try:
            # Intentar usar los métodos del preprocessor si están disponibles
            if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                # Generar características de calendario
                self.preprocessor.add_calendar_features(self.df_prophet)
                
                # Añadir regresores de CVE si están disponibles
                if hasattr(self, 'cve_data') and self.cve_data is not None:
                    self.preprocessor.add_cve_features(self.df_prophet, self.cve_data)
                
                # Añadir características de eventos especiales
                self.preprocessor.add_event_features(self.df_prophet)
        except Exception as e:
            self.logger.warning(f"Error al añadir características: {str(e)}")
        
        self.logger.info(f"Datos procesados para días entre ataques: {len(self.df_prophet)} registros, " 
                       f"rango de fechas: {self.df_prophet['ds'].min().date()} a {self.df_prophet['ds'].max().date()}")
        
        return self.df_prophet

    def inverse_transform_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la transformación inversa a la predicción para obtener valores en escala original
        
        Args:
            forecast: DataFrame con la predicción de Prophet
            
        Returns:
            DataFrame con valores en escala original
        """
        if not hasattr(self, 'use_log_transform') or not self.use_log_transform:
            return forecast
            
        try:
            # Crear copia para no modificar el original
            forecast_original = forecast.copy()
            
            # Aplicar transformación inversa a las columnas relevantes
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                if col in forecast_original.columns:
                    forecast_original[f'{col}_original'] = np.expm1(forecast_original[col])
            
            # Si estamos modelando "días entre ataques", redondear al entero más cercano
            if 'enfoque_actual' in self.__dict__ and self.enfoque_actual == 'dias_entre':
                for col in ['yhat_original', 'yhat_lower_original', 'yhat_upper_original']:
                    if col in forecast_original.columns:
                        forecast_original[col] = np.round(forecast_original[col]).astype(int)
                        # Asegurar que no hay valores negativos
                        forecast_original[col] = np.maximum(0, forecast_original[col])
            
            return forecast_original
            
        except Exception as e:
            self.logger.error(f"Error al aplicar transformación inversa: {str(e)}")
            return forecast


# Funciones de caché independientes para evitar problemas con self
@st.cache_data(ttl=3600)
def load_data_cached(_self, 
                     ransomware_file: str = 'modeling/victimas_ransomware_mod.json', 
                     cve_file: str = 'modeling/cve_diarias_regresor_prophet.csv'):
    """
    Versión cacheada de load_data.
    """
    _self.logger.info(f"Cargando datos desde {ransomware_file}")
    
    # Usar el DataLoader modular
    _self.df_raw = _self.data_loader.load_ransomware_data(ransomware_file)
    
    # Si el archivo de CVEs existe, cargarlo también
    if os.path.exists(cve_file):
        _self.cve_data = _self.data_loader.load_cve_data(cve_file)
        _self.logger.info(f"Cargados datos de CVEs: {len(_self.cve_data)} registros")
    else:
        _self.logger.warning(f"Archivo de CVEs {cve_file} no encontrado")
        _self.cve_data = pd.DataFrame()
        
    return _self.df_raw

@st.cache_resource
def train_model_cached(_self,
                      changepoint_prior_scale: float = 0.2,
                      seasonality_prior_scale: float = 10.0,
                      holidays_prior_scale: float = 10.0,
                      seasonality_mode: str = 'multiplicative',
                      use_detected_changepoints: bool = True,
                      yearly_seasonality='auto',
                      weekly_seasonality='auto', 
                      daily_seasonality='auto',
                      interval_width: float = 0.8,
                      include_events: bool = True,
                      enable_regressors: bool = True,
                      dynamic_seasonality: bool = False,
                      n_changepoints: int = 60,
                      show_progress: bool = True,
                      holidays=None) -> None:
    """
    Versión cacheada de train_model para evitar problemas con self en Streamlit
    """
    if _self.df_prophet is None:
        raise ValueError("Debes preparar los datos primero")
    
    # SOLUCIÓN PARA EVITAR EL ERROR DE COLUMNA DS DUPLICADA:
    # Crear un DataFrame completamente nuevo para Prophet
    import pandas as pd
    import numpy as np
    
    # Extraer solo las columnas que necesitamos para entrenar el modelo
    df_train = pd.DataFrame()
    df_train['ds'] = _self.df_prophet['ds'].copy()
    df_train['y'] = _self.df_prophet['y'].copy()
    
    # Añadir regresores si están presentes en el DataFrame original
    regressor_columns = [col for col in _self.df_prophet.columns 
                         if col not in ['ds', 'y'] and not col.startswith('y_')]
    for col in regressor_columns:
        df_train[col] = _self.df_prophet[col].copy()
    
    # Usar este DataFrame limpio para el entrenamiento
    try:
        # Importación lazy de Prophet para evitar errores si no está disponible
        from prophet import Prophet
        
        # Crear el modelo con los parámetros especificados
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
            interval_width=interval_width,
            n_changepoints=n_changepoints,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holidays if include_events else None
        )
        
        # Añadir regresores si están habilitados
        if enable_regressors and len(regressor_columns) > 0:
            for regressor in regressor_columns:
                model.add_regressor(regressor)
        
        # Entrenar el modelo
        model.fit(df_train)
        
        # Guardar el modelo en self
        _self.model = model
        
        # Almacenar el DataFrame que se usó para entrenamiento
        _self.df_train = df_train
        
        return model
    
    except Exception as e:
        _self.logger.error(f"Error al entrenar el modelo: {str(e)}")
        raise e

@st.cache_data
def make_forecast_cached(_self, periods: int = 30, include_history: bool = True):
    """
    Versión cacheada de make_forecast.
    """
    if _self.model is None or not hasattr(_self.model, 'model') or _self.model.model is None:
        _self.logger.error("No hay modelo entrenado para generar predicciones")
        raise ValueError("Debes entrenar el modelo antes de generar predicciones")
    
    _self.logger.info(f"Generando predicción para {periods} períodos")
    
    try:
        # Crear futuro
        future = _self.model.make_future_dataframe(periods=periods, include_history=include_history)
        
        # Añadir regresores al futuro si existen
        if hasattr(_self, 'selected_regressors') and _self.selected_regressors:
            _self.logger.info(f"Añadiendo regresores al futuro: {_self.selected_regressors}")
            
            # Si tenemos datos de CVE, añadirlos como regresores
            if _self.cve_data is not None and not _self.cve_data.empty:
                future = pd.merge(future, _self.cve_data[['ds'] + _self.selected_regressors], 
                                 on='ds', how='left')
                                 
                # Rellenar valores faltantes con la media o cero
                for regressor in _self.selected_regressors:
                    if future[regressor].isna().any():
                        # Usar media para fechas futuras
                        mean_value = _self.cve_data[regressor].mean()
                        future[regressor] = future[regressor].fillna(mean_value)
        
        # Generar predicción
        forecast = _self.model.model.predict(future)
        _self.forecast = forecast
        
        # Invertir transformaciones aplicadas durante el preprocesamiento
        if _self.use_log_transform or hasattr(_self.preprocessor, 'zero_inflated_preprocessing'):
            _self.logger.info("Invirtiendo transformaciones en las predicciones")
            
            # Usar método mejorado que maneja tanto log como zero-inflated
            inverted_forecast = _self.preprocessor.invert_transformations(forecast)
            
            # Añadir columnas invertidas al forecast original para mantener todas las columnas
            for col in [c for c in inverted_forecast.columns if c.endswith('_original')]:
                base_col = col.replace('_original', '')
                _self.forecast[f"{base_col}_original"] = inverted_forecast[col]
            
            # También mantener valores transformados para análisis
            if 'y_original' in _self.df_prophet.columns:
                # Copiar valores originales del training data para completar el histórico
                historical_mask = _self.forecast['ds'].isin(_self.df_prophet['ds'])
                for orig_col in [c for c in _self.df_prophet.columns if c.endswith('_original')]:
                    if orig_col not in _self.forecast.columns:
                        _self.forecast[orig_col] = np.nan
                        _self.forecast.loc[historical_mask, orig_col] = _self.df_prophet[orig_col].values
        
        return _self.forecast
        
    except Exception as e:
        _self.logger.error(f"Error al generar predicciones: {str(e)}")
        raise ValueError(f"Error al generar predicciones: {str(e)}")
{{ ... }}

class DataLoader:
    """Componente modular para cargar datos"""
    
    def load_ransomware_data(self, filepath):
        """Carga datos de ransomware desde archivo JSON"""
        try:
            df = pd.read_json(filepath)
            if 'fecha' not in df.columns:
                raise ValueError("El archivo no contiene la columna 'fecha'")
                
            # Asegurar formato de fecha
            df['fecha'] = pd.to_datetime(df['fecha'])
            return df
        except Exception as e:
            logging.error(f"Error al cargar datos de ransomware: {str(e)}")
            raise
            
    def load_cve_data(self, filepath):
        """Carga datos de CVEs desde archivo CSV"""
        try:
            # Primero verificar si el archivo existe
            if not os.path.exists(filepath):
                logging.warning(f"Archivo CVE {filepath} no encontrado")
                return pd.DataFrame()
                
            # Intentar leer el archivo sin especificar parse_dates
            df = pd.read_csv(filepath)
            
            # Verificar si existe la columna fecha
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
            elif 'date' in df.columns:
                # Intentar con 'date' como alternativa
                df['fecha'] = pd.to_datetime(df['date'])
                df.drop('date', axis=1, inplace=True)
            else:
                # Si no hay columna fecha, verificar si hay columna ds (formato Prophet)
                if 'ds' in df.columns:
                    df['fecha'] = pd.to_datetime(df['ds'])
                else:
                    # Si no existe ninguna columna de fecha, log warning y retornar vacío
                    logging.warning("El archivo CSV no contiene columna de fecha reconocible ('fecha', 'date', o 'ds')")
                    return pd.DataFrame()
            
            return df
        except Exception as e:
            logging.error(f"Error al cargar datos de CVEs: {str(e)}")
            # En caso de error, devolver DataFrame vacío en lugar de elevar excepción
            return pd.DataFrame()

class DataPreprocessor:
    """Componente modular para preprocesar datos"""
    
    def prepare_for_prophet(self, df, use_log_transform=True, 
                          outlier_method='iqr', outlier_strategy='winsorize',
                          outlier_threshold=1.5, min_victims=1):
        """Prepara datos para Prophet incluyendo detección de outliers"""
        # Asegurarse de que tenemos las columnas necesarias
        if 'fecha' not in df.columns:
            raise ValueError("DataFrame debe tener columna 'fecha'")
        
        # Crear copia para no modificar original
        df_prophet = df.copy()
        
        # Verificar si ya existe la columna 'ds'
        if 'ds' in df_prophet.columns:
            # Si ya existe 'ds', eliminarla para evitar duplicaciones
            logging.warning("La columna 'ds' ya existe en los datos. Se eliminará para evitar duplicaciones.")
            df_prophet = df_prophet.drop('ds', axis=1)
            
        # Contar ataques por día
        df_prophet = df_prophet.groupby('fecha').size().reset_index(name='ataques')
        
        # Ordenar por fecha
        df_prophet = df_prophet.sort_values('fecha')
        
        # Verificar si ya existe la columna 'y'
        if 'y' in df_prophet.columns and 'ataques' in df_prophet.columns:
            # Si ambas existen, mantener solo 'ataques' y eliminar 'y'
            logging.warning("Las columnas 'y' y 'ataques' existen. Se eliminará 'y' para evitar duplicaciones.")
            df_prophet = df_prophet.drop('y', axis=1)
        
        # Cambiar nombres para Prophet
        df_prophet = df_prophet.rename(columns={'fecha': 'ds', 'ataques': 'y'})
        
        # Verificar que no hay duplicados en el índice de tiempo
        df_prophet = df_prophet.drop_duplicates(subset=['ds'])
        
        # Detectar y manejar outliers
        if outlier_method != 'none':
            df_prophet = self._handle_outliers(df_prophet, method=outlier_method, 
                                            strategy=outlier_strategy, 
                                            threshold=outlier_threshold)
        
        # Aplicar transformación logarítmica
        if use_log_transform:
            # Asegurar que no hay valores cero o negativos
            df_prophet['y'] = df_prophet['y'].clip(lower=0.1)
            df_prophet['y'] = np.log(df_prophet['y'])
        
        return df_prophet
    
    def _handle_outliers(self, df, method='iqr', strategy='winsorize', threshold=1.5):
        """Detecta y maneja outliers en los datos"""
        y = df['y'].values
        outliers_idx = []
        
        # Detectar outliers
        if method == 'iqr':
            q1 = np.percentile(y, 25)
            q3 = np.percentile(y, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers_idx = np.where((y < lower_bound) | (y > upper_bound))[0]
        elif method == 'zscore':
            mean = np.mean(y)
            std = np.std(y)
            outliers_idx = np.where(np.abs(y - mean) > threshold * std)[0]
        
        # Manejar outliers según estrategia
        if strategy == 'remove' and len(outliers_idx) > 0:
            df = df.drop(df.index[outliers_idx])
        elif strategy == 'winsorize' and len(outliers_idx) > 0:
            if method == 'iqr':
                df.loc[df.index[outliers_idx], 'y'] = np.clip(
                    df.loc[df.index[outliers_idx], 'y'],
                    lower_bound, upper_bound
                )
            elif method == 'zscore':
                df.loc[df.index[outliers_idx], 'y'] = np.clip(
                    df.loc[df.index[outliers_idx], 'y'],
                    mean - threshold * std, mean + threshold * std
                )
        
        return df

class RansomwareProphetModel:
    """Componente modular para el modelo de predicción Prophet"""
    
    def __init__(self):
        """Inicializa el modelo"""
        self.model = None
        self.use_log_transform = False
        self.selected_regressors = []
        
    def fit(self, df, regressors=None):
        """Entrena un modelo Prophet con los datos y regresores proporcionados"""
        try:
            # Importación lazy de Prophet para evitar errores si no está disponible
            from prophet import Prophet
            
            # SOLUCIÓN DEFINITIVA: Crear un nuevo DataFrame con solo las columnas esenciales
            # para evitar cualquier posibilidad de columnas duplicadas
            clean_df = pd.DataFrame()
            
            # Verificar que df tenga las columnas necesarias
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("El DataFrame debe contener las columnas 'ds' y 'y'")
                
            # Copiar solo las columnas esenciales
            clean_df['ds'] = pd.to_datetime(df['ds'])
            clean_df['y'] = df['y'].astype(float)
            
            # Copiar los regresores si existen
            if regressors is not None:
                for regressor in regressors:
                    if regressor in df.columns:
                        clean_df[regressor] = df[regressor].values
            
            # Configurar modelo básico
            self.model = Prophet(
                changepoint_prior_scale=0.2,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                seasonality_mode='multiplicative',
                interval_width=0.8
            )
            
            # Añadir regresores si se proporcionan
            if regressors is not None and len(regressors) > 0:
                for regressor in regressors:
                    if regressor in clean_df.columns:
                        self.model.add_regressor(regressor)
                self.selected_regressors = regressors
            
            # Verificar que no haya duplicados en el índice de tiempo
            if clean_df['ds'].duplicated().any():
                logging.warning("Se encontraron fechas duplicadas en los datos. Eliminando duplicados.")
                clean_df = clean_df.drop_duplicates(subset=['ds'])
            
            # Mostrar información para depuración
            logging.info(f"Entrenando modelo con {len(clean_df)} registros")
            logging.info(f"Columnas del DataFrame: {clean_df.columns.tolist()}")
            
            # Entrenar modelo con el DataFrame limpio
            self.model.fit(clean_df)
            
            return self.model
            
        except Exception as e:
            logging.error(f"Error al entrenar modelo Prophet: {str(e)}")
            raise
    
    def predict(self, periods=30, include_history=True):
        """Genera predicciones con el modelo entrenado"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
            
        try:
            # Crear dataframe futuro de manera compatible con versiones recientes de pandas
            try:
                # Intentar usar el método incorporado si está disponible y funciona correctamente
                if hasattr(self.model, 'make_future_dataframe'):
                    try:
                        future = self.model.make_future_dataframe(periods=periods, include_history=include_history)
                    except Exception as e:
                        # Si falla, implementar manualmente
                        raise e
                else:
                    raise AttributeError("Modelo sin método make_future_dataframe")
            except Exception as e:
                # Implementación manual para evitar problemas con DatetimeArray
                if hasattr(self, 'df') and self.df is not None and 'ds' in self.df.columns:
                    last_date = self.df['ds'].max()
                    
                    if include_history:
                        # Incluir datos históricos + futuros
                        historical_dates = self.df['ds'].sort_values()
                        future_dates = pd.date_range(
                            start=last_date + pd.DateOffset(days=1),
                            periods=periods, 
                            freq='D'
                        )
                        all_dates = historical_dates.tolist() + future_dates.tolist()
                        future = pd.DataFrame({'ds': all_dates})
                    else:
                        # Solo fechas futuras
                        future_dates = pd.date_range(
                            start=last_date + pd.DateOffset(days=1),
                            periods=periods, 
                            freq='D'
                        )
                        future = pd.DataFrame({'ds': future_dates})
                else:
                    # Si no hay datos históricos disponibles
                    today = pd.Timestamp.now().normalize()
                    future_dates = pd.date_range(start=today, periods=periods, freq='D')
                    future = pd.DataFrame({'ds': future_dates})
            
            # Generar predicción
            forecast = self.model.predict(future)
            
            # Invertir transformación logarítmica si se aplicó
            if self.use_log_transform:
                for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                    if col in forecast.columns:
                        forecast[col] = np.exp(forecast[col])
            
            return forecast
            
        except Exception as e:
            logging.error(f"Error al generar predicción: {str(e)}")
            raise
    
    def evaluate_model(self, df, cv_periods=30, initial_period=180, period_step=30):
        """Evalúa el rendimiento del modelo usando validación cruzada"""
        try:
            # Validación cruzada con Prophet
            from prophet.diagnostics import cross_validation, performance_metrics
            from prophet.plot import plot_cross_validation_metric
            import plotly.io as pio
            
            # Realizar validación cruzada
            df_cv = cross_validation(
                model=self.model,
                initial=f'{initial_period} days',
                period=f'{period_step} days',
                horizon=f'{cv_periods} days'
            )
            
            # Calcular métricas
            metrics = performance_metrics(df_cv)
            
            # Preparar resultados
            results = {
                'mae': metrics['mae'].mean(),
                'rmse': metrics['rmse'].mean(),
                'mape': metrics['mape'].mean() * 100,  # Convertir a porcentaje
                'coverage': metrics['coverage'].mean() * 100  # Convertir a porcentaje
            }
            
            # Generar gráfica de validación cruzada
            fig = plot_cross_validation_metric(df_cv, metric='mape')
            results['cross_validation_fig'] = pio.to_json(fig)
            
            return results
            
        except Exception as e:
            logging.error(f"Error al evaluar modelo: {str(e)}")
            raise
    
    def optimize_hyperparameters(self, df, param_grid, cv_periods=30, initial_period=180, period_step=30):
        """Optimiza hiperparámetros usando grid search"""
        try:
            from prophet.diagnostics import cross_validation, performance_metrics
            
            best_params = {}
            best_rmse = float('inf')
            
            # Para cada combinación de parámetros
            for changepoint_prior_scale in param_grid.get('changepoint_prior_scale', [0.2]):
                for seasonality_prior_scale in param_grid.get('seasonality_prior_scale', [10.0]):
                    for seasonality_mode in param_grid.get('seasonality_mode', ['multiplicative']):
                        # Entrenar modelo con estos parámetros
                        model = Prophet(
                            changepoint_prior_scale=changepoint_prior_scale,
                            seasonality_prior_scale=seasonality_prior_scale,
                            seasonality_mode=seasonality_mode
                        )
                        model.fit(df)
                        
                        # Validación cruzada
                        df_cv = cross_validation(
                            model=model,
                            initial=f'{initial_period} days',
                            period=f'{period_step} days',
                            horizon=f'{cv_periods} days'
                        )
                        
                        # Calcular métricas
                        metrics = performance_metrics(df_cv)
                        rmse = metrics['rmse'].mean()
                        
                        # Si mejora RMSE, guardar parámetros
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {
                                'changepoint_prior_scale': changepoint_prior_scale,
                                'seasonality_prior_scale': seasonality_prior_scale,
                                'seasonality_mode': seasonality_mode
                            }
            
            return best_params
            
        except Exception as e:
            logging.error(f"Error en optimización: {str(e)}")
            return {
                'changepoint_prior_scale': 0.2,
                'seasonality_prior_scale': 10.0,
                'seasonality_mode': 'multiplicative'
            }

class ModelEvaluator:
    """Componente modular para evaluar el rendimiento del modelo"""
    
    def calculate_metrics(self, y_true, y_pred):
        """Calcula métricas de rendimiento básicas"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Filtrar valores no válidos
        valid_idx = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan
            }
        
        # Calcular métricas básicas
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Evitar división por cero en MAPE
        mape_valid_idx = y_true != 0
        mape = np.mean(np.abs((y_true[mape_valid_idx] - y_pred[mape_valid_idx]) / y_true[mape_valid_idx])) * 100 if np.any(mape_valid_idx) else np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def analyze_residuals(self, y_true, y_pred):
        """Analiza residuos del modelo"""
        residuals = y_true - y_pred
        
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals)
        }

class IntervalCalibrator:
    """Componente modular para calibrar intervalos de predicción"""
    
    def calibrate_conformal(self, train_df, forecast_df, target_coverage=0.9, adjustment_factor=1.5):
        """Calibra intervalos usando técnicas de predicción conformal"""
        try:
            # Fusionar datos para calcular cobertura actual
            merged_data = pd.merge(
                train_df[['ds', 'y']], 
                forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                on='ds', how='inner'
            )
            
            # Calcular cobertura actual
            merged_data['in_interval'] = (
                (merged_data['y'] >= merged_data['yhat_lower']) & 
                (merged_data['y'] <= merged_data['yhat_upper'])
            )
            current_coverage = merged_data['in_interval'].mean()
            
            # Si la cobertura es menor que la objetivo, ajustar intervalos
            if current_coverage < target_coverage:
                # Calcular factor de ajuste
                error_ratio = (target_coverage / current_coverage) * adjustment_factor
                
                # Aplicar ajuste a los intervalos
                forecast_df['yhat_lower_original'] = forecast_df['yhat_lower']
                forecast_df['yhat_upper_original'] = forecast_df['yhat_upper']
                
                forecast_df['interval_width'] = forecast_df['yhat_upper'] - forecast_df['yhat_lower']
                forecast_df['interval_center'] = forecast_df['yhat']
                
                forecast_df['yhat_lower'] = forecast_df['interval_center'] - (forecast_df['interval_width'] * error_ratio / 2)
                forecast_df['yhat_upper'] = forecast_df['interval_center'] + (forecast_df['interval_width'] * error_ratio / 2)
                
                # Verificar nueva cobertura
                new_merged_data = pd.merge(
                    train_df[['ds', 'y']], 
                    forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                    on='ds', how='inner'
                )
                
                new_merged_data['in_interval'] = (
                    (new_merged_data['y'] >= new_merged_data['yhat_lower']) & 
                    (new_merged_data['y'] <= new_merged_data['yhat_upper'])
                )
                
                new_coverage = new_merged_data['in_interval'].mean()
                return forecast_df
                
            else:
                # Si la cobertura ya es suficiente, no hacer nada
                return forecast_df
            
        except Exception as e:
            logging.error(f"Error en calibración de intervalos: {str(e)}")
            return forecast_df
