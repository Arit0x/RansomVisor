"""
Módulo de optimizaciones avanzadas para modelos de predicción de ransomware.

Este módulo integra todas las optimizaciones avanzadas disponibles para mejorar
el rendimiento de los modelos Prophet para la predicción de ataques de ransomware.
"""

import pandas as pd
import numpy as np
import logging
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Configuración de logging
logger = logging.getLogger(__name__)

# Importar componentes específicos
try:
    # Intenta importación relativa primero
    from .feature_engineering import FeatureEngineer
    from .interval_calibrator import IntervalCalibrator
    from .hyperparameter_optimizer import BayesianOptimizer
    ADVANCED_MODULES_AVAILABLE = True
except ImportError:
    try:
        # Si falla, intenta importación absoluta
        from modeling.feature_engineering import FeatureEngineer
        from modeling.interval_calibrator import IntervalCalibrator
        from modeling.hyperparameter_optimizer import BayesianOptimizer
        ADVANCED_MODULES_AVAILABLE = True
    except ImportError:
        # Si ambas fallan, registra el error pero no muestra mensaje aún
        logger.error("No se pudieron importar los módulos necesarios para las optimizaciones avanzadas")
        ADVANCED_MODULES_AVAILABLE = False
        
        # Define clases con métodos mínimos para evitar errores
        class FeatureEngineer:
            def __init__(self): 
                pass
                
            @staticmethod
            def apply_optimal_transformation(df, method='log'):
                # Implementación mínima para evitar errores
                return df.copy(), lambda x: x
                
            @staticmethod
            def create_temporal_features(df):
                return df.copy()
                
            @staticmethod
            def add_patch_tuesday_features(df):
                return df.copy()
                
            @staticmethod
            def add_cve_features(df, cve_df):
                return df.copy()
                
            @staticmethod
            def select_optimal_features(df, target_col, max_features, corr_threshold):
                return []
                
            @staticmethod
            def reverse_transform_forecast(forecast, func):
                return forecast
                
        class IntervalCalibrator:
            def __init__(self, target_coverage=0.95): 
                self.target_coverage = target_coverage
                
            def fit(self, actual_values, predicted_values, target_coverage):
                pass
                
            def calibrate(self, forecast_df):
                return forecast_df
                
        class BayesianOptimizer:
            def __init__(self, seasonality_mode='multiplicative'):
                self.seasonality_mode = seasonality_mode
                
            def optimize_hyperparameters(self, df, regressors, target_metric, n_trials, timeout):
                # Implementación mínima para evitar errores
                return {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0}

# Configuración de logging
logger = logging.getLogger(__name__)


class RansomwareOptimizer:
    """
    Clase que integra todas las optimizaciones avanzadas para modelos de predicción
    de ransomware, incluyendo feature engineering, selección de regresores,
    optimización de hiperparámetros y calibración de intervalos.
    """
    
    def __init__(self, transform_method: str = 'log'):
        """
        Inicializa el optimizador con la configuración especificada.
        
        Args:
            transform_method: Método de transformación para la variable objetivo ('log', 'sqrt', 'none')
        """
        self.transform_method = transform_method
        self.feature_engineer = FeatureEngineer()
        self.interval_calibrator = IntervalCalibrator()
        self.bayesian_optimizer = BayesianOptimizer(seasonality_mode='multiplicative')
        self.selected_features = []
        self.best_params = {}
        self.reverse_transform_func = None
        self.is_prepared = False
        
    def prepare_features(self, 
                       df: pd.DataFrame, 
                       cve_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepara las características para el modelo, aplicando transformaciones y
        feature engineering avanzado.
        
        Args:
            df: DataFrame con datos de ransomware (debe tener columnas 'ds' y 'y')
            cve_df: DataFrame con datos de CVE (opcional)
            
        Returns:
            DataFrame con características preparadas
        """
        logger.info("Preparando características avanzadas...")
        
        # 1. Aplicar transformación óptima
        if 'y' not in df.columns or 'ds' not in df.columns:
            raise ValueError("El DataFrame debe contener columnas 'ds' y 'y'")
            
        transformed_df, self.reverse_transform_func = self.feature_engineer.apply_optimal_transformation(
            df, method=self.transform_method
        )
        
        # 2. Crear características temporales básicas
        df_features = self.feature_engineer.create_temporal_features(transformed_df)
        
        # 3. Añadir características de Patch Tuesday
        df_features = self.feature_engineer.add_patch_tuesday_features(df_features)
        
        # 4. Añadir características de CVE si están disponibles
        if cve_df is not None:
            df_features = self.feature_engineer.add_cve_features(df_features, cve_df)
        
        # Marcar como preparado
        self.is_prepared = True
        
        return df_features
    
    def select_optimal_regressors(self, 
                                df: pd.DataFrame, 
                                target_col: str = 'y', 
                                max_features: int = 10,
                                correlation_threshold: float = 0.1) -> List[str]:
        """
        Selecciona los regresores óptimos para el modelo basándose en
        correlación con la variable objetivo.
        
        Args:
            df: DataFrame con características y target
            target_col: Nombre de la columna objetivo
            max_features: Número máximo de características a seleccionar
            correlation_threshold: Umbral mínimo de correlación
            
        Returns:
            Lista de nombres de los regresores seleccionados
        """
        logger.info(f"Seleccionando regresores óptimos (umbral={correlation_threshold})...")
        
        # Usar la funcionalidad del FeatureEngineer
        self.selected_features = self.feature_engineer.select_optimal_features(
            df, 
            target_col=target_col,
            max_features=max_features,
            corr_threshold=correlation_threshold
        )
        
        if self.selected_features:
            logger.info(f"Seleccionados {len(self.selected_features)} regresores óptimos: {self.selected_features}")
        else:
            logger.warning("No se encontraron regresores con correlación significativa")
        
        return self.selected_features
    
    def optimize_hyperparameters(self, 
                               df: pd.DataFrame, 
                               regressors: List[str] = None,
                               target_metric: str = 'smape',
                               n_trials: int = 25,
                               timeout: int = 600) -> Dict:
        """
        Optimiza los hiperparámetros del modelo usando Bayesian Optimization.
        
        Args:
            df: DataFrame con datos
            regressors: Lista de nombres de regresores
            target_metric: Métrica a optimizar
            n_trials: Número de combinaciones a probar
            timeout: Tiempo máximo en segundos
            
        Returns:
            Diccionario con los mejores parámetros encontrados
        """
        logger.info(f"Optimizando hiperparámetros ({n_trials} trials, timeout={timeout}s)...")
        
        # Optimizar hiperparámetros
        opt_results = self.bayesian_optimizer.optimize_hyperparameters(
            df=df,
            regressors=regressors or self.selected_features,
            target_metric=target_metric,
            n_trials=n_trials,
            timeout=timeout
        )
        
        if opt_results and 'best_params' in opt_results:
            # Guardar los mejores parámetros
            self.best_params = opt_results['best_params']
            logger.info(f"Optimización completada. Mejores parámetros: {self.best_params}")
            
            return self.best_params
        else:
            logger.warning("La optimización no produjo resultados válidos")
            return {}
    
    def calibrate_intervals(self, 
                          model: Prophet, 
                          actual_values: pd.Series, 
                          predicted_values: pd.Series,
                          target_coverage: float = 0.95) -> Prophet:
        """
        Calibra los intervalos de predicción del modelo usando métodos conformal.
        
        Args:
            model: Modelo Prophet entrenado
            actual_values: Valores reales observados
            predicted_values: Valores predichos por el modelo
            target_coverage: Nivel de cobertura objetivo
            
        Returns:
            Modelo Prophet con intervalos calibrados
        """
        logger.info(f"Calibrando intervalos de predicción (cobertura={target_coverage})...")
        
        # Ajustar el calibrador con los datos históricos
        self.interval_calibrator.fit(
            actual_values=actual_values,
            predicted_values=predicted_values,
            target_coverage=target_coverage
        )
        
        # No podemos modificar el modelo Prophet directamente para calibrar los intervalos,
        # en su lugar, guardaremos el calibrador para aplicarlo a las predicciones
        logger.info("Calibración completada. Se aplicará a las predicciones futuras.")
        
        return model
    
    def apply_calibration_to_forecast(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la calibración de intervalos a un DataFrame de predicción.
        
        Args:
            forecast_df: DataFrame con predicciones de Prophet
            
        Returns:
            DataFrame con intervalos calibrados
        """
        # Verificar que el calibrador ha sido entrenado
        if not hasattr(self.interval_calibrator, 'lower_adjustment') or self.interval_calibrator.lower_adjustment is None:
            logger.warning("El calibrador no ha sido entrenado. Usando intervalos originales.")
            return forecast_df
        
        # Aplicar calibración
        return self.interval_calibrator.calibrate(forecast_df)
    
    def create_optimized_model(self, 
                             df: pd.DataFrame, 
                             cve_df: pd.DataFrame = None,
                             use_optimal_regressors: bool = True,
                             use_bayesian_optimization: bool = True,
                             use_interval_calibration: bool = True,
                             correlation_threshold: float = 0.1,
                             optimization_trials: int = 25) -> Tuple[Prophet, Dict]:
        """
        Crea y entrena un modelo completamente optimizado aplicando todas las
        optimizaciones disponibles.
        
        Args:
            df: DataFrame con datos de ransomware
            cve_df: DataFrame con datos de CVE (opcional)
            use_optimal_regressors: Si usar selección óptima de regresores
            use_bayesian_optimization: Si usar optimización bayesiana
            use_interval_calibration: Si calibrar intervalos
            correlation_threshold: Umbral de correlación para regresores
            optimization_trials: Número de trials para optimización
            
        Returns:
            Tupla con (modelo entrenado, diccionario de resultados)
        """
        try:
            # 1. Preparar características
            df_features = self.prepare_features(df, cve_df)
            
            # 2. Seleccionar regresores óptimos si está habilitado
            regressors = []
            if use_optimal_regressors:
                regressors = self.select_optimal_regressors(
                    df_features,
                    correlation_threshold=correlation_threshold
                )
            
            # 3. Optimizar hiperparámetros si está habilitado
            params = {}
            if use_bayesian_optimization:
                params = self.optimize_hyperparameters(
                    df=df_features,
                    regressors=regressors,
                    n_trials=optimization_trials
                )
            
            # Si la optimización falló o no está habilitada, usar parámetros por defecto
            if not params:
                params = {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10.0,
                    'holidays_prior_scale': 10.0,
                    'seasonality_mode': 'multiplicative',
                    'interval_width': 0.9
                }
            
            # 4. Crear y entrenar modelo
            model = Prophet(
                changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=params.get('holidays_prior_scale', 10.0),
                seasonality_mode=params.get('seasonality_mode', 'multiplicative'),
                interval_width=params.get('interval_width', 0.9)
            )
            
            # Añadir regresores al modelo
            if regressors:
                for regressor in regressors:
                    if regressor in df_features.columns:
                        model.add_regressor(regressor)
            
            # Entrenar modelo
            model.fit(df_features)
            
            # 5. Calibrar intervalos si está habilitado
            if use_interval_calibration:
                # Generar predicciones para los datos de entrenamiento
                train_future = model.make_future_dataframe(periods=0)
                
                # Añadir regresores al dataframe futuro
                if regressors:
                    for regressor in regressors:
                        if regressor in df_features.columns:
                            train_future[regressor] = df_features[regressor].values
                
                # Predecir
                train_forecast = model.predict(train_future)
                
                # Calibrar intervalos
                model = self.calibrate_intervals(
                    model=model,
                    actual_values=df_features['y'],
                    predicted_values=train_forecast['yhat']
                )
            
            # 6. Preparar resultados
            results = {
                'params': params,
                'regressors': regressors,
                'features_shape': df_features.shape,
                'reverse_transform_func': self.reverse_transform_func,
                'has_calibration': use_interval_calibration
            }
            
            return model, results
            
        except Exception as e:
            logger.error(f"Error al crear modelo optimizado: {str(e)}")
            raise
    
    def apply_advanced_optimizations(self, 
                                   df: pd.DataFrame, 
                                   model: Optional[Prophet] = None, 
                                   cve_df: Optional[pd.DataFrame] = None,
                                   params: Optional[Dict] = None) -> Tuple[Prophet, Dict]:
        """
        Aplica todas las optimizaciones avanzadas a un modelo existente o crea uno nuevo.
        
        Args:
            df: DataFrame con datos
            model: Modelo Prophet existente (opcional)
            cve_df: DataFrame con datos de CVE (opcional)
            params: Parámetros para personalizar las optimizaciones (opcional)
            
        Returns:
            Tupla con (modelo optimizado, diccionario de resultados)
        """
        # Configurar parámetros por defecto
        default_params = {
            'use_optimal_regressors': True,
            'use_bayesian_optimization': True,
            'use_interval_calibration': True,
            'correlation_threshold': 0.1,
            'optimization_trials': 25,
            'transform_method': self.transform_method
        }
        
        # Combinar con parámetros proporcionados
        if params:
            for key, value in params.items():
                default_params[key] = value
        
        # Si se proporciona un modelo existente, aplicar optimizaciones de forma incremental
        if model is not None:
            logger.info("Aplicando optimizaciones a modelo existente...")
            # TODO: Implementar optimización incremental de modelo existente
            raise NotImplementedError("La optimización de modelos existentes no está implementada aún")
        
        # Si no hay modelo, crear uno nuevo completamente optimizado
        return self.create_optimized_model(
            df=df,
            cve_df=cve_df,
            use_optimal_regressors=default_params['use_optimal_regressors'],
            use_bayesian_optimization=default_params['use_bayesian_optimization'],
            use_interval_calibration=default_params['use_interval_calibration'],
            correlation_threshold=default_params['correlation_threshold'],
            optimization_trials=default_params['optimization_trials']
        )
    
    def predict_with_optimized_model(self, 
                                   model: Prophet, 
                                   periods: int = 30,
                                   include_history: bool = True,
                                   df_features: Optional[pd.DataFrame] = None,
                                   apply_inverse_transform: bool = True) -> pd.DataFrame:
        """
        Genera predicciones con un modelo optimizado, aplicando las transformaciones
        y calibraciones necesarias.
        
        Args:
            model: Modelo Prophet optimizado
            periods: Número de períodos a predecir
            include_history: Si incluir datos históricos
            df_features: DataFrame con características (si ya existe)
            apply_inverse_transform: Si aplicar transformación inversa
            
        Returns:
            DataFrame con predicciones
        """
        # 1. Crear dataframe futuro
        future = model.make_future_dataframe(periods=periods, include_history=include_history)
        
        # 2. Añadir regresores al dataframe futuro si es necesario
        if self.selected_features and df_features is not None:
            for regressor in self.selected_features:
                if regressor in df_features.columns:
                    # Para datos históricos, copiar valores
                    future_with_history = pd.merge(
                        future[['ds']], 
                        df_features[['ds', regressor]],
                        on='ds', 
                        how='left'
                    )
                    
                    # Para fechas futuras, usar estrategias apropiadas
                    if future_with_history[regressor].isna().any():
                        last_values = df_features[regressor].tail(28).values  # Últimos 28 días
                        
                        # Identificar valores faltantes
                        missing_mask = future_with_history[regressor].isna()
                        missing_count = missing_mask.sum()
                        
                        if 'day_of_week' in regressor or 'is_weekend' in regressor:
                            # Calcular valores de calendario para fechas futuras
                            missing_dates = future_with_history.loc[missing_mask, 'ds']
                            if 'day_of_week' in regressor:
                                future_with_history.loc[missing_mask, regressor] = missing_dates.dt.dayofweek
                            elif 'is_weekend' in regressor:
                                future_with_history.loc[missing_mask, regressor] = missing_dates.dt.dayofweek.isin([5, 6]).astype(int)
                        else:
                            # Para otros regresores, usar ciclo de últimos valores
                            future_with_history.loc[missing_mask, regressor] = np.tile(
                                last_values,
                                int(np.ceil(missing_count / len(last_values)))
                            )[:missing_count]
                    
                    # Copiar al dataframe futuro
                    future[regressor] = future_with_history[regressor].values
        
        # 3. Generar predicción
        forecast = model.predict(future)
        
        # 4. Aplicar calibración de intervalos si está disponible
        if hasattr(self.interval_calibrator, 'lower_adjustment') and self.interval_calibrator.lower_adjustment is not None:
            forecast = self.apply_calibration_to_forecast(forecast)
        
        # 5. Aplicar transformación inversa si está habilitada
        if apply_inverse_transform and self.reverse_transform_func is not None:
            forecast = self.feature_engineer.reverse_transform_forecast(
                forecast, 
                self.reverse_transform_func
            )
        
        return forecast


# Función para crear un optimizador desde un diccionario de configuración
def create_optimizer_from_config(config: Dict) -> RansomwareOptimizer:
    """
    Crea un optimizador con la configuración especificada.
    
    Args:
        config: Diccionario con parámetros de configuración
        
    Returns:
        Instancia de RansomwareOptimizer
    """
    # Extraer parámetros relevantes
    transform_method = config.get('transform_method', 'log')
    
    # Crear optimizador
    optimizer = RansomwareOptimizer(transform_method=transform_method)
    
    return optimizer


# Función global para aplicar optimizaciones avanzadas
def apply_advanced_optimizations(forecaster, df=None, enable_regressor_selection=True, 
                               enable_bayesian_opt=True, enable_interval_calibration=True):
    """
    Aplica optimizaciones avanzadas al forecaster existente.
    
    Args:
        forecaster: Instancia de RansomwareForecaster
        df: DataFrame con datos (opcional)
        enable_regressor_selection: Si habilitar selección de regresores
        enable_bayesian_opt: Si habilitar optimización bayesiana
        enable_interval_calibration: Si habilitar calibración de intervalos
        
    Returns:
        Forecaster optimizado
    """
    try:
        # Crear optimizador
        optimizer = RansomwareOptimizer()
        
        # Obtener DataFrame
        if df is None:
            if hasattr(forecaster, 'df_prophet') and forecaster.df_prophet is not None:
                df = forecaster.df_prophet
            else:
                logger.error("No hay datos disponibles para optimización")
                return forecaster
        
        # Obtener datos de CVE si están disponibles
        cve_df = None
        if hasattr(forecaster, 'df_cve') and forecaster.df_cve is not None:
            cve_df = forecaster.df_cve
        
        # Aplicar optimizaciones
        params = {
            'use_optimal_regressors': enable_regressor_selection,
            'use_bayesian_optimization': enable_bayesian_opt,
            'use_interval_calibration': enable_interval_calibration
        }
        
        # Crear modelo optimizado
        model, results = optimizer.apply_advanced_optimizations(
            df=df,
            cve_df=cve_df,
            params=params
        )
        
        # Actualizar forecaster con modelo optimizado
        if hasattr(forecaster, 'model'):
            forecaster.model = model
        
        # Guardar información adicional
        if hasattr(forecaster, 'reverse_transform_func'):
            forecaster.reverse_transform_func = optimizer.reverse_transform_func
        
        if hasattr(forecaster, 'interval_calibrator'):
            forecaster.interval_calibrator = optimizer.interval_calibrator
        
        logger.info("Optimizaciones avanzadas aplicadas correctamente")
        return forecaster
        
    except Exception as e:
        logger.error(f"Error al aplicar optimizaciones avanzadas: {str(e)}")
        return forecaster
