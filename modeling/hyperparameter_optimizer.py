"""
Módulo para optimización bayesiana de hiperparámetros de Prophet.

Implementa un optimizador bayesiano que utiliza Optuna para encontrar
la mejor combinación de hiperparámetros para el modelo Prophet.
"""

import pandas as pd
import numpy as np
import logging
import optuna
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from typing import Dict, List, Tuple, Optional, Union, Callable
import time
from joblib import Parallel, delayed

# Configuración de logging
logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """
    Optimizador Bayesiano para encontrar los mejores hiperparámetros
    para un modelo Prophet usando Optuna.
    """
    
    def __init__(self, 
               seasonality_mode: str = 'multiplicative',
               use_holidays: bool = True,
               mcmc_samples: int = 0):
        """
        Inicializa el optimizador
        
        Args:
            seasonality_mode: Modo de estacionalidad ('additive' o 'multiplicative')
            use_holidays: Si se deben incluir holidays en la optimización
            mcmc_samples: Número de muestras MCMC para intervalos bayesianos
        """
        self.seasonality_mode = seasonality_mode
        self.use_holidays = use_holidays
        self.mcmc_samples = mcmc_samples
        self.best_params = None
        self.study = None
        self.opt_history = []
        self.is_optimized = False
    
    def _create_model(self, params: Dict) -> Prophet:
        """
        Crea un modelo Prophet con los parámetros especificados
        
        Args:
            params: Diccionario con parámetros del modelo
            
        Returns:
            Modelo Prophet configurado
        """
        # Parámetros básicos
        model = Prophet(
            changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=params.get('holidays_prior_scale', 10.0),
            seasonality_mode=params.get('seasonality_mode', self.seasonality_mode),
            changepoint_range=params.get('changepoint_range', 0.8),
            interval_width=params.get('interval_width', 0.95),
            mcmc_samples=self.mcmc_samples
        )
        
        # Configurar estacionalidades
        if 'yearly_seasonality' in params:
            if isinstance(params['yearly_seasonality'], bool):
                model.yearly_seasonality = params['yearly_seasonality']
            else:
                model.add_seasonality(
                    name='yearly',
                    period=365.25,
                    fourier_order=int(params['yearly_seasonality'])
                )
        
        if 'weekly_seasonality' in params:
            if isinstance(params['weekly_seasonality'], bool):
                model.weekly_seasonality = params['weekly_seasonality']
            else:
                model.add_seasonality(
                    name='weekly',
                    period=7,
                    fourier_order=int(params['weekly_seasonality'])
                )
                
        if 'quarterly_seasonality' in params and params['quarterly_seasonality']:
            model.add_seasonality(
                name='quarterly',
                period=365.25/4,
                fourier_order=int(params['quarterly_seasonality'])
            )
            
        if 'monthly_seasonality' in params and params['monthly_seasonality']:
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=int(params['monthly_seasonality'])
            )
        
        if 'daily_seasonality' in params:
            model.daily_seasonality = params['daily_seasonality']
        
        return model
    
    def _objective(self, 
                  trial: optuna.Trial, 
                  df: pd.DataFrame, 
                  regressors: List[str] = None,
                  initial: str = '365 days',
                  period: str = '30 days', 
                  horizon: str = '30 days',
                  metric: str = 'smape') -> float:
        """
        Función objetivo para Optuna
        
        Args:
            trial: Objeto trial de Optuna
            df: DataFrame con datos
            regressors: Lista de nombres de regresores
            initial: Período inicial para validación cruzada
            period: Período entre cortes de validación cruzada
            horizon: Horizonte de predicción
            metric: Métrica a optimizar
            
        Returns:
            Valor de la métrica (a minimizar)
        """
        # Definir espacio de búsqueda de hiperparámetros
        params = {
            # Escalas de prior (sensibilidad a cambios)
            'changepoint_prior_scale': trial.suggest_float('cps', 0.001, 0.5, log=True),
            'seasonality_prior_scale': trial.suggest_float('sps', 0.1, 20, log=True),
            
            # Modo de estacionalidad
            'seasonality_mode': trial.suggest_categorical('mode', ['additive', 'multiplicative']),
            
            # Rango de changepoints
            'changepoint_range': trial.suggest_float('cpr', 0.7, 0.95),
            
            # Fourier orders para estacionalidades
            'yearly_seasonality': trial.suggest_int('yearly', 5, 20),
            'weekly_seasonality': trial.suggest_int('weekly', 3, 10),
            
            # Estacionalidades adicionales
            'quarterly_seasonality': trial.suggest_int('quarterly', 0, 10),
            'monthly_seasonality': trial.suggest_int('monthly', 0, 8),
            
            # Intervalos de confianza
            'interval_width': trial.suggest_float('interval', 0.8, 0.95)
        }
        
        # Agregar holidays si están habilitados
        if self.use_holidays:
            params['holidays_prior_scale'] = trial.suggest_float('hps', 0.1, 20, log=True)
        
        # Registrar combinación actual
        logger.info(f"Evaluando: {params}")
        
        # Crear y entrenar modelo
        try:
            model = self._create_model(params)
            
            # Añadir regresores
            if regressors:
                for regressor in regressors:
                    model.add_regressor(regressor)
            
            # Entrenar modelo
            model.fit(df)
            
            # Validación cruzada
            df_cv = cross_validation(
                model, initial=initial, period=period, horizon=horizon
            )
            
            # Calcular métricas
            if len(df_cv) == 0:
                logger.warning("Validación cruzada no produjo resultados, posible error en parámetros")
                return float('inf')
                
            df_metrics = performance_metrics(df_cv)
            
            # Obtener la métrica solicitada
            metric_value = df_metrics.loc[0, metric]
            
            # Guardar en historial
            self.opt_history.append({
                'params': params.copy(),
                'metric': metric_value
            })
            
            return metric_value
            
        except Exception as e:
            logger.error(f"Error en objective: {str(e)}")
            return float('inf')
            
    def optimize_hyperparameters(self, 
                               df: pd.DataFrame, 
                               regressors: List[str] = None,
                               target_metric: str = 'smape',
                               n_trials: int = 25,
                               timeout: int = 600,
                               initial: str = '365 days',
                               period: str = '30 days',
                               horizon: str = '30 days',
                               study_name: str = None) -> Dict:
        """
        Optimiza hiperparámetros usando Optuna
        
        Args:
            df: DataFrame con datos
            regressors: Lista de nombres de regresores
            target_metric: Métrica a optimizar
            n_trials: Número de combinaciones a probar
            timeout: Tiempo máximo en segundos
            initial: Período inicial para validación cruzada
            period: Período entre cortes de validación cruzada
            horizon: Horizonte de predicción
            study_name: Nombre del estudio
            
        Returns:
            Diccionario con mejores parámetros y métricas
        """
        if study_name is None:
            study_name = f"prophet_opt_{time.strftime('%Y%m%d_%H%M%S')}"
            
        logger.info(f"Iniciando optimización bayesiana: {study_name}")
        logger.info(f"Target: {target_metric}, Trials: {n_trials}, Timeout: {timeout}s")
        
        # Limpiar historial
        self.opt_history = []
        
        # Crear estudio de Optuna
        self.study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Iniciar optimización
        start_time = time.time()
        
        try:
            self.study.optimize(
                lambda trial: self._objective(
                    trial, df, regressors, initial, period, horizon, target_metric
                ),
                n_trials=n_trials,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            # Obtener mejores parámetros
            self.best_params = self.study.best_params
            
            # Guardar como objeto de clase
            best_value = self.study.best_value
            
            # Crear y entrenar modelo con los mejores parámetros
            model = self._create_model(self.best_params)
            
            # Añadir regresores
            if regressors:
                for regressor in regressors:
                    model.add_regressor(regressor)
            
            # Entrenar modelo
            model.fit(df)
            
            # Validación cruzada final para todas las métricas
            df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
            df_metrics = performance_metrics(df_cv)
            
            logger.info(f"Optimización completada en {duration:.1f}s")
            logger.info(f"Mejor {target_metric}: {best_value:.4f}")
            logger.info(f"Mejores parámetros: {self.best_params}")
            
            self.is_optimized = True
            
            # Devolver resultados
            return {
                'best_params': self.best_params,
                'best_model': model,
                'metrics': df_metrics.to_dict('records')[0],
                'cross_validation': df_cv,
                'n_trials': len(self.study.trials),
                'duration': duration,
                'study': self.study
            }
            
        except Exception as e:
            logger.error(f"Error en optimize_hyperparameters: {str(e)}")
            return None
            
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Obtiene el historial de optimización
        
        Returns:
            DataFrame con historial de optimización
        """
        if not self.opt_history:
            return pd.DataFrame()
            
        # Extraer información
        history = []
        
        for i, item in enumerate(self.opt_history):
            entry = {'trial': i}
            entry.update(item['params'])
            entry['metric'] = item['metric']
            history.append(entry)
            
        return pd.DataFrame(history)
        
    def create_best_model(self, df: pd.DataFrame, regressors: List[str] = None) -> Prophet:
        """
        Crea un modelo con los mejores parámetros y lo entrena
        
        Args:
            df: DataFrame con datos
            regressors: Lista de nombres de regresores
            
        Returns:
            Modelo Prophet entrenado
        """
        if not self.is_optimized:
            raise ValueError("El optimizador debe ser ejecutado primero con optimize_hyperparameters")
            
        # Crear modelo
        model = self._create_model(self.best_params)
        
        # Añadir regresores
        if regressors:
            for regressor in regressors:
                model.add_regressor(regressor)
        
        # Entrenar modelo
        model.fit(df)
        
        return model


@st.cache_resource
def optimize_prophet_hyperparameters(_self, 
                                   df: pd.DataFrame, 
                                   regressors: List[str] = None,
                                   target_metric: str = 'smape',
                                   n_trials: int = 25,
                                   timeout: int = 600,
                                   seasonality_mode: str = 'multiplicative',
                                   initial: str = '365 days') -> Dict:
    """
    Función cacheada para optimizar hiperparámetros de Prophet
    
    Args:
        _self: Parámetro para compatibilidad con Streamlit (no usado)
        df: DataFrame con datos
        regressors: Lista de nombres de regresores
        target_metric: Métrica a optimizar
        n_trials: Número de combinaciones a probar
        timeout: Tiempo máximo en segundos
        seasonality_mode: Modo de estacionalidad predeterminado
        initial: Período inicial para validación cruzada
        
    Returns:
        Diccionario con resultados de la optimización
    """
    try:
        # Crear optimizador
        optimizer = BayesianOptimizer(seasonality_mode=seasonality_mode)
        
        # Optimizar hiperparámetros
        results = optimizer.optimize_hyperparameters(
            df=df,
            regressors=regressors,
            target_metric=target_metric,
            n_trials=n_trials,
            timeout=timeout,
            initial=initial
        )
        
        return results
    except Exception as e:
        logger.error(f"Error en optimize_prophet_hyperparameters: {str(e)}")
        return None
