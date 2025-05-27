"""
Módulo para calibración de intervalos de predicción.

Este módulo proporciona funcionalidad para calibrar los intervalos de
predicción generados por modelos Prophet, garantizando que la cobertura real
se aproxime a la cobertura objetivo.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.model_selection import TimeSeriesSplit
import streamlit as st
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta

class IntervalCalibrator:
    """
    Calibra los intervalos de predicción para garantizar una cobertura precisa
    y adaptada a las características de los datos.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calibration_factor = 1.0
        self.coverage_target = 0.9  # 90% por defecto
        self.is_calibrated = False
        self.calibration_history = []
        
    def calibrate_intervals(self, 
                          df_train: pd.DataFrame, 
                          df_test: pd.DataFrame, 
                          forecast: pd.DataFrame, 
                          target_coverage: float = 0.9,
                          method: str = 'adaptive') -> Tuple[pd.DataFrame, float]:
        """
        Calibra los intervalos de predicción para lograr la cobertura objetivo.
        
        Args:
            df_train: DataFrame con datos de entrenamiento
            df_test: DataFrame con datos de prueba (contiene valores reales)
            forecast: DataFrame con predicciones y sus intervalos
            target_coverage: Cobertura objetivo (fracción)
            method: Método de calibración ('simple', 'adaptive', 'conformal')
            
        Returns:
            DataFrame con intervalos calibrados y factor de calibración
        """
        # Verificar que tenemos columnas necesarias
        required_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        if not all(col in forecast.columns for col in required_cols):
            self.logger.error(f"Faltan columnas requeridas en forecast: {[col for col in required_cols if col not in forecast.columns]}")
            return forecast, 1.0
            
        if 'ds' not in df_test.columns or 'y' not in df_test.columns:
            self.logger.error("df_test debe contener columnas 'ds' y 'y'")
            return forecast, 1.0
            
        # Registrar el objetivo de calibración
        self.coverage_target = target_coverage
        self.logger.info(f"Calibrando intervalos para cobertura objetivo: {target_coverage:.2%}")
            
        # Elegir método de calibración
        if method == 'simple':
            calibrated, factor = self._simple_scaling_calibration(df_test, forecast, target_coverage)
        elif method == 'conformal':
            calibrated, factor = self._conformal_prediction_calibration(df_train, df_test, forecast, target_coverage)
        else:  # method == 'adaptive'
            calibrated, factor = self._adaptive_calibration(df_train, df_test, forecast, target_coverage)
            
        # Registrar resultados
        self.calibration_factor = factor
        self.is_calibrated = True
        
        # Añadir al historial de calibración
        self.calibration_history.append({
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'method': method,
            'target_coverage': target_coverage,
            'factor': factor,
            'data_points': len(df_test)
        })
        
        self.logger.info(f"Calibración completada. Factor: {factor:.3f}")
        return calibrated, factor
        
    def _simple_scaling_calibration(self, 
                                  df_test: pd.DataFrame, 
                                  forecast: pd.DataFrame, 
                                  target_coverage: float) -> Tuple[pd.DataFrame, float]:
        """
        Calibra intervalos usando un factor de escala simple.
        
        Args:
            df_test: DataFrame con datos de prueba
            forecast: DataFrame con predicciones e intervalos
            target_coverage: Cobertura objetivo
            
        Returns:
            DataFrame con intervalos calibrados y factor de calibración
        """
        # Fusionar predicciones con valores reales
        merged = pd.merge(df_test[['ds', 'y']], forecast, on='ds', how='inner')
        
        if len(merged) == 0:
            self.logger.warning("No hay fechas comunes entre df_test y forecast")
            return forecast, 1.0
            
        # Función para calcular la cobertura con un factor de escala dado
        def calculate_coverage(scale_factor):
            # Calcular intervalos ajustados
            y_true = merged['y'].values
            y_pred = merged['yhat'].values
            half_width = (merged['yhat_upper'].values - merged['yhat_lower'].values) / 2
            lower_bound = y_pred - half_width * scale_factor
            upper_bound = y_pred + half_width * scale_factor
            
            # Calcular cobertura
            covered = ((y_true >= lower_bound) & (y_true <= upper_bound)).mean()
            
            # Devolver diferencia con cobertura objetivo (para minimizar)
            return abs(covered - target_coverage)
            
        # Encontrar factor óptimo usando optimización
        try:
            result = minimize_scalar(
                calculate_coverage,
                bounds=(0.1, 5.0),
                method='bounded'
            )
            optimal_factor = result.x
            
            # Verificar que el resultado es razonable
            if not (0.1 <= optimal_factor <= 5.0):
                self.logger.warning(f"Factor de calibración fuera de rango ({optimal_factor}), ajustando a valores límite")
                optimal_factor = max(0.1, min(5.0, optimal_factor))
        except Exception as e:
            self.logger.error(f"Error en optimización: {str(e)}")
            optimal_factor = 1.0
            
        # Aplicar calibración
        calibrated = forecast.copy()
        y_pred = calibrated['yhat'].values
        half_width = (calibrated['yhat_upper'].values - calibrated['yhat_lower'].values) / 2
        calibrated['yhat_lower'] = y_pred - half_width * optimal_factor
        calibrated['yhat_upper'] = y_pred + half_width * optimal_factor
        
        return calibrated, optimal_factor
        
    def _conformal_prediction_calibration(self, 
                                        df_train: pd.DataFrame, 
                                        df_test: pd.DataFrame, 
                                        forecast: pd.DataFrame, 
                                        target_coverage: float) -> Tuple[pd.DataFrame, float]:
        """
        Calibra intervalos usando predicción conformal, ajustando basado en errores históricos.
        
        Args:
            df_train: DataFrame con datos de entrenamiento
            df_test: DataFrame con datos de prueba
            forecast: DataFrame con predicciones e intervalos
            target_coverage: Cobertura objetivo
            
        Returns:
            DataFrame con intervalos calibrados y factor equivalente
        """
        # Fusionar datos reales con predicciones
        merged = pd.merge(df_test[['ds', 'y']], forecast, on='ds', how='inner')
        
        if len(merged) == 0:
            self.logger.warning("No hay fechas comunes entre df_test y forecast")
            return forecast, 1.0
            
        try:
            # Calcular errores absolutos
            merged['abs_error'] = np.abs(merged['y'] - merged['yhat'])
            
            # Calcular cuantil correspondiente a la cobertura objetivo
            # Para cobertura 90%, necesitamos cuantil 95% (errores simétricos)
            alpha = 1 - target_coverage
            conformal_quantile = np.quantile(merged['abs_error'].values, 1 - alpha/2)
            
            # Aplicar calibración conformal
            calibrated = forecast.copy()
            calibrated['yhat_lower'] = calibrated['yhat'] - conformal_quantile
            calibrated['yhat_upper'] = calibrated['yhat'] + conformal_quantile
            
            # Calcular factor equivalente (para compatibilidad)
            original_width = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
            new_width = (calibrated['yhat_upper'] - calibrated['yhat_lower']).mean()
            equivalent_factor = new_width / original_width if original_width > 0 else 1.0
            
            return calibrated, equivalent_factor
            
        except Exception as e:
            self.logger.error(f"Error en calibración conformal: {str(e)}")
            return forecast, 1.0
    
    def _adaptive_calibration(self, 
                            df_train: pd.DataFrame, 
                            df_test: pd.DataFrame, 
                            forecast: pd.DataFrame, 
                            target_coverage: float) -> Tuple[pd.DataFrame, float]:
        """
        Calibra intervalos de forma adaptativa según las características de los datos.
        
        Args:
            df_train: DataFrame con datos de entrenamiento
            df_test: DataFrame con datos de prueba
            forecast: DataFrame con predicciones e intervalos
            target_coverage: Cobertura objetivo
            
        Returns:
            DataFrame con intervalos calibrados y factor de calibración
        """
        # Analizar características de los datos para decidir método óptimo
        merged = pd.merge(df_test[['ds', 'y']], forecast, on='ds', how='inner')
        
        if len(merged) == 0:
            self.logger.warning("No hay fechas comunes entre df_test y forecast")
            return forecast, 1.0
            
        try:
            # Detectar características clave de los datos
            has_zeros = (merged['y'] == 0).any()
            volatility = merged['y'].std() / merged['y'].mean() if merged['y'].mean() > 0 else float('inf')
            is_high_volatility = volatility > 1.0
            error_skewness = np.abs(merged['y'] - merged['yhat']).skew()
            has_skewed_errors = abs(error_skewness) > 1.0
            
            self.logger.info(f"Características de datos: zeros={has_zeros}, volatility={volatility:.2f}, error_skew={error_skewness:.2f}")
            
            # Seleccionar método según características
            if has_zeros and is_high_volatility:
                # Datos con ceros y alta volatilidad: usar intervalos asimétricos
                return self._asymmetric_calibration(merged, forecast, target_coverage)
            elif has_skewed_errors:
                # Errores sesgados: usar predicción conformal
                return self._conformal_prediction_calibration(df_train, df_test, forecast, target_coverage)
            else:
                # Caso estándar: usar calibración simple
                return self._simple_scaling_calibration(df_test, forecast, target_coverage)
                
        except Exception as e:
            self.logger.error(f"Error en calibración adaptativa: {str(e)}")
            # Fallback a método simple
            return self._simple_scaling_calibration(df_test, forecast, target_coverage)
    
    def _asymmetric_calibration(self, 
                              merged: pd.DataFrame, 
                              forecast: pd.DataFrame, 
                              target_coverage: float) -> Tuple[pd.DataFrame, float]:
        """
        Calibra intervalos de forma asimétrica para datos con ceros o alta volatilidad.
        
        Args:
            merged: DataFrame con datos reales y predicciones
            forecast: DataFrame con predicciones e intervalos originales
            target_coverage: Cobertura objetivo
            
        Returns:
            DataFrame con intervalos calibrados y factor equivalente
        """
        try:
            # Calcular errores
            errors = merged['y'] - merged['yhat']
            
            # Calcular cuantiles para límites inferior y superior
            alpha = 1 - target_coverage
            lower_quantile = np.quantile(errors, alpha/2)
            upper_quantile = np.quantile(errors, 1 - alpha/2)
            
            # Aplicar calibración asimétrica
            calibrated = forecast.copy()
            calibrated['yhat_lower'] = calibrated['yhat'] + lower_quantile
            calibrated['yhat_upper'] = calibrated['yhat'] + upper_quantile
            
            # Asegurar que yhat_lower no sea negativo para datos que no pueden ser negativos
            if (merged['y'] >= 0).all():
                calibrated['yhat_lower'] = np.maximum(0, calibrated['yhat_lower'])
                
            # Calcular factor equivalente (promedio de factores superior e inferior)
            original_width = forecast['yhat_upper'] - forecast['yhat_lower']
            new_width = calibrated['yhat_upper'] - calibrated['yhat_lower']
            equivalent_factor = (new_width / original_width).mean() if original_width.mean() > 0 else 1.0
            
            return calibrated, equivalent_factor
            
        except Exception as e:
            self.logger.error(f"Error en calibración asimétrica: {str(e)}")
            return forecast, 1.0
    
    def apply_calibration(self, forecast: pd.DataFrame, factor: Optional[float] = None) -> pd.DataFrame:
        """
        Aplica un factor de calibración a una predicción.
        
        Args:
            forecast: DataFrame con predicciones e intervalos
            factor: Factor de calibración (si es None, usa el último calculado)
            
        Returns:
            DataFrame con intervalos calibrados
        """
        if factor is None:
            factor = self.calibration_factor
            
        if not self.is_calibrated and factor == 1.0:
            self.logger.warning("Aplicando calibración sin haber calibrado previamente")
            
        # Aplicar calibración
        calibrated = forecast.copy()
        y_pred = calibrated['yhat'].values
        half_width = (calibrated['yhat_upper'].values - calibrated['yhat_lower'].values) / 2
        calibrated['yhat_lower'] = y_pred - half_width * factor
        calibrated['yhat_upper'] = y_pred + half_width * factor
        
        return calibrated
    
    def get_calibration_info(self) -> Dict:
        """
        Obtiene información sobre la calibración actual.
        
        Returns:
            Diccionario con información de calibración
        """
        return {
            'is_calibrated': self.is_calibrated,
            'calibration_factor': self.calibration_factor,
            'target_coverage': self.coverage_target,
            'history': self.calibration_history
        }

@st.cache_data
def calibrate_prophet_intervals(_self, actual_values: pd.Series, predicted_values: pd.Series, 
                             target_coverage: float = 0.95) -> Dict:
    """
    Función cacheada para calibrar intervalos de predicción.
    
    Args:
        _self: Parámetro para compatibilidad con Streamlit (no usado)
        actual_values: Valores reales observados
        predicted_values: Valores predichos por el modelo
        target_coverage: Nivel de cobertura objetivo (0-1)
        
    Returns:
        Diccionario con información de calibración
    """
    try:
        # Crear y ajustar calibrador
        calibrator = IntervalCalibrator()
        df_train = pd.DataFrame({'ds': actual_values.index, 'y': actual_values.values})
        df_test = pd.DataFrame({'ds': actual_values.index, 'y': actual_values.values})
        forecast = pd.DataFrame({'ds': actual_values.index, 'yhat': predicted_values.values, 
                                'yhat_lower': predicted_values.values, 'yhat_upper': predicted_values.values})
        calibrated, factor = calibrator.calibrate_intervals(df_train, df_test, forecast, target_coverage)
        
        # Devolver información de calibración
        return calibrator.get_calibration_info()
    except Exception as e:
        logging.error(f"Error al calibrar intervalos: {str(e)}")
        return None
