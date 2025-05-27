import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st
from prophet import Prophet

class IntervalCalibrator:
    """
    Componente modular para calibrar intervalos de predicción
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.coverage_target = 0.9  # Cobertura objetivo por defecto (90%)
        
    def calibrate_conformal(self, train_df: pd.DataFrame, forecast_df: pd.DataFrame, 
                          target_coverage: float = 0.9, 
                          adjustment_factor: float = 1.5) -> pd.DataFrame:
        """
        Calibra intervalos usando técnicas de predicción conformal
        
        Args:
            train_df: DataFrame con datos de entrenamiento
            forecast_df: DataFrame con predicciones
            target_coverage: Cobertura objetivo (0-1)
            adjustment_factor: Factor de ajuste para la calibración
            
        Returns:
            DataFrame con intervalos calibrados
        """
        self.logger.info(f"Calibrando intervalos para cobertura objetivo={target_coverage*100:.0f}%")
        
        # Verificar que tenemos datos suficientes
        if train_df is None or len(train_df) < 10:
            self.logger.warning("Datos insuficientes para calibración")
            return forecast_df
        
        # Calibración empírica basada en residuos históricos
        try:
            # Paso 1: Validación para obtener errores de predicción históricos
            calibration_df = forecast_df[forecast_df['ds'].isin(train_df['ds'])].copy()
            
            # Unir con datos reales para calcular errores
            calibration_df = pd.merge(
                calibration_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                train_df[['ds', 'y']], 
                on='ds', 
                how='inner'
            )
            
            # Si no hay suficientes datos para calibración, retornar sin cambios
            if len(calibration_df) < 10:
                self.logger.warning(f"Solo {len(calibration_df)} puntos para calibración. Omitiendo.")
                return forecast_df
            
            # Paso 2: Calcular errores de calibración
            calibration_df['error'] = calibration_df['y'] - calibration_df['yhat']
            calibration_df['abs_error'] = np.abs(calibration_df['error'])
            
            # Paso 3: Calcular cuantiles de errores para la cobertura objetivo
            alpha = 1 - target_coverage
            error_quantile = calibration_df['abs_error'].quantile(1 - alpha/2)
            
            # Aplicar un factor de ajuste adicional para corregir sesgos
            error_quantile *= adjustment_factor
            
            # Paso 4: Calcular cobertura actual
            calibration_df['in_range'] = ((calibration_df['y'] >= calibration_df['yhat_lower']) & 
                                       (calibration_df['y'] <= calibration_df['yhat_upper']))
            current_coverage = calibration_df['in_range'].mean()
            
            self.logger.info(f"Cobertura actual: {current_coverage*100:.1f}%, objetivo: {target_coverage*100:.1f}%")
            
            # Paso 5: Ajustar intervalos empíricamente
            if current_coverage < target_coverage:
                # Ampliar intervalos
                width_factor = target_coverage / max(current_coverage, 0.01)
                self.logger.info(f"Ampliando intervalos por factor {width_factor:.2f}")
                
                # Crear copia del forecast para no modificar el original
                calibrated_forecast = forecast_df.copy()
                
                # Ampliar intervalos
                calibrated_forecast['yhat_lower_calibrated'] = calibrated_forecast['yhat'] - (
                    (calibrated_forecast['yhat'] - calibrated_forecast['yhat_lower']) * width_factor)
                calibrated_forecast['yhat_upper_calibrated'] = calibrated_forecast['yhat'] + (
                    (calibrated_forecast['yhat_upper'] - calibrated_forecast['yhat']) * width_factor)
                
                # Reemplazar intervalos originales
                calibrated_forecast['yhat_lower'] = calibrated_forecast['yhat_lower_calibrated']
                calibrated_forecast['yhat_upper'] = calibrated_forecast['yhat_upper_calibrated']
                
                # Eliminar columnas temporales
                calibrated_forecast = calibrated_forecast.drop(['yhat_lower_calibrated', 'yhat_upper_calibrated'], axis=1)
                
            elif current_coverage > target_coverage + 0.1:  # Permitir un margen de 10%
                # Reducir intervalos si son muy conservadores
                width_factor = target_coverage / current_coverage
                self.logger.info(f"Reduciendo intervalos por factor {width_factor:.2f}")
                
                # Crear copia del forecast para no modificar el original
                calibrated_forecast = forecast_df.copy()
                
                # Reducir intervalos
                calibrated_forecast['yhat_lower_calibrated'] = calibrated_forecast['yhat'] - (
                    (calibrated_forecast['yhat'] - calibrated_forecast['yhat_lower']) * width_factor)
                calibrated_forecast['yhat_upper_calibrated'] = calibrated_forecast['yhat'] + (
                    (calibrated_forecast['yhat_upper'] - calibrated_forecast['yhat']) * width_factor)
                
                # Reemplazar intervalos originales
                calibrated_forecast['yhat_lower'] = calibrated_forecast['yhat_lower_calibrated']
                calibrated_forecast['yhat_upper'] = calibrated_forecast['yhat_upper_calibrated']
                
                # Eliminar columnas temporales
                calibrated_forecast = calibrated_forecast.drop(['yhat_lower_calibrated', 'yhat_upper_calibrated'], axis=1)
            else:
                self.logger.info("Cobertura ya es adecuada, manteniendo intervalos originales")
                calibrated_forecast = forecast_df.copy()
                
            # Añadir estadísticas de calibración
            calibrated_forecast.attrs['calibration'] = {
                'target_coverage': target_coverage,
                'original_coverage': current_coverage,
                'adjustment_factor': width_factor if 'width_factor' in locals() else 1.0,
                'error_quantile': error_quantile
            }
            
            return calibrated_forecast
            
        except Exception as e:
            self.logger.error(f"Error en calibración de intervalos: {str(e)}")
            return forecast_df
            
    def empirical_calibration(self, model: Prophet, train_df: pd.DataFrame, 
                           forecast_df: pd.DataFrame, 
                           target_coverage: float = 0.9) -> pd.DataFrame:
        """
        Realiza calibración empírica de intervalos basada en el rendimiento del modelo
        
        Args:
            model: Modelo Prophet entrenado
            train_df: DataFrame con datos de entrenamiento
            forecast_df: DataFrame con predicciones
            target_coverage: Cobertura objetivo (0-1)
            
        Returns:
            DataFrame con intervalos calibrados
        """
        self.logger.info(f"Iniciando calibración empírica para cobertura={target_coverage*100:.0f}%")
        
        try:
            # Paso 1: Realizar validación cruzada rápida
            from prophet.diagnostics import cross_validation
            
            # Ajustar horizonte al 20% de los datos de entrenamiento
            n_train = len(train_df)
            horizon_days = max(min(n_train // 5, 30), 7)  # Entre 7 y 30 días, o 20% de los datos
            
            # Calcular tamaño del conjunto de entrenamiento inicial (70% de los datos)
            initial_days = max(int(n_train * 0.7), 30)
            
            # Realizar validación cruzada
            cv_results = cross_validation(
                model=model, 
                initial=f'{initial_days} days',
                horizon=f'{horizon_days} days',
                period=f'{max(horizon_days // 3, 1)} days'
            )
            
            # Paso 2: Calcular factores de escala para intervalos
            cv_results['in_range'] = ((cv_results['y'] >= cv_results['yhat_lower']) & 
                                   (cv_results['y'] <= cv_results['yhat_upper']))
            
            # Calcular cobertura actual
            current_coverage = cv_results['in_range'].mean()
            
            # Si la cobertura actual está lejos del objetivo, ajustar intervalos
            if abs(current_coverage - target_coverage) > 0.05:  # Si difiere en más del 5%
                # Calcular factor de escala empírico
                scale_factor = np.sqrt(target_coverage / max(current_coverage, 0.01))
                
                # Limitar el factor de escala para evitar cambios extremos
                scale_factor = max(min(scale_factor, 2.0), 0.5)
                
                self.logger.info(f"Aplicando escala {scale_factor:.2f} a intervalos (cobertura actual: {current_coverage*100:.1f}%)")
                
                # Calibrar intervalos
                calibrated_forecast = forecast_df.copy()
                
                # Aplicar factor de escala a los intervalos
                calibrated_forecast['width'] = calibrated_forecast['yhat_upper'] - calibrated_forecast['yhat_lower']
                calibrated_forecast['half_width'] = calibrated_forecast['width'] / 2
                calibrated_forecast['yhat_lower'] = calibrated_forecast['yhat'] - (calibrated_forecast['half_width'] * scale_factor)
                calibrated_forecast['yhat_upper'] = calibrated_forecast['yhat'] + (calibrated_forecast['half_width'] * scale_factor)
                
                # Eliminar columnas temporales
                calibrated_forecast = calibrated_forecast.drop(['width', 'half_width'], axis=1)
                
                # Añadir información de calibración
                calibrated_forecast.attrs['calibration'] = {
                    'method': 'empirical',
                    'target_coverage': target_coverage,
                    'original_coverage': current_coverage,
                    'scale_factor': scale_factor
                }
                
                return calibrated_forecast
                
            else:
                self.logger.info(f"Cobertura actual ({current_coverage*100:.1f}%) cercana al objetivo, manteniendo intervalos")
                forecast_df.attrs['calibration'] = {
                    'method': 'none',
                    'original_coverage': current_coverage,
                }
                return forecast_df
                
        except Exception as e:
            self.logger.error(f"Error en calibración empírica: {str(e)}")
            return forecast_df
    
    def quantile_based_calibration(self, train_df: pd.DataFrame, 
                                forecast_df: pd.DataFrame, 
                                target_coverage: float = 0.9,
                                rolling_window: int = 30) -> pd.DataFrame:
        """
        Calibra intervalos basados en cuantiles de errores recientes
        
        Args:
            train_df: DataFrame con datos de entrenamiento
            forecast_df: DataFrame con predicciones
            target_coverage: Cobertura objetivo (0-1)
            rolling_window: Ventana para calcular errores recientes
            
        Returns:
            DataFrame con intervalos calibrados
        """
        self.logger.info(f"Calibrando intervalos con método de cuantiles (target={target_coverage*100:.0f}%)")
        
        try:
            # Extraer datos históricos
            train_subset = train_df.copy()
            
            # Nos aseguramos que train_subset tenga 'ds' y 'y'
            if 'ds' not in train_subset.columns or 'y' not in train_subset.columns:
                self.logger.error("Los datos de entrenamiento deben tener columnas 'ds' y 'y'")
                return forecast_df
                
            # Ordenar por fecha
            train_subset = train_subset.sort_values('ds')
            
            # Calcular tendencia reciente con rolling mean
            window_size = min(rolling_window, len(train_subset))
            train_subset['y_trend'] = train_subset['y'].rolling(window=window_size, min_periods=1).mean()
            
            # Calcular errores respecto a la tendencia
            train_subset['y_error'] = train_subset['y'] - train_subset['y_trend']
            
            # Calcular error absoluto
            train_subset['y_abs_error'] = np.abs(train_subset['y_error'])
            
            # Calcular cuantil de error absoluto para la cobertura objetivo
            alpha = 1 - target_coverage
            q = train_subset['y_abs_error'].quantile(1 - alpha/2)
            
            # Aplicar a intervalos de predicción
            calibrated = forecast_df.copy()
            
            # Calcular nuevos intervalos
            calibrated['yhat_lower'] = calibrated['yhat'] - q
            calibrated['yhat_upper'] = calibrated['yhat'] + q
            
            # Asegurarse que los intervalos no sean negativos para conteos
            if (train_subset['y'] >= 0).all():
                calibrated['yhat_lower'] = calibrated['yhat_lower'].clip(lower=0)
                
            # Registrar información de calibración
            self.logger.info(f"Ajustados intervalos usando cuantil {q:.2f} para cobertura {target_coverage*100:.0f}%")
            
            return calibrated
            
        except Exception as e:
            self.logger.error(f"Error en calibración por cuantiles: {str(e)}")
            return forecast_df
