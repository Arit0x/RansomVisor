import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

# Configuración básica de logging
logger = logging.getLogger(__name__)

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calcula el Symmetric Mean Absolute Percentage Error, que es robusto a valores cero.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        epsilon: Pequeño valor para evitar división por cero
        
    Returns:
        SMAPE como valor entre 0 y 1 (multiplicar por 100 para porcentaje)
    """
    # Asegurar que son arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Eliminar NaN si existen
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.all(mask):
        logger.warning(f"Se eliminaron {np.sum(~mask)} valores NaN del cálculo de SMAPE")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    # Verificar que hay datos suficientes
    if len(y_true) == 0:
        logger.error("No hay datos válidos para calcular SMAPE")
        return 0.0
    
    # Calcular SMAPE con protección contra división por cero
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape = 2.0 * np.mean(np.abs(y_pred - y_true) / denominator)
    return smape

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     interval_lower: Optional[np.ndarray] = None,
                     interval_upper: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calcula métricas de rendimiento del modelo, centralizada para toda la aplicación
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        interval_lower: Límite inferior del intervalo de predicción (opcional)
        interval_upper: Límite superior del intervalo de predicción (opcional)
        
    Returns:
        Diccionario con métricas de rendimiento (mae, rmse, smape, r2, dir_acc, coverage)
    """
    metrics = {}
    
    # Verificación inicial de argumentos
    if y_true is None or y_pred is None:
        logger.error("Los valores de entrada no pueden ser None")
        return {
            'error': "Los valores de entrada no pueden ser None",
            'mae': np.nan,
            'rmse': np.nan,
            'smape': np.nan,
            'r2': np.nan,
            'dir_acc': np.nan,
            'coverage': 0.0
        }
    
    # Asegurar que los arrays son numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Verificar si hay datos
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.error("Los arrays de entrada están vacíos")
        return {
            'error': "Los arrays de entrada están vacíos",
            'mae': np.nan,
            'rmse': np.nan,
            'smape': np.nan,
            'r2': np.nan,
            'dir_acc': np.nan,
            'coverage': 0.0
        }
    
    # Verificar si las longitudes coinciden
    if len(y_true) != len(y_pred):
        logger.error(f"Las longitudes de los arrays no coinciden: {len(y_true)} vs {len(y_pred)}")
        return {
            'error': f"Las longitudes de los arrays no coinciden: {len(y_true)} vs {len(y_pred)}",
            'mae': np.nan,
            'rmse': np.nan,
            'smape': np.nan,
            'r2': np.nan,
            'dir_acc': np.nan,
            'coverage': 0.0
        }
    
    # Eliminar NaN si existen
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.all(mask):
        logger.warning(f"Se eliminaron {np.sum(~mask)} valores NaN de {len(y_true)} totales")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if interval_lower is not None:
            interval_lower = np.array(interval_lower)[mask]
        if interval_upper is not None:
            interval_upper = np.array(interval_upper)[mask]
    
    # Verificar nuevamente después de eliminar NaN
    if len(y_true) == 0:
        logger.error("No hay datos válidos para evaluar después de eliminar NaN")
        return {
            'error': "No hay datos válidos para evaluar después de eliminar NaN",
            'mae': np.nan,
            'rmse': np.nan,
            'smape': np.nan,
            'r2': np.nan,
            'dir_acc': np.nan,
            'coverage': 0.0
        }
    
    # Calcular métricas básicas
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Error al calcular métricas básicas con sklearn: {str(e)}. Usando cálculo manual.")
        # Cálculo manual como respaldo
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        metrics['r2'] = 0.0  # Valor por defecto si falla
    
    # Calcular SMAPE usando nuestra función centralizada
    try:
        metrics['smape'] = calculate_smape(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Error al calcular SMAPE: {str(e)}. Usando cálculo manual.")
        epsilon = 1e-8
        denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
        metrics['smape'] = 2.0 * np.mean(np.abs(y_pred - y_true) / denominator)
        
    # Precisión de dirección (¿la predicción sigue la tendencia?)
    try:
        if len(y_true) > 1:
            direction_true = np.sign(np.diff(y_true))
            direction_pred = np.sign(np.diff(y_pred))
            # Manejar caso donde direction_true es cero
            valid_directions = direction_true != 0
            if np.sum(valid_directions) > 0:
                metrics['dir_acc'] = np.mean(direction_true[valid_directions] == direction_pred[valid_directions])
            else:
                metrics['dir_acc'] = 0.0
        else:
            metrics['dir_acc'] = 0.0
    except Exception as e:
        logger.warning(f"Error al calcular DIR_ACC: {str(e)}")
        metrics['dir_acc'] = 0.0
    
    # Evaluación de intervalos si están disponibles
    metrics['coverage'] = 0.0  # Valor predeterminado
    
    if interval_lower is not None and interval_upper is not None:
        try:
            # Asegurar que los intervalos sean arrays
            interval_lower = np.array(interval_lower)
            interval_upper = np.array(interval_upper)
            
            # Verificar longitudes
            if len(interval_lower) != len(y_true) or len(interval_upper) != len(y_true):
                logger.warning(f"Las longitudes de los intervalos no coinciden: {len(interval_lower)}, {len(interval_upper)} vs {len(y_true)}")
                # Recortar a la longitud más corta
                min_len = min(len(interval_lower), len(interval_upper), len(y_true))
                y_true = y_true[:min_len]
                interval_lower = interval_lower[:min_len]
                interval_upper = interval_upper[:min_len]
            
            # Verificar que los intervalos son válidos (no NaN y lower <= upper)
            valid_intervals = ~(np.isnan(interval_lower) | np.isnan(interval_upper))
            valid_intervals = valid_intervals & (interval_lower <= interval_upper)
            
            if np.any(valid_intervals):
                # Solo evaluar puntos con intervalos válidos
                valid_y = y_true[valid_intervals]
                valid_lower = interval_lower[valid_intervals]
                valid_upper = interval_upper[valid_intervals]
                
                # Cobertura del intervalo (porcentaje de valores reales dentro del intervalo)
                in_interval = ((valid_y >= valid_lower) & (valid_y <= valid_upper))
                metrics['coverage'] = np.mean(in_interval)
                
                # Añadir métricas adicionales sobre intervalos
                interval_widths = valid_upper - valid_lower
                metrics['interval_width_avg'] = np.mean(interval_widths)
                
                yhat_mean = np.mean(y_pred)
                if abs(yhat_mean) > 1e-8:
                    metrics['interval_width_relative'] = metrics['interval_width_avg'] / yhat_mean
                else:
                    metrics['interval_width_relative'] = np.nan
            else:
                logger.warning("No hay intervalos válidos para calcular cobertura")
                metrics['coverage'] = 0.0
                metrics['interval_width_avg'] = 0.0
                metrics['interval_width_relative'] = np.nan
        except Exception as e:
            logger.warning(f"Error al calcular cobertura: {str(e)}")
            metrics['coverage'] = 0.0
            metrics['interval_width_avg'] = 0.0
            metrics['interval_width_relative'] = np.nan
    
    # Verificar que no haya NaN en las métricas finales
    for key in metrics:
        if isinstance(metrics[key], (int, float)) and np.isnan(metrics[key]):
            logger.warning(f"Métrica {key} es NaN, sustituyendo por 0")
            metrics[key] = 0.0
    
    # Logging para depuración
    logger.info(f"Métricas calculadas en metrics.py: MAE={metrics.get('mae', np.nan):.4f}, SMAPE={metrics.get('smape', np.nan):.4f}, Cobertura={metrics.get('coverage', 0)*100:.1f}%")
                
    return metrics

def calculate_anomaly_score(y_true: np.ndarray, y_pred: np.ndarray, 
                           window_size: int = 7) -> np.ndarray:
    """
    Calcula una puntuación de anomalía basada en la desviación del valor predicho
    respecto al valor real, considerando la volatilidad reciente.
    
    Útil para identificar ataques ransomware que representan desviaciones significativas.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        window_size: Tamaño de la ventana para calcular la volatilidad
        
    Returns:
        Array con puntuación de anomalía para cada punto
    """
    # Asegurar que son arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcular errores
    errors = np.abs(y_true - y_pred)
    
    # Calcular volatilidad histórica usando ventana móvil
    volatility = np.zeros_like(errors)
    
    for i in range(len(errors)):
        start_idx = max(0, i - window_size)
        window_data = y_true[start_idx:i+1]
        if len(window_data) > 1:
            volatility[i] = np.std(window_data)
        else:
            # Si no hay suficientes datos, usar desviación estándar global
            volatility[i] = np.std(y_true) if len(y_true) > 1 else 1.0
    
    # Evitar división por cero
    volatility = np.maximum(volatility, 1e-8)
    
    # Normalizar errores por volatilidad para obtener puntuación de anomalía
    anomaly_scores = errors / volatility
    
    return anomaly_scores

def detect_attack_pattern_change(historical_data: np.ndarray, 
                               prediction_window: np.ndarray,
                               sensitivity: float = 0.8) -> Dict[str, Any]:
    """
    Detecta cambios significativos en patrones de ataque que podrían
    indicar nuevas campañas o variantes de ransomware.
    
    Args:
        historical_data: Datos históricos de ataques
        prediction_window: Ventana de predicción a analizar
        sensitivity: Sensibilidad de la detección (0-1)
        
    Returns:
        Diccionario con resultados de la detección
    """
    # Convertir a arrays numpy
    historical_data = np.array(historical_data)
    prediction_window = np.array(prediction_window)
    
    # Calcular estadísticas de datos históricos
    hist_mean = np.mean(historical_data)
    hist_std = np.std(historical_data)
    hist_median = np.median(historical_data)
    
    # Calcular estadísticas de ventana de predicción
    pred_mean = np.mean(prediction_window)
    pred_std = np.std(prediction_window)
    pred_median = np.median(prediction_window)
    
    # Determinar umbrales basados en sensibilidad
    threshold_factor = 2.0 * (1.0 + sensitivity)
    
    # Comprobar cambios significativos
    mean_change = abs(pred_mean - hist_mean) > (threshold_factor * hist_std)
    var_change = pred_std > (threshold_factor * hist_std)
    
    # Calcular tendencia y aceleración
    if len(prediction_window) > 1:
        pred_diff = np.diff(prediction_window)
        pred_trend = np.mean(pred_diff)
    else:
        pred_trend = 0
        
    if len(historical_data) > 1:
        hist_diff = np.diff(historical_data)
        hist_trend = np.mean(hist_diff)
        trend_change = abs(pred_trend) > (threshold_factor * abs(hist_trend)) if hist_trend != 0 else pred_trend != 0
    else:
        hist_trend = 0
        trend_change = False
    
    # Determinar si hay un cambio de patrón general
    pattern_change = mean_change or var_change or trend_change
    
    return {
        'pattern_change_detected': pattern_change,
        'mean_change': mean_change,
        'variance_change': var_change,
        'trend_change': trend_change,
        'historical_stats': {
            'mean': hist_mean,
            'std': hist_std,
            'median': hist_median,
            'trend': hist_trend
        },
        'prediction_stats': {
            'mean': pred_mean,
            'std': pred_std,
            'median': pred_median,
            'trend': pred_trend
        }
    }

def analyze_ransomware_seasonality(components_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analiza componentes estacionales específicos para ransomware,
    identificando patrones como ataques de fin de semana o de fin de mes.
    
    Args:
        components_df: DataFrame con componentes del modelo (de Prophet)
        
    Returns:
        Diccionario con análisis de estacionalidad relevante para ransomware
    """
    results = {}
    
    try:
        # Analizar estacionalidad semanal (si está disponible)
        if 'weekly' in components_df.columns:
            weekly = components_df['weekly'].values
            weekdays = weekly[:7]  # Asumiendo que los primeros 7 valores son para cada día de la semana
            
            # Identificar días con mayor actividad
            high_activity_days = np.argsort(weekdays)[::-1][:2]  # Top 2 días
            
            # Determinar si hay preferencia por fin de semana
            weekend_effect = np.mean(weekdays[5:7]) - np.mean(weekdays[0:5])
            weekend_preference = weekend_effect > 0
            
            results['weekly'] = {
                'high_activity_days': high_activity_days.tolist(),
                'weekend_effect': weekend_effect,
                'weekend_preference': weekend_preference
            }
        
        # Analizar estacionalidad mensual (si está disponible)
        if 'yearly' in components_df.columns:
            yearly = components_df['yearly'].values
            # Asumiendo que los datos cubren un año (365 días)
            if len(yearly) >= 365:
                # Dividir en meses aproximadamente
                months = [yearly[i:i+30] for i in range(0, 360, 30)]
                month_means = [np.mean(m) for m in months]
                
                # Identificar meses con mayor actividad
                high_activity_months = np.argsort(month_means)[::-1][:3]  # Top 3 meses
                
                # Comprobar patrones de fin de año (Nov-Dic) comunes en ransomware
                year_end_effect = np.mean(month_means[10:12]) - np.mean(month_means[0:10])
                
                results['monthly'] = {
                    'high_activity_months': high_activity_months.tolist(),
                    'year_end_effect': year_end_effect,
                    'year_end_preference': year_end_effect > 0
                }
                
        # Análisis de eventos especiales si está disponible
        if 'holidays' in components_df.columns:
            holidays = components_df['holidays'].values
            if np.any(holidays):
                holiday_impact = np.mean(holidays[holidays != 0])
                results['holidays'] = {
                    'impact': holiday_impact,
                    'significant': abs(holiday_impact) > 0.1  # Umbral arbitrario para significancia
                }
                
    except Exception as e:
        logger.error(f"Error en análisis de estacionalidad: {str(e)}")
        results['error'] = str(e)
    
    return results
