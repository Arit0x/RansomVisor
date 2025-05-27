import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from ..models.prophet_model import ProphetForecaster
from .metrics import calculate_metrics, calculate_smape, calculate_anomaly_score, detect_attack_pattern_change, analyze_ransomware_seasonality

class ModelEvaluator:
    """
    Componente modular para la evaluación de modelos de predicción de ransomware
    """
    
    def __init__(self):
        """
        Inicializa el evaluador de modelos
        """
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(self, model: ProphetForecaster, test_df: pd.DataFrame,
                     forecast_periods: int = 30, cv_mode: bool = True, 
                     num_folds: int = 3, calibrate_intervals: bool = True) -> Dict[str, float]:
        """
        Evalúa el rendimiento de un modelo de forecasting
        
        Args:
            model: Modelo ProphetForecaster entrenado
            test_df: DataFrame con datos de prueba
            forecast_periods: Número de períodos a predecir en cada fold
            cv_mode: Si usar cross-validation temporal
            num_folds: Número de folds para cross-validation
            calibrate_intervals: Si calibrar los intervalos de predicción
            
        Returns:
            Diccionario con métricas de evaluación
        """
        self.logger.info(f"Evaluando modelo. CV: {cv_mode}, Folds: {num_folds}")
        
        if not cv_mode:
            # Evaluación simple con conjunto de prueba
            forecast = model.predict(periods=len(test_df), 
                                   include_history=False,
                                   calibrate_intervals=calibrate_intervals)
            
            # Unir predicciones con valores reales
            merged = pd.merge(
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                test_df[['ds', 'y']], 
                on='ds', how='inner'
            )
            
            if len(merged) == 0:
                self.logger.error("No hay datos comunes entre predicción y prueba")
                return {"error": "No hay datos comunes entre predicción y prueba"}
            
            # Calcular métricas usando la función centralizada
            metrics = calculate_metrics(
                merged['y'].values, 
                merged['yhat'].values,
                merged['yhat_lower'].values,
                merged['yhat_upper'].values
            )
            
            # Añadir número de puntos evaluados
            metrics['N'] = len(merged)
            
            return metrics
        else:
            # Cross-validation temporal
            metrics_list = []
            
            # Crear ventanas de entrenamiento/prueba
            df_full = pd.concat([model.train_data, test_df]).sort_values('ds')
            df_full = df_full.reset_index(drop=True)
            
            # Calcular tamaño de cada fold
            fold_size = len(df_full) // (num_folds + 1)
            if fold_size < forecast_periods:
                fold_size = forecast_periods
            
            for fold in range(num_folds):
                # Definir índices de corte
                cutoff_idx = len(df_full) - (fold + 1) * fold_size
                if cutoff_idx <= 0:
                    self.logger.warning(f"No hay suficientes datos para el fold {fold+1}")
                    continue
                
                # Dividir datos
                df_train = df_full.iloc[:cutoff_idx].copy()
                df_test = df_full.iloc[cutoff_idx:cutoff_idx + forecast_periods].copy()
                
                if len(df_train) < 10:
                    self.logger.warning(f"Fold {fold+1}: Datos de entrenamiento insuficientes ({len(df_train)})")
                    continue
                    
                if len(df_test) == 0:
                    self.logger.warning(f"Fold {fold+1}: Sin datos de prueba")
                    continue
                
                # Entrenar modelo en este fold
                fold_model = ProphetForecaster(
                    config=model.config,
                    regressors=model.regressors
                )
                
                try:
                    fold_model.train(df_train, use_regressor=model.regressors is not None)
                    
                    # Predecir
                    forecast = fold_model.predict(
                        periods=len(df_test), 
                        include_history=False,
                        calibrate_intervals=calibrate_intervals
                    )
                    
                    # Unir con datos reales
                    merged = pd.merge(
                        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                        df_test[['ds', 'y']], 
                        on='ds', how='inner'
                    )
                    
                    if len(merged) == 0:
                        self.logger.warning(f"Fold {fold+1}: Sin coincidencias entre predicción y prueba")
                        continue
                    
                    # Calcular métricas para este fold usando la función centralizada
                    fold_metrics = calculate_metrics(
                        merged['y'].values, 
                        merged['yhat'].values,
                        merged['yhat_lower'].values,
                        merged['yhat_upper'].values
                    )
                    
                    # Añadir número de puntos e información de fold
                    fold_metrics['N'] = len(merged)
                    fold_metrics['FOLD'] = fold + 1
                    
                    metrics_list.append(fold_metrics)
                    
                except Exception as e:
                    self.logger.error(f"Error en fold {fold+1}: {str(e)}")
                    continue
            
            # Promediar métricas de todos los folds
            if not metrics_list:
                self.logger.error("No se pudo evaluar en ningún fold")
                return {"error": "No se pudo evaluar en ningún fold"}
                
            avg_metrics = {}
            all_keys = set().union(*metrics_list)
            
            # Excluir claves no numéricas o de identificación
            exclude_keys = {'FOLD', 'error'}
            metric_keys = all_keys - exclude_keys
            
            # Promediar métricas numéricas
            for key in metric_keys:
                values = [m[key] for m in metrics_list if key in m and not pd.isna(m[key])]
                if values:
                    avg_metrics[key] = sum(values) / len(values)
                else:
                    avg_metrics[key] = np.nan
            
            # Añadir total de puntos evaluados
            avg_metrics['N_TOTAL'] = sum(m.get('N', 0) for m in metrics_list)
            avg_metrics['N_FOLDS'] = len(metrics_list)
            
            return avg_metrics
    
    def analyze_residuals(self, 
                        actual: Union[pd.Series, np.ndarray], 
                        predicted: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Analiza los residuos de las predicciones
        
        Args:
            actual: Valores reales
            predicted: Valores predichos
            
        Returns:
            Dict con estadísticas de residuos
        """
        # Convertir a arrays numpy si son Series
        if isinstance(actual, pd.Series):
            actual = actual.values
        if isinstance(predicted, pd.Series):
            predicted = predicted.values
            
        # Calcular residuos
        residuals = actual - predicted
        
        # Estadísticas de residuos
        stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'median': np.median(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'normality': self._check_normality(residuals)
        }
        
        return {
            'residuals': residuals,
            'stats': stats
        }
        
    def _check_normality(self, residuals: np.ndarray) -> Dict:
        """
        Comprueba si los residuos siguen una distribución normal
        
        Args:
            residuals: Array de residuos
            
        Returns:
            Dict con resultados de test de normalidad
        """
        # Implementación simple basada en asimetría y curtosis
        skew = self._calculate_skewness(residuals)
        kurtosis = self._calculate_kurtosis(residuals)
        
        # Una distribución normal tiene asimetría 0 y curtosis 3
        # Consideramos "normal" si están en rangos aceptables
        is_normal = (abs(skew) < 0.5) and (abs(kurtosis - 3) < 1)
        
        return {
            'skewness': skew,
            'kurtosis': kurtosis,
            'is_normal': is_normal
        }
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calcula la asimetría de una distribución"""
        n = len(data)
        if n < 3:
            return 0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Usar n-1 para muestra
        if std == 0:
            return 0
            
        skew = np.sum((data - mean) ** 3) / ((n - 1) * std ** 3)
        return skew
        
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calcula la curtosis de una distribución"""
        n = len(data)
        if n < 4:
            return 3  # Valor para distribución normal
            
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Usar n-1 para muestra
        if std == 0:
            return 3
            
        kurt = np.sum((data - mean) ** 4) / ((n - 1) * std ** 4)
        return kurt
    
    def evaluate_forecast(self, 
                       actual_df: pd.DataFrame,
                       forecast_df: pd.DataFrame,
                       use_log_transform: bool = False) -> Dict:
        """
        Evaluación completa de una predicción
        
        Args:
            actual_df: DataFrame con valores reales
            forecast_df: DataFrame con predicciones
            use_log_transform: Si se utilizó transformación logarítmica
            
        Returns:
            Dict con métricas y estadísticas
        """
        # Unir datos reales y predicciones
        if 'ds' not in actual_df.columns or 'ds' not in forecast_df.columns:
            raise ValueError("Ambos DataFrames deben tener columna 'ds'")
            
        merged = pd.merge(
            actual_df, forecast_df, on='ds', how='inner', suffixes=('', '_pred')
        )
        
        if merged.empty:
            raise ValueError("No hay fechas coincidentes para evaluar")
            
        # Determinar qué columnas usar según transformación
        y_col = 'y_original' if 'y_original' in actual_df.columns and use_log_transform else 'y'
        yhat_col = 'yhat_exp' if 'yhat_exp' in forecast_df.columns and use_log_transform else 'yhat'
        yhat_lower_col = 'yhat_lower_exp' if 'yhat_lower_exp' in forecast_df.columns and use_log_transform else 'yhat_lower'
        yhat_upper_col = 'yhat_upper_exp' if 'yhat_upper_exp' in forecast_df.columns and use_log_transform else 'yhat_upper'
        
        # Verificar que las columnas existen
        required_cols = [y_col, yhat_col, yhat_lower_col, yhat_upper_col]
        missing_cols = [col for col in required_cols if col not in merged.columns]
        
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
        
        # Calcular métricas usando la función centralizada
        metrics = calculate_metrics(
            merged[y_col], 
            merged[yhat_col],
            merged[yhat_lower_col],
            merged[yhat_upper_col]
        )
        
        # Análisis de residuos
        residuals = self.analyze_residuals(merged[y_col], merged[yhat_col])
        
        # Análisis de anomalías en la predicción
        anomaly_scores = calculate_anomaly_score(
            merged[y_col].values, 
            merged[yhat_col].values
        )
        
        # Detección de cambios de patrón si hay suficientes datos
        pattern_change = None
        if len(merged) > 10:
            # Dividir en histórico y predicción (asumiendo los últimos puntos son predicción)
            split_point = len(merged) // 2
            historical = merged[y_col].values[:split_point]
            prediction = merged[yhat_col].values[split_point:]
            
            if len(historical) > 5 and len(prediction) > 5:
                pattern_change = detect_attack_pattern_change(
                    historical, prediction
                )
        
        return {
            'metrics': metrics,
            'residuals': residuals,
            'anomaly_scores': anomaly_scores.tolist(),
            'pattern_change': pattern_change,
            'data_points': len(merged),
            'evaluation_period': {
                'start': merged['ds'].min(),
                'end': merged['ds'].max()
            }
        }

    def visualize_forecast(self, forecast: pd.DataFrame, actual: pd.DataFrame,
                          plot_components: bool = False,
                          title: str = "Predicción de Ransomware") -> Optional[plt.Figure]:
        """
        Visualiza las predicciones y componentes
        
        Args:
            forecast: DataFrame con predicciones
            actual: DataFrame con valores reales
            plot_components: Si visualizar los componentes (tendencia, estacionalidad)
            title: Título del gráfico
            
        Returns:
            Figure de matplotlib (opcional)
        """
        try:
            # Combinar predicciones con valores reales
            if actual is not None and len(actual) > 0:
                df_plot = pd.merge(
                    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                    actual[['ds', 'y']],
                    on='ds', how='outer'
                )
                has_actual = True
            else:
                df_plot = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                df_plot['y'] = None
                has_actual = False
                
            # Crear figura
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Configurar estilo
            sns.set_style('whitegrid')
            
            # Ordenar por fecha
            df_plot = df_plot.sort_values('ds')
            
            # Separar histórico y futuro
            if has_actual:
                history = df_plot[~df_plot['y'].isna()]
                future = df_plot[df_plot['y'].isna()]
            else:
                # Sin datos reales, todo es "futuro"
                history = pd.DataFrame()
                future = df_plot
                
            # Trazar datos históricos si existen
            if not history.empty:
                ax.plot(history['ds'], history['y'], 'ko', 
                        markersize=4, label='Actual')
                
            # Trazar la predicción
            ax.plot(df_plot['ds'], df_plot['yhat'], 'b-', 
                    linewidth=2, label='Predicción')
            
            # Trazar intervalos de confianza
            ax.fill_between(df_plot['ds'], df_plot['yhat_lower'], 
                           df_plot['yhat_upper'], color='b', alpha=0.2,
                           label='Intervalo 90%')
                
            # Añadir línea vertical para separar histórico y futuro
            if not history.empty and not future.empty:
                last_date = history['ds'].max()
                ax.axvline(x=last_date, color='r', linestyle='--', alpha=0.7)
                ax.text(last_date, ax.get_ylim()[1] * 0.9, 'Hoy', 
                        color='r', ha='right', backgroundcolor='white', alpha=0.7)
            
            # Configurar ejes y leyenda
            ax.set_xlabel('Fecha')
            ax.set_ylabel('Víctimas/Incidentes')
            ax.set_title(title)
            
            ax.legend(loc='best')
            
            # Formatear fechas en eje x
            fig.autofmt_xdate()
            
            # Añadir información de métricas si hay datos históricos
            if not history.empty:
                # Calcular métricas solo en el histórico usando la función centralizada
                smape = calculate_smape(history['y'], history['yhat'])
                
                # Calcular cobertura del intervalo
                in_interval = ((history['y'] >= history['yhat_lower']) & 
                             (history['y'] <= history['yhat_upper']))
                coverage = np.mean(in_interval) * 100
                
                # Añadir texto con métricas
                metrics_text = f"SMAPE: {smape*100:.1f}%\nCobertura: {coverage:.1f}%"
                plt.figtext(0.15, 0.15, metrics_text, 
                          backgroundcolor='white', alpha=0.8)
            
            # Añadir leyenda con información de calibración
            if 'calibration' in forecast.attrs:
                cal_info = forecast.attrs['calibration']
                if 'method' in cal_info and cal_info['method'] != 'none':
                    cal_text = f"Calibración: {cal_info['method']}\n"
                    if 'original_coverage' in cal_info:
                        cal_text += f"Cobertura original: {cal_info['original_coverage']*100:.1f}%\n"
                    plt.figtext(0.15, 0.05, cal_text, 
                              backgroundcolor='white', alpha=0.8)
            
            # Ajustar diseño
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error al visualizar forecast: {str(e)}")
            return None
    
    def plot_model_components(self, model: ProphetForecaster) -> Optional[plt.Figure]:
        """
        Visualiza los componentes del modelo (tendencia, estacionalidad)
        
        Args:
            model: Modelo ProphetForecaster entrenado
            
        Returns:
            Figure de matplotlib (opcional)
        """
        try:
            # Verificar que el modelo existe
            if model is None or model.model is None:
                self.logger.error("No hay modelo para visualizar componentes")
                return None
                
            # Obtener predicción con componentes
            forecast = model.model.predict(model.model.history)
            fig = model.model.plot_components(forecast)
            
            # Analizar la estacionalidad para ransomware
            seasonality_analysis = analyze_ransomware_seasonality(forecast)
            
            # Añadir anotaciones con los resultados del análisis si hay componentes
            if seasonality_analysis and len(seasonality_analysis) > 0:
                plt.figtext(0.01, 0.01, "Análisis de estacionalidad para ransomware:", 
                           fontsize=10, weight='bold')
                
                y_pos = 0.01
                for component, analysis in seasonality_analysis.items():
                    if component != 'error':
                        y_pos += 0.03
                        if component == 'weekly' and 'weekend_preference' in analysis:
                            pref_text = "Preferencia por fin de semana" if analysis['weekend_preference'] else "Preferencia por días laborables"
                            plt.figtext(0.02, y_pos, pref_text, fontsize=8)
                        
                        if component == 'monthly' and 'year_end_preference' in analysis:
                            pref_text = "Mayor actividad a fin de año" if analysis['year_end_preference'] else "Actividad distribuida durante el año"
                            plt.figtext(0.02, y_pos+0.03, pref_text, fontsize=8)
            
            return fig
        except Exception as e:
            self.logger.error(f"Error al visualizar componentes: {str(e)}")
            return None
    
    def plot_forecast_intervals(self, forecast: pd.DataFrame, 
                             actual: pd.DataFrame = None,
                             percentiles: List[int] = [50, 80, 95]) -> Optional[plt.Figure]:
        """
        Visualiza múltiples intervalos de predicción
        
        Args:
            forecast: DataFrame con predicciones
            actual: DataFrame con valores reales (opcional)
            percentiles: Lista de percentiles a visualizar
            
        Returns:
            Figure de matplotlib
        """
        try:
            # Crear figura
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Obtener intervalo estándar (90%)
            df_plot = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            
            # Si hay datos reales, añadirlos
            if actual is not None and len(actual) > 0:
                df_plot = pd.merge(df_plot, actual[['ds', 'y']], 
                                 on='ds', how='outer')
                has_actual = True
                
                # Trazar puntos reales
                mask = ~df_plot['y'].isna()
                ax.plot(df_plot.loc[mask, 'ds'], df_plot.loc[mask, 'y'], 
                       'ko', markersize=4, label='Actual')
            else:
                has_actual = False
            
            # Ordenar por fecha
            df_plot = df_plot.sort_values('ds')
            
            # Trazar línea de predicción
            ax.plot(df_plot['ds'], df_plot['yhat'], 'b-', 
                   linewidth=2, label='Predicción')
            
            # Trazar intervalos estándar (90%)
            ax.fill_between(df_plot['ds'], df_plot['yhat_lower'], 
                           df_plot['yhat_upper'], color='b', alpha=0.2,
                           label='Intervalo 90%')
            
            # Añadir línea vertical para hoy si hay datos históricos
            if has_actual:
                last_actual = df_plot.loc[~df_plot['y'].isna(), 'ds'].max()
                ax.axvline(x=last_actual, color='r', linestyle='--', alpha=0.7)
                ax.text(last_actual, ax.get_ylim()[1] * 0.9, 'Hoy', 
                       color='r', ha='right', backgroundcolor='white', alpha=0.7)
            
            # Configurar ejes y leyenda
            ax.set_xlabel('Fecha')
            ax.set_ylabel('Víctimas/Incidentes')
            ax.set_title('Intervalos de Predicción para Ransomware')
            
            ax.legend(loc='best')
            
            # Formatear fechas en eje x
            fig.autofmt_xdate()
            
            # Añadir información de calibración si está disponible
            if 'calibration' in forecast.attrs:
                cal_info = forecast.attrs['calibration']
                cal_text = "Información de calibración:\n"
                for key, value in cal_info.items():
                    if isinstance(value, float):
                        cal_text += f"{key}: {value:.2f}\n"
                    else:
                        cal_text += f"{key}: {value}\n"
                
                plt.figtext(0.15, 0.05, cal_text, 
                          backgroundcolor='white', alpha=0.8)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error al visualizar intervalos: {str(e)}")
            return None
    
    def evaluate_model_with_chunking(self, model: ProphetForecaster, test_df: pd.DataFrame, 
                                   chunk_size: int = 1000, **kwargs) -> Dict[str, float]:
        """
        Evalúa el modelo usando chunking para manejar conjuntos de datos grandes
        sin consumir demasiada memoria.
        
        Args:
            model: Modelo ProphetForecaster entrenado
            test_df: DataFrame con datos de prueba
            chunk_size: Tamaño de los chunks para procesar por lotes
            **kwargs: Argumentos adicionales para evaluate_model
            
        Returns:
            Diccionario con métricas de evaluación
        """
        # Si los datos son pequeños, usar la evaluación estándar
        if len(test_df) <= chunk_size:
            return self.evaluate_model(model, test_df, **kwargs)
        
        self.logger.info(f"Usando evaluación por chunks (tamaño={chunk_size}) para {len(test_df)} filas")
        
        # Dividir en chunks
        n_chunks = (len(test_df) + chunk_size - 1) // chunk_size  # Redondeo hacia arriba
        chunk_metrics = []
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(test_df))
            
            chunk_df = test_df.iloc[start_idx:end_idx].copy()
            
            # Evaluar este chunk
            self.logger.info(f"Evaluando chunk {i+1}/{n_chunks} ({len(chunk_df)} filas)")
            metrics = self.evaluate_model(model, chunk_df, **kwargs)
            
            if 'error' not in metrics:
                metrics['chunk'] = i + 1
                metrics['chunk_size'] = len(chunk_df)
                chunk_metrics.append(metrics)
        
        # Combinar métricas de todos los chunks
        if not chunk_metrics:
            return {"error": "No se pudo evaluar ningún chunk"}
            
        # Calcular promedio ponderado por número de puntos
        combined_metrics = {}
        weighted_keys = ['mae', 'rmse', 'smape', 'coverage', 'dir_acc']
        
        for key in weighted_keys:
            if all(key in m for m in chunk_metrics):
                weights = [m.get('N', 1) for m in chunk_metrics]
                values = [m[key] * w for m, w in zip(chunk_metrics, weights)]
                combined_metrics[key] = sum(values) / sum(weights)
        
        # Sumar valores totales
        combined_metrics['N_TOTAL'] = sum(m.get('N', 0) for m in chunk_metrics)
        combined_metrics['chunks_evaluated'] = len(chunk_metrics)
        
        return combined_metrics
