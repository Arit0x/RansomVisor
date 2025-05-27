import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import holidays
import streamlit as st
import os
from datetime import datetime, date
from ..evaluation.metrics import calculate_metrics, calculate_smape

class RansomwareProphetModel:
    """
    Wrapper para Prophet optimizado para predicción de ransomware
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.forecast = None
        self.metrics = None
        self.train_data = None
        self.use_log_transform = True
        self.params = {}
        
    # El decorador @st.cache_resource causa el error "Prophet object can only be fit once"
    # al reutilizar el mismo modelo entre diferentes llamadas
    # Por lo tanto, lo eliminamos para garantizar que siempre se crea un nuevo modelo
    def create_model(_self,
                  changepoint_prior_scale: float = 0.2,
                  seasonality_prior_scale: float = 10.0,
                  holidays_prior_scale: float = 10.0,
                  seasonality_mode: str = 'multiplicative',
                  n_changepoints: int = 60,
                  interval_width: float = 0.8) -> Prophet:
        """
        Crea un modelo Prophet con parámetros optimizados
        
        Args:
            changepoint_prior_scale: Escala del prior para los cambios de tendencia
            seasonality_prior_scale: Escala del prior para componentes estacionales
            holidays_prior_scale: Escala del prior para efectos de días festivos
            seasonality_mode: Modo de estacionalidad ('additive' o 'multiplicative')
            n_changepoints: Número de changepoints potenciales
            interval_width: Ancho del intervalo de predicción
            
        Returns:
            Modelo Prophet configurado
        """
        _self.logger.info(f"Creando modelo Prophet con mode={seasonality_mode}")
        
        # Guardar parámetros para referencia
        _self.params = {
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'holidays_prior_scale': holidays_prior_scale,
            'seasonality_mode': seasonality_mode,
            'n_changepoints': n_changepoints,
            'interval_width': interval_width
        }
        
        # Preparar holidays para España - método moderno
        holiday_df = None
        try:
            # Obtener los holidays de España para los próximos años
            spain_holidays = holidays.ES()
            current_year = datetime.now().year
            
            # Crear una lista de diccionarios con la estructura correcta
            holiday_list = []
            for year in range(current_year - 2, current_year + 5):  # 2 años atrás y 5 adelante
                for holiday_date, holiday_name in spain_holidays[year].items():
                    holiday_list.append({
                        'ds': pd.Timestamp(holiday_date),
                        'holiday': f"{holiday_name}_{year}",  # Nombre único
                        'lower_window': 0,
                        'upper_window': 1
                    })
            
            # Crear DataFrame con las columnas requeridas
            if holiday_list:
                holiday_df = pd.DataFrame(holiday_list)
                _self.logger.info(f"Añadiendo {len(holiday_df)} días festivos españoles al modelo")
            else:
                _self.logger.warning("No se encontraron días festivos para España")
                holiday_df = None
        except Exception as e:
            _self.logger.warning(f"Error al cargar holidays: {str(e)}. Continuando sin holidays.")
            holiday_df = None
        
        # Crear modelo base con holidays ya incluidos (método moderno)
        try:
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                seasonality_mode=seasonality_mode,
                n_changepoints=n_changepoints,
                interval_width=interval_width,
                holidays=holiday_df  # Pasar holidays directamente
            )
            
            # Añadir estacionalidad semanal mejorada
            model.add_seasonality(
                name='weekly',
                period=7,
                fourier_order=12,
                prior_scale=seasonality_prior_scale
            )
            
            # Añadir estacionalidad mensual
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=8,
                prior_scale=seasonality_prior_scale
            )
            
            return model
        except Exception as e:
            _self.logger.error(f"Error al crear modelo Prophet: {str(e)}")
            # Si falla con holidays, intentar sin ellos
            if holiday_df is not None:
                _self.logger.info("Intentando crear modelo sin holidays")
                try:
                    model = Prophet(
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale,
                        holidays_prior_scale=holidays_prior_scale,
                        seasonality_mode=seasonality_mode,
                        n_changepoints=n_changepoints,
                        interval_width=interval_width
                    )
                    return model
                except Exception as e2:
                    _self.logger.error(f"Error al crear modelo sin holidays: {str(e2)}")
                    raise e2
            else:
                raise e

    def add_regressors(self, model: Prophet, regressors: List[str]) -> Prophet:
        """
        Añade regresores externos al modelo
        
        Args:
            model: Modelo Prophet
            regressors: Lista de columnas para usar como regresores
            
        Returns:
            Modelo con regresores añadidos
        """
        for regressor in regressors:
            model.add_regressor(regressor)
            self.logger.info(f"Añadido regresor: {regressor}")
            
        return model
        
    def fit(self, df: pd.DataFrame, regressors: List[str] = None) -> None:
        """
        Entrena el modelo con los datos proporcionados
        
        Args:
            df: DataFrame en formato Prophet (ds, y)
            regressors: Lista de columnas para usar como regresores
        """
        if df is None or df.empty:
            raise ValueError("No hay datos para entrenamiento")
            
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("El DataFrame debe tener columnas 'ds' y 'y'")
            
        self.train_data = df.copy()
        
        # Determinar automáticamente el mejor modo de estacionalidad
        zero_percentage = (df['y'] == 0).mean()
        recommended_mode = 'additive' if zero_percentage > 0.1 else 'multiplicative'
        
        if recommended_mode != self.params.get('seasonality_mode', 'multiplicative'):
            self.logger.info(f"Detectada serie con {zero_percentage:.1%} de ceros. Cambiando automáticamente seasonality_mode a '{recommended_mode}'")
            # Actualizar el modo de estacionalidad
            self.params['seasonality_mode'] = recommended_mode
        
        # Optimizar n_changepoints basado en la longitud de la serie
        n_rows = len(df)
        optimal_changepoints = min(max(int(n_rows * 0.15), 10), 100)
        
        if optimal_changepoints != self.params.get('n_changepoints', 60):
            self.logger.info(f"Ajustando n_changepoints de {self.params.get('n_changepoints', 60)} a {optimal_changepoints} basado en la longitud de la serie ({n_rows} filas)")
            # Actualizar n_changepoints
            self.params['n_changepoints'] = optimal_changepoints
        
        # Verificar si hay parámetros recomendados del preprocesador para series con muchos ceros
        if hasattr(df, 'preprocessor') and hasattr(df.preprocessor, 'recommended_prophet_params'):
            recommended_params = df.preprocessor.recommended_prophet_params
            self.logger.info(f"Usando parámetros recomendados para serie con muchos ceros: {recommended_params}")
            
            # Actualizar parámetros con los recomendados
            for param, value in recommended_params.items():
                self.params[param] = value
                
        # Si la serie tiene más de 50% de ceros, ajustar parámetros automáticamente
        elif zero_percentage > 0.5:
            self.logger.info(f"Serie con alta proporción de ceros ({zero_percentage:.1%}), ajustando parámetros automáticamente")
            
            # Importar detector de outliers para usar sus recomendaciones
            from ..features.outliers import OutlierDetector
            outlier_detector = OutlierDetector()
            recommended_params = outlier_detector.get_optimal_prophet_params_for_zero_inflated(zero_percentage)
            
            # Actualizar parámetros con los recomendados
            for param, value in recommended_params.items():
                self.params[param] = value
                self.logger.info(f"Ajustando {param}: {value}")
        
        # Crear modelo
        self.model = self.create_model(
            changepoint_prior_scale=self.params.get('changepoint_prior_scale', 0.2),
            seasonality_prior_scale=self.params.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=self.params.get('holidays_prior_scale', 10.0),
            seasonality_mode=self.params.get('seasonality_mode', 'multiplicative'),
            n_changepoints=self.params.get('n_changepoints', 60),
            interval_width=self.params.get('interval_width', 0.8)
        )
        
        # Añadir regresores si existen
        if regressors and len(regressors) > 0:
            valid_regressors = [r for r in regressors if r in df.columns]
            self.model = self.add_regressors(self.model, valid_regressors)
            
        # Entrenar modelo
        self.logger.info("Entrenando modelo Prophet")
        self.model.fit(df)
        self.logger.info("Modelo entrenado correctamente")
        
        # Guardar dataframe procesado para seguimiento
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join("modeling", "datos_procesados", f"datos_preprocesados_{timestamp}.csv")
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        df.to_csv(export_path, index=False)
        self.logger.info(f"Datos de entrenamiento guardados en {export_path}")
        
    def predict(self, periods: int = 30, 
              include_history: bool = True,
              future_df: pd.DataFrame = None,
              calibrate_intervals: bool = True) -> pd.DataFrame:
        """
        Genera predicciones con el modelo entrenado
        
        Args:
            periods: Número de periodos a predecir
            include_history: Si incluir el histórico en las predicciones
            future_df: DataFrame con fechas futuras (opcional)
            calibrate_intervals: Si calibrar los intervalos de predicción
            
        Returns:
            DataFrame con predicciones
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado primero con fit()")
            
        # Generar dataframe de fechas futuras
        if future_df is None:
            try:
                # Intentar usar el método estándar (puede fallar con DatetimeArray)
                future_df = self.model.make_future_dataframe(periods=periods,
                                                          include_history=include_history,
                                                          freq='D')
            except Exception as e:
                self.logger.warning(f"Error al usar make_future_dataframe: {str(e)}")
                # Crear manualmente el DataFrame futuro de manera segura
                if hasattr(self, 'train_data') and self.train_data is not None:
                    last_date = self.train_data['ds'].max()
                    
                    if include_history:
                        # Incluir todo el historial + fechas futuras
                        historical_dates = self.train_data['ds'].tolist()  # Convertir a lista para evitar problemas con DatetimeArray
                        future_dates = pd.date_range(
                            start=last_date + pd.DateOffset(days=1),
                            periods=periods, 
                            freq='D'
                        ).tolist()
                        
                        all_dates = sorted(historical_dates + future_dates)
                        future_df = pd.DataFrame({'ds': all_dates})
                    else:
                        # Solo incluir fechas futuras
                        future_dates = pd.date_range(
                            start=last_date + pd.DateOffset(days=1),
                            periods=periods, 
                            freq='D'
                        )
                        future_df = pd.DataFrame({'ds': future_dates})
                else:
                    # Si no hay datos de entrenamiento, usar fechas desde hoy
                    today = pd.Timestamp.now().normalize()
                    dates = pd.date_range(start=today, periods=periods, freq='D')
                    future_df = pd.DataFrame({'ds': dates})
                                                  
        # Realizar predicción
        self.logger.info(f"Generando predicción para {len(future_df)} periodos")
        self.forecast = self.model.predict(future_df)
        
        # Si usamos transformación log, convertir predicciones a escala original
        if self.use_log_transform:
            self.forecast['yhat_exp'] = np.expm1(self.forecast['yhat'])
            self.forecast['yhat_lower_exp'] = np.expm1(self.forecast['yhat_lower'])
            self.forecast['yhat_upper_exp'] = np.expm1(self.forecast['yhat_upper'])
                
        # Calibrar intervalos de predicción si se solicita
        if calibrate_intervals and hasattr(self, 'train_data') and self.train_data is not None:
            from ..models.calibrator import IntervalCalibrator
            
            self.logger.info("Calibrando intervalos de predicción...")
            calibrator = IntervalCalibrator()
            
            # Seleccionar método de calibración según cantidad de datos
            if len(self.train_data) > 50:
                # Si hay suficientes datos, usar calibración empírica con CV
                self.forecast = calibrator.empirical_calibration(
                    model=self.model,
                    train_df=self.train_data, 
                    forecast_df=self.forecast, 
                    target_coverage=self.params.get('interval_width', 0.9)
                )
            else:
                # Para series cortas, usar calibración conformal más sencilla
                self.forecast = calibrator.calibrate_conformal(
                    train_df=self.train_data,
                    forecast_df=self.forecast,
                    target_coverage=self.params.get('interval_width', 0.9)
                )
            
            self.logger.info("Intervalos de predicción calibrados")
        
        return self.forecast
        
    @st.cache_data(ttl=3600)
    def evaluate_model(_self, df: pd.DataFrame = None, 
                     cv_periods: int = 30, 
                     initial_period: int = 180,
                     period_step: int = 30) -> Dict:
        """
        Realiza validación cruzada y calcula métricas de rendimiento
        
        Args:
            df: DataFrame para evaluación (usa train_data si no se proporciona)
            cv_periods: Horizonte de predicción en validación cruzada
            initial_period: Tamaño inicial para entrenar
            period_step: Paso entre periodos de validación
            
        Returns:
            Dict con métricas de evaluación
        """
        if _self.model is None:
            raise ValueError("El modelo debe ser entrenado primero")
            
        # Si no se proporciona df, usar datos de entrenamiento
        if df is None:
            if _self.train_data is None:
                raise ValueError("No hay datos de entrenamiento disponibles")
            df = _self.train_data.copy()
        
        _self.logger.info(f"Iniciando validación cruzada con horizonte={cv_periods}")
        
        # Verificar que hay suficientes datos para validación cruzada
        min_rows_needed = initial_period + cv_periods
        if len(df) < min_rows_needed:
            _self.logger.warning(
                f"Insuficientes datos para validación cruzada: {len(df)} filas, se necesitan al menos {min_rows_needed}."
                f"Se ajustarán los parámetros automáticamente."
            )
            # Ajustar parámetros para adaptarse a los datos disponibles
            available_days = len(df)
            # Usar 70% para entrenamiento, hasta 20% para validación
            initial_period = max(int(available_days * 0.7), 7)  # Al menos 7 días
            cv_periods = min(int(available_days * 0.2), 30)  # Máximo 30 días o 20% de los datos
            period_step = max(int(cv_periods / 3), 1)  # Al menos 1 día
            
            _self.logger.info(
                f"Parámetros ajustados: initial_period={initial_period}, "
                f"cv_periods={cv_periods}, period_step={period_step}"
            )
        
        try:
            # Realizar validación cruzada
            cv_results = cross_validation(
                _self.model,
                initial=f'{initial_period} days',
                period=f'{period_step} days',
                horizon=f'{cv_periods} days',
                disable_tqdm=False
            )
            
            # Comprobar que obtuvimos resultados
            if cv_results is None or cv_results.empty:
                raise ValueError("La validación cruzada no generó resultados")
                
            # Verificar si hay valores NaN en las columnas necesarias
            nan_in_y = cv_results['y'].isna().sum()
            nan_in_yhat = cv_results['yhat'].isna().sum()
            if nan_in_y > 0 or nan_in_yhat > 0:
                _self.logger.warning(f"Encontrados {nan_in_y} NaN en 'y' y {nan_in_yhat} NaN en 'yhat'. Se eliminarán.")
                cv_results = cv_results.dropna(subset=['y', 'yhat'])
                
                if cv_results.empty:
                    raise ValueError("Todos los valores son NaN después de limpiar")
            
            # Extraer arrays para cálculo de métricas
            y_true = cv_results['y'].values
            y_pred = cv_results['yhat'].values
            
            # Extraer intervalos si están disponibles
            interval_lower = None
            interval_upper = None
            if 'yhat_lower' in cv_results.columns and 'yhat_upper' in cv_results.columns:
                interval_lower = cv_results['yhat_lower'].values
                interval_upper = cv_results['yhat_upper'].values
            
            # Calcular métricas utilizando la función centralizada
            metrics_dict = calculate_metrics(
                y_true=y_true,
                y_pred=y_pred,
                interval_lower=interval_lower,
                interval_upper=interval_upper
            )
            
            # Extraer las métricas relevantes
            summary_metrics = {
                'mae': metrics_dict.get('mae', np.nan),
                'rmse': metrics_dict.get('rmse', np.nan),
                'smape': metrics_dict.get('smape', np.nan),
                'coverage': metrics_dict.get('coverage', 0.0) * 100,  # Convertir a porcentaje
                'r2': metrics_dict.get('r2', np.nan),
                'horizon': cv_periods
            }
            
            # Verificar si hay métricas con NaN y loggear advertencia
            nan_metrics = [key for key, val in summary_metrics.items() 
                           if isinstance(val, (float, int)) and (np.isnan(val) or val is None)]
            
            if nan_metrics:
                _self.logger.warning(f"Las siguientes métricas son NaN: {', '.join(nan_metrics)}")
                # Reemplazar NaN con 0 para evitar problemas en la interfaz
                for key in nan_metrics:
                    summary_metrics[key] = 0.0
            
            _self.metrics = summary_metrics
            _self.logger.info(
                f"Validación cruzada completada. MAE={summary_metrics.get('mae', 0):.4f}, "
                f"SMAPE={summary_metrics.get('smape', 0):.2f}%, "
                f"Cobertura={summary_metrics.get('coverage', 0):.1f}%"
            )
            
            # Guardar datos de CV para análisis posterior
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv_results_path = os.path.join("modeling", "resultados_cv", f"cv_results_{timestamp}.csv")
            os.makedirs(os.path.dirname(cv_results_path), exist_ok=True)
            
            try:
                cv_results.to_csv(cv_results_path, index=False)
                _self.logger.info(f"Resultados de validación cruzada guardados en {cv_results_path}")
            except Exception as e:
                _self.logger.warning(f"No se pudieron guardar resultados de CV: {str(e)}")
            
            return summary_metrics
            
        except Exception as e:
            _self.logger.error(f"Error en validación cruzada: {str(e)}")
            # Devolver métricas básicas para no romper la interfaz
            return {
                'mae': 0.0,  # Mejor usar 0 que NaN para evitar errores en la interfaz
                'rmse': 0.0,
                'smape': 0.0,
                'coverage': 0.0,
                'r2': 0.0,
                'horizon': cv_periods,
                'error': str(e)
            }
