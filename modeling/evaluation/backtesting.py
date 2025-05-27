import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prophet import Prophet
import streamlit as st

# Importar el evaluador de modelos existente
from .model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class BacktestEvaluator:
    """
    Componente para realizar backtesting (pruebas retrospectivas) del modelo
    de predicción de ransomware.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_evaluator = ModelEvaluator()
        
    def run_backtest(self, 
                   model: Prophet,
                   df: pd.DataFrame,
                   cutoff_date: datetime,
                   horizon_days: int = 30,
                   include_regressors: bool = True) -> Dict:
        """
        Realiza una prueba retrospectiva (backtest) del modelo hasta una fecha de corte
        
        Args:
            model: Modelo Prophet entrenado
            df: DataFrame con datos históricos completos
            cutoff_date: Fecha de corte para el backtest
            horizon_days: Días a predecir después de la fecha de corte
            include_regressors: Si se deben incluir regresores en la predicción
            
        Returns:
            Diccionario con resultados del backtest y métricas
        """
        self.logger.info(f"Realizando backtest con fecha de corte: {cutoff_date}")
        
        # Validación de datos
        if df is None or model is None:
            self.logger.error("No hay datos o modelo disponible para backtest")
            return {
                "success": False,
                "error": "No hay datos o modelo disponible"
            }
            
        # Convertir a datetime si es string
        if isinstance(cutoff_date, str):
            cutoff_date = pd.to_datetime(cutoff_date)
            
        # Validar que la fecha de corte esté dentro del rango de datos
        if cutoff_date <= df['ds'].min():
            return {
                "success": False,
                "error": "La fecha de corte es anterior a los datos disponibles"
            }
            
        if cutoff_date >= df['ds'].max():
            return {
                "success": False,
                "error": "La fecha de corte es posterior al último dato disponible"
            }
            
        # Datos de entrenamiento (hasta la fecha de corte)
        train_df = df[df['ds'] <= cutoff_date].copy()
        
        # Datos de validación (después de la fecha de corte, hasta horizon_days)
        end_date = cutoff_date + timedelta(days=horizon_days)
        actual_df = df[(df['ds'] > cutoff_date) & (df['ds'] <= end_date)].copy()
        
        if len(actual_df) == 0:
            return {
                "success": False,
                "error": f"No hay datos reales después de la fecha de corte para {horizon_days} días"
            }
            
        # Crear un nuevo modelo y entrenarlo con datos hasta la fecha de corte
        try:
            # Clonar configuración del modelo original
            backtest_model = Prophet(
                changepoint_prior_scale=model.changepoint_prior_scale,
                seasonality_prior_scale=model.seasonality_prior_scale,
                holidays_prior_scale=model.holidays_prior_scale,
                seasonality_mode=model.seasonality_mode,
                interval_width=model.interval_width
            )
            
            # Añadir estacionalidades y regresores si existían en el modelo original
            if hasattr(model, 'extra_regressors') and include_regressors:
                for regressor in model.extra_regressors:
                    if regressor['name'] in train_df.columns:
                        backtest_model.add_regressor(regressor['name'], 
                                                   prior_scale=regressor.get('prior_scale', 10),
                                                   mode=regressor.get('mode', 'additive'))
            
            # Entrenar con datos hasta la fecha de corte
            backtest_model.fit(train_df)
            
            # Generar predicciones para el horizonte
            future = backtest_model.make_future_dataframe(periods=horizon_days)
            
            # Añadir regresores al dataframe futuro si es necesario
            if hasattr(model, 'extra_regressors') and include_regressors:
                for regressor in model.extra_regressors:
                    if regressor['name'] in df.columns:
                        # Asegurarse de que tenemos valores de regresores para el período futuro
                        regressor_future = df[['ds', regressor['name']]]
                        future = pd.merge(future, regressor_future, on='ds', how='left')
            
            # Hacer predicción
            forecast = backtest_model.predict(future)
            
            # Unir predicciones con valores reales para comparación
            comparison_df = pd.merge(
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                df[['ds', 'y']],
                on='ds',
                how='inner'
            )
            
            # Filtrar solo el período de validación
            validation_df = comparison_df[(comparison_df['ds'] > cutoff_date) & 
                                        (comparison_df['ds'] <= end_date)].copy()
            
            # Calcular métricas de error usando el evaluador existente
            if len(validation_df) > 0:
                # Usar el evaluador de modelos existente para calcular métricas
                metrics = self.model_evaluator.calculate_metrics(
                    validation_df['y'].values,
                    validation_df['yhat'].values,
                    validation_df['yhat_lower'].values,
                    validation_df['yhat_upper'].values
                )
                
                # Añadir número de puntos evaluados
                metrics['n_points'] = len(validation_df)
                
                # Resultados finales
                return {
                    "success": True,
                    "forecast": forecast,
                    "train_df": train_df,
                    "validation_df": validation_df,
                    "cutoff_date": cutoff_date,
                    "horizon_days": horizon_days,
                    "metrics": metrics
                }
            else:
                return {
                    "success": False,
                    "error": "No hay suficientes datos para validación"
                }
            
        except Exception as e:
            self.logger.error(f"Error en backtest: {str(e)}")
            return {
                "success": False,
                "error": f"Error en backtest: {str(e)}"
            }
            
    def plot_backtest_results(self, backtest_results: Dict) -> go.Figure:
        """
        Genera una visualización de los resultados del backtest
        
        Args:
            backtest_results: Resultados del backtest generados por run_backtest
            
        Returns:
            Figura Plotly con la visualización
        """
        if not backtest_results.get("success", False):
            self.logger.error("No hay resultados de backtest válidos para visualizar")
            return None
            
        # Extraer datos
        train_df = backtest_results["train_df"]
        validation_df = backtest_results["validation_df"]
        forecast = backtest_results["forecast"]
        cutoff_date = backtest_results["cutoff_date"]
        
        # Crear figura
        fig = go.Figure()
        
        # Datos de entrenamiento (históricos)
        fig.add_trace(go.Scatter(
            x=train_df['ds'],
            y=train_df['y'],
            mode='markers',
            name='Datos de entrenamiento',
            marker=dict(color='blue', size=6),
            hovertemplate='%{x|%d/%m/%Y}: %{y:.2f}<extra></extra>'
        ))
        
        # Datos de validación (valores reales)
        fig.add_trace(go.Scatter(
            x=validation_df['ds'],
            y=validation_df['y'],
            mode='markers',
            name='Valores reales',
            marker=dict(color='green', size=8),
            hovertemplate='%{x|%d/%m/%Y}: %{y:.2f}<extra></extra>'
        ))
        
        # Predicción para el período de validación
        mask = (forecast['ds'] > cutoff_date)
        fig.add_trace(go.Scatter(
            x=forecast[mask]['ds'],
            y=forecast[mask]['yhat'],
            mode='lines',
            name='Predicción',
            line=dict(color='red', width=3),
            hovertemplate='%{x|%d/%m/%Y}: %{y:.2f}<extra></extra>'
        ))
        
        # Intervalo de confianza
        fig.add_trace(go.Scatter(
            x=forecast[mask]['ds'].tolist() + forecast[mask]['ds'].tolist()[::-1],
            y=forecast[mask]['yhat_upper'].tolist() + forecast[mask]['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalo de confianza',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Línea vertical en la fecha de corte
        fig.add_vline(
            x=cutoff_date, 
            line_width=2, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Fecha de corte",
            annotation_position="top right"
        )
        
        # Añadir etiquetas y diseño
        fig.update_layout(
            title=f"Backtest con fecha de corte: {cutoff_date.strftime('%d/%m/%Y')}",
            xaxis_title="Fecha",
            yaxis_title="Ataques ransomware",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
def create_backtest_table(backtest_results: Dict) -> pd.DataFrame:
    """
    Crea una tabla detallada con los resultados del backtest
    
    Args:
        backtest_results: Resultados del backtest
        
    Returns:
        DataFrame con los resultados detallados
    """
    if not backtest_results.get("success", False):
        return None
        
    validation_df = backtest_results["validation_df"].copy()
    
    # Calcular errores
    validation_df['error'] = validation_df['y'] - validation_df['yhat']
    validation_df['error_abs'] = np.abs(validation_df['error'])
    validation_df['error_pct'] = 100 * validation_df['error_abs'] / (np.abs(validation_df['y']) + 1e-8)
    
    # Formatear las columnas para mostrar
    result_df = validation_df[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper', 'error', 'error_abs', 'error_pct', 'in_interval']].copy()
    result_df.columns = ['Fecha', 'Valor Real', 'Predicción', 'Límite Inferior', 'Límite Superior', 
                        'Error', 'Error Abs', 'Error %', 'En Intervalo']
    
    # Ordenar por fecha
    result_df = result_df.sort_values('Fecha')
    
    return result_df
