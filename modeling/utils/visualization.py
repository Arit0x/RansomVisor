import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st

class RansomwarePlotter:
    """
    Funciones de visualización para resultados del modelo de ransomware
    utilizando Plotly para gráficos interactivos.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def plot_forecast(self, 
                    history_df: pd.DataFrame, 
                    forecast_df: pd.DataFrame,
                    use_log_transform: bool = False,
                    plot_changepoints: bool = False,
                    changepoints_data: Optional[Dict] = None) -> go.Figure:
        """
        Genera gráfica de predicción con Plotly
        
        Args:
            history_df: DataFrame con datos históricos (ds, y)
            forecast_df: DataFrame con predicciones
            use_log_transform: Si se usó transformación logarítmica
            plot_changepoints: Si mostrar los puntos de cambio
            changepoints_data: Datos de changepoints si plot_changepoints=True
            
        Returns:
            Figura de Plotly
        """
        # Determinar qué columnas usar según transformación
        y_col = 'y_original' if 'y_original' in history_df.columns and use_log_transform else 'y'
        yhat_col = 'yhat_exp' if 'yhat_exp' in forecast_df.columns and use_log_transform else 'yhat'
        yhat_lower_col = 'yhat_lower_exp' if 'yhat_lower_exp' in forecast_df.columns and use_log_transform else 'yhat_lower'
        yhat_upper_col = 'yhat_upper_exp' if 'yhat_upper_exp' in forecast_df.columns and use_log_transform else 'yhat_upper'
        
        # Crear figura
        fig = go.Figure()
        
        # Datos históricos
        fig.add_trace(go.Scatter(
            x=history_df['ds'],
            y=history_df[y_col],
            mode='markers',
            name='Histórico',
            marker=dict(color='blue', size=6)
        ))
        
        # Línea de predicción
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df[yhat_col],
            mode='lines',
            name='Predicción',
            line=dict(color='red', width=2)
        ))
        
        # Intervalo de confianza
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df[yhat_upper_col],
            mode='lines',
            name='Límite Superior',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df[yhat_lower_col],
            mode='lines',
            name='Límite Inferior',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            fill='tonexty',
            showlegend=False
        ))
        
        # Añadir changepoints si se solicitan
        if plot_changepoints and changepoints_data:
            changepoint_dates = changepoints_data.get('dates', [])
            if changepoint_dates:
                # Convertir a formato de fecha si es necesario
                if isinstance(changepoint_dates[0], str):
                    changepoint_dates = pd.to_datetime(changepoint_dates)
                
                # Obtener los valores Y para los changepoints
                y_values = []
                for date in changepoint_dates:
                    # Buscar el valor más cercano en el forecast
                    closest_date_idx = abs(forecast_df['ds'] - date).argmin()
                    y_values.append(forecast_df.iloc[closest_date_idx][yhat_col])
                
                # Añadir scatter para los changepoints
                fig.add_trace(go.Scatter(
                    x=changepoint_dates,
                    y=y_values,
                    mode='markers',
                    name='Puntos de Cambio',
                    marker=dict(color='green', size=10, symbol='x')
                ))
        
        # Formato y título
        fig.update_layout(
            title='Predicción de Ataques Ransomware',
            xaxis_title='Fecha',
            yaxis_title='Número de Ataques',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode="x unified",
            template='plotly_white'
        )
        
        return fig
        
    def plot_components(self, forecast_df: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Genera gráficos de componentes del modelo
        
        Args:
            forecast_df: DataFrame con las predicciones
            
        Returns:
            Dict de figuras Plotly para cada componente
        """
        figures = {}
        
        # Verificar que tenemos las columnas necesarias
        component_cols = ['trend', 'weekly', 'yearly', 'monthly', 'holidays']
        available_components = [col for col in component_cols if col in forecast_df.columns]
        
        if not available_components:
            self.logger.warning("No se encontraron componentes para graficar")
            return figures
        
        # 1. Tendencia
        if 'trend' in available_components:
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['trend'],
                mode='lines',
                name='Tendencia',
                line=dict(color='darkblue', width=2)
            ))
            
            trend_fig.update_layout(
                title='Componente de Tendencia',
                xaxis_title='Fecha',
                yaxis_title='Tendencia',
                template='plotly_white'
            )
            
            figures['trend'] = trend_fig
            
        # 2. Componente semanal
        if 'weekly' in available_components:
            # Agrupar por día de la semana
            forecast_df['day_of_week'] = forecast_df['ds'].dt.dayofweek
            forecast_df['day_name'] = forecast_df['ds'].dt.day_name()
            
            # Calcular media por día
            day_means = forecast_df.groupby('day_name')['weekly'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                'Friday', 'Saturday', 'Sunday'
            ])
            
            weekly_fig = go.Figure()
            weekly_fig.add_trace(go.Bar(
                x=day_means.index,
                y=day_means.values,
                marker_color='royalblue',
                name='Efecto por Día'
            ))
            
            weekly_fig.update_layout(
                title='Componente Estacional Semanal',
                xaxis_title='Día de la Semana',
                yaxis_title='Efecto',
                template='plotly_white'
            )
            
            figures['weekly'] = weekly_fig
            
        # 3. Componente mensual
        if 'monthly' in available_components:
            # Agrupar por mes
            forecast_df['month'] = forecast_df['ds'].dt.month
            forecast_df['month_name'] = forecast_df['ds'].dt.month_name()
            
            # Calcular media por mes
            month_means = forecast_df.groupby('month_name')['monthly'].mean()
            
            # Reordenar meses
            month_order = [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]
            month_means = month_means.reindex(month_order)
            
            monthly_fig = go.Figure()
            monthly_fig.add_trace(go.Bar(
                x=month_means.index,
                y=month_means.values,
                marker_color='darkgreen',
                name='Efecto por Mes'
            ))
            
            monthly_fig.update_layout(
                title='Componente Estacional Mensual',
                xaxis_title='Mes',
                yaxis_title='Efecto',
                template='plotly_white'
            )
            
            figures['monthly'] = monthly_fig
            
        # 4. Componente anual
        if 'yearly' in available_components:
            # Crear gráfico con valores por día del año
            yearly_fig = go.Figure()
            
            # Usar día del año como x y yearly como y
            forecast_df['day_of_year'] = forecast_df['ds'].dt.dayofyear
            
            # Agrupar por día del año y calcular media
            day_of_year_means = forecast_df.groupby('day_of_year')['yearly'].mean()
            
            yearly_fig.add_trace(go.Scatter(
                x=day_of_year_means.index,
                y=day_of_year_means.values,
                mode='lines',
                name='Estacionalidad Anual',
                line=dict(color='darkred', width=2)
            ))
            
            # Añadir líneas verticales para los meses
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                          'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            
            for i, month_start in enumerate(month_starts):
                yearly_fig.add_vline(x=month_start, line_dash="dash", line_width=1, 
                                    line_color="gray", opacity=0.5)
                yearly_fig.add_annotation(
                    x=month_start + 15,  # Mitad del mes aproximadamente
                    y=min(day_of_year_means.values),
                    text=month_names[i],
                    showarrow=False,
                    yshift=-20
                )
            
            yearly_fig.update_layout(
                title='Componente Estacional Anual',
                xaxis_title='Día del Año',
                yaxis_title='Efecto',
                template='plotly_white'
            )
            
            figures['yearly'] = yearly_fig
            
        # 5. Efectos de días festivos
        if 'holidays' in available_components and forecast_df['holidays'].notna().any():
            # Filtrar sólo días con efectos de holidays
            holiday_days = forecast_df[forecast_df['holidays'] != 0].copy()
            
            if not holiday_days.empty:
                # Ordenar por magnitud del efecto
                holiday_days = holiday_days.sort_values(by='holidays', ascending=False)
                
                holidays_fig = go.Figure()
                holidays_fig.add_trace(go.Bar(
                    x=holiday_days['ds'],
                    y=holiday_days['holidays'],
                    marker_color='purple',
                    name='Efecto de Festivos'
                ))
                
                holidays_fig.update_layout(
                    title='Efecto de Días Festivos',
                    xaxis_title='Fecha',
                    yaxis_title='Efecto',
                    template='plotly_white'
                )
                
                figures['holidays'] = holidays_fig
            
        return figures
        
    def plot_metrics(self, metrics: Dict) -> go.Figure:
        """
        Visualiza métricas de rendimiento del modelo
        
        Args:
            metrics: Diccionario con métricas
            
        Returns:
            Figura Plotly con visualización de métricas
        """
        if not metrics:
            return None
            
        # Extraer métricas numéricas
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_metrics[key] = value
                
        if not numeric_metrics:
            return None
            
        # Crear figura
        fig = go.Figure()
        
        # Añadir barras para cada métrica
        fig.add_trace(go.Bar(
            x=list(numeric_metrics.keys()),
            y=list(numeric_metrics.values()),
            marker_color=['royalblue' if k != 'coverage' else 'green' for k in numeric_metrics.keys()]
        ))
        
        # Personalizar layout
        fig.update_layout(
            title='Métricas de Rendimiento',
            xaxis_title='Métrica',
            yaxis_title='Valor',
            template='plotly_white'
        )
        
        return fig
        
    def plot_residuals(self, residuals: np.ndarray) -> go.Figure:
        """
        Visualiza análisis de residuos
        
        Args:
            residuals: Array de residuos
            
        Returns:
            Figura con análisis de residuos
        """
        # Crear subplots: histograma y QQ plot
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=('Distribución de Residuos', 'Residuos vs. Predicción'),
                          specs=[[{'type': 'xy'}, {'type': 'xy'}]])
        
        # Histograma de residuos
        fig.add_trace(
            go.Histogram(x=residuals, name='Residuos', marker_color='royalblue'),
            row=1, col=1
        )
        
        # Añadir línea vertical en el cero
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Scatter plot de residuos vs. predicción (simplemente los residuos en orden)
        fig.add_trace(
            go.Scatter(x=np.arange(len(residuals)), y=residuos, mode='markers',
                    name='Residuos', marker=dict(color='royalblue', size=5)),
            row=1, col=2
        )
        
        # Línea horizontal en el cero
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # Actualizar layout
        fig.update_layout(
            title_text="Análisis de Residuos",
            height=400,
            template='plotly_white'
        )
        
        return fig
