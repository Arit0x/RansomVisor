"""
Módulo de ingeniería de características para el modelo de predicción de ransomware.

Este módulo proporciona funcionalidad para aplicar transformaciones a los datos
y crear características avanzadas que mejoren la precisión de los modelos Prophet.
"""

import pandas as pd
import numpy as np
import logging
import streamlit as st
from typing import Dict, Tuple, Optional, Callable, List, Union
from datetime import datetime, timedelta

# Configuración de logging
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Clase para aplicar transformaciones y crear características avanzadas
    para mejorar la predicción de ransomware.
    """
    
    @staticmethod
    def apply_optimal_transformation(df: pd.DataFrame, method: str = 'log') -> Tuple[pd.DataFrame, Callable]:
        """
        Aplica la transformación óptima a la variable objetivo.
        
        Args:
            df: DataFrame con datos (debe tener columna 'y')
            method: Método de transformación ('log', 'sqrt', 'none')
            
        Returns:
            Tuple con DataFrame transformado y función para revertir la transformación
        """
        if 'y' not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'y'")
            
        transformed_df = df.copy()
        
        # Registrar porcentaje de ceros para diagnóstico
        zero_pct = (transformed_df['y'] == 0).mean() * 100
        logger.info(f"Porcentaje de ceros en los datos: {zero_pct:.2f}%")
        
        if method == 'log':
            # Transformación log(1+y) para manejar ceros
            transformed_df['y'] = np.log1p(transformed_df['y'])
            
            # Función para revertir la transformación
            def reverse_transform(y_transformed):
                return np.expm1(y_transformed)
                
            logger.info(f"Aplicada transformación logarítmica: log(1+y)")
            
        elif method == 'sqrt':
            # Transformación raíz cuadrada
            transformed_df['y'] = np.sqrt(transformed_df['y'])
            
            # Función para revertir la transformación
            def reverse_transform(y_transformed):
                return np.power(y_transformed, 2)
                
            logger.info(f"Aplicada transformación raíz cuadrada: sqrt(y)")
            
        else:
            # Sin transformación
            logger.info("No se aplicó transformación a los datos")
            
            def reverse_transform(y_transformed):
                return y_transformed
        
        return transformed_df, reverse_transform
    
    @staticmethod
    def reverse_transform_forecast(forecast_df: pd.DataFrame, 
                                 reverse_func: Callable,
                                 columns: List[str] = None) -> pd.DataFrame:
        """
        Revierte la transformación en un DataFrame de pronóstico.
        
        Args:
            forecast_df: DataFrame con pronósticos (yhat, yhat_lower, yhat_upper)
            reverse_func: Función para revertir la transformación
            columns: Lista de columnas a transformar (por defecto: yhat, yhat_lower, yhat_upper)
            
        Returns:
            DataFrame con valores en escala original
        """
        if columns is None:
            columns = ['yhat', 'yhat_lower', 'yhat_upper']
            
        result = forecast_df.copy()
        
        for col in columns:
            if col in result.columns:
                result[col] = reverse_func(result[col])
                
        return result
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características temporales básicas.
        
        Args:
            df: DataFrame con columna 'ds' de fechas
            
        Returns:
            DataFrame con características temporales adicionales
        """
        if 'ds' not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'ds'")
            
        result = df.copy()
        
        # Convertir a datetime si es necesario
        if not pd.api.types.is_datetime64_any_dtype(result['ds']):
            result['ds'] = pd.to_datetime(result['ds'])
        
        # Características de tiempo
        result['day_of_week'] = result['ds'].dt.dayofweek
        result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
        result['day_of_month'] = result['ds'].dt.day
        result['week_of_year'] = result['ds'].dt.isocalendar().week.astype(int)
        result['month'] = result['ds'].dt.month
        result['quarter'] = result['ds'].dt.quarter
        result['year'] = result['ds'].dt.year
        result['is_month_end'] = result['ds'].dt.is_month_end.astype(int)
        result['is_month_start'] = result['ds'].dt.is_month_start.astype(int)
        
        logger.info(f"Creadas {len(result.columns) - len(df.columns)} características temporales")
        
        return result
    
    @staticmethod
    def add_patch_tuesday_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade características relacionadas con Patch Tuesday (importante para ransomware).
        
        Args:
            df: DataFrame con columna 'ds' de fechas
            
        Returns:
            DataFrame con características de Patch Tuesday
        """
        if 'ds' not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'ds'")
            
        result = df.copy()
        
        # Convertir a datetime si es necesario
        if not pd.api.types.is_datetime64_any_dtype(result['ds']):
            result['ds'] = pd.to_datetime(result['ds'])
            
        # Identificar Patch Tuesdays (segundo martes de cada mes)
        min_date = result['ds'].min()
        max_date = result['ds'].max()
        
        # Función para verificar si una fecha es Patch Tuesday
        def is_patch_tuesday(date):
            # Es martes (1) y es el segundo martes del mes (entre día 8 y 14)
            return date.weekday() == 1 and 8 <= date.day <= 14
        
        # Encontrar todos los Patch Tuesdays en el rango
        current = min_date.replace(day=1)  # Empezar al principio del mes
        patch_tuesdays = []
        
        while current <= max_date:
            # Avanzar hasta el primer martes del mes
            while current.weekday() != 1:
                current += timedelta(days=1)
            
            # Avanzar al segundo martes
            current += timedelta(days=7)
            
            # Si todavía estamos en el rango, añadir a la lista
            if current <= max_date:
                patch_tuesdays.append(current)
            
            # Avanzar al siguiente mes
            current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        
        # Inicializar columnas relacionadas con Patch Tuesday
        result['is_patch_tuesday'] = 0
        result['days_since_patch'] = 0
        result['days_to_next_patch'] = 0
        
        # Marcar Patch Tuesdays y calcular días desde/hasta
        for i, pt_date in enumerate(patch_tuesdays):
            # Marcar el día del parche
            result.loc[result['ds'] == pt_date, 'is_patch_tuesday'] = 1
            
            # Calcular días transcurridos desde último parche
            next_pt = patch_tuesdays[i+1] if i+1 < len(patch_tuesdays) else max_date + timedelta(days=30)
            mask = (result['ds'] >= pt_date) & (result['ds'] < next_pt)
            
            result.loc[mask, 'days_since_patch'] = (result.loc[mask, 'ds'] - pt_date).dt.days
            
            # Calcular días hasta el próximo parche
            if i+1 < len(patch_tuesdays):
                result.loc[mask, 'days_to_next_patch'] = (patch_tuesdays[i+1] - result.loc[mask, 'ds']).dt.days
        
        # Crear ventana de vulnerabilidad (primeros 14 días tras parche)
        result['patch_vulnerability_window'] = ((result['days_since_patch'] > 0) & 
                                              (result['days_since_patch'] <= 14)).astype(int)
        
        # Ventana previa al parche (7 días antes)
        result['pre_patch_window'] = ((result['days_to_next_patch'] >= 0) & 
                                    (result['days_to_next_patch'] <= 7)).astype(int)
        
        logger.info(f"Creadas características de Patch Tuesday con {len(patch_tuesdays)} parches identificados")
        
        return result
    
    @staticmethod
    def add_cve_features(df: pd.DataFrame, 
                       cve_df: pd.DataFrame, 
                       date_col: str = 'date', 
                       count_col: str = 'count',
                       window_sizes: List[int] = None) -> pd.DataFrame:
        """
        Añade características derivadas de datos de CVE.
        
        Args:
            df: DataFrame principal con columna 'ds'
            cve_df: DataFrame con datos de CVE
            date_col: Nombre de la columna de fecha en cve_df
            count_col: Nombre de la columna de conteo en cve_df
            window_sizes: Tamaños de ventana para características de CVE
            
        Returns:
            DataFrame con características de CVE
        """
        if 'ds' not in df.columns:
            raise ValueError("El DataFrame principal debe contener la columna 'ds'")
            
        if date_col not in cve_df.columns or count_col not in cve_df.columns:
            raise ValueError(f"El DataFrame de CVE debe contener las columnas '{date_col}' y '{count_col}'")
            
        if window_sizes is None:
            window_sizes = [7, 14, 30, 90]
        
        result = df.copy()
        cve_data = cve_df.copy()
        
        # Preparar datos de CVE
        if not pd.api.types.is_datetime64_any_dtype(cve_data[date_col]):
            cve_data[date_col] = pd.to_datetime(cve_data[date_col])
            
        # Renombrar para uniformidad
        cve_data = cve_data.rename(columns={date_col: 'ds'})
        
        # Agregar por día y preparar para merge
        daily_cve = cve_data.groupby('ds')[count_col].sum().reset_index()
        
        # Asegurar que tenemos todas las fechas en el rango
        date_range = pd.DataFrame({'ds': pd.date_range(
            start=min(result['ds'].min(), daily_cve['ds'].min()),
            end=max(result['ds'].max(), daily_cve['ds'].max()),
            freq='D'
        )})
        
        # Combinar con el rango completo
        daily_cve = pd.merge(date_range, daily_cve, on='ds', how='left').fillna(0)
        
        # Crear características para cada ventana de tiempo
        for window in window_sizes:
            # Conteo diario
            daily_cve[f'cve_{window}d'] = daily_cve[count_col].rolling(window=window, min_periods=1).sum()
            
            # Promedio diario
            daily_cve[f'cve_{window}d_avg'] = daily_cve[count_col].rolling(window=window, min_periods=1).mean()
            
            # Tendencia (ratio de crecimiento)
            if window > 7:  # Solo para ventanas mayores
                daily_cve[f'cve_{window}d_trend'] = (daily_cve[f'cve_7d'] / 
                                                   daily_cve[f'cve_{window}d'].shift(7)).fillna(1.0)
            
            # Volatilidad
            daily_cve[f'cve_{window}d_volatility'] = (
                daily_cve[count_col].rolling(window=window, min_periods=window//2).std() / 
                (daily_cve[count_col].rolling(window=window, min_periods=window//2).mean() + 1.0)  # Evitar div/0
            ).fillna(0)
        
        # Eliminar columna original para evitar duplicados en el merge
        daily_cve = daily_cve.drop(columns=[count_col])
        
        # Fusionar con el DataFrame principal
        result = pd.merge(result, daily_cve, on='ds', how='left')
        
        # Rellenar valores faltantes
        for col in result.columns:
            if col.startswith('cve_'):
                result[col] = result[col].fillna(0)
        
        # Crear interacciones con ventanas de vulnerabilidad si existen
        if 'patch_vulnerability_window' in result.columns:
            for window in window_sizes:
                result[f'patch_cve_{window}d_interaction'] = (
                    result['patch_vulnerability_window'] * np.log1p(result[f'cve_{window}d'])
                )
        
        logger.info(f"Creadas {sum(1 for col in result.columns if col.startswith('cve_'))} características de CVE")
        
        return result
    
    @staticmethod
    def select_optimal_features(df: pd.DataFrame, 
                              target_col: str = 'y', 
                              max_features: int = 10,
                              corr_threshold: float = 0.1) -> List[str]:
        """
        Selecciona las características más predictivas.
        
        Args:
            df: DataFrame con características y target
            target_col: Nombre de la columna objetivo
            max_features: Número máximo de características a seleccionar
            corr_threshold: Umbral mínimo de correlación absoluta
            
        Returns:
            Lista con los nombres de las características seleccionadas
        """
        if target_col not in df.columns:
            raise ValueError(f"El DataFrame debe contener la columna '{target_col}'")
            
        # Calcular correlaciones con el target
        numeric_cols = df.select_dtypes(include=['number']).columns
        valid_cols = [col for col in numeric_cols if col != target_col and col != 'ds']
        
        if not valid_cols:
            logger.warning("No se encontraron columnas numéricas para selección de características")
            return []
            
        correlations = {}
        for col in valid_cols:
            correlations[col] = abs(df[[col, target_col]].corr().iloc[0, 1])
            
        # Ordenar por correlación absoluta
        sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Filtrar por umbral y limitar número
        selected = [col for col, corr in sorted_features if abs(corr) >= corr_threshold][:max_features]
        
        logger.info(f"Seleccionadas {len(selected)} características predictivas de {len(valid_cols)}")
        
        return selected
    
    @staticmethod
    def add_cybersecurity_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade características específicas para predicción de ransomware basadas en 
        conocimiento del dominio de ciberseguridad.
        
        Args:
            df: DataFrame con columna 'ds' de fechas
            
        Returns:
            DataFrame con características de ciberseguridad añadidas
        """
        if 'ds' not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'ds'")
            
        result = df.copy()
        
        # Asegurar que 'ds' es datetime
        if not pd.api.types.is_datetime64_any_dtype(result['ds']):
            result['ds'] = pd.to_datetime(result['ds'])
        
        # 1. CICLOS DE PARCHEADO Y ACTUALIZACIONES
        # ----------------------------------------
        # Patch Tuesday: segundo martes de cada mes (Microsoft)
        result['is_patch_tuesday'] = result['ds'].apply(FeatureEngineer._is_patch_tuesday)
        
        # Días desde/hasta Patch Tuesday (patrón cíclico)
        result['days_since_patch_tuesday'] = result['ds'].apply(FeatureEngineer._days_since_patch_tuesday)
        result['days_to_next_patch_tuesday'] = result['ds'].apply(FeatureEngineer._days_to_next_patch_tuesday)
        
        # Ventanas de vulnerabilidad (1-7 días después de Patch Tuesday)
        result['patch_window_phase'] = result['days_since_patch_tuesday'].apply(
            lambda x: min(x, 7) if 0 <= x <= 7 else 0
        )
        
        # Ciclos de actualización de otros proveedores
        # Oracle: publica actualizaciones trimestrales (enero, abril, julio, octubre)
        result['oracle_patch_month'] = result['ds'].dt.month.isin([1, 4, 7, 10]).astype(int)
        result['oracle_patch_week'] = ((result['ds'].dt.month.isin([1, 4, 7, 10])) & 
                                     (result['ds'].dt.day.between(15, 21))).astype(int)
        
        # 2. PATRONES ESTACIONALES DE CIBERATAQUES
        # ---------------------------------------
        # Fin de trimestre y fin de año (presupuestos y presión organizacional)
        result['quarter_end'] = result['ds'].dt.is_quarter_end.astype(int)
        result['quarter_start'] = result['ds'].dt.is_quarter_start.astype(int)
        result['year_end'] = result['ds'].dt.is_year_end.astype(int)
        
        # Mes del año como variable cíclica (mejor que categórica)
        # Usando codificación seno/coseno para representar ciclicidad
        result['month_sin'] = np.sin(2 * np.pi * result['ds'].dt.month / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['ds'].dt.month / 12)
        
        # Día de la semana como variable cíclica
        result['weekday_sin'] = np.sin(2 * np.pi * result['ds'].dt.dayofweek / 7)
        result['weekday_cos'] = np.cos(2 * np.pi * result['ds'].dt.dayofweek / 7)
        
        # 3. TEMPORADAS DE ALTO RIESGO
        # ----------------------------
        # Vacaciones: períodos donde el personal de TI puede estar reducido
        result['is_holiday_season'] = ((result['ds'].dt.month == 12) & 
                                     (result['ds'].dt.day >= 15)).astype(int)
        
        # Temporada alta de ciberataques (según patrones históricos)
        # Muchos estudios indican que el período octubre-febrero tiene más ataques
        result['high_attack_season'] = ((result['ds'].dt.month >= 10) | 
                                     (result['ds'].dt.month <= 2)).astype(int)
        
        # 4. CARACTERÍSTICAS DE MOMENTUM Y TENDENCIA
        # -----------------------------------------
        # Días desde inicio de año (captura tendencias anuales)
        result['day_of_year'] = result['ds'].dt.dayofyear
        result['day_of_year_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365.25)
        result['day_of_year_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365.25)
        
        # Día del mes como variable cíclica
        result['day_of_month_sin'] = np.sin(2 * np.pi * result['ds'].dt.day / 31)
        result['day_of_month_cos'] = np.cos(2 * np.pi * result['ds'].dt.day / 31)
        
        # 5. PERÍODOS DE CONFERENCIAS DE SEGURIDAD
        # --------------------------------------
        # Principales conferencias como Black Hat, RSA, etc. (aproximación)
        # Black Hat/Defcon (típicamente en agosto)
        result['blackhat_period'] = ((result['ds'].dt.month == 8) & 
                                   (result['ds'].dt.day.between(1, 10))).astype(int)
                                   
        # RSA Conference (típicamente en febrero/marzo)
        result['rsa_period'] = (((result['ds'].dt.month == 2) & (result['ds'].dt.day >= 20)) | 
                              ((result['ds'].dt.month == 3) & (result['ds'].dt.day <= 10))).astype(int)
        
        logger.info(f"Añadidas {len(result.columns) - len(df.columns)} características de ciberseguridad")
        
        return result
    
    @staticmethod
    def _is_patch_tuesday(date):
        """Determina si una fecha es Patch Tuesday (segundo martes del mes)"""
        # Convertir a datetime si es necesario
        if not isinstance(date, (datetime, pd.Timestamp)):
            date = pd.Timestamp(date)
            
        # El segundo martes del mes
        # Primero, encontrar el primer día del mes
        first_day = date.replace(day=1)
        
        # Encontrar el primer martes (día de semana = 1 en Python)
        days_until_first_tuesday = (1 - first_day.weekday()) % 7
        first_tuesday = first_day + pd.Timedelta(days=days_until_first_tuesday)
        
        # El segundo martes es 7 días después
        second_tuesday = first_tuesday + pd.Timedelta(days=7)
        
        return date.date() == second_tuesday.date()
    
    @staticmethod
    def _days_since_patch_tuesday(date):
        """Calcula días transcurridos desde el último Patch Tuesday"""
        if not isinstance(date, (datetime, pd.Timestamp)):
            date = pd.Timestamp(date)
            
        # Encontrar el Patch Tuesday del mes actual
        first_day = date.replace(day=1)
        days_until_first_tuesday = (1 - first_day.weekday()) % 7
        first_tuesday = first_day + pd.Timedelta(days=days_until_first_tuesday)
        patch_tuesday = first_tuesday + pd.Timedelta(days=7)
        
        # Si la fecha es anterior al Patch Tuesday de este mes,
        # necesitamos buscar el del mes anterior
        if date < patch_tuesday:
            # Ir al mes anterior
            if date.month == 1:
                previous_month = date.replace(year=date.year-1, month=12, day=1)
            else:
                previous_month = date.replace(month=date.month-1, day=1)
                
            days_until_first_tuesday = (1 - previous_month.weekday()) % 7
            first_tuesday = previous_month + pd.Timedelta(days=days_until_first_tuesday)
            patch_tuesday = first_tuesday + pd.Timedelta(days=7)
        
        # Calcular la diferencia en días
        delta = date - patch_tuesday
        return delta.days
    
    @staticmethod
    def _days_to_next_patch_tuesday(date):
        """Calcula días hasta el próximo Patch Tuesday"""
        if not isinstance(date, (datetime, pd.Timestamp)):
            date = pd.Timestamp(date)
            
        # Encontrar el Patch Tuesday del mes actual
        first_day = date.replace(day=1)
        days_until_first_tuesday = (1 - first_day.weekday()) % 7
        first_tuesday = first_day + pd.Timedelta(days=days_until_first_tuesday)
        patch_tuesday = first_tuesday + pd.Timedelta(days=7)
        
        # Si la fecha es posterior o igual al Patch Tuesday de este mes,
        # necesitamos buscar el del mes siguiente
        if date >= patch_tuesday:
            # Ir al mes siguiente
            if date.month == 12:
                next_month = date.replace(year=date.year+1, month=1, day=1)
            else:
                next_month = date.replace(month=date.month+1, day=1)
                
            days_until_first_tuesday = (1 - next_month.weekday()) % 7
            first_tuesday = next_month + pd.Timedelta(days=days_until_first_tuesday)
            patch_tuesday = first_tuesday + pd.Timedelta(days=7)
        
        # Calcular la diferencia en días
        delta = patch_tuesday - date
        return delta.days
    
    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade características temporales básicas al DataFrame para ayudar en la predicción.
        
        Args:
            df: DataFrame con columna 'ds' de fechas
            
        Returns:
            DataFrame con características temporales añadidas
        """
        if 'ds' not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'ds'")
            
        result = df.copy()
        
        # Asegurar que 'ds' es datetime
        if not pd.api.types.is_datetime64_any_dtype(result['ds']):
            result['ds'] = pd.to_datetime(result['ds'])
        
        # Características de tiempo básicas
        # --------------------------------
        
        # Día de la semana (0-6, donde 0 es lunes)
        result['dayofweek'] = result['ds'].dt.dayofweek
        
        # Día del mes (1-31)
        result['dayofmonth'] = result['ds'].dt.day
        
        # Día del año (1-366)
        result['dayofyear'] = result['ds'].dt.dayofyear
        
        # Mes (1-12)
        result['month'] = result['ds'].dt.month
        
        # Trimestre (1-4)
        result['quarter'] = result['ds'].dt.quarter
        
        # Año
        result['year'] = result['ds'].dt.year
        
        # Es fin de semana (0-1)
        result['is_weekend'] = (result['dayofweek'] >= 5).astype(int)
        
        # Es fin de mes (0-1)
        result['is_month_end'] = result['ds'].dt.is_month_end.astype(int)
        
        # Es fin de trimestre (0-1)
        result['is_quarter_end'] = result['ds'].dt.is_quarter_end.astype(int)
        
        # Es fin de año (0-1)
        result['is_year_end'] = result['ds'].dt.is_year_end.astype(int)
        
        # Codificación cíclica para variables temporales
        # ---------------------------------------------
        # Estas transformaciones capturan la naturaleza cíclica del tiempo
        
        # Día de la semana como variables cíclicas
        result['dow_sin'] = np.sin(2 * np.pi * result['dayofweek'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['dayofweek'] / 7)
        
        # Mes como variables cíclicas
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Día del año como variables cíclicas
        result['doy_sin'] = np.sin(2 * np.pi * result['dayofyear'] / 366)
        result['doy_cos'] = np.cos(2 * np.pi * result['dayofyear'] / 366)
        
        # Variables categóricas para Prophet
        # ---------------------------------
        # Prophet puede usar estas como regresores
        
        # Mes como variables one-hot
        for i in range(1, 13):
            result[f'month_{i}'] = (result['month'] == i).astype(int)
            
        # Día de la semana como variables one-hot
        for i in range(7):
            result[f'dow_{i}'] = (result['dayofweek'] == i).astype(int)
            
        return result


@st.cache_data
def prepare_ransomware_features(_self, df: pd.DataFrame, 
                               cve_df: pd.DataFrame = None,
                               transform_method: str = 'log') -> Dict:
    """
    Función cacheada para preparar características avanzadas para el modelo de ransomware.
    
    Args:
        _self: Parámetro para compatibilidad con Streamlit (no usado)
        df: DataFrame con datos de ransomware
        cve_df: DataFrame con datos de CVE (opcional)
        transform_method: Método de transformación ('log', 'sqrt', 'none')
        
    Returns:
        Diccionario con DataFrame procesado y función de transformación inversa
    """
    try:
        # 1. Aplicar transformación óptima
        transformed_df, reverse_func = FeatureEngineer.apply_optimal_transformation(
            df, method=transform_method
        )
        
        # 2. Crear características temporales
        feature_df = FeatureEngineer.create_temporal_features(transformed_df)
        
        # 3. Añadir características de Patch Tuesday
        feature_df = FeatureEngineer.add_patch_tuesday_features(feature_df)
        
        # 4. Añadir características de CVE si están disponibles
        if cve_df is not None:
            feature_df = FeatureEngineer.add_cve_features(feature_df, cve_df)
        
        # 5. Añadir características temporales básicas
        feature_df = FeatureEngineer.add_temporal_features(feature_df)
        
        # 6. Añadir características de ciberseguridad
        feature_df = FeatureEngineer.add_cybersecurity_features(feature_df)
        
        # 7. Seleccionar las mejores características
        selected_features = FeatureEngineer.select_optimal_features(feature_df)
        
        return {
            'feature_df': feature_df,
            'reverse_transform_func': reverse_func,
            'selected_features': selected_features
        }
    except Exception as e:
        logger.error(f"Error al preparar características: {str(e)}")
        return None
