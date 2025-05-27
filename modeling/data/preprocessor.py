import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import streamlit as st

class DataPreprocessor:
    """
    Preprocesa los datos para el modelo Prophet, incluyendo:
    - Agregación temporal
    - Transformación logarítmica
    - Normalización
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.use_log_transform = True
        
    def prepare_prophet_data(self, df: pd.DataFrame, 
                          date_col: str = 'fecha', 
                          value_col: str = 'ataques',
                          freq: str = 'D',
                          fill_strategy: str = 'zero') -> pd.DataFrame:
        """
        Prepara datos para Prophet según el formato requerido (ds, y)
        
        Args:
            df: DataFrame con los datos
            date_col: Nombre de la columna con fechas
            value_col: Nombre de la columna con valores a predecir
            freq: Frecuencia de agregación ('D' para diario)
            fill_strategy: Estrategia para rellenar valores faltantes ('zero', 'mean', 'median', 'interpolate', 'none')
            
        Returns:
            DataFrame en formato Prophet (ds, y)
        """
        self.logger.info(f"Preparando datos para Prophet con frecuencia {freq}")
        
        # Validar columnas
        if date_col not in df.columns:
            raise ValueError(f"Columna de fecha '{date_col}' no encontrada")
            
        if value_col not in df.columns:
            raise ValueError(f"Columna de valor '{value_col}' no encontrada")
            
        # Copiar dataframe para no modificar el original
        df_tmp = df.copy()
        
        # Asegurar que la fecha es datetime
        df_tmp[date_col] = pd.to_datetime(df_tmp[date_col])
        
        # Crear df con formato Prophet
        prophet_df = df_tmp.rename(columns={date_col: 'ds', value_col: 'y'})
        
        # Agregar por fecha si hay duplicados
        if prophet_df['ds'].duplicated().any():
            self.logger.info("Detectadas fechas duplicadas, agregando valores")
            prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
            
        # Ordenar por fecha
        prophet_df = prophet_df.sort_values('ds')
        
        # Rellenar fechas faltantes en la serie temporal
        date_range = pd.date_range(start=prophet_df['ds'].min(), 
                                  end=prophet_df['ds'].max(),
                                  freq=freq)
        
        full_df = pd.DataFrame({'ds': date_range})
        prophet_df = pd.merge(full_df, prophet_df, on='ds', how='left')
        
        # Rellenar valores faltantes según la estrategia elegida
        if fill_strategy == 'zero':
            self.logger.info("Rellenando valores faltantes con ceros")
            prophet_df['y'] = prophet_df['y'].fillna(0)
        elif fill_strategy == 'mean':
            self.logger.info("Rellenando valores faltantes con la media")
            prophet_df['y'] = prophet_df['y'].fillna(prophet_df['y'].mean())
        elif fill_strategy == 'median':
            self.logger.info("Rellenando valores faltantes con la mediana")
            prophet_df['y'] = prophet_df['y'].fillna(prophet_df['y'].median())
        elif fill_strategy == 'interpolate':
            self.logger.info("Rellenando valores faltantes mediante interpolación")
            prophet_df['y'] = prophet_df['y'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        elif fill_strategy == 'none':
            self.logger.info("Manteniendo valores faltantes como NaN")
            pass
        else:
            self.logger.warning(f"Estrategia de relleno '{fill_strategy}' no reconocida, usando ceros por defecto")
            prophet_df['y'] = prophet_df['y'].fillna(0)
        
        return prophet_df
        
    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformación logarítmica a la columna 'y'
        
        Args:
            df: DataFrame con columna 'y'
            
        Returns:
            DataFrame con 'y' transformado
        """
        if 'y' not in df.columns:
            raise ValueError("Columna 'y' no encontrada para transformación log")
            
        df_transformed = df.copy()
        
        # Aplicar log(y+1) para manejar ceros
        self.logger.info("Aplicando transformación logarítmica a valores")
        df_transformed['y_original'] = df_transformed['y'].copy()
        df_transformed['y'] = np.log1p(df_transformed['y'])
        
        return df_transformed
        
    def invert_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Invierte la transformación logarítmica
        
        Args:
            df: DataFrame con valores transformados
            
        Returns:
            DataFrame con valores en escala original
        """
        if 'y' not in df.columns:
            raise ValueError("Columna 'y' no encontrada")
            
        df_original = df.copy()
        
        # Revertir transformación log
        cols_to_transform = [col for col in df.columns if col.startswith('yhat')]
        cols_to_transform.append('y')
        
        for col in cols_to_transform:
            if col in df.columns:
                df_original[f"{col}_exp"] = np.expm1(df[col])
                
        return df_original
        
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características basadas en fecha para usar como regresores
        
        Args:
            df: DataFrame con columna 'ds'
            
        Returns:
            DataFrame con características adicionales
        """
        if 'ds' not in df.columns:
            raise ValueError("Columna 'ds' no encontrada")
            
        df_features = df.copy()
        
        # Características de fecha
        df_features['year'] = df_features['ds'].dt.year
        df_features['month'] = df_features['ds'].dt.month
        df_features['day'] = df_features['ds'].dt.day
        df_features['dayofweek'] = df_features['ds'].dt.dayofweek
        df_features['quarter'] = df_features['ds'].dt.quarter
        
        # Días festivos y características especiales
        df_features['weekend'] = (df_features['dayofweek'] >= 5).astype(int)
        df_features['month_start'] = (df_features['day'] == 1).astype(int)
        df_features['month_end'] = df_features['ds'].dt.is_month_end.astype(int)
        df_features['quarter_start'] = ((df_features['month'] - 1) % 3 == 0) & (df_features['day'] == 1)
        df_features['quarter_end'] = df_features['ds'].dt.is_quarter_end.astype(int)
        
        # Patch Tuesday (segundo martes del mes)
        def is_patch_tuesday(date):
            if date.weekday() == 1:  # 1 es martes
                # Calcula el día del primer martes
                first_day = datetime(date.year, date.month, 1)
                days_until_first_tuesday = (8 - first_day.weekday()) % 7
                first_tuesday = first_day.day + days_until_first_tuesday
                
                # El segundo martes es 7 días después
                second_tuesday = first_tuesday + 7
                return date.day == second_tuesday
            return False
            
        df_features['patch_tuesday'] = df_features['ds'].apply(is_patch_tuesday).astype(int)
        
        return df_features
        
    def prepare_for_prophet(self, df: pd.DataFrame, 
                           use_log_transform: bool = True,
                           outlier_method: str = 'iqr', 
                           outlier_strategy: str = 'winsorize',
                           outlier_threshold: float = 1.5,
                           min_victims: int = 1) -> pd.DataFrame:
        """
        Prepara los datos para el modelo Prophet:
        1. Convierte a formato Prophet (ds, y)
        2. Detecta y maneja outliers
        3. Aplica transformación logarítmica si se solicita
        
        Args:
            df: DataFrame con datos crudos
            use_log_transform: Si aplicar transformación logarítmica
            outlier_method: Método para detectar outliers ('iqr', 'zscore', 'contextual_ransomware', 'none')
            outlier_strategy: Estrategia para manejar outliers ('remove', 'cap', 'winsorize', 'ransomware', 'none')
            outlier_threshold: Umbral para detección de outliers
            min_victims: Mínimo de víctimas para considerar un día como ataque
            
        Returns:
            DataFrame preparado para Prophet
        """
        self.logger.info(f"Preparando datos para Prophet (log_transform={use_log_transform})")
        
        # Guardar configuración
        self.use_log_transform = use_log_transform
        
        # Preparar estructura para Prophet
        date_col = None
        for col in ['fecha', 'date', 'ds']:
            if col in df.columns:
                date_col = col
                break
                
        if date_col is None:
            raise ValueError("No se encontró columna de fecha ('fecha', 'date', 'ds')")
            
        # Buscar columna de valor
        value_col = None
        for col in ['ataques', 'victimas', 'count', 'valor', 'y']:
            if col in df.columns:
                value_col = col
                break
                
        if value_col is None:
            # Intentar inferir columna numérica
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]
            else:
                raise ValueError("No se encontró columna de valor")
                
        # Convertir a formato Prophet
        self.logger.info(f"Usando columnas: {date_col} -> ds, {value_col} -> y")
        df_prophet = self.prepare_prophet_data(
            df, 
            date_col=date_col, 
            value_col=value_col,
            freq='D',
            fill_strategy='zero'
        )
        
        # Filtrar por mínimo de víctimas si es necesario
        if min_victims > 1:
            self.logger.info(f"Filtrando por mínimo de {min_victims} víctimas")
            original_rows = len(df_prophet)
            df_prophet = df_prophet[df_prophet['y'] >= min_victims]
            filtered_rows = original_rows - len(df_prophet)
            self.logger.info(f"Filtradas {filtered_rows} filas con menos de {min_victims} víctimas")
            
        # Detectar ratio de ceros para ajustar estrategia
        zero_ratio = (df_prophet['y'] == 0).mean()
        self.logger.info(f"Proporción de ceros en datos: {zero_ratio:.2%}")
        
        # Aplicar procesamiento especializado para series con muchos ceros
        if zero_ratio > 0.3 and use_log_transform:
            self.logger.info("Detectada serie con muchos ceros, aplicando procesamiento especializado")
            from ..features.outliers import OutlierDetector
            outlier_detector = OutlierDetector()
            
            # Aplicar preprocesamiento especializado
            df_prophet = outlier_detector.prepare_zero_inflated_data(df_prophet, column='y')
            
            # Guardar detector para uso posterior en inversión de transformación
            self.outlier_detector = outlier_detector
            self.zero_inflated_preprocessing = True
            
            # Sugerir parámetros óptimos para Prophet basados en proporción de ceros
            optimal_params = outlier_detector.get_optimal_prophet_params_for_zero_inflated(zero_ratio)
            self.logger.info(f"Parámetros sugeridos para serie con {zero_ratio:.2%} ceros: {optimal_params}")
            self.recommended_prophet_params = optimal_params
            
        else:
            # Procesamiento estándar
            self.zero_inflated_preprocessing = False
            
            # Manejo de outliers estándar
            if outlier_method != 'none':
                df_prophet = self._handle_outliers(
                    df_prophet, 
                    method=outlier_method, 
                    strategy=outlier_strategy, 
                    threshold=outlier_threshold
                )
                
            # Aplicar transformación logarítmica si se solicita
            if use_log_transform:
                self.logger.info("Aplicando transformación logarítmica estándar")
                df_prophet = self.apply_log_transform(df_prophet)
        
        return df_prophet
        
    def _handle_outliers(self, df: pd.DataFrame, 
                        method: str = 'iqr', 
                        strategy: str = 'winsorize', 
                        threshold: float = 1.5) -> pd.DataFrame:
        """
        Detecta y maneja outliers en los datos
        
        Args:
            df: DataFrame en formato Prophet
            method: Método de detección de outliers
            strategy: Estrategia para manejar outliers
            threshold: Umbral para detección
            
        Returns:
            DataFrame con outliers procesados
        """
        from ..features.outliers import OutlierDetector
        
        self.logger.info(f"Procesando outliers: método={method}, estrategia={strategy}, umbral={threshold}")
        
        outlier_detector = OutlierDetector()
        
        # Usar la nueva detección contextual si se solicita
        if method == 'contextual_ransomware':
            outliers = outlier_detector.detect_contextual_outliers(df, column='y')
        else:
            outliers = outlier_detector.detect_outliers(
                df, method=method, column='y', threshold=threshold
            )
            
        # Usar estrategia específica para ransomware si se solicita
        if strategy == 'ransomware':
            strategy = outlier_detector.determine_ransomware_outlier_strategy(df, column='y')
            self.logger.info(f"Estrategia adaptativa para ransomware seleccionada: {strategy}")
            
        # Procesar outliers con la estrategia elegida
        df_processed = outlier_detector.handle_outliers(
            df, outliers=outliers, strategy=strategy, column='y'
        )
        
        # Guardar referencia al detector para uso posterior
        self.outlier_detector = outlier_detector
        
        return df_processed
    
    def invert_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Invierte todas las transformaciones aplicadas durante el preprocesamiento
        
        Args:
            df: DataFrame con predicciones
            
        Returns:
            DataFrame con valores en escala original
        """
        df_inverted = df.copy()
        
        # Inversión para series con muchos ceros
        if hasattr(self, 'zero_inflated_preprocessing') and self.zero_inflated_preprocessing:
            if hasattr(self, 'outlier_detector'):
                self.logger.info("Invirtiendo transformación para serie con muchos ceros")
                df_inverted = self.outlier_detector.invert_zero_inflated_transform(df_inverted)
                return df_inverted
        
        # Inversión estándar de transformación logarítmica
        if self.use_log_transform:
            self.logger.info("Invirtiendo transformación logarítmica estándar")
            df_inverted = self.invert_log_transform(df_inverted)
            
        return df_inverted
