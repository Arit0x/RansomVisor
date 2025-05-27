import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st

class OutlierDetector:
    """
    Detecta y maneja outliers en series temporales para mejorar 
    la calidad de las predicciones.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_outliers(self, df: pd.DataFrame, 
                      method: str = 'iqr',
                      column: str = 'y',
                      threshold: float = 3.0) -> Dict:
        """
        Detecta outliers en una serie temporal usando varios métodos
        
        Args:
            df: DataFrame con los datos
            method: Método de detección ('iqr', 'zscore', 'modified_zscore', 'hybrid_mad_iqr')
            column: Columna para detectar outliers
            threshold: Umbral para considerar un punto como outlier
            
        Returns:
            Dict con índices de outliers y estadísticas
        """
        self.logger.info(f"Detectando outliers con método {method}")
        
        if column not in df.columns:
            raise ValueError(f"Columna {column} no encontrada en DataFrame")
            
        values = df[column].values
        outlier_indices = []
        
        if method == 'iqr':
            # Método IQR (rango intercuartílico)
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_indices = df[
                (df[column] < lower_bound) | 
                (df[column] > upper_bound)
            ].index.tolist()
            
            stats = {
                'method': 'iqr',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }
            
        elif method == 'zscore':
            # Método Z-Score
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                self.logger.warning("Desviación estándar es 0, no se pueden calcular z-scores")
                return {'indices': [], 'stats': {'method': 'zscore', 'error': 'std=0'}}
                
            z_scores = [(y - mean) / std for y in values]
            outlier_indices = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
            
            stats = {
                'method': 'zscore',
                'mean': mean,
                'std': std,
                'threshold': threshold
            }
            
        elif method == 'modified_zscore':
            # Z-Score Modificado (más robusto a outliers)
            median = np.median(values)
            mad = np.median([abs(y - median) for y in values])
            
            if mad == 0:
                self.logger.warning("MAD es 0, no se pueden calcular z-scores modificados")
                return {'indices': [], 'stats': {'method': 'modified_zscore', 'error': 'mad=0'}}
                
            modified_z_scores = [0.6745 * (y - median) / mad if mad > 0 else 0 for y in values]
            outlier_indices = [i for i, z in enumerate(modified_z_scores) if abs(z) > threshold]
            
            stats = {
                'method': 'modified_zscore',
                'median': median,
                'mad': mad,
                'threshold': threshold
            }
            
        elif method == 'hybrid_mad_iqr':
            # Método híbrido MAD+IQR: más robusto que ambos por separado
            # Primero calculamos outliers con MAD (más resistente a extremos)
            median = np.median(values)
            mad = np.median([abs(y - median) for y in values]) * 1.4826  # Factor para equivalencia con desviación estándar
            
            if mad == 0:
                self.logger.warning("MAD es 0, usando solo IQR")
                mad_indices = []
            else:
                mad_scores = [(y - median) / mad for y in values]
                mad_indices = [i for i, z in enumerate(mad_scores) if abs(z) > threshold]
            
            # Luego calculamos outliers con IQR como segundo chequeo
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            iqr_indices = df[
                (df[column] < lower_bound) | 
                (df[column] > upper_bound)
            ].index.tolist()
            
            # Combinamos los resultados: un punto es outlier si ambos métodos lo detectan
            # o si MAD lo detecta con un umbral más alto (más conservador)
            mad_indices_conservative = []
            if mad > 0:
                mad_scores = [(y - median) / mad for y in values]
                mad_indices_conservative = [i for i, z in enumerate(mad_scores) if abs(z) > 3.5]  # Umbral más conservador
            
            # Unimos los índices (conjunto de unión)
            outlier_indices = list(set(mad_indices_conservative) | (set(mad_indices) & set(iqr_indices)))
            
            stats = {
                'method': 'hybrid_mad_iqr',
                'median': median,
                'mad': mad,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'mad_indices': len(mad_indices),
                'iqr_indices': len(iqr_indices),
                'combined_indices': len(outlier_indices),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'threshold': threshold
            }
            
        else:
            raise ValueError(f"Método de detección no soportado: {method}")
            
        self.logger.info(f"Detectados {len(outlier_indices)} outliers")
        
        return {
            'indices': outlier_indices,
            'stats': stats
        }
    
    def detect_autocorrelation(self, df: pd.DataFrame, column: str = 'y', lag: int = 7) -> float:
        """
        Detecta el nivel de autocorrelación en la serie temporal
        
        Args:
            df: DataFrame con los datos
            column: Columna para detectar autocorrelación
            lag: Retraso para calcular autocorrelación
            
        Returns:
            Valor de autocorrelación (entre -1 y 1)
        """
        if column not in df.columns:
            raise ValueError(f"Columna {column} no encontrada en DataFrame")
            
        if len(df) <= lag:
            self.logger.warning(f"Serie demasiado corta para calcular autocorrelación con lag={lag}")
            return 0.0
            
        values = df[column].values
        n = len(values)
        
        # Calcular autocorrelación
        mean = np.mean(values)
        autocorr = np.sum((values[:n-lag] - mean) * (values[lag:] - mean)) / \
                   np.sum((values - mean) ** 2)
                   
        self.logger.info(f"Autocorrelación (lag={lag}): {autocorr:.4f}")
        return autocorr
        
    def determine_adaptive_strategy(self, df: pd.DataFrame, column: str = 'y') -> str:
        """
        Determina la mejor estrategia para manejar outliers basada en características de los datos
        
        Args:
            df: DataFrame con datos
            column: Columna a analizar
            
        Returns:
            Estrategia recomendada ('cap', 'winsorize', 'interpolate', 'none')
        """
        if column not in df.columns:
            raise ValueError(f"Columna {column} no encontrada")
            
        # Calcular características clave
        zero_ratio = (df[column] == 0).mean()
        autocorr = self.detect_autocorrelation(df, column)
        
        # Verificar cantidad de ceros
        if zero_ratio > 0.3:
            self.logger.info(f"Alta proporción de ceros ({zero_ratio:.2%}), recomendando 'cap'")
            return 'cap'
            
        # Verificar autocorrelación
        if abs(autocorr) > 0.7:
            self.logger.info(f"Alta autocorrelación ({autocorr:.4f}), evitando 'interpolate'")
            return 'winsorize'  # Evitar 'interpolate' con alta autocorrelación
            
        # Calcular densidad de outliers
        outliers = self.detect_outliers(df, method='hybrid_mad_iqr', column=column, threshold=3.5)
        outlier_ratio = len(outliers['indices']) / len(df)
        
        if outlier_ratio < 0.05:
            self.logger.info(f"Pocos outliers ({outlier_ratio:.2%}), recomendando 'winsorize'")
            return 'winsorize'
        elif outlier_ratio < 0.1 and abs(autocorr) < 0.5:
            self.logger.info(f"Proporción moderada de outliers ({outlier_ratio:.2%}) con autocorrelación moderada, recomendando 'interpolate'")
            return 'interpolate'
        else:
            self.logger.info(f"Alta proporción de outliers ({outlier_ratio:.2%}), recomendando 'cap'")
            return 'cap'
        
    def handle_outliers(self, df: pd.DataFrame, 
                       outliers: Dict,
                       strategy: str = 'winsorize',
                       column: str = 'y') -> pd.DataFrame:
        """
        Aplica estrategia para manejar outliers
        
        Args:
            df: DataFrame con datos
            outliers: Dict con índices de outliers y estadísticas
            strategy: Estrategia ('remove', 'winsorize', 'mean', 'median', 'interpolate', 'adaptive')
            column: Columna a procesar
            
        Returns:
            DataFrame con outliers tratados
        """
        self.logger.info(f"Aplicando estrategia '{strategy}' para outliers")
        
        if column not in df.columns:
            raise ValueError(f"Columna {column} no encontrada")
            
        df_processed = df.copy()
        indices = outliers.get('indices', [])
        stats = outliers.get('stats', {})
        
        if not indices:
            self.logger.info("No se encontraron outliers para procesar")
            return df_processed
            
        # Añadir bandera is_outlier para seguimiento
        if 'is_outlier' not in df_processed.columns:
            df_processed['is_outlier'] = False
        df_processed.loc[indices, 'is_outlier'] = True
        
        # Si es estrategia adaptativa, determinar la mejor estrategia
        if strategy == 'adaptive':
            strategy = self.determine_adaptive_strategy(df_processed, column)
            self.logger.info(f"Estrategia adaptativa seleccionada: {strategy}")
        
        if strategy == 'remove':
            # Eliminar outliers
            df_processed = df_processed.drop(indices)
            self.logger.info(f"Eliminados {len(indices)} outliers")
            
        elif strategy == 'winsorize':
            # Recortar valores extremos a los límites
            method = stats.get('method', '')
            
            if method == 'iqr':
                lower = stats.get('lower_bound')
                upper = stats.get('upper_bound')
                
                df_processed.loc[indices, column] = df_processed.loc[indices, column].clip(
                    lower=lower, upper=upper
                )
                
            elif method in ['zscore', 'modified_zscore', 'hybrid_mad_iqr']:
                for idx in indices:
                    if df_processed.loc[idx, column] > stats.get('mean', stats.get('median', 0)):
                        # Valores por encima de la media/mediana
                        df_processed.loc[idx, column] = stats.get('mean', stats.get('median', 0)) + \
                                                      stats.get('std', stats.get('mad', 0)) * stats.get('threshold', 3)
                    else:
                        # Valores por debajo de la media/mediana
                        df_processed.loc[idx, column] = stats.get('mean', stats.get('median', 0)) - \
                                                      stats.get('std', stats.get('mad', 0)) * stats.get('threshold', 3)
        
        elif strategy == 'cap':
            # Truncar a percentil 95/5
            p95 = np.percentile(df_processed[column], 95)
            p05 = np.percentile(df_processed[column], 5)
            
            for idx in indices:
                if df_processed.loc[idx, column] > p95:
                    df_processed.loc[idx, column] = p95
                elif df_processed.loc[idx, column] < p05:
                    df_processed.loc[idx, column] = p05
        
        elif strategy == 'mean':
            # Reemplazar con la media
            mean_value = df_processed[column].mean()
            df_processed.loc[indices, column] = mean_value
            
        elif strategy == 'median':
            # Reemplazar con la mediana
            median_value = df_processed[column].median()
            df_processed.loc[indices, column] = median_value
        
        elif strategy == 'interpolate':
            # Verificar si hay alta autocorrelación
            autocorr = self.detect_autocorrelation(df, column)
            
            if abs(autocorr) > 0.7:
                self.logger.warning(f"Alta autocorrelación ({autocorr:.4f}), cambiando de 'interpolate' a 'winsorize'")
                # Redirigir a winsorize para evitar "peinar" picos genuinos
                return self.handle_outliers(df, outliers, 'winsorize', column)
                
            # Reemplazar con interpolación
            # Guardamos valores originales
            original_values = df_processed.loc[indices, column].copy()
            
            # Marcar outliers como NaN para interpolar
            df_processed.loc[indices, column] = np.nan
            
            # Interpolar valores
            df_processed[column] = df_processed[column].interpolate(method='time')
            
            # Rellenar extremos si quedaron NaN
            df_processed[column] = df_processed[column].fillna(method='ffill').fillna(method='bfill')
            
            # Verificar si quedó algún NaN (por si todos son outliers)
            if df_processed[column].isna().any():
                df_processed.loc[indices, column] = original_values
                self.logger.warning("No se pudo interpolar, restaurando valores originales")
            
        else:
            raise ValueError(f"Estrategia no soportada: {strategy}")
            
        return df_processed
        
    def process_outliers(self, df: pd.DataFrame, 
                        column: str = 'y',
                        method: str = 'hybrid_mad_iqr',
                        strategy: str = 'adaptive',
                        threshold: float = 3.5) -> pd.DataFrame:
        """
        Procesa outliers en un solo paso: detecta y maneja usando estrategia óptima
        
        Args:
            df: DataFrame con datos
            column: Columna a procesar
            method: Método de detección
            strategy: Estrategia de manejo (usar 'adaptive' para selección automática)
            threshold: Umbral para detección
            
        Returns:
            DataFrame procesado con outliers tratados y marcados
        """
        # Detectar outliers
        outliers = self.detect_outliers(df, method=method, column=column, threshold=threshold)
        
        # Manejar outliers con la estrategia seleccionada
        df_processed = self.handle_outliers(df, outliers, strategy=strategy, column=column)
        
        # Verificar patrones específicos en outliers
        if 'is_outlier' in df_processed.columns and df_processed['is_outlier'].sum() > 0:
            self.analyze_outlier_patterns(df_processed, column)
            
        return df_processed
        
    def analyze_outlier_patterns(self, df: pd.DataFrame, column: str = 'y') -> Dict:
        """
        Analiza patrones en los outliers detectados
        
        Args:
            df: DataFrame con datos y columna is_outlier
            column: Columna original analizada
            
        Returns:
            Diccionario con patrones detectados
        """
        if 'is_outlier' not in df.columns:
            return {'error': 'No hay columna is_outlier en el DataFrame'}
            
        if 'ds' not in df.columns:
            return {'error': 'No hay columna de fecha (ds) en el DataFrame'}
            
        # Extraer outliers
        outliers_df = df[df['is_outlier']].copy()
        
        patterns = {}
        
        # Verificar patrones temporales
        if len(outliers_df) > 0:
            # Patrones por día de la semana
            dow_counts = outliers_df['ds'].dt.dayofweek.value_counts().to_dict()
            # Normalizar por total de cada día en el dataset
            for day in range(7):
                total_day = (df['ds'].dt.dayofweek == day).sum()
                if total_day > 0 and day in dow_counts:
                    dow_counts[day] = dow_counts[day] / total_day
                    
            patterns['day_of_week'] = dow_counts
            
            # Patrones por mes
            month_counts = outliers_df['ds'].dt.month.value_counts().to_dict()
            # Normalizar por total de cada mes
            for month in range(1, 13):
                total_month = (df['ds'].dt.month == month).sum()
                if total_month > 0 and month in month_counts:
                    month_counts[month] = month_counts[month] / total_month
                    
            patterns['month'] = month_counts
            
            # Verificar si hay patrón post patch tuesday
            if 'patch_tuesday' in df.columns:
                # Calcular días desde el último patch tuesday
                post_patch = []
                for idx in outliers_df.index:
                    date = df.loc[idx, 'ds']
                    # Buscar el patch tuesday anterior más cercano
                    patch_dates = df[df['patch_tuesday'] == 1]['ds']
                    prev_patches = patch_dates[patch_dates < date]
                    if not prev_patches.empty:
                        last_patch = prev_patches.max()
                        days_since = (date - last_patch).days
                        post_patch.append(days_since)
                        
                if post_patch:
                    # Analizar distribución de días post-patch
                    patterns['days_since_patch'] = {
                        'mean': np.mean(post_patch),
                        'median': np.median(post_patch),
                        'min': min(post_patch),
                        'max': max(post_patch),
                        'values': post_patch
                    }
                    
                    # Verificar si hay concentración en los primeros días post-patch
                    early_patch = [d for d in post_patch if d <= 3]
                    if len(early_patch) / len(post_patch) > 0.4:
                        self.logger.warning(f"Se detectaron {len(early_patch)} outliers en los primeros 3 días post-patch tuesday")
                        patterns['post_patch_concentration'] = True
            
        return patterns

    def detect_contextual_outliers(self, df: pd.DataFrame, column: str = 'y') -> Dict:
        """
        Detecta outliers considerando el contexto temporal específico de ransomware
        
        Args:
            df: DataFrame con datos
            column: Columna a analizar
            
        Returns:
            Dict con índices de outliers contextuales
        """
        self.logger.info("Aplicando detección contextual de outliers para ransomware")
        
        if 'ds' not in df.columns:
            self.logger.error("Columna 'ds' requerida para detección contextual")
            return {'indices': [], 'stats': {'method': 'contextual_ransomware', 'error': 'No ds column'}}
            
        # 1. Segmentar por día de la semana (los ataques siguen patrones semanales)
        dow_groups = df.groupby(df['ds'].dt.dayofweek)
        
        # 2. Detectar outliers dentro de cada segmento
        contextual_outliers = []
        
        for dow, group in dow_groups:
            # Usar MAD para cada día de la semana separadamente
            values = group[column].values
            if len(values) < 5:  # Necesitamos suficientes puntos
                continue
                
            median = np.median(values)
            mad = np.median(np.abs(values - median)) * 1.4826  # Factor para equivalencia
            
            if mad == 0:  # Evitar división por cero
                continue
                
            # Calcular scores para este día de la semana
            z_scores = np.abs(values - median) / mad
            
            # Índices de outliers para este segmento
            segment_outliers = group[z_scores > 3.5].index.tolist()
            contextual_outliers.extend(segment_outliers)
        
        # 3. Añadir detección de picos en fines de semana (generalmente menos actividad)
        weekend_idx = df[df['ds'].dt.dayofweek >= 5].index
        weekend_values = df.loc[weekend_idx, column]
        
        # Umbral específico para fines de semana (más estricto)
        if len(weekend_values) > 3:  # Suficientes datos de fin de semana
            weekend_median = weekend_values.median()
            weekend_mad = np.median(np.abs(weekend_values - weekend_median)) * 1.4826
            
            if weekend_mad > 0:
                weekend_scores = np.abs(weekend_values - weekend_median) / weekend_mad
                weekend_outliers = weekend_idx[weekend_scores > 2.5].tolist()  # Umbral más estricto
                contextual_outliers.extend(weekend_outliers)
        
        # 4. Considerar eventos conocidos de ransomware (específico para el dominio)
        known_events = {
            # Ejemplos de grandes ataques conocidos
            '2017-05-12': 'WannaCry',  # WannaCry
            '2017-06-27': 'NotPetya',  # NotPetya
            '2021-07-02': 'Kaseya',    # Kaseya VSA
        }
        
        # Excluir eventos conocidos de outliers
        for date_str, event in known_events.items():
            date = pd.to_datetime(date_str)
            matching_idx = df[df['ds'] == date].index
            for idx in matching_idx:
                if idx in contextual_outliers:
                    contextual_outliers.remove(idx)
        
        self.logger.info(f"Detectados {len(contextual_outliers)} outliers contextuales")
        
        return {
            'indices': contextual_outliers,
            'stats': {
                'method': 'contextual_ransomware',
                'segment_count': len(dow_groups),
                'weekend_threshold': 2.5,
                'weekday_threshold': 3.5
            }
        }
    
    def determine_ransomware_outlier_strategy(self, df: pd.DataFrame, column: str = 'y') -> str:
        """
        Determina la mejor estrategia para outliers específicamente en series de ransomware
        
        Args:
            df: DataFrame con datos
            column: Columna a analizar
            
        Returns:
            Estrategia recomendada para ransomware
        """
        self.logger.info("Determinando estrategia específica para outliers de ransomware")
        
        if column not in df.columns:
            self.logger.error(f"Columna {column} no encontrada")
            return 'winsorize'  # Estrategia por defecto
            
        # 1. Analizar proporción de ceros - crucial en datos de ransomware
        zero_ratio = (df[column] == 0).mean()
        
        # 2. Detectar autocorrelación semanal (importante en ransomware)
        weekly_autocorr = 0
        if len(df) > 7:
            weekly_autocorr = df[column].autocorr(lag=7)
        
        # 3. Verificar distribución extrema (común en ataques)
        p95 = np.percentile(df[column], 95)
        median = df[column].median()
        extremity_ratio = p95 / (median + 0.001)  # Evitar división por cero
        
        # 4. Detectar tendencia reciente (últimos 30 días)
        recent_trend = 1.0
        if len(df) > 60:
            recent_avg = df[column].iloc[-30:].mean()
            previous_avg = df[column].iloc[-60:-30].mean() 
            if previous_avg > 0:
                recent_trend = recent_avg / previous_avg
            else:
                recent_trend = 1.0 if recent_avg == 0 else 2.0
        
        # Lógica específica para series de ransomware
        self.logger.info(f"Análisis de serie: zero_ratio={zero_ratio:.2f}, weekly_autocorr={weekly_autocorr:.2f}, extremity_ratio={extremity_ratio:.2f}")
        
        if zero_ratio > 0.7:  # Series muy dispersas con mayoría de ceros
            self.logger.info("Serie muy dispersa (>70% ceros): aplicando estrategia 'cap'")
            return 'cap'  # Mejor mantener los picos limitados pero presentes
        elif zero_ratio > 0.3:
            if extremity_ratio > 10:  # Distribución muy extrema
                self.logger.info("Serie moderadamente dispersa con valores extremos: aplicando 'winsorize'")
                return 'winsorize'  # Limitar valores extremos
            else:
                self.logger.info("Serie moderadamente dispersa sin extremos: aplicando 'median'")
                return 'median'  # Tendencia central sin eliminar
        elif abs(weekly_autocorr) > 0.6:  # Fuerte patrón semanal
            if recent_trend > 1.5:  # Tendencia creciente
                self.logger.info("Serie con fuerte patrón semanal y tendencia creciente: aplicando 'cap'")
                return 'cap'  # Preservar tendencia limitando extremos
            else:
                self.logger.info("Serie con fuerte patrón semanal sin tendencia creciente: aplicando 'winsorize'")
                return 'winsorize'
        else:
            # Para datos más regulares
            strategy = 'interpolate' if extremity_ratio < 5 else 'winsorize'
            self.logger.info(f"Serie regular: aplicando '{strategy}'")
            return strategy
    
    def prepare_zero_inflated_data(self, df: pd.DataFrame, column: str = 'y') -> pd.DataFrame:
        """
        Preprocesamiento especializado para series con muchos ceros (común en ransomware)
        
        Args:
            df: DataFrame con datos
            column: Columna a procesar
            
        Returns:
            DataFrame con transformaciones optimizadas para series con ceros
        """
        if column not in df.columns:
            self.logger.error(f"Columna {column} no encontrada")
            return df
            
        # Análisis inicial de la densidad de ceros
        zero_ratio = (df[column] == 0).mean()
        self.logger.info(f"Serie con {zero_ratio:.2%} de valores cero")
        
        df_processed = df.copy()
        
        # 1. Definir estrategia óptima según densidad de ceros
        if zero_ratio > 0.8:  # Series extremadamente dispersas
            self.logger.info("Aplicando estrategia para series extremadamente dispersas")
            
            # Crear indicador binario (ocurrencia de ataque)
            df_processed['attack_occurred'] = (df_processed[column] > 0).astype(int)
            
            # Aplicar transformación solo a valores no cero
            non_zero_mask = df_processed[column] > 0
            if non_zero_mask.sum() > 0:  # Si hay valores no cero
                df_processed.loc[non_zero_mask, 'value_when_positive'] = np.log1p(df_processed.loc[non_zero_mask, column])
            else:
                df_processed['value_when_positive'] = 0
            
            # Calcular promedio local (ventana 7 días) para estabilizar
            df_processed['local_avg'] = df_processed[column].rolling(window=7, min_periods=1, center=True).mean()
            
            # Usar transformación logarítmica modificada (más simple y robusta)
            df_processed['y_transformed'] = np.log1p(df_processed['local_avg'])
            
            # Guardar transformación para posterior inversión
            self.zero_transform_type = 'extreme_sparse'
            self.zero_ratio = zero_ratio
            
        elif zero_ratio > 0.5:  # Series moderadamente dispersas
            self.logger.info("Aplicando estrategia para series moderadamente dispersas")
            
            # Aplicar regularización con suavizado para estabilizar la serie
            window_size = max(7, int(len(df) * 0.05))  # Al menos 7 días o 5% de los datos
            
            # Suavizado adaptativo: menor ventana cerca de valores no cero
            smoothed_values = []
            for i in range(len(df)):
                # Calcular distancia al valor no cero más cercano
                non_zero_idx = np.where(df[column].values != 0)[0]
                if len(non_zero_idx) == 0:
                    # No hay valores no cero
                    smoothed_values.append(0)
                    continue
                    
                closest_non_zero = min(abs(non_zero_idx - i))
                # Ajustar ventana: más pequeña cerca de valores no cero
                adaptive_window = max(3, int(window_size / (closest_non_zero + 1)))
                
                # Calcular promedio local con ventana adaptativa
                start_idx = max(0, i - adaptive_window)
                end_idx = min(len(df), i + adaptive_window + 1)
                local_values = df[column].values[start_idx:end_idx]
                smoothed_values.append(np.mean(local_values))
            
            df_processed['y_smoothed'] = smoothed_values
            
            # Transformación logarítmica modificada para manejar ceros
            df_processed['y_transformed'] = np.log1p(df_processed['y_smoothed'])
            
            # Guardar transformación para posterior inversión
            self.zero_transform_type = 'moderate_sparse'
            self.zero_ratio = zero_ratio
            
        else:  # Series con densidad normal de ceros
            self.logger.info("Aplicando transformación estándar (log1p)")
            
            # Transformación logarítmica estándar
            df_processed['y_transformed'] = np.log1p(df_processed[column])
            
            # Guardar transformación para posterior inversión
            self.zero_transform_type = 'standard'
            self.zero_ratio = zero_ratio
        
        # Reemplazar valores para modelado
        df_processed['y_original'] = df_processed[column].copy()
        df_processed[column] = df_processed['y_transformed']
        
        return df_processed
    
    def invert_zero_inflated_transform(self, df: pd.DataFrame, column: str = 'y') -> pd.DataFrame:
        """
        Invierte la transformación aplicada a series con muchos ceros
        
        Args:
            df: DataFrame con predicciones transformadas
            column: Columna a procesar
            
        Returns:
            DataFrame con valores en escala original
        """
        if not hasattr(self, 'zero_transform_type'):
            self.logger.warning("No se encontró información de transformación, aplicando expm1 estándar")
            # Aplicar inversión estándar
            df_inverted = df.copy()
            for col in [c for c in df.columns if c.startswith('yhat') or c == column]:
                if col in df.columns:
                    df_inverted[f"{col}_original"] = np.expm1(df[col])
            return df_inverted
        
        df_inverted = df.copy()
        
        # Invertir según el tipo de transformación aplicada
        if self.zero_transform_type == 'extreme_sparse':
            for col in [c for c in df.columns if c.startswith('yhat') or c == column]:
                if col in df.columns:
                    # Aplicar inversión de log1p
                    df_inverted[f"{col}_original"] = np.expm1(df[col])
                    
                    # Ajustar predicciones muy pequeñas a cero
                    small_values_mask = df_inverted[f"{col}_original"] < (0.5 / (1 - self.zero_ratio))
                    df_inverted.loc[small_values_mask, f"{col}_original"] = 0
                    
        elif self.zero_transform_type == 'moderate_sparse':
            for col in [c for c in df.columns if c.startswith('yhat') or c == column]:
                if col in df.columns:
                    # Aplicar inversión de log1p
                    df_inverted[f"{col}_original"] = np.expm1(df[col])
                    
                    # Redondear pequeños valores a cero
                    small_values_mask = df_inverted[f"{col}_original"] < 0.5
                    df_inverted.loc[small_values_mask, f"{col}_original"] = 0
                    
        else:  # 'standard'
            for col in [c for c in df.columns if c.startswith('yhat') or c == column]:
                if col in df.columns:
                    df_inverted[f"{col}_original"] = np.expm1(df[col])
        
        return df_inverted
    
    def get_optimal_prophet_params_for_zero_inflated(self, zero_ratio: float) -> Dict:
        """
        Obtiene parámetros óptimos de Prophet para series con muchos ceros
        
        Args:
            zero_ratio: Proporción de ceros en la serie
            
        Returns:
            Dict con parámetros optimizados
        """
        if zero_ratio > 0.8:  # Series extremadamente dispersas
            return {
                'seasonality_mode': 'additive',       # Mejor para series con muchos ceros
                'changepoint_prior_scale': 0.01,      # Menos flexible para evitar sobreajuste
                'seasonality_prior_scale': 0.1,       # Estacionalidad débil
                'holidays_prior_scale': 5.0,          # Mayor énfasis en eventos especiales
                'n_changepoints': 10,                 # Pocos puntos de cambio
                'interval_width': 0.95,               # Intervalos más amplios
            }
        elif zero_ratio > 0.5:  # Series moderadamente dispersas
            return {
                'seasonality_mode': 'additive',       # Mejor para series con ceros
                'changepoint_prior_scale': 0.05,      # Flexibilidad moderada
                'seasonality_prior_scale': 1.0,       # Estacionalidad moderada
                'holidays_prior_scale': 10.0,         # Énfasis en eventos especiales
                'n_changepoints': 25,                 # Cantidad moderada
                'interval_width': 0.9,                # Intervalos estándar
            }
        else:  # Series con pocos ceros
            return {
                'seasonality_mode': 'multiplicative', # Mejor para series sin muchos ceros
                'changepoint_prior_scale': 0.2,       # Mayor flexibilidad
                'seasonality_prior_scale': 10.0,      # Estacionalidad fuerte
                'holidays_prior_scale': 10.0,         # Énfasis estándar en eventos
                'n_changepoints': 50,                 # Muchos puntos de cambio
                'interval_width': 0.8                 # Intervalos más ajustados
            }
