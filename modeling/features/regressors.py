import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

class RegressorGenerator:
    """
    Genera y selecciona regresores para el modelo de Prophet basado en 
    su relevancia estadística y estabilidad.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.selected_regressors = []
    
    def generate_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera características basadas en fechas
        
        Args:
            df: DataFrame con columna 'ds'
            
        Returns:
            DataFrame con características adicionales
        """
        if 'ds' not in df.columns:
            raise ValueError("El DataFrame debe tener una columna 'ds'")
            
        df_features = df.copy()
        
        # Características básicas de fecha
        df_features['month'] = df_features['ds'].dt.month
        df_features['dayofweek'] = df_features['ds'].dt.dayofweek
        df_features['quarter'] = df_features['ds'].dt.quarter
        df_features['year'] = df_features['ds'].dt.year
        df_features['day'] = df_features['ds'].dt.day
        df_features['week'] = df_features['ds'].dt.isocalendar().week
        
        # Características derivadas
        df_features['is_weekend'] = (df_features['dayofweek'] >= 5).astype(int)
        df_features['is_month_start'] = df_features['ds'].dt.is_month_start.astype(int)
        df_features['is_month_end'] = df_features['ds'].dt.is_month_end.astype(int)
        df_features['is_quarter_start'] = df_features['ds'].dt.is_quarter_start.astype(int)
        df_features['is_quarter_end'] = df_features['ds'].dt.is_quarter_end.astype(int)
        df_features['is_year_start'] = df_features['ds'].dt.is_year_start.astype(int)
        df_features['is_year_end'] = df_features['ds'].dt.is_year_end.astype(int)
        
        return df_features
    
    def add_holidays_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade efectos de días festivos, incluyendo vísperas y días posteriores
        
        Args:
            df: DataFrame con características de fecha
            
        Returns:
            DataFrame con características de días festivos
        """
        # Implementación simplificada, ya que Prophet maneja holidays internamente
        return df
    
    def add_external_regressors(self, df: pd.DataFrame, 
                               cve_data: Optional[pd.DataFrame] = None, 
                               other_regressors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Añade regresores externos al DataFrame
        
        Args:
            df: DataFrame principal
            cve_data: DataFrame con datos de CVEs
            other_regressors: Diccionario de regresores adicionales
            
        Returns:
            DataFrame con regresores añadidos
        """
        df_result = df.copy()
        
        # Añadir datos de CVEs si están disponibles
        if cve_data is not None and not cve_data.empty:
            self.logger.info("Añadiendo datos de CVEs como regresores")
            # Verificar columnas mínimas necesarias
            if 'ds' in cve_data.columns:
                # Buscar columnas numéricas para usar como regresores
                numeric_cols = cve_data.select_dtypes(include=[np.number]).columns
                
                if numeric_cols.empty:
                    self.logger.warning("Datos de CVEs no contienen columnas numéricas")
                else:
                    # Fusionar con el DataFrame principal
                    for col in numeric_cols:
                        if col != 'ds':  # Evitar duplicar columna ds
                            # Crear un DataFrame temporal con solo ds y la columna numérica
                            temp_df = cve_data[['ds', col]].copy()
                            # Fusionar con el DataFrame principal
                            df_result = pd.merge(df_result, temp_df, on='ds', how='left')
                            # Rellenar valores faltantes
                            df_result[col] = df_result[col].fillna(df_result[col].median())
        
        # Añadir otros regresores externos
        if other_regressors:
            for name, regressors_df in other_regressors.items():
                if 'ds' in regressors_df.columns:
                    # Fusionar con DataFrame principal
                    df_result = pd.merge(df_result, regressors_df, on='ds', how='left')
        
        return df_result
        
    def select_optimal_regressors(self, df: pd.DataFrame, 
                               target_col: str = 'y',
                               correlation_threshold: float = 0.1, 
                               vif_threshold: float = 5.0, 
                               max_regressors: int = 10) -> List[str]:
        """
        Selecciona regresores óptimos basados en correlación, multicolinealidad e importancia
        
        Args:
            df: DataFrame con regresores candidatos
            target_col: Columna objetivo para predecir
            correlation_threshold: Umbral mínimo de correlación absoluta
            vif_threshold: Umbral máximo de Factor de Inflación de Varianza
            max_regressors: Número máximo de regresores a seleccionar
            
        Returns:
            Lista de nombres de regresores seleccionados
        """
        # Verificar columnas
        if target_col not in df.columns:
            raise ValueError(f"Columna objetivo '{target_col}' no encontrada en el DataFrame")
            
        # Identificar posibles regresores (columnas numéricas excepto ds y y)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        potential_regressors = [col for col in numeric_cols if col not in ['y', target_col]]
        
        if not potential_regressors:
            self.logger.warning("No se encontraron regresores potenciales")
            return []
        
        # PASO 1: EVALUACIÓN INICIAL DE REGRESORES
        # ----------------------------------------
        # 1.1 Correlación y Mutual Information
        correlations = {}
        mi_scores = {}
        stability_scores = {}
        
        for col in potential_regressors:
            # Correlación de Spearman (robusta a valores atípicos y relaciones no lineales)
            corr = df[col].corr(df[target_col], method='spearman')
            correlations[col] = abs(corr)
            
            # Mutual Information (captura relaciones no lineales)
            try:
                # Preparar datos para mutual information
                X = df[col].values.reshape(-1, 1)
                y = df[target_col].values
                
                # Calcular mutual information entre el regresor y el target
                mi = mutual_info_regression(X, y, random_state=42)[0]
                mi_scores[col] = mi
            except Exception as e:
                self.logger.warning(f"Error al calcular mutual information para {col}: {str(e)}")
                mi_scores[col] = 0
            
            # Evaluación de estabilidad del regresor (usando submuestras)
            try:
                stability = self._evaluate_regressor_stability(df, col, target_col)
                stability_scores[col] = stability
            except Exception as e:
                self.logger.warning(f"Error al evaluar estabilidad para {col}: {str(e)}")
                stability_scores[col] = 0
        
        # Normalizar puntuaciones de MI y estabilidad
        if mi_scores:
            max_mi = max(mi_scores.values()) if mi_scores.values() else 1
            mi_scores = {k: v/max_mi for k, v in mi_scores.items()}
        
        if stability_scores:
            max_stab = max(stability_scores.values()) if stability_scores.values() else 1
            stability_scores = {k: v/max_stab for k, v in stability_scores.items()}
        
        # Calcular puntuación combinada (correlación + MI + estabilidad)
        combined_scores = {}
        for col in potential_regressors:
            # Ponderación: 40% correlación + 40% MI + 20% estabilidad
            combined_scores[col] = (
                0.4 * correlations.get(col, 0) + 
                0.4 * mi_scores.get(col, 0) + 
                0.2 * stability_scores.get(col, 0)
            )
        
        # Filtrar regresores por puntuación combinada
        min_score = correlation_threshold * 0.4  # Proporcional al umbral de correlación
        filtered_regressors = [col for col, score in combined_scores.items() 
                              if score >= min_score]
        
        self.logger.info(f"Regresores con puntuación combinada > {min_score:.3f}: {len(filtered_regressors)} de {len(potential_regressors)}")
        
        if not filtered_regressors:
            self.logger.warning(f"Ningún regresor supera el umbral de puntuación combinada ({min_score:.3f})")
            # Retornar los top 3 regresores por puntuación combinada si ninguno supera el umbral
            if potential_regressors:
                top_regressors = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                return [x[0] for x in top_regressors]
            return []
        
        # PASO 2: SELECCIÓN FINAL CON CONTROL DE MULTICOLINEALIDAD
        # --------------------------------------------------------
        # Ordenar regresores por puntuación combinada
        sorted_regressors = sorted(
            [(col, combined_scores[col]) for col in filtered_regressors], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Seleccionar regresores controlando multicolinealidad
        selected_regressors = []
        remaining = sorted_regressors.copy()
        
        # Comenzar con el regresor mejor puntuado
        if remaining:
            selected_regressors.append(remaining[0][0])
            remaining = remaining[1:]
        
        # Añadir regresores adicionales mientras se controla multicolinealidad
        while remaining and len(selected_regressors) < max_regressors:
            vif_too_high = False
            
            for i, (candidate, _) in enumerate(remaining[:]):
                test_set = selected_regressors + [candidate]
                
                if len(test_set) >= 2:  # VIF requiere al menos 2 variables
                    try:
                        # Preparar datos para VIF
                        X = df[test_set].copy()
                        # Rellenar NaN para evitar errores
                        X = X.fillna(X.median())
                        
                        # Calcular VIF para cada regresor
                        vif = pd.DataFrame()
                        vif["Variable"] = X.columns
                        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                        
                        # Verificar si algún VIF excede el umbral
                        if vif["VIF"].max() > vif_threshold:
                            vif_too_high = True
                            break
                    except Exception as e:
                        self.logger.warning(f"Error al calcular VIF: {str(e)}")
                        continue
            
            if vif_too_high:
                # Si hay multicolinealidad, pasar al siguiente candidato
                remaining.pop(0)
            else:
                # Añadir el candidato a los regresores seleccionados
                selected_regressors.append(remaining[0][0])
                remaining.pop(0)
        
        self.logger.info(f"Regresores seleccionados finales: {selected_regressors}")
        self.selected_regressors = selected_regressors
        return selected_regressors
    
    def _evaluate_regressor_stability(self, df, regressor_col, target_col, n_samples=5, sample_size=0.8):
        """
        Evalúa la estabilidad del regresor usando múltiples submuestras
        
        Args:
            df: DataFrame con datos
            regressor_col: Nombre del regresor a evaluar
            target_col: Nombre de la columna objetivo
            n_samples: Número de submuestras a evaluar
            sample_size: Tamaño de cada submuestra (fracción del dataset)
            
        Returns:
            Puntuación de estabilidad (0-1, donde 1 es más estable)
        """
        # Lista para almacenar correlaciones en cada submuestra
        correlations = []
        
        # Generar múltiples submuestras
        for _ in range(n_samples):
            # Muestra aleatoria
            sample_idx = np.random.choice(
                df.index, 
                size=int(len(df) * sample_size), 
                replace=False
            )
            sample_df = df.loc[sample_idx]
            
            # Calcular correlación en la submuestra
            corr = sample_df[regressor_col].corr(sample_df[target_col], method='spearman')
            correlations.append(abs(corr))
        
        # Calcular consistencia de las correlaciones
        if not correlations:
            return 0
        
        # Usar el coeficiente de variación (invertido) como medida de estabilidad
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        # Evitar división por cero
        if mean_corr == 0:
            return 0
        
        # Coeficiente de variación (menor es mejor, así que lo invertimos)
        cv = std_corr / mean_corr
        stability = 1 / (1 + cv)  # Transformación para que esté entre 0 y 1
        
        return stability
