import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st

class DataLoader:
    """
    Carga y valida los datos de entrada para el modelo de predicción de ransomware.
    Incluye métodos para cargar datos desde JSON o CSV y validar su estructura.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @st.cache_data(ttl=3600)
    def load_ransomware_data(_self, file_path: str) -> pd.DataFrame:
        """
        Carga datos de ataques ransomware desde un archivo JSON o CSV.
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            DataFrame con los datos cargados y validados
        
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si los datos no cumplen con el esquema requerido
        """
        # Validar existencia de archivos
        if not os.path.exists(file_path):
            error_msg = f"El archivo {file_path} no existe"
            _self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        _self.logger.info(f"Cargando datos de ransomware desde {file_path}")
        
        # Determinar formato por extensión
        if file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Formato de archivo no soportado: {file_path}")
            
        # Validar esquema básico
        required_columns = ['fecha']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Faltan columnas requeridas: {missing_columns}"
            _self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Convertir fecha a datetime
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
            
        return df
    
    @st.cache_data(ttl=3600)
    def load_cve_data(_self, file_path: str) -> pd.DataFrame:
        """
        Carga datos de CVEs para usar como regresores.
        
        Args:
            file_path: Ruta al archivo de CVEs
            
        Returns:
            DataFrame con los datos de CVEs
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato de los datos no es el esperado
        """
        if not os.path.exists(file_path):
            _self.logger.warning(f"Archivo de CVEs {file_path} no encontrado")
            raise FileNotFoundError(f"El archivo de CVEs {file_path} no existe")
            
        _self.logger.info(f"Cargando datos de CVEs desde {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Validar columna de fecha
            if 'ds' not in df.columns:
                _self.logger.warning("El archivo de CVEs no contiene columna 'ds'")
                
                # Buscar una columna de fecha alternativa
                date_columns = [col for col in df.columns if 'date' in col.lower() or 'fecha' in col.lower()]
                if date_columns:
                    _self.logger.info(f"Usando columna {date_columns[0]} como fecha")
                    df = df.rename(columns={date_columns[0]: 'ds'})
                else:
                    raise ValueError("No se encontró columna de fecha en el archivo de CVEs")
                
            # Convertir a datetime
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Validar que tenga al menos una columna con datos numéricos para usar como regresor
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                raise ValueError("El archivo de CVEs no contiene columnas numéricas para usar como regresores")
                
            return df
            
        except pd.errors.EmptyDataError:
            _self.logger.error("El archivo de CVEs está vacío")
            raise ValueError("El archivo de CVEs está vacío")
            
        except pd.errors.ParserError:
            _self.logger.error("Error al parsear el archivo de CVEs, formato incorrecto")
            raise ValueError("Formato incorrecto en el archivo de CVEs")
            
        except Exception as e:
            # Para errores inesperados, registrar el error pero evitar interrumpir completamente el flujo
            _self.logger.error(f"Error desconocido al cargar datos de CVEs: {str(e)}")
            # Crear un mensaje detallado para depuración
            error_msg = f"Error al cargar datos de CVEs: {str(e)}\n"
            error_msg += "Se continuará sin datos de regresores externos."
            _self.logger.warning(error_msg)
            
            # Devolver un DataFrame con estructura pero sin datos
            # Esto permite que el flujo continúe pero marca claramente que hubo un problema
            return pd.DataFrame(columns=['ds', 'cve_count'])
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Valida que el DataFrame tenga una estructura adecuada
        
        Args:
            df: DataFrame a validar
            
        Returns:
            bool: True si el DataFrame es válido, False en caso contrario
        """
        try:
            # Verificar que el DataFrame no está vacío
            if df is None or df.empty:
                self.logger.error("El DataFrame está vacío")
                return False
                
            # Verificar tipos de datos esperados
            if 'fecha' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['fecha']):
                    self.logger.error("La columna 'fecha' debe ser de tipo datetime")
                    return False
                    
            if 'ataques' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['ataques']):
                    self.logger.error("La columna 'ataques' debe ser numérica")
                    return False
            
            # Verificar si hay valores negativos en columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if (df[col] < 0).any():
                    self.logger.warning(f"La columna {col} contiene valores negativos")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error durante la validación de datos: {str(e)}")
            return False
