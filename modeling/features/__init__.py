"""
Submódulo para la ingeniería de características, incluyendo detección de outliers
y generación de regresores para el modelo.
"""

from .outliers import OutlierDetector
from .regressors import RegressorGenerator

__all__ = ['OutlierDetector', 'RegressorGenerator']
