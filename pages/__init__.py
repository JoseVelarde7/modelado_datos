"""
Pages package initialization
"""

from .home import create_home_content
from .definicion_problema import create_definicion_content
from .analisis_estadistico import create_analisis_content

__all__ = [
    'create_home_content',
    'create_definicion_content',
    'create_analisis_content'
]
