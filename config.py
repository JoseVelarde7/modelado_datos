"""
=============================================================================
CONFIGURACIÓN GLOBAL - OLIST E-COMMERCE PROJECT
=============================================================================
Colores, estilos y constantes utilizadas en toda la aplicación.
"""

# =============================================================================
# COLORES DEL TEMA (MODO OSCURO)
# =============================================================================
COLORS = {
    'background': '#0a0e27',
    'card': '#1a1f3a',
    'primary': '#00d4ff',
    'secondary': '#7b2cbf',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'info': '#3b82f6',
    'text': '#e2e8f0',
    'text_muted': '#94a3b8',
    'border': '#2d3748'
}

# =============================================================================
# ESTILOS DEL SIDEBAR
# =============================================================================
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '280px',
    'padding': '30px 20px',
    'background': COLORS['card'],
    'borderRight': f'3px solid {COLORS["primary"]}',
    'overflowY': 'auto',
    'zIndex': 1000
}

# =============================================================================
# ESTILOS DEL CONTENIDO PRINCIPAL
# =============================================================================
CONTENT_STYLE = {
    'marginLeft': '300px',
    'marginRight': '20px',
    'padding': '30px',
    'background': COLORS['background'],
    'minHeight': '100vh',
    'color': COLORS['text']  # 🔧 AGREGADO: Color de texto por defecto
}

# =============================================================================
# ESTILOS DE NAVEGACIÓN (CORREGIDO)
# =============================================================================
NAV_ITEM_STYLE = {
    'padding': '15px 20px',
    'margin': '8px 0',
    'borderRadius': '10px',
    'cursor': 'pointer',
    'transition': 'all 0.3s ease',
    'border': f'1px solid {COLORS["border"]}',
    'background': 'transparent',
    'color': COLORS['text']  # 🔧 AGREGADO
}

NAV_ITEM_ACTIVE_STYLE = {
    'padding': '15px 20px',
    'margin': '8px 0',
    'borderRadius': '10px',
    'cursor': 'pointer',
    'transition': 'all 0.3s ease',
    'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
    'border': f'1px solid {COLORS["primary"]}',
    'boxShadow': f'0 0 20px {COLORS["primary"]}33',
    'transform': 'translateX(5px)',
    'color': COLORS['text']  # 🔧 AGREGADO
}

# =============================================================================
# ESTILOS DE CARDS
# =============================================================================
CARD_STYLE = {
    'background': COLORS['card'],
    'border': f'2px solid {COLORS["border"]}',
    'borderRadius': '12px',
    'marginBottom': '20px',
    'transition': 'all 0.3s ease',
    'color': COLORS['text']  # 🔧 AGREGADO
}

CARD_HEADER_STYLE = {
    'background': COLORS['card'],
    'borderBottom': f'2px solid {COLORS["primary"]}',
    'padding': '20px',
    'color': COLORS['text']  # 🔧 AGREGADO
}

# =============================================================================
# INFORMACIÓN DEL PROYECTO
# =============================================================================
PROJECT_INFO = {
    'title': 'Análisis E-commerce Olist',
    'subtitle': 'Satisfacción del Cliente 2016-2018',
    'period': '2016-2018',
    'total_records': '110,013',
    'total_sellers': '~3,000',
    'target_accuracy': '>80%'
}

# =============================================================================
# ESTRUCTURA DE NAVEGACIÓN
# =============================================================================
NAVIGATION_ITEMS = [
    {'label': 'Inicio', 'icon': '🏠', 'path': '/', 'id': 'home'},
    {'label': 'Definición del Problema', 'icon': '📋', 'path': '/definicion', 'id': 'definicion'},
    {'label': 'Análisis Estadístico', 'icon': '📊', 'path': '/analisis', 'id': 'analisis'},
    {'label': 'Técnica Analítica', 'icon': '🔬', 'path': '/tecnica', 'id': 'tecnica'},
    {'label': 'Comparación', 'icon': '⚖️', 'path': '/comparacion', 'id': 'comparacion'},
    {'label': 'Optimización', 'icon': '🎯', 'path': '/optimizacion', 'id': 'optimizacion'}
]

# =============================================================================
# CONFIGURACIÓN DE PLOTLY
# =============================================================================
PLOTLY_CONFIG = {
    'displayModeBar': False,
    'responsive': True
}

PLOTLY_LAYOUT_TEMPLATE = {
    'paper_bgcolor': COLORS['background'],
    'plot_bgcolor': COLORS['card'],
    'font': {'color': COLORS['text'], 'size': 12},
    'title': {
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'color': COLORS['text']}
    },
    'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50}
}
