"""
=============================================================================
HEADER COMPONENT - OLIST E-COMMERCE PROJECT
=============================================================================
Componente de encabezado para cada página.
"""

from dash import html
from config import COLORS


def create_page_header(title, subtitle=None, icon='📊', show_divider=True):
    """
    Crea un encabezado estándar para las páginas.

    Parameters:
    -----------
    title : str
        Título principal de la página
    subtitle : str, optional
        Subtítulo descriptivo
    icon : str
        Emoji/icono para el título
    show_divider : bool
        Si mostrar línea divisoria

    Returns:
    --------
    html.Div
        Componente de encabezado
    """

    components = [
        html.H1([
            html.Span(icon, style={
                'fontSize': '56px',
                'marginRight': '20px',
                'verticalAlign': 'middle'
            }),
            title
        ], style={
            'textAlign': 'center',
            'color': COLORS['primary'],
            'marginBottom': '10px' if subtitle else '30px',
            'fontWeight': 'bold',
            'fontSize': '42px'
        })
    ]

    if subtitle:
        components.append(
            html.P(subtitle, style={
                'textAlign': 'center',
                'color': COLORS['text_muted'],
                'fontSize': '18px',
                'marginBottom': '30px',
                'lineHeight': '1.6'
            })
        )

    if show_divider:
        components.append(
            html.Hr(style={
                'borderColor': COLORS['primary'],
                'borderWidth': '2px',
                'marginBottom': '40px',
                'width': '80%',
                'margin': '30px auto'
            })
        )

    return html.Div(components)


def create_section_header(title, icon='📌', color=None):
    """
    Crea un encabezado para subsecciones dentro de una página.

    Parameters:
    -----------
    title : str
        Título de la sección
    icon : str
        Emoji/icono
    color : str, optional
        Color personalizado (por defecto usa secondary)

    Returns:
    --------
    html.Div
        Componente de encabezado de sección
    """

    section_color = color or COLORS['secondary']

    return html.Div([
        html.H3([
            html.Span(icon, style={
                'fontSize': '36px',
                'marginRight': '15px',
                'verticalAlign': 'middle'
            }),
            title
        ], style={
            'color': section_color,
            'marginBottom': '20px',
            'fontWeight': 'bold',
            'paddingBottom': '10px',
            'borderBottom': f'2px solid {section_color}'
        })
    ], style={'marginTop': '40px', 'marginBottom': '30px'})


def create_info_banner(message, icon='ℹ️', banner_type='info'):
    """
    Crea un banner informativo.

    Parameters:
    -----------
    message : str
        Mensaje a mostrar
    icon : str
        Emoji/icono
    banner_type : str
        Tipo: 'info', 'success', 'warning', 'danger'

    Returns:
    --------
    html.Div
        Banner informativo
    """

    colors_map = {
        'info': COLORS['info'],
        'success': COLORS['success'],
        'warning': COLORS['warning'],
        'danger': COLORS['danger']
    }

    banner_color = colors_map.get(banner_type, COLORS['info'])

    return html.Div([
        html.Div([
            html.Span(icon, style={
                'fontSize': '28px',
                'marginRight': '15px'
            }),
            html.Span(message, style={
                'fontSize': '16px',
                'lineHeight': '1.6'
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'color': COLORS['text']
        })
    ], style={
        'padding': '20px',
        'background': f'rgba({int(banner_color[1:3], 16)}, {int(banner_color[3:5], 16)}, {int(banner_color[5:7], 16)}, 0.1)',
        'border': f'2px solid {banner_color}',
        'borderRadius': '12px',
        'marginBottom': '30px'
    })


def create_breadcrumb(items):
    """
    Crea breadcrumb de navegación.

    Parameters:
    -----------
    items : list of dict
        Lista con {'label': str, 'active': bool}

    Returns:
    --------
    html.Div
        Breadcrumb de navegación
    """

    breadcrumb_items = []

    for i, item in enumerate(items):
        if item.get('active', False):
            breadcrumb_items.append(
                html.Span(item['label'], style={
                    'color': COLORS['primary'],
                    'fontWeight': 'bold'
                })
            )
        else:
            breadcrumb_items.append(
                html.Span(item['label'], style={
                    'color': COLORS['text_muted']
                })
            )

        if i < len(items) - 1:
            breadcrumb_items.append(
                html.Span(' → ', style={
                    'color': COLORS['text_muted'],
                    'margin': '0 10px'
                })
            )

    return html.Div(
        breadcrumb_items,
        style={
            'fontSize': '14px',
            'marginBottom': '30px',
            'padding': '10px 20px',
            'background': COLORS['card'],
            'borderRadius': '8px',
            'border': f'1px solid {COLORS["border"]}'
        }
    )
