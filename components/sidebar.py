"""
=============================================================================
SIDEBAR COMPONENT - OLIST E-COMMERCE PROJECT
=============================================================================
Componente de navegación lateral.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from config import COLORS, SIDEBAR_STYLE, PROJECT_INFO, NAVIGATION_ITEMS


def create_sidebar():
    """
    Crea el componente sidebar con navegación.

    Returns:
    --------
    html.Div
        Componente sidebar completo
    """

    return html.Div([
        # Logo y título
        html.Div([
            html.Div([
                html.H2('🛒', style={
                    'fontSize': '64px',
                    'margin': '0',
                    'textAlign': 'center',
                    'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
                    'webkitBackgroundClip': 'text',
                    'webkitTextFillColor': 'transparent',
                    'backgroundClip': 'text'
                }),
                html.H3('Olist', style={
                    'textAlign': 'center',
                    'color': COLORS['primary'],
                    'margin': '10px 0',
                    'fontWeight': 'bold',
                    'fontSize': '28px'
                }),
                html.P('E-commerce Analytics', style={
                    'textAlign': 'center',
                    'color': COLORS['text_muted'],
                    'fontSize': '14px',
                    'margin': '0'
                })
            ], style={
                'padding': '20px',
                'background': f'rgba(0, 212, 255, 0.05)',
                'borderRadius': '12px',
                'border': f'1px solid {COLORS["primary"]}',
                'marginBottom': '30px'
            })
        ]),

        # Navegación
        html.Div([
            html.H5('📑 Navegación', style={
                'color': COLORS['text_muted'],
                'fontSize': '12px',
                'textTransform': 'uppercase',
                'letterSpacing': '2px',
                'marginBottom': '15px',
                'fontWeight': 'bold'
            }),

            html.Div([
                dcc.Link(
                    html.Div([
                        html.Div([
                            html.Span(item['icon'], style={
                                'fontSize': '20px',
                                'marginRight': '12px',
                                'display': 'inline-block',
                                'width': '24px',
                                'textAlign': 'center'
                            }),
                            html.Span(item['label'], style={
                                'fontSize': '14px',
                                'fontWeight': '500',
                                'display': 'inline-block',
                                'verticalAlign': 'middle'
                            })
                        ], style={
                            'display': 'flex',
                            'alignItems': 'center',
                            'color': COLORS['text']
                        })
                    ], style={
                        'padding': '15px 20px',
                        'margin': '8px 0',
                        'borderRadius': '10px',
                        'cursor': 'pointer',
                        'transition': 'all 0.3s ease',
                        'border': f'1px solid {COLORS["border"]}',
                        'background': 'transparent'
                    }, id=f"nav-{item['id']}"),
                    href=item['path'],
                    style={'textDecoration': 'none'}
                )
                for item in NAVIGATION_ITEMS
            ], id='nav-items-container')
        ], style={'marginBottom': '30px'}),

        html.Hr(style={
            'borderColor': COLORS['border'],
            'margin': '30px 0'
        }),

        # Información del proyecto
        html.Div([
            html.H5('📊 Info del Proyecto', style={
                'color': COLORS['text_muted'],
                'fontSize': '12px',
                'textTransform': 'uppercase',
                'letterSpacing': '2px',
                'marginBottom': '15px',
                'fontWeight': 'bold'
            }),

            html.Div([
                html.Div([
                    html.Span('📦 ', style={'fontSize': '18px', 'marginRight': '8px'}),
                    html.Span('Registros: ', style={'color': COLORS['text_muted'], 'fontSize': '13px'}),
                    html.Strong(PROJECT_INFO['total_records'], style={'color': COLORS['primary'], 'fontSize': '14px'})
                ], style={'marginBottom': '10px', 'color': COLORS['text']}),

                html.Div([
                    html.Span('📅 ', style={'fontSize': '18px', 'marginRight': '8px'}),
                    html.Span('Período: ', style={'color': COLORS['text_muted'], 'fontSize': '13px'}),
                    html.Strong(PROJECT_INFO['period'], style={'color': COLORS['success'], 'fontSize': '14px'})
                ], style={'marginBottom': '10px', 'color': COLORS['text']}),

                html.Div([
                    html.Span('🏪 ', style={'fontSize': '18px', 'marginRight': '8px'}),
                    html.Span('Sellers: ', style={'color': COLORS['text_muted'], 'fontSize': '13px'}),
                    html.Strong(PROJECT_INFO['total_sellers'], style={'color': COLORS['warning'], 'fontSize': '14px'})
                ], style={'marginBottom': '10px', 'color': COLORS['text']}),

                html.Div([
                    html.Span('🎯 ', style={'fontSize': '18px', 'marginRight': '8px'}),
                    html.Span('Target: ', style={'color': COLORS['text_muted'], 'fontSize': '13px'}),
                    html.Strong(PROJECT_INFO['target_accuracy'], style={'color': COLORS['danger'], 'fontSize': '14px'})
                ], style={'color': COLORS['text']})
            ], style={
                'padding': '15px',
                'background': f'rgba(123, 44, 191, 0.1)',
                'borderRadius': '10px',
                'border': f'1px solid {COLORS["secondary"]}'
            })
        ]),

        html.Hr(style={
            'borderColor': COLORS['border'],
            'margin': '30px 0'
        }),

        # Footer
        html.Div([
            html.Small('© 2024 Olist Analytics', style={
                'color': COLORS['text_muted'],
                'display': 'block',
                'textAlign': 'center',
                'fontSize': '11px'
            }),
            html.Small('Proyecto de Análisis', style={
                'color': COLORS['text_muted'],
                'display': 'block',
                'textAlign': 'center',
                'fontSize': '11px',
                'marginTop': '5px'
            })
        ])

    ], style=SIDEBAR_STYLE, id='sidebar')