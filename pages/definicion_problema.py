"""
=============================================================================
DEFINICIÓN DEL PROBLEMA PAGE
=============================================================================
Página con la pregunta de investigación, objetivos y alcance.
"""

from dash import html
import dash_bootstrap_components as dbc
from config import COLORS
from components.header import create_page_header, create_section_header, create_info_banner


def create_definicion_content():
    """
    Crea el contenido de la página de definición del problema.

    Returns:
    --------
    html.Div
        Contenido completo de la página
    """

    return html.Div([

        # Header
        create_page_header(
            title='Definición del Problema',
            subtitle='Planteamiento de la investigación y objetivos del análisis',
            icon='📋'
        ),

        # Banner informativo
        # create_info_banner(
        #     'Esta sección establece el marco conceptual y los objetivos que guían todo el análisis',
        #     icon='💡',
        #     banner_type='info'
        # ),

        # Pregunta de investigación
        create_section_header('❓ Pregunta de Investigación', color=COLORS['primary']),

        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.P([
                        html.Strong('¿Cómo se comportan los clientes del e-commerce Olist en Brasil ',
                                    style={'fontSize': '20px', 'color': COLORS['primary']}),
                        'en términos de ',
                        html.Strong('SATISFACCIÓN, PATRONES DE COMPRA y PREFERENCIAS, '),
                        'y qué ',
                        html.Strong('FACTORES OPERACIONALES y GEOGRÁFICOS '),
                        'determinan una experiencia exitosa durante el período 2016-2018?'
                    ], style={
                        'fontSize': '18px',
                        'lineHeight': '2',
                        'textAlign': 'center',
                        'padding': '30px'
                    })
                ], style={
                    'background': f'linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 44, 191, 0.1) 100%)',
                    'borderRadius': '12px',
                    'border': f'2px solid {COLORS["primary"]}'
                })
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '40px'}),

        # Objetivos
        create_section_header('🎯 Objetivos del Análisis', color=COLORS['success']),

        # Objetivo General
        dbc.Card([
            dbc.CardHeader([
                html.H4('🎯 Objetivo General', style={'margin': '0', 'color': COLORS['success']})
            ], style={'background': COLORS['card'], 'borderBottom': f'2px solid {COLORS["success"]}'}),
            dbc.CardBody([
                html.P([
                    'Predecir la ',
                    html.Strong('satisfacción del cliente '),
                    'en el marketplace Olist mediante el análisis de variables operacionales, transaccionales y geográficas, ',
                    'alcanzando una precisión ',
                    html.Strong('superior al 80% '),
                    'para identificar factores críticos que impulsen mejoras estratégicas en la experiencia del cliente.'
                ], style={'fontSize': '16px', 'lineHeight': '1.8'})
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'marginBottom': '30px'}),

        # Objetivos Específicos
        dbc.Card([
            dbc.CardHeader([
                html.H4('📌 Objetivos Específicos', style={'margin': '0', 'color': COLORS['warning']})
            ], style={'background': COLORS['card'], 'borderBottom': f'2px solid {COLORS["warning"]}'}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H5('📊 Descriptivo', style={'color': COLORS['primary'], 'marginBottom': '15px'}),
                            html.P(
                                'Caracterizar el comportamiento de compra, patrones de satisfacción y distribución geográfica de clientes y vendedores.',
                                style={'fontSize': '15px', 'lineHeight': '1.6'})
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H5('🔍 Exploratorio', style={'color': COLORS['info'], 'marginBottom': '15px'}),
                            html.P(
                                'Identificar relaciones entre variables operacionales (tiempos de entrega, precios) y niveles de satisfacción.',
                                style={'fontSize': '15px', 'lineHeight': '1.6'})
                        ])
                    ], width=6)
                ], style={'marginBottom': '20px'}),

                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H5('🔬 Inferencial', style={'color': COLORS['success'], 'marginBottom': '15px'}),
                            html.P(
                                'Validar hipótesis sobre el impacto de variables clave en la satisfacción mediante pruebas estadísticas.',
                                style={'fontSize': '15px', 'lineHeight': '1.6'})
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H5('🤖 Predictivo', style={'color': COLORS['danger'], 'marginBottom': '15px'}),
                            html.P(
                                'Desarrollar modelos de machine learning para predecir satisfacción con accuracy >80% y recomendar acciones.',
                                style={'fontSize': '15px', 'lineHeight': '1.6'})
                        ])
                    ], width=6)
                ])
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["warning"]}', 'marginBottom': '40px'}),

        # Tipo de análisis
        create_section_header('📈 Tipo de Análisis', color=COLORS['secondary']),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3('📊', style={'fontSize': '56px', 'textAlign': 'center'}),
                        html.H4('Descriptivo',
                                style={'color': COLORS['primary'], 'textAlign': 'center', 'marginBottom': '15px'}),
                        html.P('Resumen de datos históricos mediante estadísticas y visualizaciones',
                               style={'textAlign': 'center', 'color': COLORS['text_muted']})
                    ])
                ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'height': '100%'})
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3('🔍', style={'fontSize': '56px', 'textAlign': 'center'}),
                        html.H4('Exploratorio',
                                style={'color': COLORS['success'], 'textAlign': 'center', 'marginBottom': '15px'}),
                        html.P('Identificación de patrones, correlaciones y anomalías en los datos',
                               style={'textAlign': 'center', 'color': COLORS['text_muted']})
                    ])
                ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'height': '100%'})
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3('🔬', style={'fontSize': '56px', 'textAlign': 'center'}),
                        html.H4('Inferencial',
                                style={'color': COLORS['warning'], 'textAlign': 'center', 'marginBottom': '15px'}),
                        html.P('Validación de hipótesis mediante pruebas estadísticas rigurosas',
                               style={'textAlign': 'center', 'color': COLORS['text_muted']})
                    ])
                ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["warning"]}', 'height': '100%'})
            ], width=4)
        ], style={'marginBottom': '40px'}),

        # Dataset y variables
        create_section_header('📦 Dataset y Variables', color=COLORS['info']),

        dbc.Card([
            dbc.CardBody([
                # KPIs del dataset
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('110,013',
                                    style={'color': COLORS['primary'], 'fontSize': '32px', 'fontWeight': 'bold'}),
                            html.P('Registros', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3('2016-2018',
                                    style={'color': COLORS['success'], 'fontSize': '32px', 'fontWeight': 'bold'}),
                            html.P('Período', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3('~3,000',
                                    style={'color': COLORS['warning'], 'fontSize': '32px', 'fontWeight': 'bold'}),
                            html.P('Vendedores', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3('50+', style={'color': COLORS['danger'], 'fontSize': '32px', 'fontWeight': 'bold'}),
                            html.P('Variables', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center'})
                    ], width=3)
                ], style={'marginBottom': '30px'}),

                html.Hr(style={'borderColor': COLORS['border']}),

                # Categorías de variables
                html.H5('📊 Variables por Categoría:',
                        style={'color': COLORS['primary'], 'marginTop': '20px', 'marginBottom': '20px'}),

                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6('💰 Transaccionales', style={'color': COLORS['warning'], 'marginBottom': '10px'}),
                            html.Ul([
                                html.Li('price'),
                                html.Li('payment_value'),
                                html.Li('order_total_value'),
                                html.Li('freight_value'),
                                html.Li('payment_installments')
                            ], style={'fontSize': '14px', 'color': COLORS['text_muted']})
                        ])
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H6('🚚 Operacionales', style={'color': COLORS['info'], 'marginBottom': '10px'}),
                            html.Ul([
                                html.Li('delivery_time_days'),
                                html.Li('delivery_delay_days'),
                                html.Li('on_time_delivery'),
                                html.Li('shipping_limit_date')
                            ], style={'fontSize': '14px', 'color': COLORS['text_muted']})
                        ])
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H6('⭐ Satisfacción', style={'color': COLORS['success'], 'marginBottom': '10px'}),
                            html.Ul([
                                html.Li('review_score'),
                                html.Li('satisfaction_level'),
                                html.Li('review_comment_message')
                            ], style={'fontSize': '14px', 'color': COLORS['text_muted']})
                        ])
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H6('📦 Producto', style={'color': COLORS['danger'], 'marginBottom': '10px'}),
                            html.Ul([
                                html.Li('product_category'),
                                html.Li('product_weight_kg'),
                                html.Li('product_photos_qty'),
                                html.Li('product_volume_cm3')
                            ], style={'fontSize': '14px', 'color': COLORS['text_muted']})
                        ])
                    ], width=3)
                ])
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["info"]}'})
    ])
