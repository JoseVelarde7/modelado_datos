"""
=============================================================================
HOME PAGE - CONTEXTO DEL PROYECTO
=============================================================================
Página de inicio con presentación del proyecto Olist.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from config import COLORS
from components.header import create_page_header, create_section_header


def create_home_content():
    """
    Crea el contenido de la página de inicio/contexto.

    Returns:
    --------
    html.Div
        Contenido completo de la página home
    """

    return html.Div([

        # Header principal
        create_page_header(
            title='Análisis E-commerce Olist',
            subtitle='Análisis de Satisfacción del Cliente | Brasil 2016-2018',
            icon='🛒',
            show_divider=True
        ),

        # Hero section
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2('📊 Proyecto de Modelado de datos 1', style={
                            'color': COLORS['primary'],
                            'marginBottom': '20px',
                            'fontWeight': 'bold'
                        }),
                        html.P([
                            'Análisis integral del comportamiento de clientes en el marketplace Olist de Brasil. ',
                            html.Strong('110,013 transacciones'), ' analizadas durante el período ',
                            html.Strong('2016-2018'), ' para identificar ',
                            html.Strong('factores críticos de satisfacción del cliente.')
                        ], style={
                            'fontSize': '18px',
                            'lineHeight': '1.8',
                            'color': COLORS['text']
                        })
                    ])
                ], width=12)
            ])
        ], style={
            'padding': '40px',
            'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["background"]} 100%)',
            'borderRadius': '16px',
            'border': f'2px solid {COLORS["primary"]}',
            'marginBottom': '40px',
            'boxShadow': f'0 0 30px {COLORS["primary"]}22'
        }),

        # Sección: ¿Qué es Olist?
        create_section_header('🛒 ¿Qué es Olist?', icon='🛒', color=COLORS['primary']),

        dbc.Card([
            dbc.CardBody([
                html.P([
                    html.Strong('Olist ', style={'fontSize': '20px', 'color': COLORS['primary']}),
                    'es la plataforma de ',
                    html.Strong('marketplace '),
                    'en Brasil que conecta pequeños y medianos comerciantes (PYMES) con clientes en todo el país. ',
                    'Funciona como un ',
                    html.Strong('hub centralizado '),
                    'que permite a vendedores locales acceder a la infraestructura de grandes marketplaces sin necesidad de inversión propia.'
                ], style={'fontSize': '16px', 'lineHeight': '1.8', 'marginBottom': '20px'}),

                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H2('🇧🇷', style={'fontSize': '56px', 'margin': '0'}),
                            html.H5('Brasil',
                                    style={'color': COLORS['success'], 'marginTop': '10px', 'fontWeight': 'bold'}),
                            html.P('Mayor mercado digital de Latinoamérica',
                                   style={'color': COLORS['text_muted'], 'fontSize': '14px'})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H2('🏪', style={'fontSize': '56px', 'margin': '0'}),
                            html.H5('Marketplace',
                                    style={'color': COLORS['warning'], 'marginTop': '10px', 'fontWeight': 'bold'}),
                            html.P('Plataforma integradora de comercio',
                                   style={'color': COLORS['text_muted'], 'fontSize': '14px'})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H2('👥', style={'fontSize': '56px', 'margin': '0'}),
                            html.H5('PYMES',
                                    style={'color': COLORS['info'], 'marginTop': '10px', 'fontWeight': 'bold'}),
                            html.P('Pequeñas y medianas empresas',
                                   style={'color': COLORS['text_muted'], 'fontSize': '14px'})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H2('📦', style={'fontSize': '56px', 'margin': '0'}),
                            html.H5('E-commerce',
                                    style={'color': COLORS['danger'], 'marginTop': '10px', 'fontWeight': 'bold'}),
                            html.P('Comercio electrónico',
                                   style={'color': COLORS['text_muted'], 'fontSize': '14px'})
                        ], style={'textAlign': 'center'})
                    ], width=3)
                ], style={'marginTop': '30px'})
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '40px'}),

        # Sección: El auge del E-commerce
        create_section_header('📈 El Auge del E-commerce en Brasil', icon='📈', color=COLORS['success']),

        dbc.Card([
            dbc.CardBody([
                html.P([
                    'El e-commerce en Brasil ha experimentado un ',
                    html.Strong('crecimiento exponencial '),
                    'en los últimos años, consolidándose como el ',
                    html.Strong('mercado digital más grande de América Latina. '),
                    'Factores como la ',
                    html.Strong('penetración de internet, adopción de pagos digitales, '),
                    'y mejoras en ',
                    html.Strong('logística '),
                    'han impulsado este sector.'
                ], style={'fontSize': '16px', 'lineHeight': '1.8', 'marginBottom': '25px'}),

                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4('📱', style={'fontSize': '48px'}),
                            html.H5('Conectividad', style={'color': COLORS['primary']}),
                            html.P('85%+ penetración de internet móvil', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H4('💳', style={'fontSize': '48px'}),
                            html.H5('Pagos Digitales', style={'color': COLORS['primary']}),
                            html.P('Crecimiento en métodos alternativos', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H4('🚚', style={'fontSize': '48px'}),
                            html.H5('Logística', style={'color': COLORS['primary']}),
                            html.P('Mejoras en tiempos de entrega', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H4('🔒', style={'fontSize': '48px'}),
                            html.H5('Confianza', style={'color': COLORS['primary']}),
                            html.P('Mayor seguridad en transacciones', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3)
                ])
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'marginBottom': '40px'}),

        # Sección: ¿Por qué analizar satisfacción?
        create_section_header('💡 ¿Por Qué Analizar la Satisfacción del Cliente?', icon='💡', color=COLORS['warning']),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3('🎯', style={'fontSize': '48px', 'textAlign': 'center', 'margin': '0'}),
                        html.H4('Retención de Clientes',
                                style={'color': COLORS['primary'], 'textAlign': 'center', 'margin': '15px 0'}),
                        html.P('Clientes satisfechos tienen 5x más probabilidad de volver a comprar',
                               style={'color': COLORS['text_muted'], 'textAlign': 'center', 'fontSize': '14px'})
                    ])
                ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'height': '100%'})
            ], width=6, className='mb-4'),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3('⭐', style={'fontSize': '48px', 'textAlign': 'center', 'margin': '0'}),
                        html.H4('Reputación de Marca',
                                style={'color': COLORS['success'], 'textAlign': 'center', 'margin': '15px 0'}),
                        html.P('Reviews positivos mejoran visibilidad y confianza del marketplace',
                               style={'color': COLORS['text_muted'], 'textAlign': 'center', 'fontSize': '14px'})
                    ])
                ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'height': '100%'})
            ], width=6, className='mb-4'),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3('💰', style={'fontSize': '48px', 'textAlign': 'center', 'margin': '0'}),
                        html.H4('Incremento en Ventas',
                                style={'color': COLORS['warning'], 'textAlign': 'center', 'margin': '15px 0'}),
                        html.P('Clientes felices recomiendan y generan word-of-mouth positivo',
                               style={'color': COLORS['text_muted'], 'textAlign': 'center', 'fontSize': '14px'})
                    ])
                ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["warning"]}', 'height': '100%'})
            ], width=6, className='mb-4'),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3('📊', style={'fontSize': '48px', 'textAlign': 'center', 'margin': '0'}),
                        html.H4('Decisiones Data-Driven',
                                style={'color': COLORS['danger'], 'textAlign': 'center', 'margin': '15px 0'}),
                        html.P('Identificar factores críticos permite optimización basada en datos',
                               style={'color': COLORS['text_muted'], 'textAlign': 'center', 'fontSize': '14px'})
                    ])
                ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["danger"]}', 'height': '100%'})
            ], width=6, className='mb-4')
        ]),

        # Sección: Alcance del proyecto
        create_section_header('🎯 Alcance del Proyecto', icon='🎯', color=COLORS['secondary']),

        dbc.Card([
            dbc.CardBody([
                html.P([
                    'Este proyecto analiza el comportamiento de clientes de Olist durante el período ',
                    html.Strong('2016-2018 '),
                    'con el objetivo de ',
                    html.Strong('identificar patrones de satisfacción, factores operacionales y geográficos '),
                    'que determinan una experiencia exitosa.'
                ], style={'fontSize': '16px', 'lineHeight': '1.8', 'marginBottom': '30px'}),

                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H2('110,013',
                                    style={'color': COLORS['primary'], 'fontSize': '36px', 'fontWeight': 'bold',
                                           'margin': '0'}),
                            html.P('Órdenes Analizadas', style={'color': COLORS['text_muted'], 'marginTop': '10px'})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H2('2016-2018',
                                    style={'color': COLORS['success'], 'fontSize': '36px', 'fontWeight': 'bold',
                                           'margin': '0'}),
                            html.P('Período de Análisis', style={'color': COLORS['text_muted'], 'marginTop': '10px'})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H2('~3,000',
                                    style={'color': COLORS['warning'], 'fontSize': '36px', 'fontWeight': 'bold',
                                           'margin': '0'}),
                            html.P('Vendedores PYMES', style={'color': COLORS['text_muted'], 'marginTop': '10px'})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H2('>80%', style={'color': COLORS['danger'], 'fontSize': '36px', 'fontWeight': 'bold',
                                                   'margin': '0'}),
                            html.P('Target Accuracy', style={'color': COLORS['text_muted'], 'marginTop': '10px'})
                        ], style={'textAlign': 'center'})
                    ], width=3)
                ])
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["secondary"]}', 'marginBottom': '40px'}),

        # CTA
        # html.Div([
        #     html.H4('🚀 Comienza la Exploración',
        #             style={'color': COLORS['primary'], 'textAlign': 'center', 'marginBottom': '20px'}),
        #     html.P('Navega por las secciones para descubrir insights clave sobre satisfacción del cliente',
        #            style={'color': COLORS['text_muted'], 'textAlign': 'center', 'fontSize': '16px'})
        # ], style={
        #     'padding': '40px',
        #     'background': f'rgba(0, 212, 255, 0.05)',
        #     'borderRadius': '12px',
        #     'border': f'2px dashed {COLORS["primary"]}',
        #     'marginTop': '40px'
        # })
    ])
