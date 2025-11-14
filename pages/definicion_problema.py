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
        # create_section_header('📦 Dataset y Variables', color=COLORS['info']),

        # dbc.Card([
        #     dbc.CardBody([
        #         # KPIs del dataset
        #         dbc.Row([
        #             dbc.Col([
        #                 html.Div([
        #                     html.H3('110,013',
        #                             style={'color': COLORS['primary'], 'fontSize': '32px', 'fontWeight': 'bold'}),
        #                     html.P('Registros', style={'color': COLORS['text_muted']})
        #                 ], style={'textAlign': 'center'})
        #             ], width=3),
        #             dbc.Col([
        #                 html.Div([
        #                     html.H3('2016-2018',
        #                             style={'color': COLORS['success'], 'fontSize': '32px', 'fontWeight': 'bold'}),
        #                     html.P('Período', style={'color': COLORS['text_muted']})
        #                 ], style={'textAlign': 'center'})
        #             ], width=3),
        #             dbc.Col([
        #                 html.Div([
        #                     html.H3('~3,000',
        #                             style={'color': COLORS['warning'], 'fontSize': '32px', 'fontWeight': 'bold'}),
        #                     html.P('Vendedores', style={'color': COLORS['text_muted']})
        #                 ], style={'textAlign': 'center'})
        #             ], width=3),
        #             dbc.Col([
        #                 html.Div([
        #                     html.H3('50+', style={'color': COLORS['danger'], 'fontSize': '32px', 'fontWeight': 'bold'}),
        #                     html.P('Variables', style={'color': COLORS['text_muted']})
        #                 ], style={'textAlign': 'center'})
        #             ], width=3)
        #         ], style={'marginBottom': '30px'}),
        #
        #         html.Hr(style={'borderColor': COLORS['border']}),
        #
        #         # Categorías de variables
        #         html.H5('📊 Variables por Categoría:',
        #                 style={'color': COLORS['primary'], 'marginTop': '20px', 'marginBottom': '20px'}),
        #
        #         dbc.Row([
        #             dbc.Col([
        #                 html.Div([
        #                     html.H6('💰 Transaccionales', style={'color': COLORS['warning'], 'marginBottom': '10px'}),
        #                     html.Ul([
        #                         html.Li('price'),
        #                         html.Li('payment_value'),
        #                         html.Li('order_total_value'),
        #                         html.Li('freight_value'),
        #                         html.Li('payment_installments')
        #                     ], style={'fontSize': '14px', 'color': COLORS['text_muted']})
        #                 ])
        #             ], width=3),
        #             dbc.Col([
        #                 html.Div([
        #                     html.H6('🚚 Operacionales', style={'color': COLORS['info'], 'marginBottom': '10px'}),
        #                     html.Ul([
        #                         html.Li('delivery_time_days'),
        #                         html.Li('delivery_delay_days'),
        #                         html.Li('on_time_delivery'),
        #                         html.Li('shipping_limit_date')
        #                     ], style={'fontSize': '14px', 'color': COLORS['text_muted']})
        #                 ])
        #             ], width=3),
        #             dbc.Col([
        #                 html.Div([
        #                     html.H6('⭐ Satisfacción', style={'color': COLORS['success'], 'marginBottom': '10px'}),
        #                     html.Ul([
        #                         html.Li('review_score'),
        #                         html.Li('satisfaction_level'),
        #                         html.Li('review_comment_message')
        #                     ], style={'fontSize': '14px', 'color': COLORS['text_muted']})
        #                 ])
        #             ], width=3),
        #             dbc.Col([
        #                 html.Div([
        #                     html.H6('📦 Producto', style={'color': COLORS['danger'], 'marginBottom': '10px'}),
        #                     html.Ul([
        #                         html.Li('product_category'),
        #                         html.Li('product_weight_kg'),
        #                         html.Li('product_photos_qty'),
        #                         html.Li('product_volume_cm3')
        #                     ], style={'fontSize': '14px', 'color': COLORS['text_muted']})
        #                 ])
        #             ], width=3)
        #         ])
        #     ])
        # ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["info"]}', 'marginBottom': '40px'}),

        # =====================================================================
        # NUEVA SECCIÓN: ANÁLISIS DETALLADO DEL DATASET
        # =====================================================================

        create_section_header('🔍 Análisis Detallado del Dataset', color=COLORS['primary']),

        # Tablas fuente
        dbc.Card([
            dbc.CardHeader([
                html.H4('📊 Tablas de Datos Fuente', style={'margin': '0', 'color': COLORS['info']})
            ], style={'background': COLORS['card'], 'borderBottom': f'2px solid {COLORS["info"]}'}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.P([html.Strong('Orders: '), '99,441 registros × 8 columnas'],
                                   style={'fontSize': '15px', 'marginBottom': '8px'}),
                            html.P([html.Strong('Order Items: '), '112,650 registros × 7 columnas'],
                                   style={'fontSize': '15px', 'marginBottom': '8px'}),
                            html.P([html.Strong('Order Reviews: '), '99,224 registros × 7 columnas'],
                                   style={'fontSize': '15px', 'marginBottom': '8px'}),
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.P([html.Strong('Order Payments: '), '103,886 registros × 5 columnas'],
                                   style={'fontSize': '15px', 'marginBottom': '8px'}),
                            html.P([html.Strong('Products: '), '32,951 registros × 9 columnas'],
                                   style={'fontSize': '15px', 'marginBottom': '8px'}),
                            html.P([html.Strong('Customers: '), '99,441 registros × 5 columnas'],
                                   style={'fontSize': '15px', 'marginBottom': '8px'}),
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.P([html.Strong('Sellers: '), '3,095 registros × 4 columnas'],
                                   style={'fontSize': '15px', 'marginBottom': '8px'}),
                            html.P([html.Strong('Geolocation: '), '1,000,163 registros × 5 columnas'],
                                   style={'fontSize': '15px', 'marginBottom': '8px'}),
                            html.P([html.Strong('Category Translation: '), '71 registros × 2 columnas'],
                                   style={'fontSize': '15px', 'marginBottom': '8px'}),
                        ])
                    ], width=4)
                ])
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["info"]}', 'marginBottom': '30px'}),

        # Dimensiones consolidadas
        dbc.Card([
            dbc.CardHeader([
                html.H4('📐 Dimensiones del Dataset Consolidado', style={'margin': '0', 'color': COLORS['success']})
            ], style={'background': COLORS['card'], 'borderBottom': f'2px solid {COLORS["success"]}'}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H2('110,013',
                                    style={'color': COLORS['primary'], 'fontWeight': 'bold', 'marginBottom': '5px'}),
                            html.P('Registros (Filas)', style={'color': COLORS['text_muted'], 'fontSize': '14px'})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H2('51',
                                    style={'color': COLORS['warning'], 'fontWeight': 'bold', 'marginBottom': '5px'}),
                            html.P('Variables (Columnas)', style={'color': COLORS['text_muted'], 'fontSize': '14px'})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H2('5,610,663',
                                    style={'color': COLORS['danger'], 'fontWeight': 'bold', 'marginBottom': '5px'}),
                            html.P('Total de Celdas', style={'color': COLORS['text_muted'], 'fontSize': '14px'})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H2('95,832',
                                    style={'color': COLORS['info'], 'fontWeight': 'bold', 'marginBottom': '5px'}),
                            html.P('Órdenes Únicas', style={'color': COLORS['text_muted'], 'fontSize': '14px'})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3)
                ])
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'marginBottom': '30px'}),

        # Identificadores únicos
        dbc.Card([
            dbc.CardHeader([
                html.H4('🔑 Identificadores Únicos', style={'margin': '0', 'color': COLORS['warning']})
            ], style={'background': COLORS['card'], 'borderBottom': f'2px solid {COLORS["warning"]}'}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('95,832', style={'color': COLORS['primary'], 'fontWeight': 'bold'}),
                            html.P('Clientes Únicos', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center', 'padding': '15px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3('32,072', style={'color': COLORS['success'], 'fontWeight': 'bold'}),
                            html.P('Productos Únicos', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center', 'padding': '15px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3('2,965', style={'color': COLORS['warning'], 'fontWeight': 'bold'}),
                            html.P('Vendedores Únicos', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center', 'padding': '15px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3('92,755', style={'color': COLORS['danger'], 'fontWeight': 'bold'}),
                            html.P('Clientes ID Únicos', style={'color': COLORS['text_muted']})
                        ], style={'textAlign': 'center', 'padding': '15px'})
                    ], width=3)
                ])
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["warning"]}', 'marginBottom': '30px'}),

        # Clasificación detallada de variables
        dbc.Card([
            dbc.CardHeader([
                html.H4('📊 Clasificación Detallada de Variables', style={'margin': '0', 'color': COLORS['primary']})
            ], style={'background': COLORS['card'], 'borderBottom': f'2px solid {COLORS["primary"]}'}),
            dbc.CardBody([

                # Variables Numéricas Continuas
                html.H5('📈 Variables Numéricas Continuas (19)',
                        style={'color': COLORS['info'], 'marginBottom': '15px', 'marginTop': '10px'}),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.P([html.Strong('Transaccionales:')],
                                   style={'color': COLORS['warning'], 'marginBottom': '5px'}),
                            html.Ul([
                                html.Li('price: [R$0.85 - R$6,735.00]'),
                                html.Li('payment_value: [R$9.59 - R$13,664.08]'),
                                html.Li('order_total_value: [R$6.08 - R$6,929.31]'),
                                html.Li('freight_value: [R$0.00 - R$409.68]'),
                                html.Li('payment_installments: [0 - 24 cuotas]'),
                            ], style={'fontSize': '13px', 'lineHeight': '1.6'})
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.P([html.Strong('Operacionales:')],
                                   style={'color': COLORS['success'], 'marginBottom': '5px'}),
                            html.Ul([
                                html.Li('delivery_time_days: [0.53 - 208.35 días]'),
                                html.Li('delivery_delay_days: [-146.02 a 188.98 días]'),
                                html.Li('freight_price_ratio: [0.00 - 26.24]'),
                            ], style={'fontSize': '13px', 'lineHeight': '1.6'}),
                            html.P([html.Strong('Producto:')],
                                   style={'color': COLORS['danger'], 'marginBottom': '5px', 'marginTop': '15px'}),
                            html.Ul([
                                html.Li('product_weight_kg: [0.00 - 40.42 kg]'),
                                html.Li('product_volume_cm3: [168 - 296,208 cm³]'),
                                html.Li('product_height_cm: [2 - 105 cm]'),
                            ], style={'fontSize': '13px', 'lineHeight': '1.6'})
                        ])
                    ], width=6)
                ], style={'marginBottom': '25px'}),

                html.Hr(style={'borderColor': COLORS['border'], 'margin': '25px 0'}),

                # Variables Numéricas Discretas
                html.H5('🔢 Variables Numéricas Discretas (6)',
                        style={'color': COLORS['success'], 'marginBottom': '15px'}),
                dbc.Row([
                    dbc.Col([
                        html.Ul([
                            html.Li([html.Strong('review_score: '), '5 valores únicos (1-5 estrellas)']),
                            html.Li([html.Strong('product_photos_qty: '), '19 valores únicos (1-20 fotos)']),
                            html.Li([html.Strong('on_time_delivery: '), '2 valores (0=tarde, 1=a tiempo)']),
                        ], style={'fontSize': '14px', 'lineHeight': '1.8'})
                    ], width=6),
                    dbc.Col([
                        html.Ul([
                            html.Li([html.Strong('purchase_year: '), '3 valores (2016, 2017, 2018)']),
                            html.Li([html.Strong('purchase_month: '), '12 valores (1-12)']),
                            html.Li([html.Strong('purchase_day_of_week: '), '7 valores (0-6)']),
                        ], style={'fontSize': '14px', 'lineHeight': '1.8'})
                    ], width=6)
                ], style={'marginBottom': '25px'}),

                html.Hr(style={'borderColor': COLORS['border'], 'margin': '25px 0'}),

                # Variables Categóricas
                html.H5('🏷️ Variables Categóricas (12)',
                        style={'color': COLORS['warning'], 'marginBottom': '15px'}),
                dbc.Row([
                    dbc.Col([
                        html.Ul([
                            html.Li([html.Strong('satisfaction_level: '), '3 categorías']),
                            html.Li([html.Strong('payment_type: '), '4 categorías']),
                            html.Li([html.Strong('customer_state: '), '27 estados']),
                            html.Li([html.Strong('seller_state: '), '22 estados']),
                            html.Li([html.Strong('customer_city: '), '4,083 ciudades']),
                            html.Li([html.Strong('seller_city: '), '594 ciudades']),
                        ], style={'fontSize': '14px', 'lineHeight': '1.8'})
                    ], width=6),
                    dbc.Col([
                        html.Ul([
                            html.Li([html.Strong('product_category_name: '), '73 categorías']),
                            html.Li([html.Strong('product_category_english: '), '71 categorías']),
                            html.Li([html.Strong('order_status: '), '1 valor (delivered)']),
                            html.Li([html.Strong('review_creation_date: '), '627 fechas']),
                            html.Li([html.Strong('review_answer_timestamp: '), '95,493 timestamps']),
                            html.Li([html.Strong('shipping_limit_date: '), '90,758 fechas']),
                        ], style={'fontSize': '14px', 'lineHeight': '1.8'})
                    ], width=6)
                ], style={'marginBottom': '25px'}),

                html.Hr(style={'borderColor': COLORS['border'], 'margin': '25px 0'}),

                # Variables de Fecha y Texto
                dbc.Row([
                    dbc.Col([
                        html.H5('📅 Variables de Fecha/Tiempo (5)',
                                style={'color': COLORS['danger'], 'marginBottom': '15px'}),
                        html.Ul([
                            html.Li('order_purchase_timestamp'),
                            html.Li('order_approved_at'),
                            html.Li('order_delivered_carrier_date'),
                            html.Li('order_delivered_customer_date'),
                            html.Li('order_estimated_delivery_date'),
                        ], style={'fontSize': '14px', 'lineHeight': '1.8'})
                    ], width=6),
                    dbc.Col([
                        html.H5('📝 Variables de Texto (2)',
                                style={'color': COLORS['secondary'], 'marginBottom': '15px'}),
                        html.Ul([
                            html.Li('review_comment_title'),
                            html.Li('review_comment_message'),
                        ], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px'}),
                        html.H5('🔑 Variables Identificadoras (7)',
                                style={'color': COLORS['info'], 'marginBottom': '15px', 'marginTop': '20px'}),
                        html.Ul([
                            html.Li('order_id, customer_id, product_id'),
                            html.Li('seller_id, review_id, order_item_id'),
                            html.Li('customer_unique_id'),
                        ], style={'fontSize': '14px', 'lineHeight': '1.8'})
                    ], width=6)
                ])
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '40px'}),

        # Resumen técnico
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H5('📌 Resumen del Dataset',
                            style={'color': COLORS['primary'], 'marginBottom': '20px', 'textAlign': 'center'}),
                    html.P([
                        'El dataset consolidado integra ',
                        html.Strong('9 tablas relacionales '),
                        'del marketplace Olist, resultando en ',
                        html.Strong('110,013 transacciones completas '),
                        'con ',
                        html.Strong('51 variables '),
                        'clasificadas en: 19 numéricas continuas (precios, tiempos, dimensiones), ',
                        '6 numéricas discretas (ratings, fechas), 12 categóricas (ubicaciones, categorías), ',
                        '5 temporales y 2 de texto. ',
                        'La variable objetivo ',
                        html.Strong('satisfaction_level '),
                        'deriva de review_score y categoriza clientes en: Satisfecho (score 4-5), ',
                        'Neutro (score 3) e Insatisfecho (score 1-2).'
                    ], style={
                        'fontSize': '15px',
                        'lineHeight': '1.9',
                        'textAlign': 'justify',
                        'color': COLORS['text_muted']
                    })
                ], style={
                    'background': f'linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(123, 44, 191, 0.05) 100%)',
                    'padding': '25px',
                    'borderRadius': '10px',
                    'border': f'1px solid {COLORS["border"]}'
                })
            ])
        ], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["secondary"]}'})

    ])
