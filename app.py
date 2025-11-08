"""
=============================================================================
APLICACIÓN PRINCIPAL - OLIST E-COMMERCE PROJECT
=============================================================================
Dashboard Dash con navegación multipágina y análisis de satisfacción del cliente.
"""

import sys

sys.path.append('/home/claude/olist_project')

from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

# Imports del proyecto
from config import COLORS, CONTENT_STYLE
from components.sidebar import create_sidebar
from pages.home import create_home_content
from pages.definicion_problema import create_definicion_content
from pages.analisis_estadistico import create_analisis_content
from pages.tecnica_analitica import create_tecnica_content
from data_loader import load_data

# =============================================================================
# ESTILOS CSS GLOBALES
# =============================================================================

EXTERNAL_STYLESHEETS = [
	dbc.themes.BOOTSTRAP,
	{
		'href': 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
		'rel': 'stylesheet'
	}
]

# CSS personalizado para toda la aplicación
CUSTOM_CSS = f"""
    body {{
        background-color: {COLORS['background']} !important;
        color: {COLORS['text']} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }}

    * {{
        color: {COLORS['text']};
    }}

    p, span, div, li, td, th, label {{
        color: {COLORS['text']} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['text']} !important;
    }}

    .card {{
        background-color: {COLORS['card']} !important;
        color: {COLORS['text']} !important;
    }}

    .card-body {{
        color: {COLORS['text']} !important;
    }}

    .alert {{
        color: {COLORS['text']} !important;
    }}

    a {{
        text-decoration: none !important;
    }}

    strong {{
        color: {COLORS['text']} !important;
    }}

    em {{
        color: {COLORS['text']} !important;
    }}
"""

# =============================================================================
# INICIALIZACIÓN DE LA APP
# =============================================================================

app = Dash(
	__name__,
	external_stylesheets=EXTERNAL_STYLESHEETS,
	suppress_callback_exceptions=True,
	title='Olist E-commerce Analytics'
)

# Inyectar CSS personalizado
app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            {CUSTOM_CSS}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

# Cargar datos una vez al inicio
print("🚀 Inicializando aplicación...")
df = load_data()
print("✅ Datos cargados exitosamente\n")

# =============================================================================
# LAYOUT PRINCIPAL
# =============================================================================

app.layout = html.Div([
	dcc.Location(id='url', refresh=False),
	create_sidebar(),
	html.Div(id='page-content', style=CONTENT_STYLE)
], style={'background': COLORS['background']})


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
	Output('page-content', 'children'),
	Input('url', 'pathname')
)
def display_page(pathname):
	"""
    Callback principal de navegación.
    """

	print(f"📍 Navegando a: {pathname}")

	if pathname == '/definicion':
		return create_definicion_content()
	elif pathname == '/analisis':
		return create_analisis_content(df)
	elif pathname == '/tecnica':
		return create_tecnica_content()
	elif pathname == '/comparacion':
		return create_placeholder_page('⚖️', 'Comparación de Modelos', 'Evaluación y comparación de técnicas competidoras')
	elif pathname == '/optimizacion':
		return create_placeholder_page('🎯', 'Optimización', 'Refinamiento y mejora del modelo seleccionado')
	else:
		return create_home_content()


def create_placeholder_page(icon, title, description):
	"""
    Crea una página placeholder para secciones pendientes.
    """

	return html.Div([
		html.Div([
			html.H1([
				html.Span(icon, style={'fontSize': '80px', 'marginRight': '20px'}),
				title
			], style={
				'textAlign': 'center',
				'color': COLORS['primary'],
				'marginBottom': '20px'
			}),
			html.P(description, style={
				'textAlign': 'center',
				'color': COLORS['text_muted'],
				'fontSize': '20px',
				'marginBottom': '40px'
			})
		]),

		dbc.Alert([
			html.H4('🚧 Sección En Desarrollo', className='alert-heading', style={'color': COLORS['text']}),
			html.P([
				'Esta sección está pendiente de implementación. ',
				'Por favor, vuelve más tarde o navega a otras secciones disponibles.'
			], style={'marginBottom': '0', 'color': COLORS['text']})
		], color='warning', style={'fontSize': '16px', 'maxWidth': '800px', 'margin': '0 auto'})
	])


@callback(
	[Output(f'nav-{item["id"]}', 'style') for item in [
		{'id': 'home'}, {'id': 'definicion'}, {'id': 'analisis'},
		{'id': 'tecnica'}, {'id': 'comparacion'}, {'id': 'optimizacion'}
	]],
	Input('url', 'pathname')
)
def update_nav_styles(pathname):
	"""
    Actualiza los estilos de navegación según la página activa.
    """

	from config import NAV_ITEM_STYLE, NAV_ITEM_ACTIVE_STYLE

	routes = {
		'/': 0,
		'/definicion': 1,
		'/analisis': 2,
		'/tecnica': 3,
		'/comparacion': 4,
		'/optimizacion': 5
	}

	active_index = routes.get(pathname, 0)

	styles = []
	for i in range(6):
		if i == active_index:
			styles.append(NAV_ITEM_ACTIVE_STYLE)
		else:
			styles.append(NAV_ITEM_STYLE)

	return styles


# =============================================================================
# EJECUTAR APLICACIÓN
# =============================================================================

if __name__ == '__main__':
	print("\n" + "=" * 60)
	print("🚀 OLIST E-COMMERCE ANALYTICS DASHBOARD")
	print("=" * 60)
	print(f"📊 Dataset: {len(df):,} registros cargados")
	print(f"🌐 Servidor: http://127.0.0.1:8050")
	print("=" * 60 + "\n")

	app.run_server(debug=True, host='0.0.0.0', port=8050)
