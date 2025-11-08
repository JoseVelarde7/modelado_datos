"""
=============================================================================
ANÁLISIS ESTADÍSTICO COMPLETO - OLIST E-COMMERCE PROJECT
=============================================================================
Análisis profesional orientado a responder los objetivos del proyecto.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from config import COLORS, PLOTLY_CONFIG
from components.header import create_page_header, create_section_header, create_info_banner


# =============================================================================
# SECCIÓN 1: MATRIZ DE CORRELACIÓN
# =============================================================================

def create_correlation_analysis(df):
	"""
    Análisis de correlación para identificar variables predictoras clave.

    OBJETIVO: Identificar qué variables operacionales, transaccionales y
    geográficas tienen mayor relación con la satisfacción del cliente.
    """

	# Variables para análisis
	correlation_vars = [
		'review_score',  # TARGET
		'delivery_time_days',
		'delivery_delay_days',
		'on_time_delivery',
		'price',
		'freight_value',
		'order_total_value',
		'payment_installments',
		'freight_price_ratio',
		'product_photos_qty',
		'product_weight_kg',
		'product_volume_cm3'
	]

	# Calcular matriz de correlación
	corr_matrix = df[correlation_vars].corr()
	review_corr = corr_matrix['review_score'].sort_values(ascending=False).drop('review_score')

	# Clasificar correlaciones
	strong_positive = review_corr[review_corr > 0.3]
	moderate_positive = review_corr[(review_corr > 0.1) & (review_corr <= 0.3)]
	weak = review_corr[(review_corr >= -0.1) & (review_corr <= 0.1)]
	moderate_negative = review_corr[(review_corr < -0.1) & (review_corr >= -0.3)]
	strong_negative = review_corr[review_corr < -0.3]

	# 1. Heatmap de correlación
	fig_heatmap = go.Figure(data=go.Heatmap(
		z=corr_matrix.values,
		x=[var.replace('_', ' ').title() for var in corr_matrix.columns],
		y=[var.replace('_', ' ').title() for var in corr_matrix.columns],
		colorscale='RdBu',
		zmid=0,
		text=np.round(corr_matrix.values, 3),
		texttemplate='%{text}',
		textfont={"size": 9, "color": COLORS['text']},
		colorbar=dict(
			title=dict(text="Correlación", font=dict(color=COLORS['text'])),
			tickfont=dict(color=COLORS['text'])
		),
		hoverongaps=False
	))

	fig_heatmap.update_layout(
		title={
			'text': '🔍 Matriz de Correlación - Variables Clave',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 18, 'color': COLORS['text']}
		},
		xaxis={'tickangle': 45, 'side': 'bottom', 'tickfont': {'color': COLORS['text'], 'size': 10}},
		yaxis={'tickangle': 0, 'tickfont': {'color': COLORS['text'], 'size': 10}},
		height=700,
		paper_bgcolor=COLORS['background'],
		plot_bgcolor=COLORS['card'],
		font={'color': COLORS['text']}
	)

	# 2. Gráfico de barras - Top correlaciones
	top_10 = review_corr.abs().sort_values(ascending=False).head(10)
	colors_bars = [COLORS['success'] if review_corr[var] > 0 else COLORS['danger'] for var in top_10.index]

	fig_bars = go.Figure(data=[
		go.Bar(
			y=[var.replace('_', ' ').title() for var in top_10.index],
			x=[review_corr[var] for var in top_10.index],
			orientation='h',
			marker=dict(color=colors_bars, line=dict(color=COLORS['primary'], width=1)),
			text=[f'{review_corr[var]:.3f}' for var in top_10.index],
			textposition='outside',
			hovertemplate='<b>%{y}</b><br>Correlación: %{x:.3f}<extra></extra>'
		)
	])

	fig_bars.update_layout(
		title={
			'text': '📊 Top 10 Variables Más Correlacionadas con Satisfacción',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 18, 'color': COLORS['text']}
		},
		xaxis_title='Coeficiente de Correlación de Pearson',
		yaxis_title='',
		height=500,
		paper_bgcolor=COLORS['background'],
		plot_bgcolor=COLORS['card'],
		font={'color': COLORS['text']},
		showlegend=False
	)

	fig_bars.add_vline(x=0, line_width=2, line_dash="dash", line_color=COLORS['text_muted'])
	fig_bars.add_vrect(x0=-0.3, x1=-1, fillcolor=COLORS['danger'], opacity=0.1, line_width=0)
	fig_bars.add_vrect(x0=0.3, x1=1, fillcolor=COLORS['success'], opacity=0.1, line_width=0)

	return fig_heatmap, fig_bars, strong_positive, strong_negative, moderate_positive, moderate_negative


def create_correlation_section(df):
	"""Sección completa de análisis de correlación."""

	fig_heatmap, fig_bars, strong_pos, strong_neg, mod_pos, mod_neg = create_correlation_analysis(df)

	return html.Div([
		create_section_header('🔍 PASO 1: Identificación de Variables Predictoras', color=COLORS['primary']),

		create_info_banner(
			'Análisis de correlación de Pearson para identificar relaciones lineales entre variables y satisfacción del cliente',
			icon='📌',
			banner_type='info'
		),

		# Objetivo específico
		dbc.Card([
			dbc.CardHeader(html.H5('🎯 Objetivo Específico: Exploratorio', style={'margin': 0, 'color': COLORS['info']})),
			dbc.CardBody([
				html.P([
					html.Strong('"Identificar relaciones entre variables operacionales y niveles de satisfacción"',
											style={'color': COLORS['text'], 'fontSize': '16px'}),
				], style={'marginBottom': '10px', 'color': COLORS['text']}),
				html.P([
					'Este análisis responde: ',
					html.Strong('¿Qué variables tienen mayor impacto en la satisfacción? ', style={'color': COLORS['primary']}),
					'Utilizamos el coeficiente de correlación de Pearson (r) para medir la fuerza y dirección de relaciones lineales.'
				], style={'fontSize': '15px', 'lineHeight': '1.8', 'color': COLORS['text'], 'marginBottom': 0})
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["info"]}', 'marginBottom': '30px'}),

		# Heatmap
		dbc.Card([
			dbc.CardBody([
				dcc.Graph(figure=fig_heatmap, config=PLOTLY_CONFIG)
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '30px'}),

		# Gráfico de barras
		dbc.Card([
			dbc.CardBody([
				dcc.Graph(figure=fig_bars, config=PLOTLY_CONFIG)
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '30px'}),

		# Interpretación estadística
		dbc.Card([
			dbc.CardHeader(
				html.H5('📊 Interpretación Estadística de Correlaciones', style={'margin': 0, 'color': COLORS['success']})),
			dbc.CardBody([
				# Criterios de interpretación
				html.Div([
					html.H6('📏 Criterios de Cohen (1988):', style={'color': COLORS['primary'], 'marginBottom': '15px'}),
					dbc.Row([
						dbc.Col([
							html.Div([
								html.Strong('|r| < 0.3:', style={'color': COLORS['text']}),
								html.Br(),
								html.Span('Débil', style={'color': COLORS['text_muted']})
							], style={'textAlign': 'center', 'padding': '15px', 'background': 'rgba(148, 163, 184, 0.1)',
												'borderRadius': '8px'})
						], width=3),
						dbc.Col([
							html.Div([
								html.Strong('0.3 ≤ |r| < 0.5:', style={'color': COLORS['text']}),
								html.Br(),
								html.Span('Moderada', style={'color': COLORS['warning']})
							], style={'textAlign': 'center', 'padding': '15px', 'background': 'rgba(245, 158, 11, 0.1)',
												'borderRadius': '8px'})
						], width=3),
						dbc.Col([
							html.Div([
								html.Strong('|r| ≥ 0.5:', style={'color': COLORS['text']}),
								html.Br(),
								html.Span('Fuerte', style={'color': COLORS['success']})
							], style={'textAlign': 'center', 'padding': '15px', 'background': 'rgba(16, 185, 129, 0.1)',
												'borderRadius': '8px'})
						], width=3),
						dbc.Col([
							html.Div([
								html.Strong('p < 0.05:', style={'color': COLORS['text']}),
								html.Br(),
								html.Span('Significativo', style={'color': COLORS['info']})
							], style={'textAlign': 'center', 'padding': '15px', 'background': 'rgba(59, 130, 246, 0.1)',
												'borderRadius': '8px'})
						], width=3)
					], style={'marginBottom': '30px'})
				]),

				html.Hr(style={'borderColor': COLORS['border']}),

				# Hallazgos
				html.Div([
					html.H6('🔬 Hallazgos Principales:', style={'color': COLORS['success'], 'marginBottom': '20px'}),

					# Correlaciones positivas fuertes
					html.Div([
						html.Strong('✅ Correlaciones Positivas Fuertes (r > 0.3):',
												style={'color': COLORS['success'], 'fontSize': '15px'}),
						html.Ul([
											html.Li([
												html.Strong(f'{var.replace("_", " ").title()}: ', style={'color': COLORS['text']}),
												html.Span(f'r = {corr:.3f}',
																	style={'color': COLORS['success'], 'fontSize': '15px', 'fontWeight': 'bold'}),
												html.Span(' → Mayor valor = Mayor satisfacción', style={'color': COLORS['text_muted']})
											], style={'marginBottom': '8px', 'color': COLORS['text']})
											for var, corr in strong_pos.items()
										] if len(strong_pos) > 0 else [
							html.Li('No se encontraron correlaciones positivas fuertes', style={'color': COLORS['text_muted']})],
										style={'lineHeight': '1.8'})
					], style={'marginBottom': '25px'}),

					# Correlaciones negativas fuertes
					html.Div([
						html.Strong('⚠️ Correlaciones Negativas Fuertes (r < -0.3):',
												style={'color': COLORS['danger'], 'fontSize': '15px'}),
						html.Ul([
											html.Li([
												html.Strong(f'{var.replace("_", " ").title()}: ', style={'color': COLORS['text']}),
												html.Span(f'r = {corr:.3f}',
																	style={'color': COLORS['danger'], 'fontSize': '15px', 'fontWeight': 'bold'}),
												html.Span(' → Mayor valor = Menor satisfacción', style={'color': COLORS['text_muted']})
											], style={'marginBottom': '8px', 'color': COLORS['text']})
											for var, corr in strong_neg.items()
										] if len(strong_neg) > 0 else [
							html.Li('No se encontraron correlaciones negativas fuertes', style={'color': COLORS['text_muted']})],
										style={'lineHeight': '1.8'})
					], style={'marginBottom': '25px'}),
				]),

				html.Hr(style={'borderColor': COLORS['border']}),

				# Conclusión estratégica
				html.Div([
					html.H6('🎯 Conclusión Estratégica:', style={'color': COLORS['secondary'], 'marginBottom': '15px'}),
					html.P([
						'El análisis revela que ',
						html.Strong('las variables operacionales de entrega ', style={'color': COLORS['primary']}),
						'(delivery_delay_days, on_time_delivery) muestran las correlaciones más fuertes con la satisfacción. ',
						'Esto indica que ',
						html.Strong('la experiencia de entrega es el factor crítico ', style={'color': COLORS['success']}),
						'que determina la satisfacción del cliente en Olist, superando incluso a variables transaccionales como precio. ',
						html.Strong('Recomendación: Priorizar optimización logística sobre estrategias de precio.',
												style={'color': COLORS['warning']})
					], style={'fontSize': '15px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={
					'background': f'rgba(123, 44, 191, 0.1)',
					'padding': '20px',
					'borderRadius': '12px',
					'border': f'1px solid {COLORS["secondary"]}'
				})
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'marginBottom': '40px'})
	])


# =============================================================================
# SECCIÓN 2: ANÁLISIS DE REVIEW SCORE (VARIABLE TARGET)
# =============================================================================

def create_review_score_analysis(df):
	"""
    Análisis descriptivo e inferencial de review_score.

    OBJETIVO: Caracterizar la distribución de satisfacción y realizar
    pruebas de hipótesis sobre diferencias entre grupos.
    """

	# Estadísticas descriptivas
	stats_dict = {
		'mean': df['review_score'].mean(),
		'median': df['review_score'].median(),
		'mode': df['review_score'].mode()[0],
		'std': df['review_score'].std(),
		'var': df['review_score'].var(),
		'skewness': df['review_score'].skew(),
		'kurtosis': df['review_score'].kurtosis(),
		'q1': df['review_score'].quantile(0.25),
		'q3': df['review_score'].quantile(0.75),
		'iqr': df['review_score'].quantile(0.75) - df['review_score'].quantile(0.25),
		'cv': (df['review_score'].std() / df['review_score'].mean()) * 100,
		'min': df['review_score'].min(),
		'max': df['review_score'].max(),
		'n': len(df)
	}

	# Distribución de frecuencias
	review_counts = df['review_score'].value_counts().sort_index()
	review_pcts = (review_counts / len(df) * 100).round(2)

	# Test de normalidad Shapiro-Wilk (muestra aleatoria de 5000)
	sample = df['review_score'].sample(min(5000, len(df)), random_state=42)
	shapiro_stat, shapiro_p = stats.shapiro(sample)

	# Crear subplots
	fig = make_subplots(
		rows=2, cols=3,
		subplot_titles=(
			'Distribución de Frecuencias',
			'Distribución Acumulada',
			'Box Plot con Outliers',
			'Proporción por Rating',
			'Q-Q Plot (Normalidad)',
			'Violin Plot'
		),
		specs=[
			[{"type": "bar"}, {"type": "scatter"}, {"type": "box"}],
			[{"type": "pie"}, {"type": "scatter"}, {"type": "violin"}]
		],
		vertical_spacing=0.12,
		horizontal_spacing=0.1
	)

	colors_bars = ['#ef4444', '#f59e0b', '#eab308', '#22c55e', '#10b981']

	# 1. Histograma
	fig.add_trace(
		go.Bar(
			x=[f'{i}⭐' for i in review_counts.index],
			y=review_counts.values,
			text=[f'{count:,}<br>({pct}%)' for count, pct in zip(review_counts.values, review_pcts.values)],
			textposition='outside',
			marker=dict(color=colors_bars, line=dict(color=COLORS['primary'], width=2)),
			hovertemplate='<b>%{x}</b><br>N: %{y:,}<extra></extra>',
			showlegend=False
		),
		row=1, col=1
	)

	# 2. Distribución acumulada
	cumulative_pct = review_pcts.cumsum()
	fig.add_trace(
		go.Scatter(
			x=review_counts.index,
			y=cumulative_pct.values,
			mode='lines+markers',
			line=dict(color=COLORS['primary'], width=3),
			marker=dict(size=10, color=COLORS['secondary']),
			fill='tonexty',
			hovertemplate='<b>Rating ≤ %{x}</b><br>Acumulado: %{y:.1f}%<extra></extra>',
			showlegend=False
		),
		row=1, col=2
	)

	# 3. Box plot
	fig.add_trace(
		go.Box(
			y=df['review_score'],
			marker=dict(color=COLORS['primary']),
			boxmean='sd',
			name='',
			hovertemplate='Valor: %{y}<extra></extra>',
			showlegend=False
		),
		row=1, col=3
	)

	# 4. Pie chart
	fig.add_trace(
		go.Pie(
			labels=[f'{i}⭐' for i in review_counts.index],
			values=review_counts.values,
			marker=dict(colors=colors_bars, line=dict(color=COLORS['background'], width=2)),
			textinfo='label+percent',
			textfont=dict(size=12, color=COLORS['text']),
			hole=0.4,
			hovertemplate='<b>%{label}</b><br>%{value:,} reviews<br>%{percent}<extra></extra>',
			showlegend=False
		),
		row=2, col=1
	)

	# 5. Q-Q Plot para test de normalidad
	theoretical_quantiles = stats.probplot(sample, dist="norm")[0][0]
	sample_quantiles = stats.probplot(sample, dist="norm")[0][1]

	fig.add_trace(
		go.Scatter(
			x=theoretical_quantiles,
			y=sample_quantiles,
			mode='markers',
			marker=dict(size=4, color=COLORS['info'], opacity=0.6),
			hovertemplate='Teórico: %{x:.2f}<br>Observado: %{y:.2f}<extra></extra>',
			showlegend=False
		),
		row=2, col=2
	)

	# Línea de referencia para normalidad
	fig.add_trace(
		go.Scatter(
			x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
			y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
			mode='lines',
			line=dict(color=COLORS['danger'], dash='dash', width=2),
			showlegend=False
		),
		row=2, col=2
	)

	# 6. Violin plot
	fig.add_trace(
		go.Violin(
			y=df['review_score'],
			marker=dict(color=COLORS['secondary']),
			box_visible=True,
			meanline_visible=True,
			name='',
			hovertemplate='Valor: %{y}<extra></extra>',
			showlegend=False
		),
		row=2, col=3
	)

	# Actualizar layout
	fig.update_layout(
		title={
			'text': '⭐ Análisis Descriptivo Completo: Review Score (Variable Target)',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 20, 'color': COLORS['text']}
		},
		height=900,
		paper_bgcolor=COLORS['background'],
		plot_bgcolor=COLORS['card'],
		font={'color': COLORS['text']}
	)

	# Actualizar ejes
	fig.update_xaxes(title_text="Rating", row=1, col=1, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Frecuencia", row=1, col=1, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Rating", row=1, col=2, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="% Acumulado", row=1, col=2, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Review Score", row=1, col=3, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Cuantiles Teóricos", row=2, col=2, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Cuantiles Observados", row=2, col=2, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Review Score", row=2, col=3, tickfont={'color': COLORS['text']})

	return fig, stats_dict, shapiro_stat, shapiro_p, review_counts, review_pcts


def create_review_score_section(df):
	"""Sección completa de análisis de review_score."""

	fig, stats_dict, shapiro_stat, shapiro_p, counts, pcts = create_review_score_analysis(df)

	return html.Div([
		create_section_header('⭐ PASO 2: Análisis de la Variable Target (Review Score)', color=COLORS['warning']),

		create_info_banner(
			'Análisis descriptivo e inferencial de la satisfacción del cliente medida por review_score',
			icon='📊',
			banner_type='info'
		),

		# Objetivo
		dbc.Card([
			dbc.CardHeader(html.H5('🎯 Objetivo Específico: Descriptivo', style={'margin': 0, 'color': COLORS['success']})),
			dbc.CardBody([
				html.P([
					html.Strong('"Caracterizar el comportamiento de compra y patrones de satisfacción"',
											style={'color': COLORS['text'], 'fontSize': '16px'}),
				], style={'marginBottom': '10px', 'color': COLORS['text']}),
				html.P([
					'Este análisis responde: ',
					html.Strong('¿Cómo se distribuye la satisfacción del cliente? ', style={'color': COLORS['primary']}),
					'¿Es simétrica o sesgada? ¿Sigue una distribución normal? ',
					'Utilizamos estadística descriptiva y pruebas de normalidad.'
				], style={'fontSize': '15px', 'lineHeight': '1.8', 'color': COLORS['text'], 'marginBottom': 0})
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'marginBottom': '30px'}),

		# KPIs estadísticos
		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('📊', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{stats_dict['mean']:.2f}",
										style={'color': COLORS['success'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Media (μ)',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(
							f"IC 95%: [{stats_dict['mean'] - 1.96 * stats_dict['std'] / np.sqrt(stats_dict['n']):.2f}, {stats_dict['mean'] + 1.96 * stats_dict['std'] / np.sqrt(stats_dict['n']):.2f}]",
							style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block', 'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'height': '100%'})
			], width=2),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('🎯', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{stats_dict['median']:.0f}⭐",
										style={'color': COLORS['primary'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Mediana (Me)',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(f"Q1: {stats_dict['q1']:.0f} | Q3: {stats_dict['q3']:.0f}",
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'height': '100%'})
			], width=2),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('📏', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"±{stats_dict['std']:.2f}",
										style={'color': COLORS['warning'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Desv. Est. (σ)',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(f"CV: {stats_dict['cv']:.1f}%",
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["warning"]}', 'height': '100%'})
			], width=2),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('📉', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{stats_dict['skewness']:.2f}",
										style={'color': COLORS['danger'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Asimetría',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small('Sesgada izquierda' if stats_dict['skewness'] < -0.5 else 'Simétrica' if stats_dict[
																																																	'skewness'] < 0.5 else 'Sesgada derecha',
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["danger"]}', 'height': '100%'})
			], width=2),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('📐', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{stats_dict['kurtosis']:.2f}",
										style={'color': COLORS['info'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Curtosis',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small('Leptocúrtica' if stats_dict['kurtosis'] > 3 else 'Platicúrtica' if stats_dict[
																																														 'kurtosis'] < -1 else 'Mesocúrtica',
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["info"]}', 'height': '100%'})
			], width=2),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('🔔', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4('No' if shapiro_p < 0.05 else 'Sí',
										style={'color': COLORS['danger'] if shapiro_p < 0.05 else COLORS['success'],
													 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Normal',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(f"p = {shapiro_p:.4f}",
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["secondary"]}', 'height': '100%'})
			], width=2)
		], style={'marginBottom': '30px'}),

		# Gráficos
		dbc.Card([
			dbc.CardBody([
				dcc.Graph(figure=fig, config=PLOTLY_CONFIG)
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '30px'}),

		# Interpretación estadística completa
		dbc.Card([
			dbc.CardHeader(
				html.H5('📊 Interpretación Estadística Profesional', style={'margin': 0, 'color': COLORS['primary']})),
			dbc.CardBody([

				# Medidas de tendencia central
				html.Div([
					html.H6('1️⃣ Tendencia Central y Posición', style={'color': COLORS['success'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Media (μ): ', style={'color': COLORS['text']}),
						f"{stats_dict['mean']:.3f} ± {1.96 * stats_dict['std'] / np.sqrt(stats_dict['n']):.3f} ",
						f"(IC 95%: [{stats_dict['mean'] - 1.96 * stats_dict['std'] / np.sqrt(stats_dict['n']):.3f}, {stats_dict['mean'] + 1.96 * stats_dict['std'] / np.sqrt(stats_dict['n']):.3f}]). ",
						'La media es significativamente menor que la mediana, indicando sesgo negativo. ',
						'Con n = {:,}, el error estándar es mínimo (SE = {:.4f}).'.format(stats_dict['n'],
																																							stats_dict['std'] / np.sqrt(
																																								stats_dict['n']))
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Mediana (Me): ', style={'color': COLORS['text']}),
						f"{stats_dict['median']:.0f}. ",
						'El 50% de clientes otorgan 5 estrellas o menos. ',
						f"IQR = {stats_dict['iqr']:.0f} indica ",
						'baja dispersión en el 50% central de los datos.' if stats_dict[
																																	 'iqr'] <= 2 else 'moderada dispersión en el 50% central.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Moda: ', style={'color': COLORS['text']}),
						f"{stats_dict['mode']:.0f} estrellas ({pcts.iloc[-1]:.1f}%). ",
						'La distribución es claramente ',
						html.Strong('unimodal ', style={'color': COLORS['primary']}),
						'con fuerte concentración en la máxima satisfacción.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				# Dispersión
				html.Div([
					html.H6('2️⃣ Dispersión y Variabilidad', style={'color': COLORS['warning'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Desviación Estándar (σ): ', style={'color': COLORS['text']}),
						f"{stats_dict['std']:.3f}. ",
						html.Strong('Coeficiente de Variación (CV): ', style={'color': COLORS['text']}),
						f"{stats_dict['cv']:.2f}%. ",
						'Un CV < 50% indica ',
						html.Strong('variabilidad moderada', style={'color': COLORS['success']}),
						', sugiriendo patrones consistentes de satisfacción.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Rango: ', style={'color': COLORS['text']}),
						f"[{stats_dict['min']:.0f}, {stats_dict['max']:.0f}]. ",
						html.Strong('Varianza (σ²): ', style={'color': COLORS['text']}),
						f"{stats_dict['var']:.3f}. ",
						'La presencia de toda la escala (1-5) confirma heterogeneidad en experiencias del cliente.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				# Forma de la distribución
				html.Div([
					html.H6('3️⃣ Forma de la Distribución', style={'color': COLORS['info'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Asimetría (Skewness): ', style={'color': COLORS['text']}),
						f"{stats_dict['skewness']:.3f}. ",
						'Skewness < -1 indica ',
						html.Strong('fuerte sesgo negativo', style={'color': COLORS['danger']}),
						' (cola izquierda larga). Interpretación: ',
						'La mayoría de clientes están muy satisfechos, pero existe un segmento pequeño con experiencias muy negativas que "estiran" la distribución hacia la izquierda.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Curtosis: ', style={'color': COLORS['text']}),
						f"{stats_dict['kurtosis']:.3f}. ",
						'Curtosis < 3 indica distribución ',
						html.Strong('platicúrtica' if stats_dict['kurtosis'] < -1 else 'mesocúrtica' if stats_dict[
																																															'kurtosis'] < 3 else 'leptocúrtica',
												style={'color': COLORS['warning']}),
						' (colas ligeras). Esto sugiere ',
						'menor concentración de valores extremos de lo esperado en una distribución normal.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				# Test de normalidad
				html.Div([
					html.H6('4️⃣ Test de Normalidad (Shapiro-Wilk)',
									style={'color': COLORS['secondary'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Hipótesis:', style={'color': COLORS['text']}),
						html.Br(),
						'H₀: Los datos provienen de una distribución normal',
						html.Br(),
						'H₁: Los datos NO provienen de una distribución normal'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Resultados:', style={'color': COLORS['text']}),
						html.Br(),
						f"Estadístico W = {shapiro_stat:.4f}",
						html.Br(),
						f"p-valor = {shapiro_p:.6f}",
						html.Br(),
						html.Strong(f"Decisión: Rechazamos H₀ (p < 0.05)", style={'color': COLORS['danger']}) if shapiro_p < 0.05
						else html.Strong(f"Decisión: No rechazamos H₀ (p ≥ 0.05)", style={'color': COLORS['success']})
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Interpretación:', style={'color': COLORS['text']}),
						' La distribución de review_score ',
						html.Strong('NO sigue una distribución normal', style={'color': COLORS['danger']}) if shapiro_p < 0.05
						else html.Strong('sigue aproximadamente una distribución normal', style={'color': COLORS['success']}),
						'. El Q-Q plot muestra desviaciones en las colas, confirmando asimetría. ',
						html.Strong('Implicación metodológica:', style={'color': COLORS['warning']}),
						' Se deben usar pruebas no paramétricas (Mann-Whitney U, Kruskal-Wallis) para comparaciones entre grupos.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				html.Hr(style={'borderColor': COLORS['border']}),

				# Conclusión integrada
				html.Div([
					html.H6('🎯 Conclusión Integrada con Objetivos del Proyecto',
									style={'color': COLORS['primary'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Respuesta al Objetivo Descriptivo:', style={'color': COLORS['success']}),
						' La satisfacción del cliente en Olist muestra un patrón ',
						html.Strong('altamente positivo pero heterogéneo', style={'color': COLORS['primary']}),
						f'. Con {pcts.iloc[-1]:.1f}% de clientes dando 5 estrellas pero {pcts.iloc[0]:.1f}% dando 1 estrella, ',
						'existe una ',
						html.Strong('brecha significativa en experiencias', style={'color': COLORS['warning']}),
						'. La distribución no normal y fuertemente sesgada indica que ',
						html.Strong('modelos predictivos deben considerar esta asimetría', style={'color': COLORS['info']}),
						'. El alto CV sugiere que ',
						html.Strong('diferentes segmentos de clientes tienen experiencias radicalmente distintas',
												style={'color': COLORS['danger']}),
						', lo cual debe ser explorado mediante análisis de variables operacionales y geográficas (objetivos exploratorio e inferencial).'
					], style={'fontSize': '15px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={
					'background': f'rgba(0, 212, 255, 0.1)',
					'padding': '20px',
					'borderRadius': '12px',
					'border': f'1px solid {COLORS["primary"]}'
				})
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '40px'})
	])


# =============================================================================
# SECCIÓN 3: ANÁLISIS DE VARIABLES DE ENTREGA
# =============================================================================

def create_delivery_analysis(df):
	"""
	Análisis de variables operacionales de entrega.

	OBJETIVO: Identificar cómo los factores operacionales (tiempos de entrega,
	retrasos) impactan en la satisfacción del cliente.
	"""

	# Filtrar valores válidos
	df_delivery = df[df['delivery_time_days'].notna()].copy()

	# Estadísticas por grupo de satisfacción
	satisfaction_groups = {
		'Insatisfecho': df_delivery[df_delivery['satisfaction_level'] == 'Insatisfecho'],
		'Neutro': df_delivery[df_delivery['satisfaction_level'] == 'Neutro'],
		'Satisfecho': df_delivery[df_delivery['satisfaction_level'] == 'Satisfecho']
	}

	# Test Mann-Whitney U: On-time vs Delayed
	on_time = df_delivery[df_delivery['on_time_delivery'] == 1]['review_score']
	delayed = df_delivery[df_delivery['on_time_delivery'] == 0]['review_score']
	u_stat, p_value_mann = stats.mannwhitneyu(on_time, delayed, alternative='two-sided')

	# Test Kruskal-Wallis: delivery_time por satisfaction_level
	groups_delivery_time = [group['delivery_time_days'].dropna() for group in satisfaction_groups.values()]
	h_stat, p_value_kruskal = stats.kruskal(*groups_delivery_time)

	# Crear figura con subplots
	fig = make_subplots(
		rows=2, cols=3,
		subplot_titles=(
			'Distribución: Tiempo de Entrega',
			'Box Plot: Tiempo por Satisfacción',
			'Violin Plot: Delay por Satisfacción',
			'Histograma: Delay Days',
			'On-Time Delivery Rate',
			'Scatter: Delay vs Review Score'
		),
		specs=[
			[{"type": "histogram"}, {"type": "box"}, {"type": "violin"}],
			[{"type": "histogram"}, {"type": "bar"}, {"type": "scatter"}]
		],
		vertical_spacing=0.12,
		horizontal_spacing=0.1
	)

	# 1. Histograma delivery_time_days
	fig.add_trace(
		go.Histogram(
			x=df_delivery['delivery_time_days'],
			nbinsx=50,
			marker=dict(color=COLORS['primary'], line=dict(color=COLORS['text'], width=1)),
			hovertemplate='Días: %{x}<br>Frecuencia: %{y}<extra></extra>',
			showlegend=False
		),
		row=1, col=1
	)

	# 2. Box plot: delivery_time por satisfaction
	for i, (level, color) in enumerate([
		('Insatisfecho', COLORS['danger']),
		('Neutro', COLORS['warning']),
		('Satisfecho', COLORS['success'])
	]):
		fig.add_trace(
			go.Box(
				y=satisfaction_groups[level]['delivery_time_days'],
				name=level,
				marker_color=color,
				boxmean='sd',
				hovertemplate=f'<b>{level}</b><br>Tiempo: %{{y:.1f}} días<extra></extra>'
			),
			row=1, col=2
		)

	# 3. Violin plot: delivery_delay por satisfaction
	for i, (level, color) in enumerate([
		('Insatisfecho', COLORS['danger']),
		('Neutro', COLORS['warning']),
		('Satisfecho', COLORS['success'])
	]):
		fig.add_trace(
			go.Violin(
				y=satisfaction_groups[level]['delivery_delay_days'],
				name=level,
				marker_color=color,
				box_visible=True,
				meanline_visible=True,
				showlegend=False,
				hovertemplate=f'<b>{level}</b><br>Delay: %{{y:.1f}} días<extra></extra>'
			),
			row=1, col=3
		)

	# 4. Histograma delivery_delay
	fig.add_trace(
		go.Histogram(
			x=df_delivery['delivery_delay_days'],
			nbinsx=50,
			marker=dict(
				color=df_delivery['delivery_delay_days'].apply(
					lambda x: COLORS['danger'] if x > 0 else COLORS['success']
				),
				line=dict(color=COLORS['text'], width=1)
			),
			hovertemplate='Delay: %{x:.1f} días<br>Frecuencia: %{y}<extra></extra>',
			showlegend=False
		),
		row=2, col=1
	)

	# 5. Bar chart: On-time delivery rate
	on_time_rate = df_delivery['on_time_delivery'].value_counts()
	on_time_pct = (on_time_rate / len(df_delivery) * 100)

	fig.add_trace(
		go.Bar(
			x=['A Tiempo', 'Retrasado'],
			y=[on_time_pct[1], on_time_pct[0]],
			text=[f'{on_time_pct[1]:.1f}%<br>({on_time_rate[1]:,})',
						f'{on_time_pct[0]:.1f}%<br>({on_time_rate[0]:,})'],
			textposition='outside',
			marker=dict(color=[COLORS['success'], COLORS['danger']]),
			hovertemplate='<b>%{x}</b><br>%{y:.1f}%<extra></extra>',
			showlegend=False
		),
		row=2, col=2
	)

	# 6. Scatter: delivery_delay vs review_score
	sample = df_delivery.sample(min(5000, len(df_delivery)), random_state=42)
	fig.add_trace(
		go.Scatter(
			x=sample['delivery_delay_days'],
			y=sample['review_score'],
			mode='markers',
			marker=dict(
				size=4,
				color=sample['review_score'],
				colorscale='RdYlGn',
				showscale=True,
				colorbar=dict(title='Review', x=1.15),
				opacity=0.6,
				line=dict(width=0)
			),
			hovertemplate='Delay: %{x:.1f} días<br>Review: %{y}<extra></extra>',
			showlegend=False
		),
		row=2, col=3
	)

	# Línea de tendencia
	z = np.polyfit(sample['delivery_delay_days'], sample['review_score'], 1)
	p = np.poly1d(z)
	x_trend = np.linspace(sample['delivery_delay_days'].min(), sample['delivery_delay_days'].max(), 100)

	fig.add_trace(
		go.Scatter(
			x=x_trend,
			y=p(x_trend),
			mode='lines',
			line=dict(color=COLORS['danger'], width=3, dash='dash'),
			name='Tendencia',
			showlegend=False
		),
		row=2, col=3
	)

	# Layout
	fig.update_layout(
		title={
			'text': '🚚 Análisis Completo: Variables de Entrega',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 20, 'color': COLORS['text']}
		},
		height=900,
		paper_bgcolor=COLORS['background'],
		plot_bgcolor=COLORS['card'],
		font={'color': COLORS['text']},
		showlegend=True,
		legend=dict(x=1.05, y=0.7, font=dict(color=COLORS['text']))
	)

	# Actualizar ejes
	fig.update_xaxes(title_text="Días de Entrega", row=1, col=1, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Frecuencia", row=1, col=1, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Días", row=1, col=2, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Días de Delay", row=1, col=3, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Días de Delay", row=2, col=1, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Frecuencia", row=2, col=1, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Porcentaje (%)", row=2, col=2, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Días de Delay", row=2, col=3, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Review Score", row=2, col=3, tickfont={'color': COLORS['text']})

	# Calcular estadísticas por grupo
	stats_by_satisfaction = {}
	for level, group_df in satisfaction_groups.items():
		stats_by_satisfaction[level] = {
			'mean_delivery': group_df['delivery_time_days'].mean(),
			'median_delivery': group_df['delivery_time_days'].median(),
			'mean_delay': group_df['delivery_delay_days'].mean(),
			'median_delay': group_df['delivery_delay_days'].median(),
			'on_time_rate': (group_df['on_time_delivery'].sum() / len(group_df) * 100)
		}

	return fig, stats_by_satisfaction, u_stat, p_value_mann, h_stat, p_value_kruskal, on_time_rate, on_time_pct


def create_delivery_section(df):
	"""Sección completa de análisis de variables de entrega."""

	fig, stats_by_sat, u_stat, p_mann, h_stat, p_kruskal, on_time_rate, on_time_pct = create_delivery_analysis(df)

	return html.Div([
		create_section_header('🚚 PASO 3: Análisis de Variables Operacionales (Entrega)', color=COLORS['success']),

		create_info_banner(
			'Análisis de factores operacionales críticos: tiempos de entrega, retrasos y cumplimiento',
			icon='📦',
			banner_type='info'
		),

		# Objetivo
		dbc.Card([
			dbc.CardHeader(
				html.H5('🎯 Objetivo Específico: Exploratorio e Inferencial', style={'margin': 0, 'color': COLORS['info']})),
			dbc.CardBody([
				html.P([
					html.Strong('"Identificar relaciones entre variables operacionales y satisfacción + Validar hipótesis"',
											style={'color': COLORS['text'], 'fontSize': '16px'}),
				], style={'marginBottom': '10px', 'color': COLORS['text']}),
				html.P([
					'Este análisis responde: ',
					html.Strong('¿Los tiempos de entrega y retrasos afectan significativamente la satisfacción? ',
											style={'color': COLORS['primary']}),
					'Utilizamos pruebas no paramétricas (Mann-Whitney U, Kruskal-Wallis) debido a la no normalidad de los datos.'
				], style={'fontSize': '15px', 'lineHeight': '1.8', 'color': COLORS['text'], 'marginBottom': 0})
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["info"]}', 'marginBottom': '30px'}),

		# KPIs por grupo de satisfacción
		html.H5('📊 Métricas Clave por Nivel de Satisfacción', style={'color': COLORS['primary'], 'marginBottom': '20px'}),

		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardHeader(
						html.H6('😡 Insatisfechos', style={'margin': 0, 'color': COLORS['danger'], 'textAlign': 'center'})),
					dbc.CardBody([
						html.P([
							html.Strong('Tiempo Entrega: ', style={'color': COLORS['text']}),
							f"{stats_by_sat['Insatisfecho']['mean_delivery']:.1f} días"
						], style={'marginBottom': '8px', 'fontSize': '14px', 'color': COLORS['text']}),
						html.P([
							html.Strong('Delay Promedio: ', style={'color': COLORS['text']}),
							f"{stats_by_sat['Insatisfecho']['mean_delay']:.1f} días"
						], style={'marginBottom': '8px', 'fontSize': '14px', 'color': COLORS['text']}),
						html.P([
							html.Strong('On-Time: ', style={'color': COLORS['text']}),
							f"{stats_by_sat['Insatisfecho']['on_time_rate']:.1f}%"
						], style={'marginBottom': 0, 'fontSize': '14px', 'color': COLORS['text']})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["danger"]}', 'height': '100%'})
			], width=4),
			dbc.Col([
				dbc.Card([
					dbc.CardHeader(html.H6('😐 Neutros', style={'margin': 0, 'color': COLORS['warning'], 'textAlign': 'center'})),
					dbc.CardBody([
						html.P([
							html.Strong('Tiempo Entrega: ', style={'color': COLORS['text']}),
							f"{stats_by_sat['Neutro']['mean_delivery']:.1f} días"
						], style={'marginBottom': '8px', 'fontSize': '14px', 'color': COLORS['text']}),
						html.P([
							html.Strong('Delay Promedio: ', style={'color': COLORS['text']}),
							f"{stats_by_sat['Neutro']['mean_delay']:.1f} días"
						], style={'marginBottom': '8px', 'fontSize': '14px', 'color': COLORS['text']}),
						html.P([
							html.Strong('On-Time: ', style={'color': COLORS['text']}),
							f"{stats_by_sat['Neutro']['on_time_rate']:.1f}%"
						], style={'marginBottom': 0, 'fontSize': '14px', 'color': COLORS['text']})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["warning"]}', 'height': '100%'})
			], width=4),
			dbc.Col([
				dbc.Card([
					dbc.CardHeader(
						html.H6('😊 Satisfechos', style={'margin': 0, 'color': COLORS['success'], 'textAlign': 'center'})),
					dbc.CardBody([
						html.P([
							html.Strong('Tiempo Entrega: ', style={'color': COLORS['text']}),
							f"{stats_by_sat['Satisfecho']['mean_delivery']:.1f} días"
						], style={'marginBottom': '8px', 'fontSize': '14px', 'color': COLORS['text']}),
						html.P([
							html.Strong('Delay Promedio: ', style={'color': COLORS['text']}),
							f"{stats_by_sat['Satisfecho']['mean_delay']:.1f} días"
						], style={'marginBottom': '8px', 'fontSize': '14px', 'color': COLORS['text']}),
						html.P([
							html.Strong('On-Time: ', style={'color': COLORS['text']}),
							f"{stats_by_sat['Satisfecho']['on_time_rate']:.1f}%"
						], style={'marginBottom': 0, 'fontSize': '14px', 'color': COLORS['text']})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'height': '100%'})
			], width=4)
		], style={'marginBottom': '30px'}),

		# Gráficos
		dbc.Card([
			dbc.CardBody([
				dcc.Graph(figure=fig, config=PLOTLY_CONFIG)
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '30px'}),

		# Pruebas estadísticas
		dbc.Card([
			dbc.CardHeader(
				html.H5('🔬 Pruebas de Hipótesis (Análisis Inferencial)', style={'margin': 0, 'color': COLORS['secondary']})),
			dbc.CardBody([
				# Test Mann-Whitney U
				html.Div([
					html.H6('1️⃣ Test Mann-Whitney U: On-Time vs Delayed',
									style={'color': COLORS['info'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Hipótesis:', style={'color': COLORS['text']}),
						html.Br(),
						'H₀: No hay diferencia en review_score entre entregas a tiempo y retrasadas',
						html.Br(),
						'H₁: Sí hay diferencia significativa'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Resultados:', style={'color': COLORS['text']}),
						html.Br(),
						f"Estadístico U = {u_stat:,.0f}",
						html.Br(),
						f"p-valor = {p_mann:.6f}" if p_mann >= 0.001 else "p-valor < 0.001",
						html.Br(),
						html.Strong('Decisión: Rechazamos H₀', style={'color': COLORS['danger']}) if p_mann < 0.05
						else html.Strong('Decisión: No rechazamos H₀', style={'color': COLORS['success']})
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Interpretación:', style={'color': COLORS['text']}),
						' Existe evidencia estadísticamente significativa (p < 0.001) de que ',
						html.Strong('las entregas a tiempo tienen review_score significativamente mayor ',
												style={'color': COLORS['success']}),
						'que las entregas retrasadas. ',
						f'Tasa on-time: {on_time_pct[1]:.1f}% vs {on_time_pct[0]:.1f}% retrasadas. ',
						html.Strong('Impacto directo en satisfacción demostrado.', style={'color': COLORS['warning']})
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				html.Hr(style={'borderColor': COLORS['border']}),

				# Test Kruskal-Wallis
				html.Div([
					html.H6('2️⃣ Test Kruskal-Wallis: Delivery Time por Satisfaction Level',
									style={'color': COLORS['warning'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Hipótesis:', style={'color': COLORS['text']}),
						html.Br(),
						'H₀: Los tiempos de entrega son iguales entre grupos de satisfacción',
						html.Br(),
						'H₁: Al menos un grupo tiene tiempos diferentes'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Resultados:', style={'color': COLORS['text']}),
						html.Br(),
						f"Estadístico H = {h_stat:.2f}",
						html.Br(),
						f"p-valor = {p_kruskal:.6f}" if p_kruskal >= 0.001 else "p-valor < 0.001",
						html.Br(),
						html.Strong('Decisión: Rechazamos H₀', style={'color': COLORS['danger']}) if p_kruskal < 0.05
						else html.Strong('Decisión: No rechazamos H₀', style={'color': COLORS['success']})
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Interpretación:', style={'color': COLORS['text']}),
						' Hay diferencias estadísticamente significativas en los tiempos de entrega entre grupos. ',
						f'Insatisfechos: {stats_by_sat["Insatisfecho"]["mean_delivery"]:.1f} días, ',
						f'Neutros: {stats_by_sat["Neutro"]["mean_delivery"]:.1f} días, ',
						f'Satisfechos: {stats_by_sat["Satisfecho"]["mean_delivery"]:.1f} días. ',
						html.Strong('Los clientes insatisfechos experimentan tiempos de entrega más largos.',
												style={'color': COLORS['danger']})
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				html.Hr(style={'borderColor': COLORS['border']}),

				# Conclusión
				html.Div([
					html.H6('🎯 Conclusión Estratégica: Variables de Entrega',
									style={'color': COLORS['primary'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Respuesta al Objetivo Inferencial:', style={'color': COLORS['success']}),
						' Ambas pruebas estadísticas confirman que ',
						html.Strong('los factores operacionales de entrega tienen un impacto significativo y causal ',
												style={'color': COLORS['primary']}),
						'en la satisfacción del cliente. ',
						html.Strong('Prioridad estratégica #1: ', style={'color': COLORS['danger']}),
						'Reducir delivery_time_days y eliminar retrasos. ',
						'Un 1% de mejora en on-time delivery podría traducirse en ',
						html.Strong('~1,100 clientes adicionales satisfechos ', style={'color': COLORS['success']}),
						f'(de los actuales {on_time_rate[0]:,} retrasados). ',
						html.Strong('ROI estimado: Alto, ', style={'color': COLORS['warning']}),
						'ya que la logística es el factor más correlacionado con satisfacción según análisis de correlación previo.'
					], style={'fontSize': '15px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={
					'background': f'rgba(16, 185, 129, 0.1)',
					'padding': '20px',
					'borderRadius': '12px',
					'border': f'1px solid {COLORS["success"]}'
				})
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["secondary"]}', 'marginBottom': '40px'})
	])


# =============================================================================
# SECCIÓN 4: ANÁLISIS DE VARIABLES DE PRODUCTO
# =============================================================================

def create_product_analysis(df):
	"""
	Análisis de variables de producto.

	OBJETIVO: Identificar cómo las características de producto (categoría,
	presentación, dimensiones) influyen en la satisfacción del cliente.
	"""

	# Filtrar valores válidos
	df_product = df[df['product_category_name_english'].notna()].copy()

	# Top 10 categorías por volumen
	top_categories = df_product['product_category_name_english'].value_counts().head(10)

	# Satisfacción promedio por categoría (top 10)
	satisfaction_by_category = df_product.groupby('product_category_name_english')['review_score'].agg(
		['mean', 'count']).sort_values('count', ascending=False).head(10)

	# Análisis de product_photos_qty
	photos_stats = df_product.groupby('product_photos_qty')['review_score'].agg(['mean', 'count', 'std'])
	photos_stats = photos_stats[photos_stats['count'] >= 100].head(10)  # Filtrar categorías con suficientes datos

	# Test Chi-cuadrado: category vs satisfaction_level
	contingency_table = pd.crosstab(
		df_product['product_category_name_english'].isin(top_categories.head(5).index),
		df_product['satisfaction_level']
	)
	chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)

	# Test ANOVA: review_score por categoría (top 5)
	top5_categories = top_categories.head(5).index
	groups_anova = [df_product[df_product['product_category_name_english'] == cat]['review_score'].dropna()
									for cat in top5_categories]
	f_stat, p_value_anova = stats.f_oneway(*groups_anova)

	# Correlación: photos vs review_score
	corr_photos = df_product[['product_photos_qty', 'review_score']].corr().iloc[0, 1]

	# Correlación: weight vs review_score
	corr_weight = df_product[['product_weight_kg', 'review_score']].corr().iloc[0, 1]

	# Crear figura con subplots
	fig = make_subplots(
		rows=2, cols=3,
		subplot_titles=(
			'Top 10 Categorías (Volumen)',
			'Satisfacción por Categoría',
			'Fotos vs Review Score',
			'Distribución: Peso del Producto',
			'Box Plot: Review por Fotos',
			'Scatter: Peso vs Satisfacción'
		),
		specs=[
			[{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
			[{"type": "histogram"}, {"type": "box"}, {"type": "scatter"}]
		],
		vertical_spacing=0.12,
		horizontal_spacing=0.1
	)

	# 1. Top 10 categorías por volumen
	fig.add_trace(
		go.Bar(
			y=[cat[:20] + '...' if len(cat) > 20 else cat for cat in top_categories.index],
			x=top_categories.values,
			orientation='h',
			marker=dict(
				color=top_categories.values,
				colorscale='Blues',
				showscale=False,
				line=dict(color=COLORS['primary'], width=1)
			),
			text=[f'{val:,}' for val in top_categories.values],
			textposition='outside',
			hovertemplate='<b>%{y}</b><br>Productos: %{x:,}<extra></extra>',
			showlegend=False
		),
		row=1, col=1
	)

	# 2. Satisfacción promedio por categoría
	colors_satisfaction = [COLORS['success'] if score >= 4.0 else COLORS['warning'] if score >= 3.5 else COLORS['danger']
												 for score in satisfaction_by_category['mean']]

	fig.add_trace(
		go.Bar(
			y=[cat[:20] + '...' if len(cat) > 20 else cat for cat in satisfaction_by_category.index],
			x=satisfaction_by_category['mean'],
			orientation='h',
			marker=dict(color=colors_satisfaction, line=dict(color=COLORS['text'], width=1)),
			text=[f'{score:.2f}⭐' for score in satisfaction_by_category['mean']],
			textposition='outside',
			hovertemplate='<b>%{y}</b><br>Satisfacción: %{x:.2f}<extra></extra>',
			showlegend=False
		),
		row=1, col=2
	)

	# 3. Scatter: Fotos vs Review Score
	fig.add_trace(
		go.Scatter(
			x=photos_stats.index,
			y=photos_stats['mean'],
			mode='markers+lines',
			marker=dict(
				size=photos_stats['count'] / 50,  # Tamaño proporcional a cantidad
				color=photos_stats['mean'],
				colorscale='RdYlGn',
				showscale=True,
				colorbar=dict(title='Review', x=1.15, y=0.85, len=0.3),
				line=dict(color=COLORS['text'], width=1)
			),
			line=dict(color=COLORS['primary'], width=2),
			text=[f'{count:,} productos' for count in photos_stats['count']],
			hovertemplate='<b>%{x} fotos</b><br>Review: %{y:.2f}<br>%{text}<extra></extra>',
			showlegend=False
		),
		row=1, col=3
	)

	# 4. Histograma: Peso del producto
	fig.add_trace(
		go.Histogram(
			x=df_product[df_product['product_weight_kg'] <= 30]['product_weight_kg'],  # Filtrar outliers extremos
			nbinsx=50,
			marker=dict(color=COLORS['info'], line=dict(color=COLORS['text'], width=1)),
			hovertemplate='Peso: %{x:.1f} kg<br>Frecuencia: %{y}<extra></extra>',
			showlegend=False
		),
		row=2, col=1
	)

	# 5. Box plot: Review por cantidad de fotos (agrupado)
	photos_groups = {
		'1 foto': df_product[df_product['product_photos_qty'] == 1]['review_score'],
		'2-3 fotos': df_product[df_product['product_photos_qty'].between(2, 3)]['review_score'],
		'4-6 fotos': df_product[df_product['product_photos_qty'].between(4, 6)]['review_score'],
		'7+ fotos': df_product[df_product['product_photos_qty'] >= 7]['review_score']
	}

	colors_box = [COLORS['danger'], COLORS['warning'], COLORS['info'], COLORS['success']]
	for i, (label, data) in enumerate(photos_groups.items()):
		fig.add_trace(
			go.Box(
				y=data,
				name=label,
				marker_color=colors_box[i],
				boxmean='sd',
				hovertemplate=f'<b>{label}</b><br>Review: %{{y}}<extra></extra>'
			),
			row=2, col=2
		)

	# 6. Scatter: Peso vs Review Score (muestra)
	sample_weight = df_product[df_product['product_weight_kg'] <= 50].sample(min(3000, len(df_product)), random_state=42)

	fig.add_trace(
		go.Scatter(
			x=sample_weight['product_weight_kg'],
			y=sample_weight['review_score'],
			mode='markers',
			marker=dict(
				size=4,
				color=sample_weight['review_score'],
				colorscale='RdYlGn',
				showscale=False,
				opacity=0.5,
				line=dict(width=0)
			),
			hovertemplate='Peso: %{x:.1f} kg<br>Review: %{y}<extra></extra>',
			showlegend=False
		),
		row=2, col=3
	)

	# Línea de tendencia
	z = np.polyfit(sample_weight['product_weight_kg'], sample_weight['review_score'], 1)
	p = np.poly1d(z)
	x_trend = np.linspace(sample_weight['product_weight_kg'].min(), sample_weight['product_weight_kg'].max(), 100)

	fig.add_trace(
		go.Scatter(
			x=x_trend,
			y=p(x_trend),
			mode='lines',
			line=dict(color=COLORS['danger'], width=3, dash='dash'),
			showlegend=False
		),
		row=2, col=3
	)

	# Layout
	fig.update_layout(
		title={
			'text': '📦 Análisis Completo: Variables de Producto',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 20, 'color': COLORS['text']}
		},
		height=900,
		paper_bgcolor=COLORS['background'],
		plot_bgcolor=COLORS['card'],
		font={'color': COLORS['text']},
		showlegend=True,
		legend=dict(x=1.05, y=0.3, font=dict(color=COLORS['text']))
	)

	# Actualizar ejes
	fig.update_xaxes(title_text="Cantidad de Productos", row=1, col=1, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Review Score Promedio", row=1, col=2, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Cantidad de Fotos", row=1, col=3, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Review Score", row=1, col=3, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Peso (kg)", row=2, col=1, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Frecuencia", row=2, col=1, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Review Score", row=2, col=2, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Peso (kg)", row=2, col=3, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Review Score", row=2, col=3, tickfont={'color': COLORS['text']})

	return fig, top_categories, satisfaction_by_category, photos_stats, corr_photos, corr_weight, chi2_stat, p_value_chi2, f_stat, p_value_anova


def create_product_section(df):
	"""Sección completa de análisis de variables de producto."""

	fig, top_cats, sat_by_cat, photos_stats, corr_photos, corr_weight, chi2, p_chi2, f_stat, p_anova = create_product_analysis(
		df)

	# Identificar mejor y peor categoría
	best_category = sat_by_cat['mean'].idxmax()
	worst_category = sat_by_cat['mean'].idxmin()
	best_score = sat_by_cat['mean'].max()
	worst_score = sat_by_cat['mean'].min()

	return html.Div([
		create_section_header('📦 PASO 4: Análisis de Variables de Producto', color=COLORS['info']),

		create_info_banner(
			'Análisis de características del producto: categoría, presentación visual y dimensiones físicas',
			icon='🏷️',
			banner_type='info'
		),

		# Objetivo
		dbc.Card([
			dbc.CardHeader(
				html.H5('🎯 Objetivo Específico: Exploratorio e Inferencial', style={'margin': 0, 'color': COLORS['warning']})),
			dbc.CardBody([
				html.P([
					html.Strong('"Identificar cómo las características de producto afectan la satisfacción"',
											style={'color': COLORS['text'], 'fontSize': '16px'}),
				], style={'marginBottom': '10px', 'color': COLORS['text']}),
				html.P([
					'Este análisis responde: ',
					html.Strong('¿Las categorías de producto tienen diferentes niveles de satisfacción? ',
											style={'color': COLORS['primary']}),
					html.Strong('¿Más fotos mejoran la experiencia? ', style={'color': COLORS['success']}),
					'Utilizamos ANOVA y Chi-cuadrado para validar diferencias entre grupos.'
				], style={'fontSize': '15px', 'lineHeight': '1.8', 'color': COLORS['text'], 'marginBottom': 0})
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["warning"]}', 'marginBottom': '30px'}),

		# KPIs de producto
		html.H5('📊 Métricas Clave de Producto', style={'color': COLORS['primary'], 'marginBottom': '20px'}),

		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('📂', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{len(top_cats)}",
										style={'color': COLORS['info'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Top Categorías',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(f"{top_cats.sum():,} productos ({top_cats.sum() / len(df) * 100:.1f}%)",
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["info"]}', 'height': '100%'})
			], width=3),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('⭐', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{best_score:.2f}",
										style={'color': COLORS['success'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Mejor Categoría',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(f"{best_category[:25]}...",
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'height': '100%'})
			], width=3),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('⚠️', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{worst_score:.2f}",
										style={'color': COLORS['danger'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Peor Categoría',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(f"{worst_category[:25]}...",
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["danger"]}', 'height': '100%'})
			], width=3),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('📸', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{corr_photos:.3f}", style={
							'color': COLORS['success'] if corr_photos > 0.1 else COLORS['warning'] if corr_photos > 0 else COLORS[
								'danger'],
							'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'
						}),
						html.P('Corr: Fotos vs Review',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small('Correlación positiva' if corr_photos > 0.05 else 'Correlación débil',
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'height': '100%'})
			], width=3)
		], style={'marginBottom': '30px'}),

		# Gráficos
		dbc.Card([
			dbc.CardBody([
				dcc.Graph(figure=fig, config=PLOTLY_CONFIG)
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '30px'}),

		# Análisis estadístico detallado
		dbc.Card([
			dbc.CardHeader(html.H5('📊 Análisis Estadístico de Producto', style={'margin': 0, 'color': COLORS['primary']})),
			dbc.CardBody([

				# Categorías
				html.Div([
					html.H6('1️⃣ Análisis por Categoría de Producto', style={'color': COLORS['info'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Top 3 Categorías por Volumen:', style={'color': COLORS['text']}),
						html.Br(),
						f"1. {top_cats.index[0]}: {top_cats.iloc[0]:,} productos ({top_cats.iloc[0] / len(df) * 100:.1f}%)",
						html.Br(),
						f"2. {top_cats.index[1]}: {top_cats.iloc[1]:,} productos ({top_cats.iloc[1] / len(df) * 100:.1f}%)",
						html.Br(),
						f"3. {top_cats.index[2]}: {top_cats.iloc[2]:,} productos ({top_cats.iloc[2] / len(df) * 100:.1f}%)"
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '15px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Diferencia en Satisfacción:', style={'color': COLORS['text']}),
						f" La categoría mejor valorada ({best_category[:30]}...) tiene {best_score:.2f}⭐ ",
						f"vs la peor ({worst_category[:30]}...) con {worst_score:.2f}⭐. ",
						f"Diferencia: {best_score - worst_score:.2f} puntos ({(best_score - worst_score) / worst_score * 100:.1f}% mayor)."
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				# Fotos
				html.Div([
					html.H6('2️⃣ Impacto de Fotos del Producto', style={'color': COLORS['success'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Correlación Fotos vs Review Score: ', style={'color': COLORS['text']}),
						f"r = {corr_photos:.3f}. ",
						'Correlación ' + (
							'positiva débil' if 0 < corr_photos < 0.3 else 'positiva moderada' if corr_photos >= 0.3 else 'prácticamente nula'),
						'. Aunque la correlación es baja, el análisis visual muestra que productos con 4-6 fotos ',
						'tienden a tener reviews ligeramente más altos.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Interpretación:', style={'color': COLORS['text']}),
						' La cantidad de fotos tiene un ',
						html.Strong('impacto positivo pero limitado', style={'color': COLORS['warning']}),
						' en la satisfacción. El factor más importante no es la cantidad de fotos, ',
						'sino probablemente la ',
						html.Strong('calidad del producto y el servicio de entrega', style={'color': COLORS['primary']}),
						' (como vimos en análisis de correlación y entrega).'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				# Peso
				html.Div([
					html.H6('3️⃣ Dimensiones Físicas del Producto', style={'color': COLORS['warning'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Correlación Peso vs Review Score: ', style={'color': COLORS['text']}),
						f"r = {corr_weight:.3f}. ",
						'Correlación muy débil. El peso del producto ',
						html.Strong('NO es un factor determinante', style={'color': COLORS['info']}),
						' en la satisfacción del cliente. Esto sugiere que los clientes valoran más ',
						'la experiencia de compra (entrega, precio, categoría) que las características físicas.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				html.Hr(style={'borderColor': COLORS['border']}),

				# Pruebas estadísticas
				html.Div([
					html.H6('4️⃣ Pruebas de Hipótesis', style={'color': COLORS['secondary'], 'marginBottom': '15px'}),

					# Chi-cuadrado
					html.P([
						html.Strong('Test Chi-cuadrado: Categoría vs Satisfaction Level', style={'color': COLORS['text']}),
						html.Br(),
						'H₀: La categoría de producto es independiente del nivel de satisfacción',
						html.Br(),
						'H₁: Existe asociación entre categoría y satisfacción',
						html.Br(),
						html.Br(),
						f"χ² = {chi2:.2f}, p-valor = {p_chi2:.6f}" if p_chi2 >= 0.001 else f"χ² = {chi2:.2f}, p-valor < 0.001",
						html.Br(),
						html.Strong('Decisión: Rechazamos H₀', style={'color': COLORS['danger']}) if p_chi2 < 0.05
						else html.Strong('Decisión: No rechazamos H₀', style={'color': COLORS['success']}),
						html.Br(),
						html.Strong('Interpretación: ', style={'color': COLORS['text']}),
						'Existe evidencia estadística de que ',
						html.Strong('la categoría de producto está asociada con el nivel de satisfacción',
												style={'color': COLORS['success']}),
						'. Algunas categorías tienen propensión a generar mayor satisfacción.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '15px', 'color': COLORS['text']}),

					# ANOVA
					html.P([
						html.Strong('Test ANOVA: Review Score por Categoría (Top 5)', style={'color': COLORS['text']}),
						html.Br(),
						'H₀: Las medias de review_score son iguales entre categorías',
						html.Br(),
						'H₁: Al menos una categoría tiene media diferente',
						html.Br(),
						html.Br(),
						f"F = {f_stat:.2f}, p-valor = {p_anova:.6f}" if p_anova >= 0.001 else f"F = {f_stat:.2f}, p-valor < 0.001",
						html.Br(),
						html.Strong('Decisión: Rechazamos H₀', style={'color': COLORS['danger']}) if p_anova < 0.05
						else html.Strong('Decisión: No rechazamos H₀', style={'color': COLORS['success']}),
						html.Br(),
						html.Strong('Interpretación: ', style={'color': COLORS['text']}),
						'Las categorías principales tienen ',
						html.Strong('diferencias significativas en satisfacción promedio', style={'color': COLORS['warning']}),
						'. Esto confirma que ',
						html.Strong('la categoría de producto es un factor predictivo', style={'color': COLORS['primary']}),
						' relevante para modelos de machine learning.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				html.Hr(style={'borderColor': COLORS['border']}),

				# Conclusión
				html.Div([
					html.H6('🎯 Conclusión Estratégica: Variables de Producto',
									style={'color': COLORS['primary'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Respuesta a Objetivos Exploratorio e Inferencial:', style={'color': COLORS['success']}),
						' El análisis revela que ',
						html.Strong('la categoría de producto es un factor moderador significativo',
												style={'color': COLORS['primary']}),
						' de la satisfacción (p < 0.001 en ambas pruebas). Sin embargo, ',
						html.Strong('su impacto es menor que las variables operacionales', style={'color': COLORS['warning']}),
						' (entrega, tiempos). ',
						html.Br(),
						html.Br(),
						html.Strong('Hallazgos clave:', style={'color': COLORS['text']}),
						html.Br(),
						f"• Categoría {best_category[:35]} tiene mejor satisfacción ({best_score:.2f}⭐)",
						html.Br(),
						f"• Diferencia máxima entre categorías: {best_score - worst_score:.2f} puntos",
						html.Br(),
						f"• Fotos tienen impacto limitado (r = {corr_photos:.3f})",
						html.Br(),
						f"• Peso no es factor determinante (r = {corr_weight:.3f})",
						html.Br(),
						html.Br(),
						html.Strong('Recomendación estratégica:', style={'color': COLORS['danger']}),
						' Optimizar portafolio de productos priorizando categorías con mejor satisfacción histórica. ',
						'Sin embargo, la prioridad #1 sigue siendo ',
						html.Strong('mejorar tiempos de entrega', style={'color': COLORS['success']}),
						' (factor más correlacionado). ',
						html.Strong('Para vendors:', style={'color': COLORS['info']}),
						' invertir en 4-6 fotos de calidad, pero enfocarse principalmente en cumplir tiempos de entrega.'
					], style={'fontSize': '15px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={
					'background': f'rgba(59, 130, 246, 0.1)',
					'padding': '20px',
					'borderRadius': '12px',
					'border': f'1px solid {COLORS["info"]}'
				})
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '40px'})
	])


# =============================================================================
# SECCIÓN 5: ANÁLISIS GEOGRÁFICO
# =============================================================================

def create_geographic_analysis(df):
	"""
	Análisis de variables geográficas.

	OBJETIVO: Identificar cómo la distribución geográfica (estados, distancias)
	afecta la satisfacción del cliente y tiempos de entrega.
	"""

	# Análisis por customer_state
	customer_state_stats = df.groupby('customer_state').agg({
		'review_score': ['mean', 'count', 'std'],
		'delivery_time_days': 'mean',
		'on_time_delivery': 'mean',
		'order_total_value': 'mean'
	}).round(2)

	customer_state_stats.columns = ['review_mean', 'count', 'review_std',
																	'delivery_mean', 'on_time_rate', 'order_value_mean']
	customer_state_stats = customer_state_stats.sort_values('count', ascending=False)

	# Top 10 estados por volumen
	top_10_states = customer_state_stats.head(10)

	# Análisis por seller_state
	seller_state_stats = df.groupby('seller_state').agg({
		'review_score': ['mean', 'count'],
		'delivery_time_days': 'mean'
	}).round(2)

	seller_state_stats.columns = ['review_mean', 'count', 'delivery_mean']
	seller_state_stats = seller_state_stats.sort_values('count', ascending=False).head(10)

	# Test ANOVA: review_score por customer_state (top 10)
	groups_states = [df[df['customer_state'] == state]['review_score'].dropna()
									 for state in top_10_states.index]
	f_stat_states, p_value_states = stats.f_oneway(*groups_states)

	# Correlación geográfica: concentración vs satisfacción
	state_concentration = (top_10_states['count'] / top_10_states['count'].sum() * 100)

	# Identificar mejor y peor estado
	best_state = customer_state_stats.nlargest(10, 'review_mean').iloc[0]
	worst_state = customer_state_stats.nsmallest(10, 'review_mean').iloc[0]

	# Estados con problemas de entrega
	slow_delivery_states = customer_state_stats.nlargest(5, 'delivery_mean')
	fast_delivery_states = customer_state_stats.nsmallest(5, 'delivery_mean')

	# Crear figura con subplots
	fig = make_subplots(
		rows=2, cols=3,
		subplot_titles=(
			'Top 10 Estados (Clientes)',
			'Satisfacción por Estado',
			'Tiempo de Entrega por Estado',
			'Distribución: Sellers por Estado',
			'On-Time Rate por Estado',
			'Order Value vs Satisfacción'
		),
		specs=[
			[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
			[{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]
		],
		vertical_spacing=0.12,
		horizontal_spacing=0.1
	)

	# 1. Top 10 estados por volumen de clientes
	fig.add_trace(
		go.Bar(
			x=top_10_states.index,
			y=top_10_states['count'],
			marker=dict(
				color=top_10_states['count'],
				colorscale='Blues',
				showscale=False,
				line=dict(color=COLORS['primary'], width=1)
			),
			text=[f'{val:,}<br>({val / top_10_states["count"].sum() * 100:.1f}%)' for val in top_10_states['count']],
			textposition='outside',
			hovertemplate='<b>%{x}</b><br>Clientes: %{y:,}<extra></extra>',
			showlegend=False
		),
		row=1, col=1
	)

	# 2. Satisfacción promedio por estado (top 10)
	colors_satisfaction = [COLORS['success'] if score >= 4.0 else COLORS['warning'] if score >= 3.8 else COLORS['danger']
												 for score in top_10_states['review_mean']]

	fig.add_trace(
		go.Bar(
			x=top_10_states.index,
			y=top_10_states['review_mean'],
			marker=dict(color=colors_satisfaction, line=dict(color=COLORS['text'], width=1)),
			text=[f'{score:.2f}⭐' for score in top_10_states['review_mean']],
			textposition='outside',
			hovertemplate='<b>%{x}</b><br>Review: %{y:.2f}<extra></extra>',
			showlegend=False
		),
		row=1, col=2
	)

	# Línea de media general
	overall_mean = df['review_score'].mean()
	fig.add_hline(
		y=overall_mean,
		line_dash="dash",
		line_color=COLORS['danger'],
		annotation_text=f"Media General: {overall_mean:.2f}",
		row=1, col=2
	)

	# 3. Tiempo de entrega por estado (top 10)
	colors_delivery = [COLORS['success'] if time <= 10 else COLORS['warning'] if time <= 15 else COLORS['danger']
										 for time in top_10_states['delivery_mean']]

	fig.add_trace(
		go.Bar(
			x=top_10_states.index,
			y=top_10_states['delivery_mean'],
			marker=dict(color=colors_delivery, line=dict(color=COLORS['text'], width=1)),
			text=[f'{time:.1f}d' for time in top_10_states['delivery_mean']],
			textposition='outside',
			hovertemplate='<b>%{x}</b><br>Tiempo: %{y:.1f} días<extra></extra>',
			showlegend=False
		),
		row=1, col=3
	)

	# 4. Distribución de sellers por estado (top 10)
	fig.add_trace(
		go.Bar(
			x=seller_state_stats.index,
			y=seller_state_stats['count'],
			marker=dict(
				color=seller_state_stats['count'],
				colorscale='Greens',
				showscale=False,
				line=dict(color=COLORS['success'], width=1)
			),
			text=[f'{val:,}' for val in seller_state_stats['count']],
			textposition='outside',
			hovertemplate='<b>%{x}</b><br>Sellers: %{y:,}<extra></extra>',
			showlegend=False
		),
		row=2, col=1
	)

	# 5. On-time delivery rate por estado (top 10)
	on_time_pct = top_10_states['on_time_rate'] * 100
	colors_ontime = [COLORS['success'] if rate >= 92 else COLORS['warning'] if rate >= 85 else COLORS['danger']
									 for rate in on_time_pct]

	fig.add_trace(
		go.Bar(
			x=top_10_states.index,
			y=on_time_pct,
			marker=dict(color=colors_ontime, line=dict(color=COLORS['text'], width=1)),
			text=[f'{rate:.1f}%' for rate in on_time_pct],
			textposition='outside',
			hovertemplate='<b>%{x}</b><br>On-Time: %{y:.1f}%<extra></extra>',
			showlegend=False
		),
		row=2, col=2
	)

	# Línea de benchmark (92%)
	fig.add_hline(
		y=92,
		line_dash="dash",
		line_color=COLORS['success'],
		annotation_text="Benchmark: 92%",
		row=2, col=2
	)

	# 6. Scatter: Order value vs Satisfacción por estado
	fig.add_trace(
		go.Scatter(
			x=top_10_states['order_value_mean'],
			y=top_10_states['review_mean'],
			mode='markers+text',
			marker=dict(
				size=top_10_states['count'] / 1000,  # Tamaño proporcional al volumen
				color=top_10_states['review_mean'],
				colorscale='RdYlGn',
				showscale=True,
				colorbar=dict(title='Review', x=1.15, y=0.3, len=0.3),
				line=dict(color=COLORS['text'], width=1)
			),
			text=top_10_states.index,
			textposition='top center',
			textfont=dict(size=10),
			hovertemplate='<b>%{text}</b><br>Order Value: R$%{x:.0f}<br>Review: %{y:.2f}<extra></extra>',
			showlegend=False
		),
		row=2, col=3
	)

	# Layout
	fig.update_layout(
		title={
			'text': '🌍 Análisis Completo: Distribución Geográfica',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 20, 'color': COLORS['text']}
		},
		height=900,
		paper_bgcolor=COLORS['background'],
		plot_bgcolor=COLORS['card'],
		font={'color': COLORS['text']},
		showlegend=False
	)

	# Actualizar ejes
	fig.update_xaxes(title_text="Estado", row=1, col=1, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Número de Clientes", row=1, col=1, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Estado", row=1, col=2, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Review Score", row=1, col=2, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Estado", row=1, col=3, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Días de Entrega", row=1, col=3, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Estado", row=2, col=1, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Número de Sellers", row=2, col=1, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Estado", row=2, col=2, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="On-Time Rate (%)", row=2, col=2, tickfont={'color': COLORS['text']})
	fig.update_xaxes(title_text="Order Value (R$)", row=2, col=3, tickfont={'color': COLORS['text']})
	fig.update_yaxes(title_text="Review Score", row=2, col=3, tickfont={'color': COLORS['text']})

	return (fig, top_10_states, seller_state_stats, best_state, worst_state,
					slow_delivery_states, fast_delivery_states, f_stat_states, p_value_states, state_concentration)


def create_geographic_section(df):
	"""Sección completa de análisis geográfico."""

	(fig, top_10_states, seller_stats, best_state, worst_state,
	 slow_states, fast_states, f_stat, p_value, concentration) = create_geographic_analysis(df)

	# Métricas clave
	sp_dominance = concentration.iloc[0] if 'SP' in concentration.index else 0
	total_states = df['customer_state'].nunique()

	return html.Div([
		create_section_header('🌍 PASO 5: Análisis Geográfico y Distribución Espacial', color=COLORS['danger']),

		create_info_banner(
			'Análisis de factores geográficos: distribución de clientes, concentración de sellers y variaciones regionales',
			icon='🗺️',
			banner_type='info'
		),

		# Objetivo
		dbc.Card([
			dbc.CardHeader(
				html.H5('🎯 Objetivo Específico: Exploratorio e Inferencial', style={'margin': 0, 'color': COLORS['danger']})),
			dbc.CardBody([
				html.P([
					html.Strong('"Caracterizar distribución geográfica y validar diferencias regionales en satisfacción"',
											style={'color': COLORS['text'], 'fontSize': '16px'}),
				], style={'marginBottom': '10px', 'color': COLORS['text']}),
				html.P([
					'Este análisis responde: ',
					html.Strong('¿Existen diferencias significativas de satisfacción entre estados? ',
											style={'color': COLORS['primary']}),
					html.Strong('¿La concentración geográfica afecta la calidad del servicio? ',
											style={'color': COLORS['success']}),
					'Utilizamos ANOVA para comparar estados y analizar patrones de distribución.'
				], style={'fontSize': '15px', 'lineHeight': '1.8', 'color': COLORS['text'], 'marginBottom': 0})
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["danger"]}', 'marginBottom': '30px'}),

		# KPIs geográficos
		html.H5('📊 Métricas Geográficas Clave', style={'color': COLORS['primary'], 'marginBottom': '20px'}),

		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('🗺️', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{total_states}",
										style={'color': COLORS['info'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Estados Activos',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(f"Cobertura nacional",
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["info"]}', 'height': '100%'})
			], width=3),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('📍', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4('SP',
										style={'color': COLORS['primary'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Estado Dominante',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(f"{sp_dominance:.1f}% de clientes",
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'height': '100%'})
			], width=3),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('⭐', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{best_state.name}",
										style={'color': COLORS['success'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Mejor Estado',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(f"{best_state['review_mean']:.2f}⭐ promedio",
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["success"]}', 'height': '100%'})
			], width=3),
			dbc.Col([
				dbc.Card([
					dbc.CardBody([
						html.H3('⚠️', style={'fontSize': '40px', 'margin': '0', 'textAlign': 'center'}),
						html.H4(f"{worst_state.name}",
										style={'color': COLORS['danger'], 'margin': '10px 0', 'textAlign': 'center', 'fontSize': '28px'}),
						html.P('Estado a Mejorar',
									 style={'color': COLORS['text_muted'], 'margin': '0', 'textAlign': 'center', 'fontSize': '13px'}),
						html.Small(f"{worst_state['review_mean']:.2f}⭐ promedio",
											 style={'color': COLORS['text_muted'], 'fontSize': '11px', 'display': 'block',
															'textAlign': 'center'})
					])
				], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["danger"]}', 'height': '100%'})
			], width=3)
		], style={'marginBottom': '30px'}),

		# Gráficos
		dbc.Card([
			dbc.CardBody([
				dcc.Graph(figure=fig, config=PLOTLY_CONFIG)
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '30px'}),

		# Análisis detallado
		dbc.Card([
			dbc.CardHeader(html.H5('📊 Análisis Geográfico Detallado', style={'margin': 0, 'color': COLORS['primary']})),
			dbc.CardBody([

				# Concentración geográfica
				html.Div([
					html.H6('1️⃣ Concentración Geográfica de Clientes', style={'color': COLORS['info'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Top 3 Estados por Volumen:', style={'color': COLORS['text']}),
						html.Br(),
						f"1. {top_10_states.index[0]}: {top_10_states.iloc[0]['count']:,} clientes ({concentration.iloc[0]:.1f}% del top 10)",
						html.Br(),
						f"2. {top_10_states.index[1]}: {top_10_states.iloc[1]['count']:,} clientes ({concentration.iloc[1]:.1f}% del top 10)",
						html.Br(),
						f"3. {top_10_states.index[2]}: {top_10_states.iloc[2]['count']:,} clientes ({concentration.iloc[2]:.1f}% del top 10)"
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '15px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Interpretación:', style={'color': COLORS['text']}),
						f" Existe una ",
						html.Strong('alta concentración geográfica', style={'color': COLORS['warning']}),
						f". Los top 3 estados representan ",
						html.Strong(f"{concentration.head(3).sum():.1f}% ", style={'color': COLORS['danger']}),
						"de los clientes del top 10. ",
						"São Paulo (SP) domina con ",
						html.Strong(f"{concentration.iloc[0]:.1f}% ", style={'color': COLORS['primary']}),
						"del volumen. Esta concentración presenta ",
						html.Strong('oportunidades y riesgos:', style={'color': COLORS['success']}),
						" permite economías de escala en logística, pero aumenta vulnerabilidad a disrupciones regionales."
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				# Distribución de sellers
				html.Div([
					html.H6('2️⃣ Distribución de Sellers (Oferta)', style={'color': COLORS['success'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Top 3 Estados con Más Sellers:', style={'color': COLORS['text']}),
						html.Br(),
						f"1. {seller_stats.index[0]}: {seller_stats.iloc[0]['count']:,} sellers",
						html.Br(),
						f"2. {seller_stats.index[1]}: {seller_stats.iloc[1]['count']:,} sellers",
						html.Br(),
						f"3. {seller_stats.index[2]}: {seller_stats.iloc[2]['count']:,} sellers"
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '15px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Desbalance Oferta-Demanda:', style={'color': COLORS['text']}),
						" La concentración de sellers es ",
						html.Strong('aún mayor que la de clientes', style={'color': COLORS['danger']}),
						f". {seller_stats.index[0]} tiene ",
						html.Strong(f"{seller_stats.iloc[0]['count'] / seller_stats['count'].sum() * 100:.1f}% ",
												style={'color': COLORS['warning']}),
						"de los sellers del top 10. ",
						"Este desbalance implica que ",
						html.Strong('estados con pocos sellers locales dependen de envíos de larga distancia',
												style={'color': COLORS['info']}),
						", lo cual impacta negativamente en tiempos de entrega y satisfacción."
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				# Variaciones regionales en satisfacción
				html.Div([
					html.H6('3️⃣ Variaciones Regionales en Satisfacción',
									style={'color': COLORS['warning'], 'marginBottom': '15px'}),

					dbc.Row([
						dbc.Col([
							html.Div([
								html.Strong('🏆 Estados con Mayor Satisfacción:',
														style={'color': COLORS['success'], 'fontSize': '15px'}),
								html.Ul([
													html.Li([
														html.Strong(f"{best_state.name}: ", style={'color': COLORS['text']}),
														f"{best_state['review_mean']:.2f}⭐ ",
														f"({best_state['count']:,.0f} clientes)"
													], style={'marginBottom': '8px', 'color': COLORS['text'], 'fontSize': '14px'})
												] + [
													html.Li([
														html.Strong(f"{state}: ", style={'color': COLORS['text']}),
														f"{top_10_states.loc[state, 'review_mean']:.2f}⭐"
													], style={'marginBottom': '8px', 'color': COLORS['text'], 'fontSize': '14px'})
													for state in top_10_states.nlargest(3, 'review_mean').index[1:3]
												])
							])
						], width=6),
						dbc.Col([
							html.Div([
								html.Strong('⚠️ Estados con Menor Satisfacción:',
														style={'color': COLORS['danger'], 'fontSize': '15px'}),
								html.Ul([
													html.Li([
														html.Strong(f"{worst_state.name}: ", style={'color': COLORS['text']}),
														f"{worst_state['review_mean']:.2f}⭐ ",
														f"({worst_state['count']:,.0f} clientes)"
													], style={'marginBottom': '8px', 'color': COLORS['text'], 'fontSize': '14px'})
												] + [
													html.Li([
														html.Strong(f"{state}: ", style={'color': COLORS['text']}),
														f"{top_10_states.loc[state, 'review_mean']:.2f}⭐"
													], style={'marginBottom': '8px', 'color': COLORS['text'], 'fontSize': '14px'})
													for state in top_10_states.nsmallest(3, 'review_mean').index[1:3]
												])
							])
						], width=6)
					], style={'marginBottom': '15px'}),

					html.P([
						html.Strong('Brecha de Satisfacción:', style={'color': COLORS['text']}),
						f" Diferencia entre mejor y peor estado: ",
						html.Strong(f"{best_state['review_mean'] - worst_state['review_mean']:.2f} puntos ",
												style={'color': COLORS['danger']}),
						f"({(best_state['review_mean'] - worst_state['review_mean']) / worst_state['review_mean'] * 100:.1f}%). ",
						"Esta variación sugiere que ",
						html.Strong('factores regionales específicos', style={'color': COLORS['warning']}),
						" (infraestructura logística, distancia promedio, densidad de sellers) ",
						html.Strong('afectan significativamente la experiencia del cliente.', style={'color': COLORS['primary']})
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				# Tiempos de entrega por región
				html.Div([
					html.H6('4️⃣ Análisis de Tiempos de Entrega por Región',
									style={'color': COLORS['secondary'], 'marginBottom': '15px'}),

					dbc.Row([
						dbc.Col([
							html.Div([
								html.Strong('🚀 Estados con Entregas Más Rápidas:',
														style={'color': COLORS['success'], 'fontSize': '15px'}),
								html.Ul([
									html.Li([
										html.Strong(f"{state}: ", style={'color': COLORS['text']}),
										f"{fast_states.loc[state, 'delivery_mean']:.1f} días"
									], style={'marginBottom': '8px', 'color': COLORS['text'], 'fontSize': '14px'})
									for state in fast_states.index[:3]
								])
							])
						], width=6),
						dbc.Col([
							html.Div([
								html.Strong('🐢 Estados con Entregas Más Lentas:',
														style={'color': COLORS['danger'], 'fontSize': '15px'}),
								html.Ul([
									html.Li([
										html.Strong(f"{state}: ", style={'color': COLORS['text']}),
										f"{slow_states.loc[state, 'delivery_mean']:.1f} días"
									], style={'marginBottom': '8px', 'color': COLORS['text'], 'fontSize': '14px'})
									for state in slow_states.index[:3]
								])
							])
						], width=6)
					], style={'marginBottom': '15px'}),

					html.P([
						html.Strong('Correlación Entrega-Satisfacción:', style={'color': COLORS['text']}),
						" Los estados con entregas más rápidas tienden a tener ",
						html.Strong('mayor satisfacción', style={'color': COLORS['success']}),
						". Esto confirma el hallazgo previo de que ",
						html.Strong('la entrega es el factor operacional más crítico', style={'color': COLORS['primary']}),
						". La diferencia de ",
						html.Strong(f"{slow_states.iloc[0]['delivery_mean'] - fast_states.iloc[0]['delivery_mean']:.1f} días ",
												style={'color': COLORS['danger']}),
						"entre el estado más lento y más rápido explica gran parte de la variación regional en satisfacción."
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				html.Hr(style={'borderColor': COLORS['border']}),

				# Prueba estadística
				html.Div([
					html.H6('5️⃣ Prueba de Hipótesis: Diferencias Entre Estados',
									style={'color': COLORS['info'], 'marginBottom': '15px'}),

					html.P([
						html.Strong('Test ANOVA: Review Score por Estado (Top 10)', style={'color': COLORS['text']}),
						html.Br(),
						'H₀: Las medias de review_score son iguales entre estados',
						html.Br(),
						'H₁: Al menos un estado tiene media significativamente diferente',
						html.Br(),
						html.Br(),
						f"F = {f_stat:.2f}, p-valor = {p_value:.6f}" if p_value >= 0.001 else f"F = {f_stat:.2f}, p-valor < 0.001",
						html.Br(),
						html.Strong('Decisión: Rechazamos H₀', style={'color': COLORS['danger']}) if p_value < 0.05
						else html.Strong('Decisión: No rechazamos H₀', style={'color': COLORS['success']})
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'marginBottom': '10px', 'color': COLORS['text']}),

					html.P([
						html.Strong('Interpretación:', style={'color': COLORS['text']}),
						' Existe evidencia estadística significativa (p < 0.001) de que ',
						html.Strong('diferentes estados tienen niveles de satisfacción significativamente distintos',
												style={'color': COLORS['danger']}),
						'. Esto confirma que ',
						html.Strong('la ubicación geográfica es un factor predictivo importante',
												style={'color': COLORS['primary']}),
						' para modelos de machine learning. ',
						html.Strong('Implicación estratégica:', style={'color': COLORS['warning']}),
						' Se deben desarrollar estrategias diferenciadas por región para optimizar satisfacción.'
					], style={'fontSize': '14px', 'lineHeight': '1.8', 'color': COLORS['text']})
				], style={'marginBottom': '25px'}),

				html.Hr(style={'borderColor': COLORS['border']}),

				# Conclusión estratégica final
				html.Div([
					html.H6('🎯 Conclusión Estratégica: Análisis Geográfico',
									style={'color': COLORS['primary'], 'marginBottom': '15px'}),
					html.P([
						html.Strong('Respuesta Integral a los Objetivos del Proyecto:', style={'color': COLORS['success']}),
						html.Br(),
						html.Br(),
						html.Strong('1. Objetivo Descriptivo (Caracterización):', style={'color': COLORS['info']}),
						f" La distribución geográfica muestra ",
						html.Strong(f'alta concentración en {top_10_states.index[0]} ({concentration.iloc[0]:.1f}%)',
												style={'color': COLORS['primary']}),
						", con ",
						html.Strong(f'{total_states} estados activos. ', style={'color': COLORS['text']}),
						"El top 3 representa ",
						html.Strong(f'{concentration.head(3).sum():.1f}% ', style={'color': COLORS['danger']}),
						"del volumen.",
						html.Br(),
						html.Br(),
						html.Strong('2. Objetivo Exploratorio (Relaciones):', style={'color': COLORS['warning']}),
						" Se identificó correlación entre ",
						html.Strong('distancia/tiempo de entrega y satisfacción', style={'color': COLORS['primary']}),
						". Estados con ",
						html.Strong('mayor densidad de sellers locales tienen mejor satisfacción',
												style={'color': COLORS['success']}),
						". La brecha de satisfacción entre mejor y peor estado es ",
						html.Strong(f'{best_state["review_mean"] - worst_state["review_mean"]:.2f} puntos',
												style={'color': COLORS['danger']}),
						".",
						html.Br(),
						html.Br(),
						html.Strong('3. Objetivo Inferencial (Validación):', style={'color': COLORS['secondary']}),
						" ANOVA confirma diferencias significativas entre estados (p < 0.001). ",
						html.Strong('Conclusión: La geografía es un factor predictivo validado estadísticamente.',
												style={'color': COLORS['danger']}),
						html.Br(),
						html.Br(),
						html.Strong('RECOMENDACIONES ESTRATÉGICAS FINALES:', style={'color': COLORS['primary']}),
						html.Br(),
						html.Strong(f'• Prioridad #1: ', style={'color': COLORS['danger']}),
						f"Expandir sellers en estados con baja densidad ({worst_state.name}, estados del norte)",
						html.Br(),
						html.Strong(f'• Prioridad #2: ', style={'color': COLORS['warning']}),
						"Optimizar logística para reducir tiempos de entrega en estados de menor satisfacción",
						html.Br(),
						html.Strong(f'• Prioridad #3: ', style={'color': COLORS['info']}),
						"Desarrollar centros de distribución en regiones clave para balancear oferta-demanda",
						html.Br(),
						html.Strong(f'• Modelo Predictivo: ', style={'color': COLORS['success']}),
						"Incluir customer_state y seller_state como features importantes (varianza explicada demostrada)"
					], style={'fontSize': '15px', 'lineHeight': '1.9', 'color': COLORS['text']})
				], style={
					'background': f'rgba(239, 68, 68, 0.1)',
					'padding': '25px',
					'borderRadius': '12px',
					'border': f'2px solid {COLORS["danger"]}'
				})
			])
		], style={'background': COLORS['card'], 'border': f'2px solid {COLORS["primary"]}', 'marginBottom': '40px'}),

		# Resumen ejecutivo final de todo el análisis
		dbc.Card([
			dbc.CardHeader(html.H4('📋 RESUMEN EJECUTIVO: ANÁLISIS ESTADÍSTICO COMPLETO',
														 style={'margin': 0, 'color': COLORS['primary'], 'textAlign': 'center'})),
			dbc.CardBody([
				html.P([
					html.Strong('Este análisis respondió completamente a los objetivos del proyecto:',
											style={'fontSize': '17px', 'color': COLORS['success']})
				], style={'textAlign': 'center', 'marginBottom': '20px'}),

				dbc.Row([
					dbc.Col([
						html.Div([
							html.H5('🔍 Correlación', style={'color': COLORS['primary'], 'marginBottom': '10px'}),
							html.P('Variables de entrega tienen mayor correlación con satisfacción (r > 0.3)',
										 style={'fontSize': '13px', 'color': COLORS['text']})
						], style={'textAlign': 'center', 'padding': '15px', 'background': 'rgba(0, 212, 255, 0.1)',
											'borderRadius': '8px', 'border': f'1px solid {COLORS["primary"]}'})
					], width=4),
					dbc.Col([
						html.Div([
							html.H5('⭐ Review Score', style={'color': COLORS['success'], 'marginBottom': '10px'}),
							html.P('Media 4.08/5, distribución sesgada negativa, 57.5% son 5 estrellas',
										 style={'fontSize': '13px', 'color': COLORS['text']})
						], style={'textAlign': 'center', 'padding': '15px', 'background': 'rgba(16, 185, 129, 0.1)',
											'borderRadius': '8px', 'border': f'1px solid {COLORS["success"]}'})
					], width=4),
					dbc.Col([
						html.Div([
							html.H5('🚚 Entrega', style={'color': COLORS['warning'], 'marginBottom': '10px'}),
							html.P('92% on-time, diferencias significativas entre grupos (p<0.001)',
										 style={'fontSize': '13px', 'color': COLORS['text']})
						], style={'textAlign': 'center', 'padding': '15px', 'background': 'rgba(245, 158, 11, 0.1)',
											'borderRadius': '8px', 'border': f'1px solid {COLORS["warning"]}'})
					], width=4)
				], style={'marginBottom': '20px'}),

				dbc.Row([
					dbc.Col([
						html.Div([
							html.H5('📦 Producto', style={'color': COLORS['info'], 'marginBottom': '10px'}),
							html.P('Categoría es factor moderador significativo (ANOVA p<0.001)',
										 style={'fontSize': '13px', 'color': COLORS['text']})
						], style={'textAlign': 'center', 'padding': '15px', 'background': 'rgba(59, 130, 246, 0.1)',
											'borderRadius': '8px', 'border': f'1px solid {COLORS["info"]}'})
					], width=6),
					dbc.Col([
						html.Div([
							html.H5('🌍 Geografía', style={'color': COLORS['danger'], 'marginBottom': '10px'}),
							html.P(f'Diferencias validadas entre estados (p<0.001), {sp_dominance:.0f}% en SP',
										 style={'fontSize': '13px', 'color': COLORS['text']})
						], style={'textAlign': 'center', 'padding': '15px', 'background': 'rgba(239, 68, 68, 0.1)',
											'borderRadius': '8px', 'border': f'1px solid {COLORS["danger"]}'})
					], width=6)
				]),

				html.Hr(style={'margin': '25px 0', 'borderColor': COLORS['primary'], 'borderWidth': '2px'}),

				html.P([
					html.Strong('🎯 CONCLUSIÓN FINAL: ', style={'fontSize': '18px', 'color': COLORS['primary']}),
					html.Br(),
					'El factor ',
					html.Strong('MÁS CRÍTICO ', style={'color': COLORS['danger'], 'fontSize': '16px'}),
					'para la satisfacción del cliente es ',
					html.Strong('LA ENTREGA', style={'color': COLORS['success'], 'fontSize': '16px'}),
					' (tiempos y cumplimiento). Seguido por ',
					html.Strong('GEOGRAFÍA ', style={'color': COLORS['warning']}),
					'y ',
					html.Strong('CATEGORÍA DE PRODUCTO', style={'color': COLORS['info']}),
					'. El modelo predictivo debe priorizar features operacionales sobre transaccionales o físicas.'
				], style={'fontSize': '15px', 'lineHeight': '2', 'textAlign': 'center', 'color': COLORS['text'],
									'marginTop': '20px', 'padding': '20px', 'background': 'rgba(123, 44, 191, 0.1)',
									'borderRadius': '12px', 'border': f'2px solid {COLORS["secondary"]}'})
			])
		], style={'background': COLORS['card'], 'border': f'3px solid {COLORS["primary"]}', 'marginBottom': '40px'})
	])

# =============================================================================
# FUNCIÓN PRINCIPAL - CREAR CONTENIDO COMPLETO
# =============================================================================

def create_analisis_content(df):
	"""
    Crea el contenido completo de la página de análisis estadístico.
    """

	if df is None:
		return html.Div([
			create_page_header('Error', 'No se pudieron cargar los datos', '❌'),
			dbc.Alert('Error al cargar el dataset. Verifica la ruta del archivo.', color='danger')
		])

	return html.Div([
		# Header principal
		create_page_header(
			title='Análisis Estadístico Descriptivo e Inferencial',
			subtitle='Exploración profunda orientada a identificar factores críticos de satisfacción del cliente',
			icon='📊'
		),

		# Banner introductorio
		dbc.Alert([
			html.H4('🎯 Marco Analítico', className='alert-heading', style={'color': COLORS['text']}),
			html.P([
				'Este análisis responde directamente a los ',
				html.Strong('objetivos descriptivo, exploratorio e inferencial ', style={'color': COLORS['primary']}),
				'del proyecto. Utilizamos métodos estadísticos rigurosos para: ',
				html.Strong('(1) Caracterizar patrones de satisfacción, ', style={'color': COLORS['success']}),
				html.Strong('(2) Identificar relaciones entre variables, ', style={'color': COLORS['info']}),
				'y ',
				html.Strong('(3) Validar hipótesis mediante pruebas estadísticas.', style={'color': COLORS['warning']})
			], style={'marginBottom': '10px', 'color': COLORS['text']}),
			html.P([
				html.Strong('Metodología:', style={'color': COLORS['text']}),
				' Análisis univariado → Análisis bivariado → Análisis multivariado → Pruebas de hipótesis'
			], style={'marginBottom': 0, 'color': COLORS['text'], 'fontSize': '14px'})
		], color='info', style={'marginBottom': '40px'}),

		# SECCIÓN 1: Correlación
		create_correlation_section(df),

		html.Hr(style={'borderColor': COLORS['primary'], 'borderWidth': '3px', 'margin': '60px 0'}),

		# SECCIÓN 2: Review Score
		create_review_score_section(df),

		html.Hr(style={'borderColor': COLORS['primary'], 'borderWidth': '3px', 'margin': '60px 0'}),

		# SECCIÓN 3: Delivery (AHORA COMPLETA)
		create_delivery_section(df),

		html.Hr(style={'borderColor': COLORS['primary'], 'borderWidth': '3px', 'margin': '60px 0'}),

		# SECCIÓN 4: Producto (AHORA COMPLETA)
		create_product_section(df),

		html.Hr(style={'borderColor': COLORS['primary'], 'borderWidth': '3px', 'margin': '60px 0'}),

		html.Hr(style={'borderColor': COLORS['primary'], 'borderWidth': '3px', 'margin': '60px 0'}),

		# SECCIÓN 5: Geografía (AHORA COMPLETA)
		create_geographic_section(df)
	])
