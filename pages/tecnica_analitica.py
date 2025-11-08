"""
=============================================================================
PÁGINA: TÉCNICA ANALÍTICA - MACHINE LEARNING
=============================================================================
Predicción de satisfacción del cliente usando modelos pre-entrenados.
Carga instantánea con resultados pre-calculados.
"""

from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from config import COLORS


def load_model_results():
	"""Carga los resultados pre-calculados desde el JSON"""
	try:
		with open('model_results.json', 'r', encoding='utf-8') as f:
			return json.load(f)
	except FileNotFoundError:
		return None


def create_metrics_comparison_chart(results):
	"""Gráfico de comparación de métricas principales"""
	models = list(results['models'].keys())

	metrics = {
		'Accuracy': [results['models'][m]['accuracy'] for m in models],
		'Precision': [results['models'][m]['precision'] for m in models],
		'Recall': [results['models'][m]['recall'] for m in models],
		'F1-Score': [results['models'][m]['f1_score'] for m in models]
	}

	fig = go.Figure()

	colors = ['#00d4ff', '#00ff9f', '#ffd700', '#ff6b6b']

	for i, (metric_name, values) in enumerate(metrics.items()):
		fig.add_trace(go.Bar(
			name=metric_name,
			x=models,
			y=values,
			text=[f'{v:.3f}' for v in values],
			textposition='outside',
			marker_color=colors[i],
			hovertemplate=f'<b>{metric_name}</b><br>%{{x}}<br>Score: %{{y:.4f}}<extra></extra>'
		))

	fig.update_layout(
		title={
			'text': '📊 Comparación de Métricas por Modelo',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 20, 'color': COLORS['text'], 'family': 'Arial Black'}
		},
		barmode='group',
		plot_bgcolor='rgba(0,0,0,0)',
		paper_bgcolor='rgba(0,0,0,0)',
		font={'color': COLORS['text']},
		xaxis={'gridcolor': 'rgba(255,255,255,0.1)'},
		yaxis={
			'gridcolor': 'rgba(255,255,255,0.1)',
			'title': 'Score',
			'range': [0, 1]
		},
		height=500,
		hovermode='x unified',
		legend={
			'orientation': 'h',
			'yanchor': 'bottom',
			'y': 1.02,
			'xanchor': 'center',
			'x': 0.5
		}
	)

	return fig


def create_confusion_matrices(results):
	"""Matrices de confusión para todos los modelos"""
	models = list(results['models'].keys())
	classes = results['dataset_info']['target_classes']

	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=[f'{model}' for model in models],
		specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
					 [{'type': 'heatmap'}, {'type': 'heatmap'}]],
		vertical_spacing=0.15,
		horizontal_spacing=0.1
	)

	positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

	for idx, (model, pos) in enumerate(zip(models, positions)):
		cm = np.array(results['models'][model]['confusion_matrix'])

		# Normalizar para mejor visualización
		cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		fig.add_trace(
			go.Heatmap(
				z=cm_normalized,
				x=classes,
				y=classes,
				text=cm,
				texttemplate='%{text}',
				textfont={'size': 12},
				colorscale='Blues',
				showscale=(idx == 0),
				hovertemplate='Predicho: %{x}<br>Real: %{y}<br>Casos: %{text}<br>Proporción: %{z:.2%}<extra></extra>'
			),
			row=pos[0], col=pos[1]
		)

	fig.update_layout(
		title={
			'text': '🎯 Matrices de Confusión por Modelo',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 20, 'color': COLORS['text'], 'family': 'Arial Black'}
		},
		plot_bgcolor='rgba(0,0,0,0)',
		paper_bgcolor='rgba(0,0,0,0)',
		font={'color': COLORS['text']},
		height=800
	)

	return fig


def create_feature_importance_chart(results):
	"""Importancia de características para modelos basados en árboles"""
	models_with_importance = ['Random Forest', 'Gradient Boosting']

	fig = make_subplots(
		rows=1, cols=2,
		subplot_titles=models_with_importance,
		specs=[[{'type': 'bar'}, {'type': 'bar'}]]
	)

	colors_map = {
		'Random Forest': '#00ff9f',
		'Gradient Boosting': '#ffd700'
	}

	for idx, model in enumerate(models_with_importance, 1):
		importance = results['models'][model]['feature_importance']

		# Ordenar por importancia
		sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
		features = [f[0] for f in sorted_features]
		values = [f[1] for f in sorted_features]

		fig.add_trace(
			go.Bar(
				x=values,
				y=features,
				orientation='h',
				marker_color=colors_map[model],
				text=[f'{v:.3f}' for v in values],
				textposition='outside',
				hovertemplate='<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>',
				showlegend=False
			),
			row=1, col=idx
		)

	fig.update_layout(
		title={
			'text': '🔍 Importancia de Variables (Feature Importance)',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 20, 'color': COLORS['text'], 'family': 'Arial Black'}
		},
		plot_bgcolor='rgba(0,0,0,0)',
		paper_bgcolor='rgba(0,0,0,0)',
		font={'color': COLORS['text']},
		height=500
	)

	fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', title='Importancia')
	fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

	return fig


def create_cv_scores_chart(results):
	"""Cross-validation scores comparison"""
	models = list(results['models'].keys())

	fig = go.Figure()

	colors = ['#00d4ff', '#00ff9f', '#ffd700', '#ff6b6b']

	for idx, model in enumerate(models):
		cv_scores = results['models'][model]['cv_scores']

		fig.add_trace(go.Box(
			y=cv_scores,
			name=model,
			marker_color=colors[idx],
			boxmean='sd',
			hovertemplate=f'<b>{model}</b><br>Score: %{{y:.4f}}<extra></extra>'
		))

	fig.update_layout(
		title={
			'text': '📈 Cross-Validation Scores (5-Fold)',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 20, 'color': COLORS['text'], 'family': 'Arial Black'}
		},
		plot_bgcolor='rgba(0,0,0,0)',
		paper_bgcolor='rgba(0,0,0,0)',
		font={'color': COLORS['text']},
		yaxis={
			'gridcolor': 'rgba(255,255,255,0.1)',
			'title': 'Accuracy Score',
			'range': [0.78, 0.82]
		},
		xaxis={'gridcolor': 'rgba(255,255,255,0.1)'},
		height=500,
		showlegend=False
	)

	return fig


def create_roc_comparison_chart(results):
	"""Comparación de ROC-AUC scores"""
	models = list(results['models'].keys())
	roc_scores = [results['models'][m]['roc_auc'] for m in models]

	sorted_data = sorted(zip(models, roc_scores), key=lambda x: x[1], reverse=True)
	models_sorted = [d[0] for d in sorted_data]
	scores_sorted = [d[1] for d in sorted_data]

	colors = ['#00ff9f' if s > 0.72 else '#ffd700' if s > 0.70 else '#ff6b6b'
						for s in scores_sorted]

	fig = go.Figure()

	fig.add_trace(go.Bar(
		x=models_sorted,
		y=scores_sorted,
		marker_color=colors,
		text=[f'{s:.4f}' for s in scores_sorted],
		textposition='outside',
		hovertemplate='<b>%{x}</b><br>ROC-AUC: %{y:.4f}<extra></extra>'
	))

	fig.add_hline(
		y=0.7,
		line_dash="dash",
		line_color="white",
		annotation_text="Umbral Bueno (0.70)",
		annotation_position="right"
	)

	fig.update_layout(
		title={
			'text': '🎯 ROC-AUC Score Comparison',
			'x': 0.5,
			'xanchor': 'center',
			'font': {'size': 20, 'color': COLORS['text'], 'family': 'Arial Black'}
		},
		plot_bgcolor='rgba(0,0,0,0)',
		paper_bgcolor='rgba(0,0,0,0)',
		font={'color': COLORS['text']},
		xaxis={'gridcolor': 'rgba(255,255,255,0.1)'},
		yaxis={
			'gridcolor': 'rgba(255,255,255,0.1)',
			'title': 'ROC-AUC Score',
			'range': [0.65, 0.80]
		},
		height=500
	)

	return fig


def create_model_ranking_table(results):
	"""Tabla de ranking de modelos usando componentes Dash nativos"""
	models = list(results['models'].keys())

	data = []
	for model in models:
		data.append({
			'model': model,
			'accuracy': results['models'][model]['accuracy'],
			'f1': results['models'][model]['f1_score'],
			'roc_auc': results['models'][model]['roc_auc'],
			'cv_mean': results['models'][model]['cv_mean'],
			'time': results['models'][model]['training_time']
		})

	data.sort(key=lambda x: x['accuracy'], reverse=True)

	medals = ['🥇', '🥈', '🥉', '4️⃣']

	# Header de la tabla
	header = html.Tr([
		html.Th('🏆 Rank', style={'padding': '15px', 'textAlign': 'left', 'color': '#00d4ff'}),
		html.Th('Modelo', style={'padding': '15px', 'textAlign': 'left', 'color': '#00d4ff'}),
		html.Th('Accuracy', style={'padding': '15px', 'textAlign': 'center', 'color': '#00d4ff'}),
		html.Th('F1-Score', style={'padding': '15px', 'textAlign': 'center', 'color': '#00d4ff'}),
		html.Th('ROC-AUC', style={'padding': '15px', 'textAlign': 'center', 'color': '#00d4ff'}),
		html.Th('CV Score', style={'padding': '15px', 'textAlign': 'center', 'color': '#00d4ff'}),
		html.Th('⏱️ Tiempo (s)', style={'padding': '15px', 'textAlign': 'center', 'color': '#00d4ff'})
	], style={'background': 'rgba(0,212,255,0.2)', 'borderBottom': '2px solid #00d4ff'})

	# Filas de la tabla
	rows = []
	for idx, row in enumerate(data):
		bg_color = 'rgba(0,255,159,0.1)' if idx == 0 else 'rgba(255,255,255,0.03)'
		rows.append(html.Tr([
			html.Td(medals[idx], style={'padding': '15px', 'color': '#ffffff', 'fontSize': '20px'}),
			html.Td(row['model'], style={'padding': '15px', 'color': '#ffffff', 'fontWeight': 'bold'}),
			html.Td(f"{row['accuracy']:.4f}",
							style={'padding': '15px', 'textAlign': 'center', 'color': '#00ff9f', 'fontWeight': 'bold'}),
			html.Td(f"{row['f1']:.4f}", style={'padding': '15px', 'textAlign': 'center', 'color': '#ffd700'}),
			html.Td(f"{row['roc_auc']:.4f}", style={'padding': '15px', 'textAlign': 'center', 'color': '#00d4ff'}),
			html.Td(f"{row['cv_mean']:.4f}", style={'padding': '15px', 'textAlign': 'center', 'color': '#ff6b6b'}),
			html.Td(f"{row['time']:.2f}", style={'padding': '15px', 'textAlign': 'center', 'color': '#ffffff'})
		], style={'background': bg_color, 'borderBottom': '1px solid rgba(255,255,255,0.1)'}))

	# Tabla completa
	return html.Div([
		html.Table([
			html.Thead(header),
			html.Tbody(rows)
		], style={'width': '100%', 'borderCollapse': 'collapse', 'background': 'rgba(255,255,255,0.05)',
							'borderRadius': '10px'})
	], style={'overflowX': 'auto'})


def create_detailed_metrics_cards(results):
	"""Cards con métricas detalladas del mejor modelo usando componentes Dash"""
	best_model = max(results['models'].items(), key=lambda x: x[1]['accuracy'])
	model_name = best_model[0]
	metrics = best_model[1]

	# Estilos comunes para cards
	card_style_base = {
		'padding': '20px',
		'borderRadius': '15px',
		'boxShadow': '0 8px 32px 0 rgba(0,255,159,0.2)'
	}

	card1_style = {**card_style_base,
								 'background': 'linear-gradient(135deg, rgba(0,255,159,0.2), rgba(0,212,255,0.2))',
								 'border': '1px solid rgba(0,255,159,0.3)'
								 }

	card2_style = {**card_style_base,
								 'background': 'linear-gradient(135deg, rgba(255,215,0,0.2), rgba(255,107,107,0.2))',
								 'border': '1px solid rgba(255,215,0,0.3)'
								 }

	card3_style = {**card_style_base,
								 'background': 'linear-gradient(135deg, rgba(0,212,255,0.2), rgba(138,43,226,0.2))',
								 'border': '1px solid rgba(0,212,255,0.3)'
								 }

	card4_style = {**card_style_base,
								 'background': 'linear-gradient(135deg, rgba(255,107,107,0.2), rgba(255,215,0,0.2))',
								 'border': '1px solid rgba(255,107,107,0.3)'
								 }

	return html.Div([
		# Grid de cards
		html.Div([
			# Card 1: Accuracy
			html.Div([
				html.Div('✅ ACCURACY',
								 style={'fontSize': '14px', 'color': '#00ff9f', 'textTransform': 'uppercase', 'marginBottom': '10px'}),
				html.Div(f"{metrics['accuracy']:.2%}", style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#ffffff'}),
				html.Div('Test Set', style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.7)', 'marginTop': '5px'})
			], style=card1_style),

			# Card 2: F1-Score
			html.Div([
				html.Div('⚖️ F1-SCORE',
								 style={'fontSize': '14px', 'color': '#ffd700', 'textTransform': 'uppercase', 'marginBottom': '10px'}),
				html.Div(f"{metrics['f1_score']:.2%}", style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#ffffff'}),
				html.Div('Weighted', style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.7)', 'marginTop': '5px'})
			], style=card2_style),

			# Card 3: ROC-AUC
			html.Div([
				html.Div('🎯 ROC-AUC',
								 style={'fontSize': '14px', 'color': '#00d4ff', 'textTransform': 'uppercase', 'marginBottom': '10px'}),
				html.Div(f"{metrics['roc_auc']:.4f}", style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#ffffff'}),
				html.Div('Multi-class OvR', style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.7)', 'marginTop': '5px'})
			], style=card3_style),

			# Card 4: CV Score
			html.Div([
				html.Div('📊 CV SCORE',
								 style={'fontSize': '14px', 'color': '#ff6b6b', 'textTransform': 'uppercase', 'marginBottom': '10px'}),
				html.Div(f"{metrics['cv_mean']:.2%}", style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#ffffff'}),
				html.Div(f"± {metrics['cv_std']:.4f}",
								 style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.7)', 'marginTop': '5px'})
			], style=card4_style)

		], style={
			'display': 'grid',
			'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
			'gap': '20px',
			'margin': '20px 0'
		}),

		# Descripción del mejor modelo
		html.Div([
			html.H4(f"🏆 Mejor Modelo: {model_name}", style={'color': '#00ff9f', 'margin': '0 0 10px 0'}),
			html.P(
				f"Este modelo alcanzó la mejor precisión en el conjunto de prueba con un accuracy de {metrics['accuracy']:.2%}, "
				f"superando el objetivo del 80%. El cross-validation score de {metrics['cv_mean']:.2%} confirma la robustez del modelo.",
				style={'color': 'rgba(255,255,255,0.8)', 'margin': '0'}
			)
		], style={
			'background': 'rgba(0,255,159,0.1)',
			'padding': '20px',
			'borderRadius': '15px',
			'borderLeft': '4px solid #00ff9f',
			'marginTop': '20px'
		})
	])


def create_tecnica_content():
	"""
	Función principal que crea el contenido de la página.
	Sigue el patrón de create_*_content() usado en el proyecto.
	"""

	# Cargar resultados
	results = load_model_results()

	if results is None:
		return html.Div([
			html.H2("⚠️ Error: Archivo model_results.json no encontrado",
							style={'color': '#ff6b6b', 'textAlign': 'center', 'marginTop': '50px'}),
			html.P("Por favor, ejecuta primero el script train_models.py",
						 style={'color': COLORS['text'], 'textAlign': 'center'})
		])

	dataset_info = results['dataset_info']

	return html.Div([

		# Header
		html.Div([
			html.H2("🤖 TÉCNICA ANALÍTICA: MACHINE LEARNING",
							style={'color': COLORS['primary'], 'textAlign': 'center', 'marginBottom': '10px',
										 'fontFamily': 'Arial Black', 'fontSize': '32px'}),
			html.P(f"Predicción de Satisfacción del Cliente | Dataset: {dataset_info['total_records']:,} registros",
						 style={'color': COLORS['text_muted'], 'textAlign': 'center', 'fontSize': '16px'}),
			html.Hr(style={'borderColor': 'rgba(0,212,255,0.3)', 'margin': '20px 0'})
		]),

		# Métricas del mejor modelo
		html.Div([
			create_detailed_metrics_cards(results)
		], style={'marginBottom': '30px'}),

		# Tabla de ranking
		html.Div([
			html.H3("📋 Ranking de Modelos",
							style={'color': COLORS['primary'], 'marginBottom': '20px', 'fontFamily': 'Arial Black'}),
			create_model_ranking_table(results)
		], style={'marginBottom': '40px'}),

		# Gráficos
		html.Div([
			dcc.Graph(figure=create_metrics_comparison_chart(results), config={'displayModeBar': False})
		], style={'marginBottom': '40px'}),

		html.Div([
			dcc.Graph(figure=create_roc_comparison_chart(results), config={'displayModeBar': False})
		], style={'marginBottom': '40px'}),

		html.Div([
			dcc.Graph(figure=create_cv_scores_chart(results), config={'displayModeBar': False})
		], style={'marginBottom': '40px'}),

		html.Div([
			dcc.Graph(figure=create_feature_importance_chart(results), config={'displayModeBar': False})
		], style={'marginBottom': '40px'}),

		html.Div([
			dcc.Graph(figure=create_confusion_matrices(results), config={'displayModeBar': False})
		], style={'marginBottom': '40px'}),

		# Insights
		html.Div([
			html.H3("💡 Insights y Conclusiones",
							style={'color': '#00ff9f', 'marginBottom': '20px', 'fontFamily': 'Arial Black'}),

			html.Div([
				html.Div([
					html.H4("🎯 Rendimiento General", style={'color': COLORS['primary']}),
					html.Ul([
						html.Li(f"Random Forest obtuvo el mejor accuracy: {results['models']['Random Forest']['accuracy']:.2%}",
										style={'color': COLORS['text'], 'marginBottom': '10px'}),
						html.Li("Todos los modelos superaron el 78% de accuracy en validación cruzada",
										style={'color': COLORS['text'], 'marginBottom': '10px'}),
						html.Li("El objetivo de >80% accuracy fue alcanzado por Random Forest",
										style={'color': COLORS['text'], 'marginBottom': '10px'})
					])
				], style={'marginBottom': '20px'}),

				html.Div([
					html.H4("🔍 Variables Clave", style={'color': '#ffd700'}),
					html.Ul([
						html.Li("delivery_delay_days es la variable más importante (23-54% de importancia)",
										style={'color': COLORS['text'], 'marginBottom': '10px'}),
						html.Li("delivery_time_days también muestra alta relevancia (8-16% de importancia)",
										style={'color': COLORS['text'], 'marginBottom': '10px'}),
						html.Li("Variables de precio (price, payment_value) tienen impacto moderado",
										style={'color': COLORS['text'], 'marginBottom': '10px'})
					])
				], style={'marginBottom': '20px'}),

				html.Div([
					html.H4("⚠️ Desafíos Identificados", style={'color': '#ff6b6b'}),
					html.Ul([
						html.Li("Clase 'Neutro' es difícil de predecir (muy pocos casos correctos)",
										style={'color': COLORS['text'], 'marginBottom': '10px'}),
						html.Li("Desbalance de clases: 76.8% Satisfechos vs 14.8% Insatisfechos vs 8.4% Neutros",
										style={'color': COLORS['text'], 'marginBottom': '10px'}),
						html.Li("Los modelos tienden a sobre-predecir la clase mayoritaria (Satisfecho)",
										style={'color': COLORS['text'], 'marginBottom': '10px'})
					])
				])

			], style={
				'background': 'rgba(255,255,255,0.05)',
				'padding': '30px',
				'borderRadius': '15px',
				'border': '1px solid rgba(255,255,255,0.1)'
			})
		], style={'marginBottom': '40px'}),

		# Footer
		html.Div([
			html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)', 'margin': '40px 0 20px 0'}),
			html.P(f"📅 Resultados generados: {results['timestamp'][:19]}",
						 style={'color': COLORS['text_muted'], 'textAlign': 'center', 'fontSize': '12px'})
		])

	], style={'padding': '20px'})
