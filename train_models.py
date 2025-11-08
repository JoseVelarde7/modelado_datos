"""
Script de Entrenamiento de Modelos - Proyecto Olist
Ejecutar UNA VEZ para generar resultados pre-calculados
Autor: Jose
Fecha: 2025-11-08
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
														 f1_score, roc_auc_score, confusion_matrix,
														 classification_report)
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("INICIANDO ENTRENAMIENTO DE MODELOS")
print("=" * 60)

# ============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================================
print("\n📂 Cargando datos...")
df = pd.read_excel('df_oficial.xlsx')  # Ajusta la ruta si es necesario

# Variables para el modelo
feature_columns = [
	'price', 'freight_value', 'payment_value', 'payment_installments',
	'product_photos_qty', 'product_weight_kg', 'product_volume_cm3',
	'delivery_time_days', 'delivery_delay_days', 'on_time_delivery'
]

target_column = 'satisfaction_level'

# Filtrar datos
df_model = df[feature_columns + [target_column]].copy()
df_model = df_model.dropna()

print(f"✅ Datos cargados: {len(df_model):,} registros")
print(f"   Variables predictoras: {len(feature_columns)}")
print(f"   Distribución del target:")
for level, count in df_model[target_column].value_counts().items():
	pct = count / len(df_model) * 100
	print(f"      • {level}: {count:,} ({pct:.2f}%)")

# ============================================================================
# 2. ENCODING Y SPLIT
# ============================================================================
print("\n⚙️  Preparando datos para entrenamiento...")

# Encoding del target
le = LabelEncoder()
y = le.fit_transform(df_model[target_column])
X = df_model[feature_columns]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train set: {len(X_train):,} | Test set: {len(X_test):,}")

# ============================================================================
# 3. ENTRENAMIENTO DE MODELOS
# ============================================================================
print("\n🤖 Entrenando modelos...")

results = {}

# Lista de modelos a entrenar
models = {
	'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
	'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
	'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
	'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
}

for model_name, model in models.items():
	print(f"\n   Entrenando {model_name}...")
	start_time = datetime.now()

	# Entrenar
	if model_name == 'Logistic Regression':
		model.fit(X_train_scaled, y_train)
		y_pred = model.predict(X_test_scaled)
		y_pred_proba = model.predict_proba(X_test_scaled)
	else:
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		y_pred_proba = model.predict_proba(X_test)

	# Calcular métricas
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
	recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
	f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

	# ROC AUC (multiclase)
	try:
		roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
	except:
		roc_auc = 0.0

	# Confusion Matrix
	cm = confusion_matrix(y_test, y_pred)

	# Classification Report
	report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

	# Feature Importance (si aplica)
	if hasattr(model, 'feature_importances_'):
		feature_imp = dict(zip(feature_columns, model.feature_importances_.tolist()))
	elif hasattr(model, 'coef_'):
		feature_imp = dict(zip(feature_columns, np.abs(model.coef_[0]).tolist()))
	else:
		feature_imp = {}

	# Cross-validation
	print(f"      Calculando cross-validation...")
	if model_name == 'Logistic Regression':
		cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
	else:
		cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

	training_time = (datetime.now() - start_time).total_seconds()

	# Guardar resultados
	results[model_name] = {
		'accuracy': float(accuracy),
		'precision': float(precision),
		'recall': float(recall),
		'f1_score': float(f1),
		'roc_auc': float(roc_auc),
		'confusion_matrix': cm.tolist(),
		'classification_report': report,
		'feature_importance': feature_imp,
		'cv_scores': cv_scores.tolist(),
		'cv_mean': float(cv_scores.mean()),
		'cv_std': float(cv_scores.std()),
		'training_time': float(training_time),
		'train_size': int(len(X_train)),
		'test_size': int(len(X_test))
	}

	print(f"      ✅ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================================
# 4. GUARDAR RESULTADOS
# ============================================================================
print("\n💾 Guardando resultados...")

# Guardar resultados en JSON
output_data = {
	'timestamp': datetime.now().isoformat(),
	'dataset_info': {
		'total_records': int(len(df_model)),
		'train_size': int(len(X_train)),
		'test_size': int(len(X_test)),
		'num_features': int(len(feature_columns)),
		'feature_names': feature_columns,
		'target_classes': le.classes_.tolist()
	},
	'models': results
}

with open('model_results.json', 'w', encoding='utf-8') as f:
	json.dump(output_data, f, indent=2, ensure_ascii=False)

print("✅ Resultados guardados en: model_results.json")

# Guardar modelos entrenados (opcional)
print("\n💾 Guardando modelos entrenados...")
for model_name, model in models.items():
	filename = f"model_{model_name.lower().replace(' ', '_')}.pkl"
	with open(filename, 'wb') as f:
		pickle.dump(model, f)
	print(f"   ✅ {filename}")

# Guardar scaler
with open('scaler.pkl', 'wb') as f:
	pickle.dump(scaler, f)
print("   ✅ scaler.pkl")

# ============================================================================
# 5. RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 60)
print("RESUMEN DE RESULTADOS")
print("=" * 60)

print("\n📊 Ranking de Modelos por Accuracy:")
ranking = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for i, (name, metrics) in enumerate(ranking, 1):
	print(f"   {i}. {name:20} - Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

print("\n🏆 Mejor Modelo:", ranking[0][0])
print(f"   📈 Accuracy: {ranking[0][1]['accuracy']:.4f}")
print(f"   📈 F1-Score: {ranking[0][1]['f1_score']:.4f}")
print(f"   📈 CV Score: {ranking[0][1]['cv_mean']:.4f} ± {ranking[0][1]['cv_std']:.4f}")

print("\n" + "=" * 60)
print("✅ ENTRENAMIENTO COMPLETADO")
print("=" * 60)
print("\n📁 Archivos generados:")
print("   • model_results.json (resultados completos)")
print("   • model_*.pkl (modelos entrenados)")
print("   • scaler.pkl (escalador)")
print("\n🎯 Siguiente paso: Pasa el contenido de 'model_results.json' para crear la página")