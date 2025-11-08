#!/usr/bin/env python3
"""
Script de Verificación Pre-Deploy
Ejecuta esto antes de subir a Render para verificar que todo esté correcto
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("🔍 VERIFICACIÓN PRE-DEPLOY PARA RENDER")
print("=" * 70)

errors = []
warnings = []
success = []


# 1. Verificar archivos necesarios
print("\n📁 Verificando archivos necesarios...")
required_files = {
	'requirements.txt': 'Dependencias de Python',
	'Procfile': 'Comando de inicio',
	'app.py': 'Aplicación principal',
	'config.py': 'Configuración',
	'data_loader.py': 'Cargador de datos',
	'model_results.json': 'Resultados de modelos'
}

for file, description in required_files.items():
	if os.path.exists(file):
		success.append(f"✅ {file} - {description}")
	else:
		errors.append(f"❌ FALTA: {file} - {description}")

optional_files = {
	'runtime.txt': 'Versión de Python',
	'render.yaml': 'Configuración de Render',
	'.gitignore': 'Archivos a ignorar'
}

for file, description in optional_files.items():
	if os.path.exists(file):
		success.append(f"✅ {file} - {description}")
	else:
		warnings.append(f"⚠️  OPCIONAL: {file} - {description}")

# 2. Verificar estructura de carpetas
print("\n📂 Verificando estructura de carpetas...")
required_dirs = ['components', 'pages']
for dir_name in required_dirs:
	if os.path.isdir(dir_name):
		success.append(f"✅ Carpeta: {dir_name}/")
		# Verificar __init__.py
		init_file = os.path.join(dir_name, '__init__.py')
		if os.path.exists(init_file):
			success.append(f"   ✅ {init_file}")
		else:
			warnings.append(f"   ⚠️  Falta: {init_file}")
	else:
		errors.append(f"❌ FALTA carpeta: {dir_name}/")

# 3. Verificar app.py
print("\n🔍 Verificando app.py...")
try:
	with open('app_x.py', 'r', encoding='utf-8') as f:
		content = f.read()

		# Verificar server = app.server
		if 'server = app.server' in content:
			success.append("✅ app.py tiene: server = app.server")
		else:
			errors.append("❌ CRÍTICO: app.py debe tener 'server = app.server'")

		# Verificar puerto dinámico
		if "os.environ.get('PORT'" in content or "os.getenv('PORT'" in content:
			success.append("✅ app.py usa puerto dinámico (PORT)")
		else:
			warnings.append("⚠️  Recomienda usar: port = int(os.environ.get('PORT', 8050))")

		# Verificar host 0.0.0.0
		if "host='0.0.0.0'" in content:
			success.append("✅ app.py usa host='0.0.0.0'")
		else:
			warnings.append("⚠️  Recomienda usar: host='0.0.0.0'")

except FileNotFoundError:
	errors.append("❌ CRÍTICO: No se encuentra app.py")
except Exception as e:
	errors.append(f"❌ Error leyendo app.py: {e}")

# 4. Verificar requirements.txt
print("\n📦 Verificando requirements.txt...")
try:
	with open('requirements.txt', 'r') as f:
		requirements = f.read()

		critical_packages = ['dash', 'plotly', 'pandas', 'gunicorn']
		for package in critical_packages:
			if package in requirements.lower():
				success.append(f"✅ Dependencia: {package}")
			else:
				errors.append(f"❌ FALTA dependencia crítica: {package}")

except FileNotFoundError:
	errors.append("❌ CRÍTICO: No se encuentra requirements.txt")

# 5. Verificar Procfile
print("\n⚙️  Verificando Procfile...")
try:
	with open('Procfile', 'r') as f:
		procfile = f.read()

		if 'gunicorn' in procfile and 'app:server' in procfile:
			success.append("✅ Procfile correcto: gunicorn app:server")
		else:
			errors.append("❌ Procfile debe contener: web: gunicorn app:server")

except FileNotFoundError:
	errors.append("❌ No se encuentra Procfile")

# 6. Verificar tamaño de archivos
print("\n📊 Verificando tamaño de archivos...")
large_files = []
for root, dirs, files in os.walk('.'):
	# Ignorar carpetas ocultas y __pycache__
	dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

	for file in files:
		if file.startswith('.'):
			continue
		filepath = os.path.join(root, file)
		try:
			size = os.path.getsize(filepath) / (1024 * 1024)  # MB
			if size > 50:
				large_files.append(f"⚠️  {filepath}: {size:.1f} MB")
		except:
			pass

if large_files:
	warnings.append("⚠️  Archivos grandes encontrados (>50MB):")
	for lf in large_files:
		warnings.append(f"   {lf}")
	warnings.append("   Considera usar almacenamiento externo para archivos >100MB")

# 7. Verificar imports
print("\n🔌 Verificando imports...")
try:
	sys.path.insert(0, os.getcwd())

	modules_to_test = [
		'config',
		'data_loader',
		'components.sidebar',
		'pages.home',
		'pages.tecnica_analitica'
	]

	for module in modules_to_test:
		try:
			__import__(module)
			success.append(f"✅ Import OK: {module}")
		except Exception as e:
			errors.append(f"❌ Error importando {module}: {str(e)[:50]}")

except Exception as e:
	errors.append(f"❌ Error verificando imports: {e}")

# 8. Verificar model_results.json
print("\n📊 Verificando model_results.json...")
try:
	import json

	with open('model_results.json', 'r') as f:
		data = json.load(f)
		if 'models' in data and 'dataset_info' in data:
			success.append(f"✅ model_results.json válido ({len(data['models'])} modelos)")
		else:
			errors.append("❌ model_results.json tiene formato incorrecto")
except FileNotFoundError:
	errors.append("❌ CRÍTICO: Falta model_results.json")
except json.JSONDecodeError:
	errors.append("❌ model_results.json no es JSON válido")
except Exception as e:
	errors.append(f"❌ Error con model_results.json: {e}")

# RESUMEN
print("\n" + "=" * 70)
print("📋 RESUMEN DE VERIFICACIÓN")
print("=" * 70)

print(f"\n✅ ÉXITOS: {len(success)}")
for s in success:
	print(f"   {s}")

if warnings:
	print(f"\n⚠️  ADVERTENCIAS: {len(warnings)}")
	for w in warnings:
		print(f"   {w}")

if errors:
	print(f"\n❌ ERRORES CRÍTICOS: {len(errors)}")
	for e in errors:
		print(f"   {e}")
	print("\n🚫 DEBES CORREGIR LOS ERRORES ANTES DE HACER DEPLOY")
	sys.exit(1)
else:
	print("\n" + "=" * 70)
	print("✅ ¡TODO LISTO PARA DEPLOY!")
	print("=" * 70)
	print("\n📝 Próximos pasos:")
	print("   1. git add .")
	print("   2. git commit -m 'Ready for deployment'")
	print("   3. git push")
	print("   4. Ir a render.com y conectar tu repo")
	print("\n🎉 ¡Buena suerte con el deploy!")
	sys.exit(0)
