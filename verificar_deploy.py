#!/usr/bin/env python3
"""
Script de verificación pre-deployment para Railway
Verifica que todos los archivos y configuraciones estén correctos
"""

import os
import sys
import json


def check_file_exists(filename, critical=True):
	"""Verificar si un archivo existe"""
	exists = os.path.isfile(filename)
	status = "✅" if exists else ("❌" if critical else "⚠️")
	print(f"{status} {filename}: {'Encontrado' if exists else 'NO ENCONTRADO'}")
	return exists


def check_app_structure(filename='app.py'):
	"""Verificar estructura correcta de app.py"""
	if not os.path.isfile(filename):
		print(f"❌ {filename} no encontrado")
		return False

	with open(filename, 'r', encoding='utf-8') as f:
		content = f.read()

	has_server = 'server = app.server' in content or 'server=app.server' in content
	has_main = '__main__' in content

	print(f"\n📋 Verificando estructura de {filename}:")
	print(f"  {'✅' if has_server else '❌'} Expone 'server' para Gunicorn: {has_server}")
	print(f"  {'✅' if has_main else '⚠️'} Tiene bloque __main__: {has_main}")

	return has_server


def check_json_file(filename='model_results.json'):
	"""Verificar que el archivo JSON sea válido"""
	if not os.path.isfile(filename):
		print(f"❌ {filename} no encontrado")
		return False

	try:
		with open(filename, 'r', encoding='utf-8') as f:
			data = json.load(f)
		print(f"✅ {filename}: JSON válido ({len(data)} keys)")
		return True
	except json.JSONDecodeError as e:
		print(f"❌ {filename}: JSON inválido - {e}")
		return False


def check_requirements(filename='requirements.txt'):
	"""Verificar requirements.txt"""
	if not os.path.isfile(filename):
		print(f"❌ {filename} no encontrado")
		return False

	required_packages = ['dash', 'pandas', 'gunicorn', 'plotly', 'scikit-learn']

	with open(filename, 'r') as f:
		content = f.read().lower()

	missing = [pkg for pkg in required_packages if pkg not in content]

	if missing:
		print(f"⚠️  Paquetes faltantes en requirements.txt: {', '.join(missing)}")
		return False

	print(f"✅ requirements.txt contiene todos los paquetes necesarios")
	return True


def check_git_status():
	"""Verificar estado de Git"""
	if not os.path.isdir('.git'):
		print("⚠️  No es un repositorio Git. Ejecuta: git init")
		return False

	# Verificar si hay cambios sin commit
	stream = os.popen('git status --porcelain')
	output = stream.read()

	if output.strip():
		print("⚠️  Hay cambios sin commit:")
		print(output)
		print("   Ejecuta: git add . && git commit -m 'mensaje'")
		return False

	print("✅ Repositorio Git limpio (todos los cambios commiteados)")
	return True


def main():
	print("=" * 60)
	print("🔍 VERIFICACIÓN PRE-DEPLOYMENT PARA RAILWAY")
	print("=" * 60)

	print("\n📁 Verificando archivos críticos:")
	critical_files = [
		('app.py', True),
		('requirements.txt', True),
		('Dockerfile', True),
		('.dockerignore', False),
		('model_results.json', True),
	]

	all_critical_exist = True
	for filename, critical in critical_files:
		if not check_file_exists(filename, critical) and critical:
			all_critical_exist = False

	print("\n" + "=" * 60)

	# Verificaciones adicionales
	checks_passed = 0
	total_checks = 4

	if check_app_structure():
		checks_passed += 1

	print("\n" + "=" * 60)

	if check_json_file():
		checks_passed += 1

	print("=" * 60)

	if check_requirements():
		checks_passed += 1

	print("=" * 60)
	print("\n📦 Verificando Git:")
	if check_git_status():
		checks_passed += 1

	# Resultado final
	print("\n" + "=" * 60)
	print("📊 RESULTADO FINAL")
	print("=" * 60)

	if all_critical_exist and checks_passed == total_checks:
		print("✅ ¡TODO LISTO PARA DEPLOYMENT!")
		print("\n🚀 Próximos pasos:")
		print("   1. git push origin main")
		print("   2. Conectar repositorio en Railway")
		print("   3. Railway detectará el Dockerfile automáticamente")
		return 0
	else:
		print(f"⚠️  {total_checks - checks_passed} verificaciones fallidas")
		print("\n🔧 Corrige los errores antes de deployar")
		return 1


if __name__ == '__main__':
	sys.exit(main())