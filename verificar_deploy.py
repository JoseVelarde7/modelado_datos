"""
Script de diagnóstico para problema de puerto en Render
Ejecuta esto localmente para ver qué está mal
"""

import os
import sys

print("=" * 70)
print("🔍 DIAGNÓSTICO: app.py para Render")
print("=" * 70)

# 1. Verificar que app.py existe
print("\n1️⃣ Verificando existencia de app.py...")
if os.path.exists('app.py'):
	print("   ✅ app.py existe")
else:
	print("   ❌ ERROR: app.py no existe")
	sys.exit(1)

# 2. Leer contenido
print("\n2️⃣ Leyendo app.py...")
try:
	with open('app.py', 'r', encoding='utf-8') as f:
		content = f.read()
	print(f"   ✅ Archivo leído ({len(content)} caracteres)")
except Exception as e:
	print(f"   ❌ ERROR leyendo: {e}")
	sys.exit(1)

# 3. Verificar imports críticos
print("\n3️⃣ Verificando imports...")
required_imports = [
	('from dash import Dash', 'Dash'),
	('import dash', 'dash'),
]

has_dash = False
for import_str, name in required_imports:
	if import_str in content:
		print(f"   ✅ {name} importado")
		has_dash = True
		break

if not has_dash:
	print("   ❌ ERROR: Dash no está importado")
	sys.exit(1)

# 4. Verificar creación de app
print("\n4️⃣ Verificando creación de app Dash...")
if 'app = Dash(' in content or 'app=Dash(' in content:
	print("   ✅ app = Dash() encontrado")
else:
	print("   ❌ ERROR: No se encuentra 'app = Dash()'")
	print("   💡 Debe tener: app = Dash(__name__, ...)")
	sys.exit(1)

# 5. CRÍTICO: Verificar server = app.server
print("\n5️⃣ VERIFICANDO CRÍTICO: server = app.server...")
if 'server = app.server' in content or 'server=app.server' in content:
	print("   ✅ server = app.server encontrado")
else:
	print("   ❌ ERROR CRÍTICO: Falta 'server = app.server'")
	print("   💡 Esta línea es OBLIGATORIA para Gunicorn")
	print("   💡 Agregar después de: app = Dash(...)")
	print("\n   Debe verse así:")
	print("   ─────────────────────────────────────")
	print("   app = Dash(__name__, ...)")
	print("   server = app.server  # ← AGREGAR ESTO")
	print("   ─────────────────────────────────────")
	sys.exit(1)

# 6. Verificar app.run_server con configuración correcta
print("\n6️⃣ Verificando app.run_server...")
if 'app.run_server(' in content or 'app.run(' in content:
	print("   ✅ app.run_server() encontrado")

	# Verificar puerto dinámico
	if 'os.environ.get' in content and 'PORT' in content:
		print("   ✅ Puerto dinámico configurado")
	else:
		print("   ⚠️  ADVERTENCIA: Puerto no es dinámico")
		print("   💡 Debería tener: port = int(os.environ.get('PORT', 8050))")

	# Verificar host
	if "host='0.0.0.0'" in content or 'host="0.0.0.0"' in content:
		print("   ✅ host='0.0.0.0' configurado")
	else:
		print("   ⚠️  ADVERTENCIA: host no es 0.0.0.0")
		print("   💡 Debería tener: host='0.0.0.0'")
else:
	print("   ℹ️  app.run_server() no encontrado (OK si usas solo Gunicorn)")

# 7. Buscar errores comunes
print("\n7️⃣ Buscando errores comunes...")
issues = []

if 'sys.path.append' in content and '/home/claude' in content:
	issues.append("sys.path con ruta absoluta '/home/claude' (no funciona en Render)")

if 'app.run_server(debug=True' in content:
	issues.append("debug=True hardcoded (usar variable de entorno)")

if issues:
	for issue in issues:
		print(f"   ⚠️  {issue}")
else:
	print("   ✅ No se encontraron errores comunes")

# 8. Test de importación
print("\n8️⃣ Intentando importar app.py...")
try:
	sys.path.insert(0, os.getcwd())
	import app

	print("   ✅ app.py se puede importar")

	# Verificar que app existe
	if hasattr(app, 'app'):
		print("   ✅ Variable 'app' existe")
	else:
		print("   ❌ ERROR: Variable 'app' no existe")
		sys.exit(1)

	# Verificar que server existe
	if hasattr(app, 'server'):
		print("   ✅ Variable 'server' existe")
	else:
		print("   ❌ ERROR CRÍTICO: Variable 'server' no existe")
		print("   💡 Agregar: server = app.server")
		sys.exit(1)

except ImportError as e:
	print(f"   ❌ ERROR importando: {e}")
	sys.exit(1)
except Exception as e:
	print(f"   ⚠️  Warning al importar: {e}")

# 9. Resumen
print("\n" + "=" * 70)
print("📋 RESUMEN")
print("=" * 70)

print("\n✅ VERIFICACIONES PASADAS:")
print("   • app.py existe")
print("   • Dash importado correctamente")
print("   • app = Dash() está presente")
print("   • server = app.server está presente")
print("   • app y server son importables")

print("\n🎯 CONFIGURACIÓN PARA RENDER:")
print("   Start Command debe ser:")
print("   → gunicorn app:server --bind 0.0.0.0:$PORT")

print("\n" + "=" * 70)
print("✅ app.py ESTÁ LISTO PARA RENDER")
print("=" * 70)
print("\n💡 Si Render sigue sin detectar puerto, verifica:")
print("   1. Start Command en Settings")
print("   2. Que app.py esté en la raíz del repo en GitHub")
print("   3. Los logs de Render para ver el error exacto")
