#!/usr/bin/env bash
# Lanzador de la GUI de AuditV en macOS (doble clic desde el Finder).
#
# El Finder no lee el PATH del shell de inicio de sesión, así que aquí se
# añaden a mano las rutas donde Homebrew y las herramientas suelen vivir.
# Para que macOS lo deje ejecutar: clic derecho -> Abrir -> Abrir (la primera
# vez), o `chmod +x iniciar_gui.command`.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for extra in /opt/homebrew/bin /usr/local/bin "$HOME/bin"; do
    [[ -d "$extra" ]] && PATH="$PATH:$extra"
done
export PATH

cd "$SCRIPT_DIR" || exit 1
if [[ ! -x "venv/bin/python" ]]; then
    echo "[ERROR] No se encontró el entorno virtual. Ejecuta primero:"
    echo "        ./instalar.sh"
    read -rp "Pulsa Enter para cerrar..." _
    exit 1
fi

echo "AuditV GUI — pulsa Ctrl+C (o cierra esta ventana) para detenerla."
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "venv/bin/python" tools/auditv_gui.py
code=$?
echo ""
echo "AuditV GUI cerrada (código $code)."
read -rp "Pulsa Enter para cerrar esta ventana..." _
exit $code
