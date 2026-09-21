#!/usr/bin/env bash
# Lanzador de la GUI de AuditV (Gradio).
# Ejecútalo directamente o con doble clic vía auditv.desktop:
#   - Abre la GUI en una ventana de terminal (konsole) para ver el progreso.
#   - El navegador se abre solo cuando el servidor ya está listo.
#   - Para detenerla: cierra la ventana o pulsa Ctrl+C.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/venv/bin/python"
GUI="$SCRIPT_DIR/tools/auditv_gui.py"

# Si aún no estamos ejecutándonos dentro de un terminal, abrir uno (KDE).
if [[ "${AUDITV_GUI_INSIDE:-0}" != "1" ]]; then
    if command -v konsole >/dev/null 2>&1; then
        exec konsole --separate --workdir "$SCRIPT_DIR" \
            -e env AUDITV_GUI_INSIDE=1 bash "$0" "$@"
    fi
    exec bash "$0" "$@"
fi

cd "$SCRIPT_DIR"
if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] No se encontró el virtualenv. Ejecuta primero:"
    echo "        python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    read -rp "Pulsa Enter para cerrar..." _
    exit 1
fi

echo "AuditV GUI — pulsa Ctrl+C (o cierra esta ventana) para detenerla."
"$VENV_PY" "$GUI" "$@"
code=$?
echo ""
echo "AuditV GUI cerrada (código $code)."
read -rp "Pulsa Enter para cerrar esta ventana..." _
exit $code