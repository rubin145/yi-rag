# observability/exporters/console.py
"""
Configurador para exportar traces a consola (desarrollo/debug)
"""
import os


def setup_console_env():
    """
    Helper para habilitar export a consola
    Útil para desarrollo y debug
    """
    os.environ["OTEL_EXPORTER_CONSOLE"] = "true"
    print("📝 Console exporter enabled")