# observability/exporters/phoenix.py
"""
Configurador para enviar traces de OpenTelemetry a Phoenix (Arize)
"""
import os


def get_phoenix_config() -> dict:
    """
    Configuración para exportar a Phoenix usando OTLP
    
    Variables de entorno necesarias:
    - PHOENIX_ENDPOINT: endpoint de Phoenix (ej: http://localhost:6006)
    - PHOENIX_API_KEY: API key de Phoenix (opcional para local)
    
    Returns:
        dict: Configuración para OTLP exporter
    """
    endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
    api_key = os.getenv("PHOENIX_API_KEY")
    
    # Phoenix acepta traces vía OTLP en /v1/traces
    otlp_endpoint = f"{endpoint}/v1/traces"
    
    headers = "Content-Type=application/x-protobuf"
    if api_key:
        headers += f",Authorization=Bearer {api_key}"
    
    return {
        "endpoint": otlp_endpoint,
        "headers": headers
    }


def setup_phoenix_env():
    """
    Helper para configurar variables de entorno para Phoenix
    Llama esta función antes de inicializar OpenTelemetry
    """
    config = get_phoenix_config()
    if config:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config["endpoint"] 
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = config["headers"]
        print("🐦‍⬛ Phoenix OTLP configuration applied")
    else:
        print("⚠️  Phoenix endpoint not configured. Set PHOENIX_ENDPOINT.")