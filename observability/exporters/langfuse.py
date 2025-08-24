# observability/exporters/langfuse.py
"""
Configurador para enviar traces de OpenTelemetry a Langfuse
"""
import os


def get_langfuse_config() -> dict:
    """
    Configuración para exportar a Langfuse usando OTLP
    
    Variables de entorno necesarias:
    - LANGFUSE_PUBLIC_KEY: clave pública de Langfuse
    - LANGFUSE_SECRET_KEY: clave secreta de Langfuse  
    - LANGFUSE_HOST: endpoint de Langfuse (opcional)
    
    Returns:
        dict: Configuración para OTLP exporter
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY") 
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    if not public_key or not secret_key:
        return {}
    
    # Langfuse acepta traces vía OTLP
    # Endpoint para OTLP traces: /api/public/ingestion/traces
    endpoint = f"{host}/api/public/ingestion/traces"
    
    # Basic auth usando public_key:secret_key
    import base64
    auth_string = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    
    return {
        "endpoint": endpoint,
        "headers": f"Authorization=Basic {auth_string}"
    }


def setup_langfuse_env():
    """
    Helper para configurar variables de entorno para Langfuse
    Llama esta función antes de inicializar OpenTelemetry
    """
    config = get_langfuse_config()
    if config:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config["endpoint"]
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = config["headers"]
        print("🦜 Langfuse OTLP configuration applied")
    else:
        print("⚠️  Langfuse credentials not found. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.")