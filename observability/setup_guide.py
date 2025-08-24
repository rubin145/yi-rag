# observability/setup_guide.py
"""
Guía de configuración para observabilidad multi-exportador
"""
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def print_setup_guide():
    print("🔧 Configuración de Observabilidad - Sistema RAG")
    print("="*60)
    
    print("\n📋 VARIABLES DE ENTORNO NECESARIAS:")
    print("-" * 40)
    
    # Langfuse
    print("\n🦜 LANGFUSE:")
    print("LANGFUSE_PUBLIC_KEY=pk-lf-...")
    print("LANGFUSE_SECRET_KEY=sk-lf-...")
    print("LANGFUSE_HOST=https://cloud.langfuse.com  # o https://us.cloud.langfuse.com")
    
    langfuse_ok = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    print(f"Estado: {'✅ Configurado' if langfuse_ok else '❌ Falta configurar'}")
    
    # Phoenix
    print("\n🐦‍⬛ PHOENIX/ARIZE:")
    print("PHOENIX_API_KEY=ak-...")
    print("PHOENIX_SPACE_ID=...")
    print("PHOENIX_ENDPOINT=https://otlp.arize.com/v1  # opcional")
    
    phoenix_ok = bool(os.getenv("PHOENIX_API_KEY") and os.getenv("PHOENIX_SPACE_ID"))
    print(f"Estado: {'✅ Configurado' if phoenix_ok else '❌ Falta configurar'}")
    
    # Console para debugging
    print("\n📝 CONSOLE (debugging local):")
    print("OTEL_EXPORTER_CONSOLE=true  # opcional")
    console_ok = os.getenv("OTEL_EXPORTER_CONSOLE", "false").lower() == "true"
    print(f"Estado: {'✅ Habilitado' if console_ok else '⚪ Deshabilitado'}")
    
    print("\n" + "="*60)
    
    # Estado actual
    print("\n📊 ESTADO ACTUAL:")
    print("-" * 20)
    
    exporters = []
    if langfuse_ok:
        exporters.append("Langfuse")
    if phoenix_ok:
        exporters.append("Phoenix")
    if console_ok:
        exporters.append("Console")
    
    if exporters:
        print(f"✅ Exportadores activos: {', '.join(exporters)}")
    else:
        print("❌ No hay exportadores configurados")
    
    print("\n🚀 PRÓXIMOS PASOS:")
    print("-" * 20)
    print("1. Configurar variables de entorno en .env")
    print("2. Ejecutar: poetry run python app/app.py")
    print("3. Hacer consultas en la interfaz Gradio")
    print("4. Verificar trazas en Langfuse y Phoenix dashboards")
    
    if langfuse_ok:
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        print(f"\n🦜 Langfuse Dashboard: {host}")
    
    if phoenix_ok:
        print("🐦‍⬛ Phoenix Dashboard: https://app.arize.com")

if __name__ == "__main__":
    print_setup_guide()
