# observability/multi_exporter.py
"""
Configurador de múltiples exportadores OpenTelemetry
Maneja simultáneamente Langfuse, Phoenix, Console y cualquier endpoint OTLP
"""
from __future__ import annotations

import os
from typing import List, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME as RESOURCE_SERVICE_NAME
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

try:
    from langfuse._client.span_processor import LangfuseSpanProcessor
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


class MultiExporterConfig:
    """Configuración centralizada para múltiples exportadores"""
    
    def __init__(self, service_name: str = "yiddish-rag"):
        self.service_name = service_name
        self.enabled = False
        self.provider: Optional[TracerProvider] = None
        self.tracer = None
        
        if not OTEL_AVAILABLE:
            print("⚠️  OpenTelemetry not installed. Skipping observability.")
            return
            
        self._setup_provider()
        self._add_exporters()
        self.enabled = True
    
    def _setup_provider(self):
        """Configurar el TracerProvider base"""
        # Check if there's already a tracer provider set
        current_provider = trace.get_tracer_provider()
        if hasattr(current_provider, '_resource') and current_provider._resource:
            print("⚠️  Using existing TracerProvider")
            self.provider = current_provider
            self.tracer = trace.get_tracer(self.service_name)
            return
        
        resource = Resource.create({
            RESOURCE_SERVICE_NAME: self.service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        })
        
        self.provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer(self.service_name)
    
    def _add_exporters(self):
        """Agregar todos los exportadores configurados"""
        exporters_added = []
        
        # 1. Console Exporter (debugging local)
        if os.getenv("OTEL_EXPORTER_CONSOLE", "false").lower() == "true":
            self.provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            exporters_added.append("Console")
        
        # 2. Langfuse Span Processor (nativo)
        if self._setup_langfuse():
            exporters_added.append("Langfuse")
        
        # 3. Phoenix/Arize OTLP Exporter
        if self._setup_phoenix():
            exporters_added.append("Phoenix")
        
        # 4. Generic OTLP Exporter (para otros backends)
        if self._setup_generic_otlp():
            exporters_added.append("Generic-OTLP")
        
        if exporters_added:
            print(f"✅ OpenTelemetry exporters enabled: {', '.join(exporters_added)}")
        else:
            print("⚠️  No OpenTelemetry exporters configured")
    
    def _setup_langfuse(self) -> bool:
        """Configurar Langfuse usando su SpanProcessor nativo"""
        if not LANGFUSE_AVAILABLE:
            print("⚠️  Langfuse OpenTelemetry package not available")
            return False
        
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if not public_key or not secret_key:
            print("ℹ️  Langfuse credentials not set, skipping Langfuse exporter")
            return False
        
        try:
            # Usar el SpanProcessor nativo de Langfuse
            langfuse_processor = LangfuseSpanProcessor(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            self.provider.add_span_processor(langfuse_processor)
            print(f"🦜 Langfuse exporter configured for {host}")
            return True
        except Exception as e:
            print(f"❌ Failed to setup Langfuse exporter: {e}")
            return False
    
    def _setup_phoenix(self) -> bool:
        """Configurar Phoenix/Arize usando OTLP"""
        # Configurar desde variables de entorno específicas de Phoenix
        phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT")
        phoenix_api_key = os.getenv("PHOENIX_API_KEY")
        phoenix_space_id = os.getenv("PHOENIX_SPACE_ID")
        
        if not phoenix_api_key or not phoenix_space_id:
            print("ℹ️  Phoenix credentials not set, skipping Phoenix exporter")
            return False
        
        try:
            # Phoenix Cloud endpoint para OTLP
            if not phoenix_endpoint:
                phoenix_endpoint = "https://otlp.arize.com/v1"
            
            # Arize específico: usar space_id como space_key
            headers = {
                "space_id": phoenix_space_id,
                "api_key": phoenix_api_key
            }
            
            print(f"🔧 DEBUG: Phoenix headers format: space_id={phoenix_space_id[:10]}..., api_key={phoenix_api_key[:10]}...")
            
            phoenix_exporter = OTLPSpanExporter(
                endpoint=phoenix_endpoint,
                headers=headers
            )
            
            self.provider.add_span_processor(BatchSpanProcessor(phoenix_exporter))
            print(f"🐦‍⬛ Phoenix exporter configured for {phoenix_endpoint}")
            return True
        except Exception as e:
            print(f"❌ Failed to setup Phoenix exporter: {e}")
            return False
    
    def _setup_generic_otlp(self) -> bool:
        """Configurar exportador OTLP genérico"""
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            return False
        
        try:
            headers = {}
            headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
            if headers_str:
                for header in headers_str.split(","):
                    if "=" in header:
                        key, value = header.strip().split("=", 1)
                        headers[key.strip()] = value.strip()
            
            otlp_exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers=headers
            )
            
            self.provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            print(f"🌐 Generic OTLP exporter configured for {endpoint}")
            return True
        except Exception as e:
            print(f"❌ Failed to setup generic OTLP exporter: {e}")
            return False
    
    def get_tracer(self):
        """Obtener el tracer configurado"""
        return self.tracer
    
    def flush(self):
        """Forzar envío de spans pendientes"""
        if self.enabled and self.provider:
            try:
                self.provider.force_flush(timeout_millis=5000)
                print("🚽 All spans flushed")
            except Exception as e:
                print(f"⚠️  Error flushing spans: {e}")


# Singleton global
_multi_exporter: Optional[MultiExporterConfig] = None

def get_multi_exporter() -> MultiExporterConfig:
    """Obtener o crear la instancia global del multi-exporter"""
    global _multi_exporter
    if _multi_exporter is None:
        _multi_exporter = MultiExporterConfig()
    return _multi_exporter
