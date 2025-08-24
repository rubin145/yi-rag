# Observabilidad con OpenTelemetry - Sistema RAG Yiddish

Sistema de observabilidad refactorizado basado en **OpenTelemetry** estándar, compatible con múltiples backends simultáneos.

## 🎯 **Backends Soportados**

**✅ Funcionando:**
- **Langfuse** - LLM observability platform (SpanProcessor nativo)
- **Phoenix (Arize Cloud)** - ML monitoring via OTLP
- **Console** - Debug local via OpenTelemetry console exporter
- **Cualquier backend OTLP** - Estándar abierto

**🔄 Planeados:**
- **HF Datasets** - Dataset propio (TODO)
- **Qdrant** - Translation feedback (TODO)

## 🚀 **Uso Básico**

```python
from observability import get_logger

# Usar logger
logger = get_logger()

# Log query event (retrieval + generation)
logger.log_query_event(
    query_id="uuid-123",
    session_id="session-456", 
    input_data={"query": "¿Qué es el conflicto?", "k": 5},
    retrieval_data={"hits": [...], "timing_ms": 120},
    generation_data={"model": "gpt-4", "answer": "...", "timing_ms": 800}
)

# Log feedback event  
logger.log_feedback_event(
    feedback_id="feedback-789",
    query_id="uuid-123",
    scope="overall",  # "overall" | "doc_retrieval" | "doc_translation"
    sentiment="up",   # "up" | "down" | "annotation_only"
    annotation="Great answer!"
)
```

## ⚙️ **Configuración por Variables de Entorno**

### **Langfuse:**
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # o https://us.cloud.langfuse.com
```

### **Phoenix (Arize Cloud):**
```bash
PHOENIX_API_KEY=ak-...
PHOENIX_SPACE_ID=U3BhY2U6...
PHOENIX_ENDPOINT=https://otlp.arize.com/v1  # opcional
```

### **Console (debugging local):**
```bash
OTEL_EXPORTER_CONSOLE=true
```

### **Generic OTLP (otros backends):**
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-backend.com/v1/traces
OTEL_EXPORTER_OTLP_HEADERS="api-key=xxx,other-header=yyy"
```

## 🔧 **Verificar Estado**

```bash
poetry run python observability/setup_guide.py
```

## 🏗️ **Arquitectura**

```
MultiLogger (observability/logger.py)
├── OpenTelemetryClient (observability/otel_client.py)
└── MultiExporterConfig (observability/multi_exporter.py)
    ├── LangfuseSpanProcessor (langfuse._client.span_processor)
    ├── OTLPSpanExporter → Phoenix/Arize
    ├── ConsoleSpanExporter (debug)
    └── OTLPSpanExporter → Generic backends
```

### **Componentes:**

1. **MultiLogger**: Interfaz principal, compatible con código existente
2. **MultiExporterConfig**: Configura múltiples exportadores simultáneamente
3. **OpenTelemetryClient**: Cliente OTel con convenciones OpenInference para RAG
4. **Exportadores específicos**: Cada backend tiene su configuración optimizada

## 🎨 **Características**

- ✅ **Múltiples exportadores simultáneos**: Langfuse + Phoenix + otros
- ✅ **OpenInference compatible**: Convenciones semánticas para RAG/LLM
- ✅ **Configuración independiente**: Cada backend se configura por separado
- ✅ **Fallback graceful**: Si un exportador falla, otros siguen funcionando
- ✅ **Performance optimizada**: Sin threading innecesario, batch processing
- ✅ **Retrocompatible**: API existente sigue funcionando

## 🧪 **Testing**

```bash
# Test básico del sistema
poetry run python -c "
from observability.logger import get_logger
logger = get_logger()
print(f'Logger enabled: {logger.enabled}')
"

# Test con eventos de ejemplo
poetry run python -c "
from observability.logger import get_logger
import uuid
logger = get_logger()
logger.log_query_event(
    query_id=str(uuid.uuid4()),
    session_id=str(uuid.uuid4()),
    input_data={'query': 'test'},
    retrieval_data={'hits': [], 'timing_ms': 100, 'vector_name': 'test', 'k': 5}
)
logger.flush()
"
```

## 🔄 **Cambios Principales vs. Versión Anterior**

### **✅ Arreglado:**
- **Langfuse**: Ahora usa `LangfuseSpanProcessor` correcto de `langfuse._client.span_processor`
- **Multi-exportadores**: Phoenix + Langfuse funcionan simultáneamente
- **Configuración limpia**: Cada exportador tiene su configuración independiente
- **Threading removido**: Ya no hay problemas de concurrencia
- **Error handling**: Falla graceful si un exportador no funciona

### **❌ Removido:**
- Threading innecesario en eventos
- Variables de entorno conflictivas
- Configuración monolítica de OTLP

## 🚀 **Próximos Pasos**

1. **Verificar trazas**: Revisar dashboards de Langfuse y Phoenix
2. **Ajustar configuración**: Phoenix requiere credenciales válidas
3. **Agregar métricas**: Complementar trazas con métricas OpenTelemetry
4. **Instrumentación automática**: Auto-instrumentar bibliotecas (OpenAI, etc.)

## 🔍 **Troubleshooting**

### Langfuse no recibe trazas:
- Verificar `LANGFUSE_PUBLIC_KEY` y `LANGFUSE_SECRET_KEY`
- Verificar conectividad a `LANGFUSE_HOST`

### Phoenix muestra PERMISSION_DENIED:
- Verificar `PHOENIX_API_KEY` y `PHOENIX_SPACE_ID`
- Verificar que las credenciales sean para el proyecto correcto

### Console no muestra trazas:
- Configurar `OTEL_EXPORTER_CONSOLE=true`
- Las trazas aparecen en stdout después de `logger.flush()`
