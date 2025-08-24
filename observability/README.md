# Observability con OpenTelemetry

Sistema de observabilidad basado en **OpenTelemetry** estándar, compatible con múltiples backends.

## 🎯 **Backends Soportados**

**Automáticamente via OpenTelemetry:**
- ✅ **Langfuse** - LLM observability platform (SDK bridge exporter)
- ✅ **Phoenix (Arize Cloud)** - ML monitoring  
- ✅ **Jaeger** - Distributed tracing
- ✅ **Console** - Debug local
- ✅ **Cualquier backend OTLP** - Estándar abierto

**Clientes específicos** (para casos especiales):
- 🔄 **HF Datasets** - Dataset propio (TODO)
- 🔄 **Qdrant** - Translation feedback (TODO)

## 🚀 **Uso Básico**

```python
from observability import get_logger, setup_langfuse_env

# 1. Configurar backend (opcional)
setup_langfuse_env()  # Lee LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

# 2. Usar logger
logger = get_logger()

# Log query event
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
LANGFUSE_PUBLIC_KEY=pk_lf_...
LANGFUSE_SECRET_KEY=sk_lf_...
LANGFUSE_HOST=https://cloud.langfuse.com  # opcional
```

### **Phoenix Cloud (Arize):**
```bash
# Variables para Cloud (recomendado)
PHOENIX_SPACE_ID=your_space_id
PHOENIX_API_KEY=your_api_key
PHOENIX_PROJECT_NAME=yi-rag  # opcional, default "yi-rag"

# Alternativa local (self-host) vía OTLP directo
# PHOENIX_ENDPOINT=http://localhost:6006
# PHOENIX_API_KEY=your_api_key  # opcional para local
```

### **Console Debug:**
```bash
OTEL_EXPORTER_CONSOLE=true
```

### **OTLP Genérico:**
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-otlp-endpoint/v1/traces
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer token,Custom=header"
```

### **Jaeger:**
```bash
JAEGER_ENDPOINT=http://localhost:14268
```

## 📊 **Atributos de Spans**

### **Query Spans:**
```
rag.query
├── rag.query.id = "uuid-123"
├── rag.session.id = "session-456" 
├── rag.query.text = "¿Qué es el conflicto?"
├── rag.query.type = "generation" | "retrieval_only"
└── spans:
    ├── rag.retrieval
    │   ├── rag.retrieval.vector_name = "dense_yi_gemini"
    │   ├── rag.retrieval.k = 5
    │   ├── rag.retrieval.timing_ms = 120
    │   └── rag.retrieval.num_results = 5
    └── rag.generation  # (opcional)
        ├── rag.generation.model = "gpt-4"
        ├── rag.generation.timing_ms = 800
        ├── rag.generation.answer_length = 245
        └── rag.generation.usage.* = tokens, cost, etc.
```

### **Feedback Spans:**
```
rag.feedback
├── rag.feedback.id = "feedback-789"
├── rag.feedback.query_id = "uuid-123"
├── rag.feedback.scope = "overall" | "doc_retrieval" | "doc_translation" 
├── rag.feedback.sentiment = "up" | "down" | "annotation_only"
└── rag.feedback.target.* = doc_idx, doc_id, etc.
```

## 🔧 **Configuración Avanzada**

### **Múltiples Backends Simultáneos:**
```python
from observability import setup_langfuse_env, setup_phoenix_env, setup_console_env

# Configurar múltiples exporters
# Para Langfuse puedes usar:
# 1) SDK bridge (automático si LANGFUSE_* está configurado)
# 2) OTLP directo (opcional):
# setup_langfuse_env()  # configura OTLP endpoint hacia Langfuse

# Para Phoenix Cloud usa PHOENIX_SPACE_ID/PHOENIX_API_KEY (arize-otel)
# Para Phoenix local puedes usar OTLP directo:
# setup_phoenix_env()
setup_console_env()   # → Console debug

# Automáticamente envía a todos los configurados
logger = get_logger()
logger.log_query_event(...)  # → Va a Langfuse + Phoenix + Console
```

### **Configuración Programática:**
```python
import os
from observability import get_logger

# Set config en runtime
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk_lf_..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk_lf_..."
os.environ["OTEL_EXPORTER_CONSOLE"] = "true"

logger = get_logger()  # Aplica configuración automáticamente
```

## 🏗️ **Arquitectura**

```
app.py
    ↓
observability/logger.py (MultiLogger)
    ↓  
observability/otel_client.py (OpenTelemetryClient)
    ↓
OpenTelemetry SDK
    ↓
┌─────────────┬─────────────┬─────────────┐
│  Langfuse   │   Phoenix   │   Jaeger    │
│   (OTLP)    │   (OTLP)    │   (OTLP)    │
└─────────────┴─────────────┴─────────────┘
```

**Ventajas:**
- ✅ **Un código, múltiples backends**
- ✅ **Estándar abierto** (OpenTelemetry)
- ✅ **Fácil cambiar/añadir backends**
- ✅ **Threading automático** (no bloquea UI)
- ✅ **Configuración declarativa** (env vars)