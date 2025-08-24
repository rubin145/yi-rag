# observability/__init__.py
from .logger import MultiLogger, get_logger
from .schemas import QueryEvent, FeedbackEvent
from .otel_client import get_otel_client

# Helpers para configurar diferentes backends
from .exporters.langfuse import setup_langfuse_env
from .exporters.phoenix import setup_phoenix_env  
from .exporters.console import setup_console_env

__all__ = [
    "MultiLogger", 
    "get_logger",
    "QueryEvent", 
    "FeedbackEvent",
    "get_otel_client",
    "setup_langfuse_env",
    "setup_phoenix_env", 
    "setup_console_env"
]