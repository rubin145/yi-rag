"""
Phoenix/Arize exporter: Canonical OpenTelemetry + OpenInference integration
"""
import os
from typing import Optional
from .schemas import QueryEvent

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from openinference.semconv.trace import SpanAttributes as OpenInferenceSpanAttributes, OpenInferenceSpanKindValues
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

class PhoenixExporter:
    def __init__(self, service_name: str = "yiddish-rag"):
        self.enabled = False
        self.tracer = None
        if not OTEL_AVAILABLE:
            print("OpenTelemetry not installed. Skipping Phoenix/Arize logging.")
            return
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        otlp_endpoint = os.getenv("PHOENIX_OTLP_ENDPOINT")
        if otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(service_name)
        self.enabled = True

    def log_query_event(self, event: QueryEvent):
        if not self.enabled or not self.tracer:
            return
        phoenix_external_model_id = (
            os.getenv("PHOENIX_EXTERNAL_MODEL_ID")
            or os.getenv("PHOENIX_MODEL_ID")
            or (getattr(event, "generation", None) and getattr(event.generation, "model", None))
            or "unknown-model"
        )
        with self.tracer.start_as_current_span(
            "rag.query",
            kind=trace.SpanKind.SERVER,
            attributes={
                OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND: "CHAIN",
                OpenInferenceSpanAttributes.INPUT_VALUE: event.input.get("query", ""),
                OpenInferenceSpanAttributes.OUTPUT_VALUE: event.generation.answer if event.generation else "No generation",
                "query.id": event.query_id,
                "external_model_id": phoenix_external_model_id,
            },
        ) as root_span:
            if event.generation:
                with self.tracer.start_as_current_span(
                    "llm",
                    kind=trace.SpanKind.INTERNAL,
                    attributes={
                        OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                        OpenInferenceSpanAttributes.LLM_MODEL_NAME: event.generation.model,
                        OpenInferenceSpanAttributes.INPUT_VALUE: event.input.get("query", ""),
                        OpenInferenceSpanAttributes.OUTPUT_VALUE: event.generation.answer,
                        "external_model_id": phoenix_external_model_id,
                    },
                ):
                    pass
            # Add retrieval span if needed (similar pattern)

    def log_feedback_event(self, event):
        if not self.enabled or not self.tracer:
            return
        phoenix_external_model_id = (
            os.getenv("PHOENIX_EXTERNAL_MODEL_ID")
            or os.getenv("PHOENIX_MODEL_ID")
            or os.getenv("MODEL_NAME")
            or "unknown-model"
        )
        with self.tracer.start_as_current_span(
            "rag.feedback",
            kind=trace.SpanKind.INTERNAL,
            attributes={
                "feedback.id": getattr(event, "feedback_id", None),
                "feedback.query_id": getattr(event, "query_id", None),
                "feedback.scope": getattr(event, "scope", None),
                "feedback.sentiment": getattr(event, "sentiment", None),
                "feedback.annotation": getattr(event, "annotation", None),
                "external_model_id": phoenix_external_model_id,
            },
        ):
            pass
