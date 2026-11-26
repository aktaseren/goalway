from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import os

def setup_tracing():
    resource = Resource.create({
        "service.name": "goalway",
    })

    provider = TracerProvider(resource=resource)
    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        ""
    )
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint)
    elif os.getenv("TRACE_CONSOLE", "").lower() in ("1", "true", "yes"):
        # Opt-in console tracing for local debugging. The default is
        # no exporter: spans are still created (trace ids appear in
        # logs and responses) without flooding stdout.
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        exporter = ConsoleSpanExporter()
    else:
        processor = BatchSpanProcessor(NoopSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        return
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


class NoopSpanExporter:
    """Drops spans; keeps tracer ids flowing to logs."""

    def export(self, spans):
        return 0  # SUCCESS

    def shutdown(self):
        return

    def force_flush(self, timeout_millis=30000):
        return True

tracer = trace.get_tracer("goalway")