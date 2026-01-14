from fastapi import FastAPI
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_client import make_asgi_app

def setup_monitoring(app: FastAPI, service_name: str) -> None:
    """
    Sets up OpenTelemetry instrumentation and Prometheus metrics exporter for a FastAPI app.
    
    Args:
        app (FastAPI): The FastAPI application instance to instrument.
        service_name (str): The name of the service identification.
    """
    resource: Resource = Resource.create({SERVICE_NAME: service_name})
    reader: PrometheusMetricReader = PrometheusMetricReader()
    provider: MeterProvider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    FastAPIInstrumentor.instrument_app(app)

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

def get_meter(name: str) -> metrics.Meter:
    """
    Returns an OTEL Meter instance.
    
    Args:
        name (str): Name of the meter.
        
    Returns:
        metrics.Meter: The meter instance.
    """
    return metrics.get_meter(name)