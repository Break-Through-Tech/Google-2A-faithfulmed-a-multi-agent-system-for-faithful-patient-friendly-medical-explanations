"""Opt-in Google Cloud observability setup for ADK."""

import os


def configure_cloud_observability(enabled: bool | None = None) -> bool:
    """Configure ADK's Cloud Trace and Cloud Logging exporters when requested.

    Set FAITHFULMED_CLOUD_OBSERVABILITY=true and configure Application Default
    Credentials before starting the agent. Content capture is not enabled here.
    """
    if enabled is None:
        enabled = os.getenv("FAITHFULMED_CLOUD_OBSERVABILITY", "").casefold() in {
            "1",
            "true",
            "yes",
        }
    if not enabled:
        return False

    from google.adk.telemetry.google_cloud import get_gcp_exporters
    from google.adk.telemetry.setup import maybe_set_otel_providers

    os.environ.setdefault("OTEL_SERVICE_NAME", "faithfulmed-readability")
    exporters = get_gcp_exporters(
        enable_cloud_tracing=True,
        enable_cloud_logging=True,
    )
    maybe_set_otel_providers([exporters])
    return True
