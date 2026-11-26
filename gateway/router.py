import time
import uuid
import requests
from fastapi import HTTPException

from gateway.cache import make_cache_key, get_cached_response, store_cached_response

from gateway.config import CACHE_ENABLED, CACHE_TTL_SECONDS

from gateway.metrics import REQUEST_COUNTER, REQUEST_LATENCY

from gateway.resilience import post_with_retries, is_circuit_open, record_success, record_failure

from gateway.providers.factory import get_provider

from gateway.conversations import create_conversation_id, add_messages, build_conversation_messages, add_assistant_message

from gateway.logging import logger

from gateway.metrics import(
    MODEL_REQUESTS, CACHE_HITS, CACHE_MISSES, RETRIES, FAILOVERS, ACTIVE_CONVERSATIONS,
)

from gateway.conversations import list_conversations

from gateway.telemetry import tracer

def forward_request(req, request, model, backend_url, provider_name="openai_compatible", failover_from=None):

    request_id = ""

    trace_id = ""

    conversation_id = req.conversation_id or create_conversation_id()

    MODEL_REQUESTS.labels(model=model).inc()

    forward_messages = build_conversation_messages(
        conversation_id,
        req.messages,
    )

    add_messages(
        conversation_id,
        req.messages,
    )

    cache_key = make_cache_key(model, req.messages)

    if CACHE_ENABLED:
        cached = get_cached_response(cache_key, CACHE_TTL_SECONDS)
        if cached:
            CACHE_HITS.inc()

            cached["gateway"]["conversation_id"] = conversation_id
            cached["gateway"]["cache"] = "hit"
            return cached
        CACHE_MISSES.inc()

    request_id = str(uuid.uuid4())
    start = time.time()

    REQUEST_COUNTER.labels(model=model).inc()

    if is_circuit_open(model):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Circuit breaker is open",
                "model": model,
                "backend_url": backend_url,
            },
        )

    try:
        latency_start = time.time()

        provider = get_provider(provider_name)
        original_messages = req.messages
        req.messages = forward_messages
        payload = provider.build_payload(req)
        req.messages = original_messages
        
        with tracer.start_as_current_span("backend_request") as span:
            trace_id = format(span.get_span_context().trace_id, "032x",
                              )

            logger.info(
                "trace_id=%s request_id=%s model=%s conversation_id=%s",
                trace_id,
                request_id,
                model,
                conversation_id,
            )

            span.set_attribute("gateway.model", model)
            span.set_attribute("gateway.backend_url", backend_url)
            span.set_attribute("gateway.conversation_id", conversation_id)
            span.set_attribute("gateway.trace_id", trace_id)

            response, retry_count = post_with_retries(
                url = backend_url,
                payload = payload,
                headers={"Content-Type": "application/json"},
                timeout=600,
                max_retries=2,
            )
                
            span.set_attribute("gateway.retry_count", retry_count)
            span.set_attribute("gateway.status", "success")
            
            elapsed_seconds = time.time() - latency_start

            REQUEST_LATENCY.labels(model=model).observe(
                elapsed_seconds,
                exemplar={"trace_id": trace_id},
            )
            
            record_success(model)
            data = response.json()
            data = provider.parse_response(data)

            assistant_content = ""

            try:
                assistant_content = (
                    data["choices"][0]["message"]["content"]
                )
            except (KeyError, IndexError, TypeError):
                assistant_content = ""

    except requests.exceptions.RequestException as e:
        record_failure(model)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Backend model service unavailable",
                "model": model,
                "backend_url": backend_url,
                "details": str(e),
            },
        )

    latency_ms = round((time.time() - start) * 1000, 2)

    if assistant_content:
        add_assistant_message(
            conversation_id,
            assistant_content,
        )

    data["gateway"] = {
        "request_id": request_id,
        "conversation_id": conversation_id,
        "trace_id": trace_id,
        "latency_ms": latency_ms,
        "retry_count": retry_count,
        "failover_from": failover_from,
        "backend_url": backend_url,
        "cache": "miss",
        "forward_message_count": len(forward_messages),
    }

    if CACHE_ENABLED:
        store_cached_response(cache_key, data)

    logger.info(
        "trace_id=%s completed latency_ms=%s backend=%s",
        trace_id,
        latency_ms,
        backend_url,
    )

    ACTIVE_CONVERSATIONS.set(
        len(list_conversations())
    )

    return data