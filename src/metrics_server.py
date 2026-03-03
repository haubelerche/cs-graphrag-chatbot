"""
Prometheus Metrics Server
=========================

FastAPI server expose metrics cho Prometheus scraping.
Đọc metrics từ Redis (shared với Chainlit app).

Usage:
    uvicorn metrics_server:app --host 0.0.0.0 --port 8001
"""

import os
import time
import logging
import statistics
import json
from typing import Optional, Any, List
from contextlib import asynccontextmanager

from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import redis


env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # Fallback to default

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Redis connection
redis_client: Optional[redis.Redis] = None
start_time = time.time()


def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client."""
    global redis_client
    if redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
            logger.info(f"Connected to Redis: {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            redis_client = None
    return redis_client


def get_redis_value(key: str, default: Any = 0) -> Any:
    """Safe get value from Redis."""
    client = get_redis_client()
    if not client:
        return default
    try:
        val = client.get(key)
        return val if val is not None else default
    except:
        return default


def get_redis_list(key: str) -> List:
    """Safe get list from Redis."""
    client = get_redis_client()
    if not client:
        return []
    try:
        raw_list = client.lrange(key, 0, -1)
        # Parse JSON strings if needed
        result = []
        for item in raw_list:
            try:
                # Try to parse as JSON first
                parsed = json.loads(item)
                if isinstance(parsed, dict) and 'value' in parsed:
                    result.append(parsed['value'])
                else:
                    result.append(parsed)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, use raw value
                result.append(item)
        return result
    except Exception as e:
        logger.warning(f"Error getting Redis list {key}: {e}")
        return []


def check_service_health(service: str) -> bool:
    """Check if a service is healthy."""
    client = get_redis_client()
    if service == "redis":
        return client is not None
    
    if service == "neo4j":
        try:
            from neo4j import GraphDatabase
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "")
            
            if not password:
                logger.warning("NEO4J_PASSWORD not set")
                return False
                
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            driver.close()
            return True
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")
            return False
    
    if service == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Metrics Server...")
    get_redis_client()
    yield
    logger.info("Shutting down Metrics Server...")


app = FastAPI(
    title="VNPT Money Chatbot Metrics",
    description="Prometheus metrics endpoint",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "redis_connected": get_redis_client() is not None
    }


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint - reads from Redis."""
    
    lines = []
    
    # ==================== Request Metrics ====================
    total_requests = int(get_redis_value("metrics:counter:requests_total", 0))
    lines.append("# HELP vnpt_requests_total Total number of requests processed")
    lines.append("# TYPE vnpt_requests_total counter")
    lines.append(f"vnpt_requests_total {total_requests}")
    
    # Requests per minute - calculate dynamically from timestamps
    rpm = float(get_redis_value("metrics:gauge:requests_per_minute", 0))
    # Also calculate from sorted set if available
    client = get_redis_client()
    if client:
        try:
            now = time.time()
            one_min_ago = now - 60
            rpm_from_ts = client.zcount("metrics:request_timestamps", one_min_ago, now)
            if rpm_from_ts > 0:
                rpm = float(rpm_from_ts)
        except Exception:
            pass
    lines.append("# HELP vnpt_requests_per_minute Current request rate per minute")
    lines.append("# TYPE vnpt_requests_per_minute gauge")
    lines.append(f"vnpt_requests_per_minute {rpm:.2f}")
    
    # Active sessions - read from gauge, fallback to counting session set
    active_sessions = int(get_redis_value("metrics:gauge:active_sessions", 0))
    if active_sessions == 0 and client:
        try:
            # Fallback: count from Redis SET
            set_count = client.scard("metrics:active_session_ids")
            if set_count and set_count > 0:
                active_sessions = set_count
        except Exception:
            pass
    lines.append("# HELP vnpt_active_sessions Number of active sessions")
    lines.append("# TYPE vnpt_active_sessions gauge")
    lines.append(f"vnpt_active_sessions {active_sessions}")
    
    # Concurrent users (from load test or real-time)
    concurrent_users = int(get_redis_value("metrics:gauge:concurrent_users", 0))
    lines.append("# HELP vnpt_concurrent_users Number of concurrent users processing requests")
    lines.append("# TYPE vnpt_concurrent_users gauge")
    lines.append(f"vnpt_concurrent_users {concurrent_users}")
    
    # Load test info
    load_test_running = int(get_redis_value("metrics:gauge:load_test_running", 0))
    load_test_max_users = int(get_redis_value("metrics:gauge:load_test_concurrent_users", 0))
    lines.append("# HELP vnpt_load_test_running Whether a load test is currently running")
    lines.append("# TYPE vnpt_load_test_running gauge")
    lines.append(f"vnpt_load_test_running {load_test_running}")
    lines.append("# HELP vnpt_load_test_max_concurrent Max concurrent users configured in load test")
    lines.append("# TYPE vnpt_load_test_max_concurrent gauge")
    lines.append(f"vnpt_load_test_max_concurrent {load_test_max_users}")
    
    # ==================== Latency Metrics ====================
    latency_values = []
    # Read from request_latency_ms (written by monitoring.py)
    raw_latencies = get_redis_list("metrics:histogram:request_latency_ms")
    for v in raw_latencies:
        try:
            latency_values.append(float(v))
        except:
            pass
    
    if latency_values:
        sorted_lat = sorted(latency_values)
        n = len(sorted_lat)
        p50 = sorted_lat[int(n * 0.5)]
        p95 = sorted_lat[min(int(n * 0.95), n-1)]
        p99 = sorted_lat[min(int(n * 0.99), n-1)]
        avg_lat = statistics.mean(latency_values)
    else:
        p50 = p95 = p99 = avg_lat = 0
    
    lines.append("# HELP vnpt_latency_ms Request latency in milliseconds")
    lines.append("# TYPE vnpt_latency_ms summary")
    lines.append(f'vnpt_latency_ms{{quantile="0.5"}} {p50:.2f}')
    lines.append(f'vnpt_latency_ms{{quantile="0.95"}} {p95:.2f}')
    lines.append(f'vnpt_latency_ms{{quantile="0.99"}} {p99:.2f}')
    lines.append("# HELP vnpt_latency_ms_avg Average latency in milliseconds")
    lines.append("# TYPE vnpt_latency_ms_avg gauge")
    lines.append(f"vnpt_latency_ms_avg {avg_lat:.2f}")
    
    # ==================== Decision Metrics ====================
    decision_types = [
        "direct_answer", "answer_with_clarify", "clarify_required",
        "escalate_personal", "escalate_out_of_scope", "escalate_max_retry",
        "escalate_low_confidence"
    ]
    
    lines.append("# HELP vnpt_decisions_total Total decisions by type")
    lines.append("# TYPE vnpt_decisions_total counter")
    
    total_decisions = 0
    direct_decisions = 0
    escalation_decisions = 0
    
    for dtype in decision_types:
        count = int(get_redis_value(f"metrics:counter:decision_{dtype}", 0))
        lines.append(f'vnpt_decisions_total{{type="{dtype}"}} {count}')
        total_decisions += count
        if dtype == "direct_answer":
            direct_decisions = count
        if dtype.startswith("escalate"):
            escalation_decisions += count
    
    # Rates
    direct_rate = direct_decisions / total_decisions if total_decisions > 0 else 0
    escalation_rate = escalation_decisions / total_decisions if total_decisions > 0 else 0
    
    lines.append("# HELP vnpt_direct_answer_rate Rate of direct answers")
    lines.append("# TYPE vnpt_direct_answer_rate gauge")
    lines.append(f"vnpt_direct_answer_rate {direct_rate:.4f}")
    
    lines.append("# HELP vnpt_escalation_rate Rate of escalations")
    lines.append("# TYPE vnpt_escalation_rate gauge")
    lines.append(f"vnpt_escalation_rate {escalation_rate:.4f}")
    
    # ==================== Confidence Metrics ====================
    confidence_values = []
    raw_conf = get_redis_list("metrics:histogram:confidence_score")
    if not raw_conf:
        raw_conf = get_redis_list("metrics:histogram:confidence")
    for v in raw_conf:
        try:
            confidence_values.append(float(v))
        except:
            pass
    
    avg_conf = statistics.mean(confidence_values) if confidence_values else 0.7
    high_conf_count = sum(1 for c in confidence_values if c >= 0.8)
    high_conf_rate = high_conf_count / len(confidence_values) if confidence_values else 0
    
    lines.append("# HELP vnpt_confidence_avg Average confidence score")
    lines.append("# TYPE vnpt_confidence_avg gauge")
    lines.append(f"vnpt_confidence_avg {avg_conf:.4f}")
    
    lines.append("# HELP vnpt_high_confidence_rate Rate of high confidence responses")
    lines.append("# TYPE vnpt_high_confidence_rate gauge")
    lines.append(f"vnpt_high_confidence_rate {high_conf_rate:.4f}")
    
    # ==================== Error Metrics ====================
    error_count = int(get_redis_value("metrics:counter:errors_total", 0))
    error_rate = error_count / total_requests if total_requests > 0 else 0
    
    lines.append("# HELP vnpt_errors_total Total number of errors")
    lines.append("# TYPE vnpt_errors_total counter")
    lines.append(f"vnpt_errors_total {error_count}")
    
    lines.append("# HELP vnpt_error_rate Error rate")
    lines.append("# TYPE vnpt_error_rate gauge")
    lines.append(f"vnpt_error_rate {error_rate:.4f}")
    
    # ==================== Health Metrics ====================
    lines.append("# HELP vnpt_service_health Service health status (1=healthy, 0=unhealthy)")
    lines.append("# TYPE vnpt_service_health gauge")
    lines.append(f'vnpt_service_health{{service="neo4j"}} {1 if check_service_health("neo4j") else 0}')
    lines.append(f'vnpt_service_health{{service="redis"}} {1 if check_service_health("redis") else 0}')
    lines.append(f'vnpt_service_health{{service="openai"}} {1 if check_service_health("openai") else 0}')
    
    # ==================== Uptime Metrics ====================
    uptime = time.time() - start_time
    lines.append("# HELP vnpt_uptime_seconds Service uptime in seconds")
    lines.append("# TYPE vnpt_uptime_seconds counter")
    lines.append(f"vnpt_uptime_seconds {uptime:.0f}")
    
    metrics_text = "\n".join(lines) + "\n"
    return Response(content=metrics_text, media_type="text/plain; charset=utf-8")


@app.get("/metrics/json")
async def json_metrics():
    """JSON metrics endpoint."""
    total_requests = int(get_redis_value("metrics:counter:requests_total", 0))
    
    # Get latency stats
    latency_values = []
    # Read from request_latency_ms (written by monitoring.py)
    for v in get_redis_list("metrics:histogram:request_latency_ms"):
        try:
            latency_values.append(float(v))
        except:
            pass
    
    if latency_values:
        sorted_lat = sorted(latency_values)
        n = len(sorted_lat)
        latency_stats = {
            "avg_ms": statistics.mean(latency_values),
            "p50_ms": sorted_lat[int(n * 0.5)],
            "p95_ms": sorted_lat[min(int(n * 0.95), n-1)],
            "p99_ms": sorted_lat[min(int(n * 0.99), n-1)]
        }
    else:
        latency_stats = {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}
    
    # Calculate RPM dynamically
    rpm = float(get_redis_value("metrics:gauge:requests_per_minute", 0))
    client = get_redis_client()
    if client:
        try:
            now = time.time()
            rpm_from_ts = client.zcount("metrics:request_timestamps", now - 60, now)
            if rpm_from_ts > 0:
                rpm = float(rpm_from_ts)
        except Exception:
            pass
    
    # Active sessions - from gauge, fallback to Redis SET
    active_sessions = int(get_redis_value("metrics:gauge:active_sessions", 0))
    if active_sessions == 0 and client:
        try:
            set_count = client.scard("metrics:active_session_ids")
            if set_count and set_count > 0:
                active_sessions = set_count
        except Exception:
            pass
    
    return {
        "timestamp": time.time(),
        "requests": {
            "total": total_requests,
            "per_minute": rpm
        },
        "latency": latency_stats,
        "errors": {
            "total": int(get_redis_value("metrics:counter:errors_total", 0))
        },
        "sessions": {
            "active": active_sessions,
            "concurrent_users": int(get_redis_value("metrics:gauge:concurrent_users", 0))
        },
        "load_test": {
            "running": bool(int(get_redis_value("metrics:gauge:load_test_running", 0))),
            "max_concurrent": int(get_redis_value("metrics:gauge:load_test_concurrent_users", 0))
        },
        "health": {
            "neo4j": check_service_health("neo4j"),
            "redis": check_service_health("redis"),
            "openai": check_service_health("openai")
        },
        "uptime_seconds": time.time() - start_time
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("METRICS_PORT", "8001"))
    uvicorn.run("metrics_server:app", host="0.0.0.0", port=port, reload=False, log_level="info")
