"""
Health check and readiness probe endpoints for Kubernetes and production monitoring.

This module provides Flask and FastAPI endpoint factories for health checks,
readiness probes, and startup probes compatible with Kubernetes.
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple


class HealthCheckManager:
    """
    Manages health check state and provides endpoints for Flask/FastAPI.

    Example with Flask:
        from flask import Flask
        from ragguard.health import HealthCheckManager

        app = Flask(__name__)
        health_manager = HealthCheckManager(retriever)

        @app.route("/health")
        def health():
            return health_manager.health_endpoint()

        @app.route("/ready")
        def ready():
            return health_manager.readiness_endpoint()

    Example with FastAPI:
        from fastapi import FastAPI
        from ragguard.health import HealthCheckManager

        app = FastAPI()
        health_manager = HealthCheckManager(retriever)

        @app.get("/health")
        def health():
            return health_manager.health_endpoint()

        @app.get("/ready")
        def ready():
            return health_manager.readiness_endpoint()
    """

    def __init__(
        self,
        retriever: Any,
        startup_timeout_seconds: int = 30,
        custom_checks: Optional[list[Callable[[], Tuple[bool, str]]]] = None
    ):
        """
        Initialize health check manager.

        Args:
            retriever: RAGGuard secure retriever instance
            startup_timeout_seconds: How long to wait for startup (default: 30)
            custom_checks: Optional list of custom health check functions
                           Each function should return (bool, str) tuple
        """
        self.retriever = retriever
        self.startup_timeout_seconds = startup_timeout_seconds
        self.custom_checks = custom_checks or []
        self._startup_time = datetime.now(timezone.utc)
        self._ready = False

    def health_endpoint(self) -> Tuple[Dict[str, Any], int]:
        """
        Liveness probe endpoint - is the service running?

        Kubernetes uses this to decide if the container should be restarted.
        This check should be fast and only verify that the service is alive,
        not that it can serve traffic.

        Returns:
            Tuple of (response_dict, http_status_code)
            - 200: Service is healthy (alive)
            - 503: Service is unhealthy (should be restarted)
        """
        try:
            # Basic liveness check - is the retriever object accessible?
            if self.retriever is None:
                return {
                    "status": "unhealthy",
                    "message": "Retriever is not initialized",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, 503

            # Check if we can access basic attributes
            _ = self.retriever.backend_name
            _ = self.retriever.collection

            return {
                "status": "healthy",
                "backend": self.retriever.backend_name,
                "collection": self.retriever.collection,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 200

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 503

    def readiness_endpoint(self) -> Tuple[Dict[str, Any], int]:
        """
        Readiness probe endpoint - can the service accept traffic?

        Kubernetes uses this to decide if the service should receive traffic.
        This check should verify that the service can actually serve requests,
        including backend connectivity.

        Returns:
            Tuple of (response_dict, http_status_code)
            - 200: Service is ready to accept traffic
            - 503: Service is not ready (don't send traffic)
        """
        try:
            # Perform comprehensive health check
            health_status = self.retriever.health_check()

            if not health_status.get("healthy", False):
                return {
                    "status": "not_ready",
                    "message": "Backend health check failed",
                    "details": health_status,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, 503

            # Run custom health checks
            for check_fn in self.custom_checks:
                try:
                    is_healthy, message = check_fn()
                    if not is_healthy:
                        return {
                            "status": "not_ready",
                            "message": f"Custom check failed: {message}",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }, 503
                except Exception as e:
                    return {
                        "status": "not_ready",
                        "message": f"Custom check error: {e!s}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }, 503

            # Mark as ready
            self._ready = True

            return {
                "status": "ready",
                "backend": health_status.get("backend"),
                "collection": health_status.get("collection"),
                "details": health_status.get("details", {}),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 200

        except Exception as e:
            return {
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 503

    def startup_endpoint(self) -> Tuple[Dict[str, Any], int]:
        """
        Startup probe endpoint - has the service finished initializing?

        Kubernetes uses this during initial container startup. Once the startup
        probe succeeds, Kubernetes switches to using readiness/liveness probes.

        This gives slow-starting applications time to initialize without being
        killed by liveness probes.

        Returns:
            Tuple of (response_dict, http_status_code)
            - 200: Service has started successfully
            - 503: Service is still starting (be patient)
        """
        try:
            # Check if we've exceeded startup timeout
            elapsed = (datetime.now(timezone.utc) - self._startup_time).total_seconds()
            if elapsed > self.startup_timeout_seconds:
                return {
                    "status": "timeout",
                    "message": f"Startup exceeded {self.startup_timeout_seconds}s timeout",
                    "elapsed_seconds": elapsed,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, 503

            # Try readiness check
            response, status_code = self.readiness_endpoint()

            if status_code == 200:
                return {
                    "status": "started",
                    "message": "Service has completed startup",
                    "elapsed_seconds": elapsed,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, 200
            else:
                return {
                    "status": "starting",
                    "message": "Service is still initializing",
                    "elapsed_seconds": elapsed,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, 503

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 503


def create_flask_health_endpoints(app, retriever: Any, prefix: str = ""):
    """
    Register health check endpoints with a Flask application.

    Args:
        app: Flask application instance
        retriever: RAGGuard secure retriever
        prefix: Optional URL prefix (e.g., "/api")

    Example:
        from flask import Flask
        from ragguard.health import create_flask_health_endpoints

        app = Flask(__name__)
        create_flask_health_endpoints(app, retriever)

        # Kubernetes livenessProbe:
        #   httpGet:
        #     path: /health
        #     port: 8080
        # Kubernetes readinessProbe:
        #   httpGet:
        #     path: /ready
        #     port: 8080
    """
    health_manager = HealthCheckManager(retriever)

    @app.route(f"{prefix}/health")
    def health():
        response, status_code = health_manager.health_endpoint()
        return response, status_code

    @app.route(f"{prefix}/ready")
    def ready():
        response, status_code = health_manager.readiness_endpoint()
        return response, status_code

    @app.route(f"{prefix}/startup")
    def startup():
        response, status_code = health_manager.startup_endpoint()
        return response, status_code

    return health_manager


def create_fastapi_health_endpoints(app, retriever: Any, prefix: str = ""):
    """
    Register health check endpoints with a FastAPI application.

    Args:
        app: FastAPI application instance
        retriever: RAGGuard secure retriever
        prefix: Optional URL prefix (e.g., "/api")

    Example:
        from fastapi import FastAPI
        from ragguard.health import create_fastapi_health_endpoints

        app = FastAPI()
        create_fastapi_health_endpoints(app, retriever)

        # Kubernetes livenessProbe:
        #   httpGet:
        #     path: /health
        #     port: 8000
        # Kubernetes readinessProbe:
        #   httpGet:
        #     path: /ready
        #     port: 8000
    """
    from fastapi import Response

    health_manager = HealthCheckManager(retriever)

    @app.get(f"{prefix}/health")
    def health(response: Response):
        result, status_code = health_manager.health_endpoint()
        response.status_code = status_code
        return result

    @app.get(f"{prefix}/ready")
    def ready(response: Response):
        result, status_code = health_manager.readiness_endpoint()
        response.status_code = status_code
        return result

    @app.get(f"{prefix}/startup")
    def startup(response: Response):
        result, status_code = health_manager.startup_endpoint()
        response.status_code = status_code
        return result

    return health_manager
