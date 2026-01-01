"""
Tests for health check endpoints.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, PropertyMock

import pytest

from ragguard.health import HealthCheckManager


def create_mock_retriever(healthy=True):
    """Create a mock retriever for testing."""
    mock_retriever = Mock()
    mock_retriever.backend_name = "qdrant"
    mock_retriever.collection = "test_collection"

    if healthy:
        mock_retriever.health_check.return_value = {
            "healthy": True,
            "backend": "qdrant",
            "collection": "test_collection",
            "details": {"document_count": 100}
        }
    else:
        mock_retriever.health_check.return_value = {
            "healthy": False,
            "backend": "qdrant",
            "error": "Connection failed"
        }

    return mock_retriever


def test_health_endpoint_healthy():
    """Test health endpoint returns 200 when service is healthy."""
    retriever = create_mock_retriever(healthy=True)
    manager = HealthCheckManager(retriever)

    response, status_code = manager.health_endpoint()

    assert status_code == 200
    assert response["status"] == "healthy"
    assert response["backend"] == "qdrant"
    assert response["collection"] == "test_collection"
    assert "timestamp" in response


def test_health_endpoint_none_retriever():
    """Test health endpoint returns 503 when retriever is None."""
    manager = HealthCheckManager(None)

    response, status_code = manager.health_endpoint()

    assert status_code == 503
    assert response["status"] == "unhealthy"
    assert "Retriever is not initialized" in response["message"]


def test_health_endpoint_exception():
    """Test health endpoint returns 503 on exception."""
    retriever = Mock()
    # Configure property to raise exception when accessed
    type(retriever).backend_name = PropertyMock(side_effect=Exception("Backend error"))

    manager = HealthCheckManager(retriever)

    response, status_code = manager.health_endpoint()

    assert status_code == 503
    assert response["status"] == "unhealthy"
    assert "Backend error" in response["error"]


def test_readiness_endpoint_ready():
    """Test readiness endpoint returns 200 when backend is healthy."""
    retriever = create_mock_retriever(healthy=True)
    manager = HealthCheckManager(retriever)

    response, status_code = manager.readiness_endpoint()

    assert status_code == 200
    assert response["status"] == "ready"
    assert response["backend"] == "qdrant"
    assert "details" in response
    assert response["details"]["document_count"] == 100


def test_readiness_endpoint_not_ready():
    """Test readiness endpoint returns 503 when backend is unhealthy."""
    retriever = create_mock_retriever(healthy=False)
    manager = HealthCheckManager(retriever)

    response, status_code = manager.readiness_endpoint()

    assert status_code == 503
    assert response["status"] == "not_ready"
    assert "Backend health check failed" in response["message"]
    assert "details" in response


def test_readiness_endpoint_custom_check_passes():
    """Test readiness endpoint with passing custom check."""
    retriever = create_mock_retriever(healthy=True)

    def custom_check():
        return True, "Custom check passed"

    manager = HealthCheckManager(retriever, custom_checks=[custom_check])

    response, status_code = manager.readiness_endpoint()

    assert status_code == 200
    assert response["status"] == "ready"


def test_readiness_endpoint_custom_check_fails():
    """Test readiness endpoint with failing custom check."""
    retriever = create_mock_retriever(healthy=True)

    def custom_check():
        return False, "Database migration pending"

    manager = HealthCheckManager(retriever, custom_checks=[custom_check])

    response, status_code = manager.readiness_endpoint()

    assert status_code == 503
    assert response["status"] == "not_ready"
    assert "Database migration pending" in response["message"]


def test_readiness_endpoint_custom_check_exception():
    """Test readiness endpoint when custom check raises exception."""
    retriever = create_mock_retriever(healthy=True)

    def custom_check():
        raise Exception("Check crashed")

    manager = HealthCheckManager(retriever, custom_checks=[custom_check])

    response, status_code = manager.readiness_endpoint()

    assert status_code == 503
    assert response["status"] == "not_ready"
    assert "Check crashed" in response["message"]


def test_startup_endpoint_success():
    """Test startup endpoint returns 200 when service has started."""
    retriever = create_mock_retriever(healthy=True)
    manager = HealthCheckManager(retriever, startup_timeout_seconds=30)

    response, status_code = manager.startup_endpoint()

    assert status_code == 200
    assert response["status"] == "started"
    assert "elapsed_seconds" in response


def test_startup_endpoint_timeout():
    """Test startup endpoint returns 503 after timeout."""
    retriever = create_mock_retriever(healthy=False)
    manager = HealthCheckManager(retriever, startup_timeout_seconds=1)

    # Set startup time to 2 seconds ago
    manager._startup_time = datetime.now(timezone.utc) - timedelta(seconds=2)

    response, status_code = manager.startup_endpoint()

    assert status_code == 503
    assert response["status"] == "timeout"
    assert "exceeded" in response["message"]


def test_startup_endpoint_still_starting():
    """Test startup endpoint returns 503 while still initializing."""
    retriever = create_mock_retriever(healthy=False)
    manager = HealthCheckManager(retriever, startup_timeout_seconds=30)

    response, status_code = manager.startup_endpoint()

    assert status_code == 503
    assert response["status"] == "starting"
    assert "still initializing" in response["message"]


def test_startup_endpoint_exception():
    """Test startup endpoint handles exceptions."""
    retriever = Mock()
    retriever.health_check = Mock(side_effect=Exception("Startup error"))

    manager = HealthCheckManager(retriever, startup_timeout_seconds=30)

    response, status_code = manager.startup_endpoint()

    # Startup endpoint calls readiness_endpoint internally,
    # which catches the exception and returns not_ready status
    assert status_code == 503
    assert response["status"] == "starting"
    assert "still initializing" in response["message"]


def test_flask_endpoints_registration():
    """Test Flask endpoints can be registered."""
    try:
        from flask import Flask

        from ragguard.health import create_flask_health_endpoints

        app = Flask(__name__)
        retriever = create_mock_retriever(healthy=True)

        health_manager = create_flask_health_endpoints(app, retriever)

        assert health_manager is not None
        assert health_manager.retriever == retriever

        # Verify routes were registered
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/health" in routes
        assert "/ready" in routes
        assert "/startup" in routes

    except ImportError:
        pytest.skip("Flask not installed")


def test_flask_endpoints_with_prefix():
    """Test Flask endpoints with URL prefix."""
    try:
        from flask import Flask

        from ragguard.health import create_flask_health_endpoints

        app = Flask(__name__)
        retriever = create_mock_retriever(healthy=True)

        create_flask_health_endpoints(app, retriever, prefix="/api/v1")

        routes = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/api/v1/health" in routes
        assert "/api/v1/ready" in routes
        assert "/api/v1/startup" in routes

    except ImportError:
        pytest.skip("Flask not installed")


def test_fastapi_endpoints_registration():
    """Test FastAPI endpoints can be registered."""
    try:
        from fastapi import FastAPI

        from ragguard.health import create_fastapi_health_endpoints

        app = FastAPI()
        retriever = create_mock_retriever(healthy=True)

        health_manager = create_fastapi_health_endpoints(app, retriever)

        assert health_manager is not None
        assert health_manager.retriever == retriever

        # Verify routes were registered
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/ready" in routes
        assert "/startup" in routes

    except ImportError:
        pytest.skip("FastAPI not installed")


def test_fastapi_endpoints_with_prefix():
    """Test FastAPI endpoints with URL prefix."""
    try:
        from fastapi import FastAPI

        from ragguard.health import create_fastapi_health_endpoints

        app = FastAPI()
        retriever = create_mock_retriever(healthy=True)

        create_fastapi_health_endpoints(app, retriever, prefix="/api/v1")

        routes = [route.path for route in app.routes]
        assert "/api/v1/health" in routes
        assert "/api/v1/ready" in routes
        assert "/api/v1/startup" in routes

    except ImportError:
        pytest.skip("FastAPI not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
