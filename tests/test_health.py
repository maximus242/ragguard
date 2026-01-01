"""
Tests to improve coverage for ragguard/health.py to 95%+.

Focuses on:
- HealthCheckManager methods
- create_flask_health_endpoints
- create_fastapi_health_endpoints
- startup_endpoint edge cases
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class TestHealthCheckManager:
    """Tests for HealthCheckManager class."""

    def create_mock_retriever(self, healthy=True, backend="qdrant", collection="test"):
        """Create a mock retriever for testing."""
        retriever = MagicMock()
        retriever.backend_name = backend
        retriever.collection = collection

        if healthy:
            retriever.health_check.return_value = {
                "healthy": True,
                "backend": backend,
                "collection": collection,
                "details": {"connection": "ok"}
            }
        else:
            retriever.health_check.return_value = {
                "healthy": False,
                "error": "Connection failed"
            }

        return retriever

    def test_health_endpoint_healthy(self):
        """Test health_endpoint returns healthy status."""
        from ragguard.health import HealthCheckManager

        retriever = self.create_mock_retriever()
        manager = HealthCheckManager(retriever)

        response, status_code = manager.health_endpoint()

        assert status_code == 200
        assert response["status"] == "healthy"
        assert response["backend"] == "qdrant"
        assert response["collection"] == "test"
        assert "timestamp" in response

    def test_health_endpoint_retriever_none(self):
        """Test health_endpoint when retriever is None."""
        from ragguard.health import HealthCheckManager

        manager = HealthCheckManager(None)

        response, status_code = manager.health_endpoint()

        assert status_code == 503
        assert response["status"] == "unhealthy"
        assert "not initialized" in response["message"]

    def test_health_endpoint_retriever_error(self):
        """Test health_endpoint when retriever raises exception."""
        from ragguard.health import HealthCheckManager

        retriever = MagicMock()
        retriever.backend_name = property(lambda self: (_ for _ in ()).throw(RuntimeError("Test error")))

        # Make backend_name raise an exception
        type(retriever).backend_name = property(lambda self: (_ for _ in ()).throw(RuntimeError("Backend error")))

        manager = HealthCheckManager(retriever)
        response, status_code = manager.health_endpoint()

        assert status_code == 503
        assert response["status"] == "unhealthy"

    def test_readiness_endpoint_ready(self):
        """Test readiness_endpoint returns ready status."""
        from ragguard.health import HealthCheckManager

        retriever = self.create_mock_retriever()
        manager = HealthCheckManager(retriever)

        response, status_code = manager.readiness_endpoint()

        assert status_code == 200
        assert response["status"] == "ready"
        assert manager._ready is True

    def test_readiness_endpoint_not_ready(self):
        """Test readiness_endpoint when backend unhealthy."""
        from ragguard.health import HealthCheckManager

        retriever = self.create_mock_retriever(healthy=False)
        manager = HealthCheckManager(retriever)

        response, status_code = manager.readiness_endpoint()

        assert status_code == 503
        assert response["status"] == "not_ready"
        assert "Backend health check failed" in response["message"]

    def test_readiness_endpoint_with_custom_checks_pass(self):
        """Test readiness_endpoint with passing custom checks."""
        from ragguard.health import HealthCheckManager

        retriever = self.create_mock_retriever()

        def custom_check():
            return True, "Custom check passed"

        manager = HealthCheckManager(retriever, custom_checks=[custom_check])

        response, status_code = manager.readiness_endpoint()

        assert status_code == 200
        assert response["status"] == "ready"

    def test_readiness_endpoint_with_custom_checks_fail(self):
        """Test readiness_endpoint with failing custom check."""
        from ragguard.health import HealthCheckManager

        retriever = self.create_mock_retriever()

        def custom_check():
            return False, "Database connection failed"

        manager = HealthCheckManager(retriever, custom_checks=[custom_check])

        response, status_code = manager.readiness_endpoint()

        assert status_code == 503
        assert response["status"] == "not_ready"
        assert "Custom check failed" in response["message"]
        assert "Database connection failed" in response["message"]

    def test_readiness_endpoint_custom_check_error(self):
        """Test readiness_endpoint when custom check raises exception."""
        from ragguard.health import HealthCheckManager

        retriever = self.create_mock_retriever()

        def failing_check():
            raise RuntimeError("Check error")

        manager = HealthCheckManager(retriever, custom_checks=[failing_check])

        response, status_code = manager.readiness_endpoint()

        assert status_code == 503
        assert response["status"] == "not_ready"
        assert "Custom check error" in response["message"]

    def test_readiness_endpoint_retriever_error(self):
        """Test readiness_endpoint when retriever.health_check raises."""
        from ragguard.health import HealthCheckManager

        retriever = MagicMock()
        retriever.health_check.side_effect = RuntimeError("Connection timeout")

        manager = HealthCheckManager(retriever)

        response, status_code = manager.readiness_endpoint()

        assert status_code == 503
        assert response["status"] == "not_ready"
        assert "error" in response

    def test_startup_endpoint_success(self):
        """Test startup_endpoint when startup completes."""
        from ragguard.health import HealthCheckManager

        retriever = self.create_mock_retriever()
        manager = HealthCheckManager(retriever)

        response, status_code = manager.startup_endpoint()

        assert status_code == 200
        assert response["status"] == "started"
        assert "elapsed_seconds" in response

    def test_startup_endpoint_still_starting(self):
        """Test startup_endpoint when service is still initializing."""
        from ragguard.health import HealthCheckManager

        retriever = self.create_mock_retriever(healthy=False)
        manager = HealthCheckManager(retriever)

        response, status_code = manager.startup_endpoint()

        assert status_code == 503
        assert response["status"] == "starting"
        assert "still initializing" in response["message"]

    def test_startup_endpoint_timeout(self):
        """Test startup_endpoint when startup timeout exceeded."""
        from ragguard.health import HealthCheckManager

        retriever = self.create_mock_retriever(healthy=False)
        manager = HealthCheckManager(retriever, startup_timeout_seconds=0)

        # Set startup time to past
        manager._startup_time = datetime.now(timezone.utc) - timedelta(seconds=60)

        response, status_code = manager.startup_endpoint()

        assert status_code == 503
        assert response["status"] == "timeout"
        assert "timeout" in response["message"].lower()

    def test_startup_endpoint_error(self):
        """Test startup_endpoint when exception occurs."""
        from ragguard.health import HealthCheckManager

        retriever = MagicMock()
        retriever.health_check.side_effect = RuntimeError("Startup error")

        manager = HealthCheckManager(retriever)

        # Force exception path by making _startup_time None
        manager._startup_time = None

        # This will cause datetime subtraction to fail
        with patch.object(manager, 'readiness_endpoint', side_effect=RuntimeError("Test error")):
            response, status_code = manager.startup_endpoint()

        assert status_code == 503
        assert response["status"] == "error"


class TestFlaskHealthEndpoints:
    """Tests for create_flask_health_endpoints function."""

    def test_flask_endpoints_registered(self):
        """Test that Flask endpoints are registered correctly."""
        from ragguard.health import create_flask_health_endpoints

        # Mock Flask app
        app = MagicMock()
        app.route = MagicMock(return_value=lambda f: f)

        retriever = MagicMock()
        retriever.backend_name = "qdrant"
        retriever.collection = "test"

        health_manager = create_flask_health_endpoints(app, retriever)

        # Check that route was called for each endpoint
        assert app.route.call_count == 3

        # Verify the routes were registered
        route_calls = [call[0][0] for call in app.route.call_args_list]
        assert "/health" in route_calls
        assert "/ready" in route_calls
        assert "/startup" in route_calls

        # Check that manager was returned
        assert health_manager is not None

    def test_flask_endpoints_with_prefix(self):
        """Test Flask endpoints with URL prefix."""
        from ragguard.health import create_flask_health_endpoints

        app = MagicMock()
        app.route = MagicMock(return_value=lambda f: f)

        retriever = MagicMock()
        retriever.backend_name = "qdrant"
        retriever.collection = "test"

        create_flask_health_endpoints(app, retriever, prefix="/api/v1")

        # Check routes have prefix
        route_calls = [call[0][0] for call in app.route.call_args_list]
        assert "/api/v1/health" in route_calls
        assert "/api/v1/ready" in route_calls
        assert "/api/v1/startup" in route_calls


class TestFastAPIHealthEndpoints:
    """Tests for create_fastapi_health_endpoints function."""

    def test_fastapi_endpoints_registered(self):
        """Test that FastAPI endpoints are registered correctly."""
        from ragguard.health import create_fastapi_health_endpoints

        # Mock FastAPI app
        app = MagicMock()
        app.get = MagicMock(return_value=lambda f: f)

        retriever = MagicMock()
        retriever.backend_name = "qdrant"
        retriever.collection = "test"

        # Mock the fastapi import
        with patch.dict('sys.modules', {'fastapi': MagicMock()}):
            health_manager = create_fastapi_health_endpoints(app, retriever)

        # Check that get was called for each endpoint
        assert app.get.call_count == 3

        # Verify the routes
        route_calls = [call[0][0] for call in app.get.call_args_list]
        assert "/health" in route_calls
        assert "/ready" in route_calls
        assert "/startup" in route_calls

    def test_fastapi_endpoints_with_prefix(self):
        """Test FastAPI endpoints with URL prefix."""
        from ragguard.health import create_fastapi_health_endpoints

        app = MagicMock()
        app.get = MagicMock(return_value=lambda f: f)

        retriever = MagicMock()
        retriever.backend_name = "qdrant"
        retriever.collection = "test"

        with patch.dict('sys.modules', {'fastapi': MagicMock()}):
            create_fastapi_health_endpoints(app, retriever, prefix="/api")

        # Check routes have prefix
        route_calls = [call[0][0] for call in app.get.call_args_list]
        assert "/api/health" in route_calls
        assert "/api/ready" in route_calls
        assert "/api/startup" in route_calls


class TestHealthCheckEdgeCases:
    """Edge case tests for health check functionality."""

    def test_multiple_custom_checks(self):
        """Test with multiple custom checks - all must pass."""
        from ragguard.health import HealthCheckManager

        retriever = MagicMock()
        retriever.backend_name = "qdrant"
        retriever.collection = "test"
        retriever.health_check.return_value = {"healthy": True}

        check_results = []

        def check1():
            check_results.append("check1")
            return True, "OK"

        def check2():
            check_results.append("check2")
            return True, "OK"

        manager = HealthCheckManager(retriever, custom_checks=[check1, check2])
        response, status_code = manager.readiness_endpoint()

        assert status_code == 200
        assert "check1" in check_results
        assert "check2" in check_results

    def test_custom_check_short_circuits_on_failure(self):
        """Test that custom checks stop on first failure."""
        from ragguard.health import HealthCheckManager

        retriever = MagicMock()
        retriever.backend_name = "qdrant"
        retriever.collection = "test"
        retriever.health_check.return_value = {"healthy": True}

        check_results = []

        def check1():
            check_results.append("check1")
            return False, "Failed"

        def check2():
            check_results.append("check2")
            return True, "OK"

        manager = HealthCheckManager(retriever, custom_checks=[check1, check2])
        response, status_code = manager.readiness_endpoint()

        assert status_code == 503
        assert "check1" in check_results
        # check2 should not be called after check1 fails
        assert "check2" not in check_results

    def test_health_details_propagated(self):
        """Test that health details are included in response."""
        from ragguard.health import HealthCheckManager

        retriever = MagicMock()
        retriever.backend_name = "qdrant"
        retriever.collection = "test"
        retriever.health_check.return_value = {
            "healthy": True,
            "backend": "qdrant",
            "collection": "test",
            "details": {
                "connection_pool": "active",
                "query_latency_ms": 5
            }
        }

        manager = HealthCheckManager(retriever)
        response, status_code = manager.readiness_endpoint()

        assert status_code == 200
        assert response["details"]["connection_pool"] == "active"
