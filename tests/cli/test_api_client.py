import pytest
from unittest.mock import patch, MagicMock

from cli.api_client import APIClient, APIError


@pytest.fixture
def api_client():
    """Create an API client for testing."""
    return APIClient(base_url="http://localhost:8000")


def test_api_client_default_url():
    """Test API client uses default URL from config."""
    client = APIClient()
    assert "localhost" in client.base_url or client.base_url is not None


def test_api_client_custom_url():
    """Test API client accepts custom URL."""
    client = APIClient(base_url="http://custom:9000")
    assert client.base_url == "http://custom:9000"


@patch("cli.api_client.httpx.Client")
def test_health_check_success(mock_client_class, api_client):
    """Test health_check returns data on success."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "healthy"}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.request.return_value = mock_response
    mock_client_class.return_value = mock_client

    result = api_client.health_check()

    assert result == {"status": "healthy"}


@patch("cli.api_client.httpx.Client")
def test_analyze_sends_correct_payload(mock_client_class, api_client):
    """Test analyze sends diff in request body."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"summary": "test"}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.request.return_value = mock_response
    mock_client_class.return_value = mock_client

    api_client.analyze("test diff", context="test context")

    mock_client.request.assert_called_once()
    call_kwargs = mock_client.request.call_args
    assert call_kwargs[1]["json"]["diff"] == "test diff"
    assert call_kwargs[1]["json"]["context"] == "test context"
