import httpx

from cli.config import settings


class APIError(Exception):
    """Raised when API request fails."""

    pass


class APIClient:
    """
    Client for communicating with Inyeon Backend.

    Usage:
        client = APIClient()
        result = client.analyze(diff_content)
    """

    def __init__(self, base_url: str | None = None, timeout: int | None = None):
        self.base_url = base_url or settings.api_url
        self.timeout = timeout or settings.timeout

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make HTTP request to backend."""
        url = f"{self.base_url}{endpoint}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException:
            raise APIError(f"Request timed out after {self.timeout}s")
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", e.response.text)
            raise APIError(f"API error ({e.response.status_code}): {detail}")
        except httpx.ConnectError:
            raise APIError(f"Cannot connect to backend at {self.base_url}")
        except Exception as e:
            raise APIError(f"Request failed: {e}")

    def health_check(self) -> dict:
        """Check backend health status."""
        return self._request("GET", "/health")

    def analyze(self, diff: str, context: str | None = None) -> dict:
        """
        Analyze a git diff.

        Args:
            diff: Git diff content.
            context: Optional additional context.

        Returns:
            Analysis result with summary, impact, etc.
        """
        payload = {"diff": diff}
        if context:
            payload["context"] = context

        return self._request("POST", "/api/v1/analyze", json=payload)

    def generate_commit(self, diff: str, issue_ref: str | None = None) -> dict:
        """
        Generate a commit message from a diff.

        Args:
            diff: Git diff content.
            issue_ref: Optional issue reference (e.g., "#234").

        Returns:
            Commit message with type, scope, subject, body, etc.
        """
        payload = {"diff": diff}
        if issue_ref:
            payload["issue_ref"] = issue_ref

        return self._request("POST", "/api/v1/generate-commit", json=payload)
