"""Exception hierarchy for LLM client errors."""


class LLMClientError(Exception):
    """Base exception for all LLM client errors."""


class AuthenticationError(LLMClientError):
    """Raised when the API key is invalid, expired, or missing."""


class RateLimitError(LLMClientError):
    """Raised when the provider rate limit has been exceeded."""


class TimeoutError(LLMClientError):
    """Raised when a request exceeds the configured timeout."""


class LLMProviderError(LLMClientError):
    """Raised for other provider-side errors (4xx/5xx)."""
