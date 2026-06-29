import os
from contextvars import ContextVar
from typing import Any

from instructor.exceptions import InstructorRetryException
from pydantic_core import ValidationError
from tenacity import RetryCallState
from tenacity.stop import stop_base

temporary_llm_key: ContextVar[str | None] = ContextVar(
    "temporary_llm_key",
    default=None,
)


def update_llm_credentials(metadata: dict[str, Any] | None):
    match metadata:
        case {"https://ichatbio.org/a2a/v1": {"temporary_llm_key": llm_key}}:
            temporary_llm_key.set(llm_key)
        case _:
            temporary_llm_key.set(None)


def get_llm_client_kwargs() -> dict[str, str]:
    metadata_llm_key = temporary_llm_key.get()
    use_proxy = os.getenv("USE_LLM_PROXY") == "true" or metadata_llm_key is not None

    if use_proxy:
        assert metadata_llm_key is not None, "Temporary LLM key is required for proxy mode"
        proxy_base_url = os.getenv("PROXY_OPENAI_BASE_URL")
        assert proxy_base_url is not None, "PROXY_OPENAI_BASE_URL environment variable must be set"
        return {"api_key": metadata_llm_key, "base_url": proxy_base_url}

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    assert openai_api_key is not None, "OPENAI_API_KEY environment variable must be set"
    assert openai_base_url is not None, "OPENAI_BASE_URL environment variable must be set"
    return {"api_key": openai_api_key, "base_url": openai_base_url}


def _get_terminal_validation_error(e: Exception):
    if isinstance(e, ValidationError):
        for error in e.errors():
            if error.get("ctx", {}).get("terminal", False):
                return error
    return None


class AIGenerationException(Exception):
    def __init__(self, e: InstructorRetryException):
        messages = []
        terminal_error = _get_terminal_validation_error(e)
        if terminal_error:
            messages.append(f"Error: {terminal_error['msg']}")
        else:
            messages.append(f"Error: AI failed to generate valid output after {e.n_attempts} attempts.")

        self.message = "\n\n".join(messages)


class StopOnTerminalErrorOrMaxAttempts(stop_base):
    """Stop when a bad value is encountered."""

    def __init__(self, max_attempts: int):
        self.max_attempts = max_attempts

    def __call__(self, retry_state: RetryCallState) -> bool:
        exception = retry_state.outcome.exception()
        if _get_terminal_validation_error(exception):
            return True
        else:
            return retry_state.attempt_number >= self.max_attempts
