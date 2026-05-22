"""
app/services/llm_service.py
---------------------------
Reusable async wrapper for the Ollama local LLM API.

Architecture note: All LLM calls in this system route through this single
service. This ensures:
  1. Timeout and retry configuration is centralized.
  2. Graceful degradation is consistent — all callers receive the same
     fallback message structure on Ollama failure.
  3. The async httpx client is created per-call (not module-level) to avoid
     connection pool issues across uvicorn worker restarts.

Design tradeoff: We use streaming=False for simplicity. For production with
large responses, consider streaming=True with Server-Sent Events to improve
perceived latency.
"""

import asyncio
from pathlib import Path

import httpx

from app.core.config import settings
from app.core.constants import LLM_UNAVAILABLE_RESPONSE
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Path to the prompts directory
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Args:
        template_name: Filename without path (e.g. 'summary_prompt.txt').

    Returns:
        The raw prompt template string.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    template_path = PROMPTS_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {template_path}. "
            "Ensure app/prompts/ directory contains the expected .txt files."
        )
    return template_path.read_text(encoding="utf-8")


async def ask_ollama(prompt: str) -> str:
    """
    Send a prompt to the local Ollama API and return the generated text.

    Uses async httpx with configurable timeout and retry logic. On failure,
    returns LLM_UNAVAILABLE_RESPONSE rather than raising — ensuring the API
    always returns a usable response even when Ollama is down.

    Args:
        prompt: The fully formatted prompt string.

    Returns:
        The LLM's generated text, or LLM_UNAVAILABLE_RESPONSE on error.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": settings.MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,   # Lower temperature = more factual, less creative
            "top_p": 0.9,
            "num_predict": 512,   # Cap response length for compliance summaries
        },
    }

    last_error: Exception | None = None

    for attempt in range(1, settings.OLLAMA_RETRIES + 2):  # +2 = initial + retries
        try:
            logger.debug(
                "Ollama request (attempt %d/%d) — model=%s, prompt_length=%d chars",
                attempt,
                settings.OLLAMA_RETRIES + 1,
                settings.MODEL_NAME,
                len(prompt),
            )

            async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

            data = response.json()
            generated_text: str = data.get("response", "").strip()

            if not generated_text:
                logger.warning("Ollama returned empty response on attempt %d", attempt)
                return LLM_UNAVAILABLE_RESPONSE

            logger.info(
                "Ollama response received: %d chars (attempt %d)",
                len(generated_text),
                attempt,
            )
            return generated_text

        except httpx.ConnectError as exc:
            last_error = exc
            logger.warning(
                "Ollama connection refused (attempt %d/%d). "
                "Is Ollama running? URL: %s",
                attempt,
                settings.OLLAMA_RETRIES + 1,
                url,
            )
        except httpx.TimeoutException as exc:
            last_error = exc
            logger.warning(
                "Ollama request timed out after %.1fs (attempt %d/%d)",
                settings.OLLAMA_TIMEOUT,
                attempt,
                settings.OLLAMA_RETRIES + 1,
            )
        except httpx.HTTPStatusError as exc:
            last_error = exc
            logger.error(
                "Ollama HTTP error %d (attempt %d): %s",
                exc.response.status_code,
                attempt,
                exc.response.text[:200],
            )
        except Exception as exc:
            last_error = exc
            logger.error(
                "Unexpected Ollama error (attempt %d): %s: %s",
                attempt,
                type(exc).__name__,
                exc,
            )

        # Brief backoff between retries
        if attempt <= settings.OLLAMA_RETRIES:
            await asyncio.sleep(1.0 * attempt)

    logger.error(
        "All %d Ollama attempts failed. Last error: %s",
        settings.OLLAMA_RETRIES + 1,
        last_error,
    )
    return LLM_UNAVAILABLE_RESPONSE


async def check_ollama_health() -> bool:
    """
    Check whether Ollama is reachable and the target model is available.

    Returns:
        True if Ollama responds successfully, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                is_available = any(
                    settings.MODEL_NAME in name for name in model_names
                )
                if not is_available:
                    logger.warning(
                        "Ollama is running but model '%s' not found. "
                        "Available: %s",
                        settings.MODEL_NAME,
                        model_names,
                    )
                return True  # Ollama is reachable even if model not pulled yet
    except Exception as exc:
        logger.debug("Ollama health check failed: %s", exc)
    return False
