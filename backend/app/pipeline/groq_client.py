"""
Shared Groq API Client for Phase 5.
Handles Vision LLM (Document Understanding) and Text LLM (Report Synthesis)
with rate-limit backoff, retry handling, model configurability, and deterministic mock fallbacks.
"""

import base64
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

from .mock_responses import get_mock_document_response, get_mock_survey_report

# Load environment variables from standard project roots
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, "../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv()

logger = logging.getLogger("groq_client")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")
DEFAULT_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")


class GroqClientError(Exception):
    """Base exception for Groq API interactions."""
    pass


class GroqRateLimitError(GroqClientError):
    """Raised when rate limits (HTTP 429) cannot be recovered."""
    pass


class GroqClient:
    """
    Robust, production-grade wrapper for the Groq Cloud API.
    Provides decoupled vision and text completions with rate-limit retries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        vision_model: Optional[str] = None,
        text_model: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: float = 45.0,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        self.vision_model = vision_model or DEFAULT_VISION_MODEL
        self.text_model = text_model or DEFAULT_TEXT_MODEL
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        self.use_mock = (not bool(self.api_key)) or (os.getenv("USE_MOCK_LLM", "false").lower() == "true")
        logger.info(f"[*] GROQ_API_KEY configured at runtime: {bool(self.api_key)}")

        if self.use_mock:
            if not self.api_key:
                logger.info("[*] GROQ_API_KEY not set. Operating in Deterministic Mock Fallback mode.")
            else:
                logger.info("[*] USE_MOCK_LLM is active. Operating in Deterministic Mock Fallback mode.")

    def is_mock_mode(self) -> bool:
        """Returns True if the client is currently running mock fallbacks."""
        return self.use_mock

    def _get_headers(self) -> Dict[str, str]:
        """Generate authorization headers without leaking keys."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AutoClaim-Phase5/2.0",
        }

    def _execute_with_retry(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an HTTP POST request to Groq API with exponential backoff on 429/5xx.
        """
        headers = self._get_headers()
        backoff = 1.5

        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 429:
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = backoff + jitter
                    logger.warning(
                        f"[!] Groq Rate Limit (429) encountered. Attempt {attempt}/{self.max_retries}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    backoff *= 2.0
                    continue

                if response.status_code in {500, 502, 503, 504}:
                    logger.warning(
                        f"[!] Transient Groq server error ({response.status_code}). "
                        f"Attempt {attempt}/{self.max_retries}. Retrying in {backoff:.2f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 1.8
                    continue

                # Unrecoverable error (400, 401, 404, etc.)
                err_text = response.text
                logger.error(f"[X] Groq API returned status {response.status_code}: {err_text[:300]}")
                raise GroqClientError(f"Groq API error ({response.status_code}): {err_text[:300]}")

            except httpx.TimeoutException:
                logger.warning(f"[!] Groq request timed out on attempt {attempt}/{self.max_retries}.")
                if attempt == self.max_retries:
                    raise GroqClientError(f"Groq request timed out after {self.max_retries} attempts.")
                time.sleep(backoff)
                backoff *= 1.5
            except httpx.RequestError as e:
                logger.error(f"[X] Network request error connecting to Groq: {e}")
                if attempt == self.max_retries:
                    raise GroqClientError(f"Failed to connect to Groq API: {e}")
                time.sleep(backoff)

        raise GroqRateLimitError(f"Max retries ({self.max_retries}) exceeded for Groq API call.")

    def text_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.25,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, float]:
        """
        Execute a text chat completion (used for Explainable Report Generation).
        Returns:
            (completion_text, latency_ms)
        """
        selected_model = model or self.text_model
        t0 = time.perf_counter()

        if self.use_mock:
            time.sleep(0.1)  # Simulate small processing time
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            # Inspect system/user messages to deliver mock report
            mock_data = get_mock_survey_report()
            if response_format and response_format.get("type") == "json_object":
                return json.dumps(mock_data["structured_data"]), latency_ms
            return mock_data["markdown_report"], latency_ms

        url = f"{GROQ_BASE_URL}/chat/completions"
        payload: Dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        data = self._execute_with_retry(url, payload)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        try:
            content = data["choices"][0]["message"]["content"]
            return content, latency_ms
        except (KeyError, IndexError) as e:
            raise GroqClientError(f"Unexpected Groq response format: {e}")

    def vision_completion(
        self,
        messages: List[Dict[str, Any]],
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        document_type: str = "rc_book",
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Tuple[str, float]:
        """
        Execute a vision-language completion for document OCR.
        Converts image bytes to base64 data URI.
        Supports automatic fallback if response_format is unsupported by the vision model.
        Returns:
            (completion_text, latency_ms)
        """
        selected_model = model or self.vision_model
        t0 = time.perf_counter()

        if self.use_mock:
            time.sleep(0.12)  # Simulate realistic fast inference
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            mock_doc = get_mock_document_response(document_type)
            return json.dumps(mock_doc), latency_ms

        # Encode image to Base64
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        image_data_uri = f"data:{mime_type};base64,{b64_img}"

        # Construct multimodal message content
        vision_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                # Combine text prompt with image payload
                vision_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri},
                        },
                    ],
                })
            else:
                vision_messages.append(msg)

        url = f"{GROQ_BASE_URL}/chat/completions"

        # Attempt with response_format={"type": "json_object"}
        payload: Dict[str, Any] = {
            "model": selected_model,
            "messages": vision_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        try:
            data = self._execute_with_retry(url, payload)
        except GroqClientError as e:
            # Check if error is due to response_format being unsupported for vision model
            err_msg = str(e).lower()
            if "response_format" in err_msg or "json_object" in err_msg or "unsupported" in err_msg:
                logger.warning(
                    f"[*] Vision model '{selected_model}' rejected response_format. "
                    f"Retrying with prompt-constrained raw JSON..."
                )
                payload.pop("response_format", None)
                data = self._execute_with_retry(url, payload)
            else:
                raise

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        try:
            content = data["choices"][0]["message"]["content"]
            return content, latency_ms
        except (KeyError, IndexError) as e:
            raise GroqClientError(f"Unexpected Groq vision response format: {e}")

    def list_available_models(self) -> List[str]:
        """Query /models endpoint to verify catalog and active models."""
        if self.use_mock:
            return [
                "llama-3.2-11b-vision-preview",
                "llama-3.2-90b-vision-preview",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mock-groq-mode-active"
            ]

        url = f"{GROQ_BASE_URL}/models"
        headers = self._get_headers()
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    models_data = resp.json().get("data", [])
                    return [m.get("id") for m in models_data if "id" in m]
                return []
        except Exception as e:
            logger.warning(f"[!] Could not query Groq /models endpoint: {e}")
            return []


# Global singleton instance
_groq_client_instance: Optional[GroqClient] = None


def get_groq_client() -> GroqClient:
    """Retrieve or initialize the global GroqClient singleton."""
    global _groq_client_instance
    if _groq_client_instance is None:
        _groq_client_instance = GroqClient()
    return _groq_client_instance
