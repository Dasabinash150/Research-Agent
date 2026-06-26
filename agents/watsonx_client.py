"""
agents/watsonx_client.py
────────────────────────
Reusable IBM watsonx.ai client wrapper.

All credentials are read exclusively from environment variables (via the
Flask Config object or python-dotenv) — nothing is hardcoded here.

Usage
-----
    from agents.watsonx_client import WatsonxClient

    client = WatsonxClient()                    # uses Flask app config
    response = client.generate("Explain RAG in one paragraph.")
    print(response)

    # Stream tokens
    for chunk in client.generate_stream("Summarise this paper: ..."):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import logging
import os
from typing import Generator, Optional

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

logger = logging.getLogger(__name__)


class WatsonxConfigError(RuntimeError):
    """Raised when required IBM credentials are missing or invalid."""


class WatsonxClient:
    """
    Thin, reusable wrapper around :class:`ibm_watsonx_ai.foundation_models.ModelInference`.

    Parameters
    ----------
    api_key:    IBM Cloud API key.  Defaults to ``IBM_API_KEY`` env var.
    project_id: watsonx.ai project ID.  Defaults to ``IBM_PROJECT_ID`` env var.
    url:        watsonx.ai service URL.  Defaults to ``IBM_URL`` env var.
    model_id:   Foundation model identifier.  Defaults to ``MODEL_ID`` env var.

    All parameters fall back to environment variables so the class works both
    inside and outside a Flask application context without modification.
    """

    # ── Sensible generation defaults (override per-call via params=) ──────────
    DEFAULT_PARAMS: dict = {
        GenParams.MAX_NEW_TOKENS: 512,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE:    0.7,
        GenParams.TOP_P:          0.9,
        GenParams.TOP_K:          50,
        GenParams.REPETITION_PENALTY: 1.1,
    }

    def __init__(
        self,
        api_key:    Optional[str] = None,
        project_id: Optional[str] = None,
        url:        Optional[str] = None,
        model_id:   Optional[str] = None,
    ) -> None:
        self._api_key    = api_key    or os.getenv("IBM_API_KEY",    "")
        self._project_id = project_id or os.getenv("IBM_PROJECT_ID", "")
        self._url        = url        or os.getenv("IBM_URL",        "https://us-south.ml.cloud.ibm.com")
        self._model_id   = model_id   or os.getenv("MODEL_ID",       "ibm/granite-13b-instruct-v2")

        self._validate_credentials()

        credentials = Credentials(
            url=self._url,
            api_key=self._api_key,
        )

        self._api_client = APIClient(credentials=credentials, project_id=self._project_id)
        self._model = ModelInference(
            model_id=self._model_id,
            api_client=self._api_client,
        )

        logger.info("WatsonxClient initialised — model: %s", self._model_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        params: Optional[dict] = None,
    ) -> str:
        """
        Generate text for *prompt* and return the full response string.

        Parameters
        ----------
        prompt: The input prompt sent to the model.
        params: Optional generation parameter overrides.
                Keys should come from ``GenTextParamsMetaNames``.

        Returns
        -------
        str — The model's generated text.
        """
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        response = self._model.generate_text(prompt=prompt, params=merged)
        return response

    def generate_stream(
        self,
        prompt: str,
        params: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """
        Stream generated tokens for *prompt*.

        Yields
        ------
        str — successive token strings as they arrive from the model.
        """
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        for chunk in self._model.generate_text_stream(prompt=prompt, params=merged):
            yield chunk

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        """Active model identifier."""
        return self._model_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_credentials(self) -> None:
        missing = [
            name
            for name, value in [
                ("IBM_API_KEY",    self._api_key),
                ("IBM_PROJECT_ID", self._project_id),
                ("IBM_URL",        self._url),
                ("MODEL_ID",       self._model_id),
            ]
            if not value
        ]
        if missing:
            raise WatsonxConfigError(
                f"Missing required IBM watsonx credentials: {', '.join(missing)}. "
                "Set them in your .env file or as environment variables."
            )
