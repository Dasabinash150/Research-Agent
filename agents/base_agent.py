"""
agents/base_agent.py
────────────────────
Abstract base class for all Research-Agent agents.

Every concrete agent (ResearchAgent, SummaryAgent, RAGAgent, …) should
inherit from BaseAgent and implement the abstract ``run()`` method.
``generate()`` is provided as a ready-to-use convenience layer on top of
WatsonxClient so subclasses never have to touch the SDK directly.

Inheritance contract
--------------------

    class MyAgent(BaseAgent):
        # Optionally override the system prompt
        SYSTEM_PROMPT = "You are a concise summariser."

        def run(self, task: str, **kwargs) -> str:
            prompt = self.build_prompt(task)
            return self.generate(prompt)

Quick standalone use
--------------------

    from agents.base_agent import BaseAgent

    class EchoAgent(BaseAgent):
        def run(self, task, **kwargs):
            return self.generate(task)

    agent = EchoAgent(name="echo")
    print(agent.run("What is RAG?"))
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Generator, Optional

from agents.watsonx_client import WatsonxClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all agents.

    Parameters
    ----------
    name:
        Human-readable identifier used in logs (defaults to the class name).
    model_id:
        Override the active IBM Granite model.  Omit to use ``MODEL_ID``
        from the environment / Flask config.
    generation_params:
        Dict of ``GenTextParamsMetaNames`` overrides applied to every
        ``generate()`` call from this agent instance.
    """

    # ── Class-level defaults (override in subclasses) ─────────────────────────

    #: Prepended to every prompt unless overridden or explicitly disabled.
    SYSTEM_PROMPT: str = (
        "You are ResearchAgent, an expert AI research assistant. "
        "Provide accurate, well-structured, and concise responses."
    )

    #: Set to False in a subclass to skip prepending SYSTEM_PROMPT.
    USE_SYSTEM_PROMPT: bool = True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        name: Optional[str] = None,
        model_id: Optional[str] = None,
        generation_params: Optional[dict] = None,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self._generation_params: dict = generation_params or {}

        # Lazily-initialised; created on first generate() call so that
        # agents can be imported and instantiated without live credentials
        # (useful for unit-testing subclass logic in isolation).
        self._client: Optional[WatsonxClient] = None
        self._model_id: Optional[str] = model_id

        logger.info("[%s] initialised", self.name)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def run(self, task: str, **kwargs) -> str:
        """
        Execute the agent's primary task.

        Parameters
        ----------
        task:   The raw task string or user query.
        kwargs: Subclass-specific extra arguments.

        Returns
        -------
        str — The agent's final response.
        """

    # ── Core generation API ───────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        params: Optional[dict] = None,
    ) -> str:
        """
        Send *prompt* to IBM Granite and return the full generated text.

        Instance-level ``generation_params`` are merged first; the optional
        per-call *params* argument takes highest precedence.

        Parameters
        ----------
        prompt: Text prompt sent to the model.
        params: Per-call generation parameter overrides.

        Returns
        -------
        str — Model response text.
        """
        merged = {**self._generation_params, **(params or {})}
        start = time.perf_counter()
        response = self._get_client().generate(prompt, params=merged)
        elapsed = time.perf_counter() - start
        logger.debug("[%s] generate() completed in %.2fs", self.name, elapsed)
        return response

    def generate_stream(
        self,
        prompt: str,
        params: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """
        Stream token chunks from IBM Granite for *prompt*.

        Yields
        ------
        str — successive token strings as they arrive.
        """
        merged = {**self._generation_params, **(params or {})}
        yield from self._get_client().generate_stream(prompt, params=merged)

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def build_prompt(self, user_input: str, context: str = "") -> str:
        """
        Assemble a full prompt from system prompt + optional context + user input.

        Parameters
        ----------
        user_input: The task or question from the caller.
        context:    Optional retrieved context (e.g. RAG chunks) inserted
                    between the system prompt and the user query.

        Returns
        -------
        str — The complete formatted prompt string.
        """
        parts: list[str] = []

        if self.USE_SYSTEM_PROMPT and self.SYSTEM_PROMPT:
            parts.append(self.SYSTEM_PROMPT.strip())

        if context:
            parts.append(f"Context:\n{context.strip()}")

        parts.append(f"Task:\n{user_input.strip()}")

        return "\n\n".join(parts)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        model = self._model_id or "(env default)"
        return f"<{self.__class__.__name__} name={self.name!r} model={model!r}>"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_client(self) -> WatsonxClient:
        """Return the shared WatsonxClient, creating it on first access."""
        if self._client is None:
            self._client = WatsonxClient(model_id=self._model_id)
        return self._client
