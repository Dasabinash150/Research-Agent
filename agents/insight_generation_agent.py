"""
agents/insight_generation_agent.py
───────────────────────────────────
Synthesises deep, actionable research insights from a topic or text.

Sections generated
------------------
1. Research Gaps      — under-explored problems and missing investigations
2. Novel Ideas        — creative, original research ideas inspired by the text
3. Future Work        — concrete next steps building on existing research
4. Recommendations    — practical guidance for researchers and practitioners

Inherits from BaseAgent; uses IBM Granite via WatsonxClient.
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SECTIONS: dict[str, dict] = {
    "research_gaps": {
        "label": "Research Gaps",
        "instruction": (
            "Identify 5–7 significant research gaps related to this topic. "
            "For each gap, explain: (a) what is missing, (b) why it matters, "
            "and (c) what evidence confirms the gap exists. "
            "Format as a numbered list."
        ),
    },
    "novel_ideas": {
        "label": "Novel Ideas",
        "instruction": (
            "Generate 4–6 original, creative research ideas inspired by this topic. "
            "For each idea describe: (a) the core concept, (b) what makes it novel, "
            "(c) how it could be tested or implemented. "
            "Be specific and forward-thinking. Format as a numbered list."
        ),
    },
    "future_work": {
        "label": "Future Work",
        "instruction": (
            "Outline 5–6 concrete future research directions that directly "
            "extend the current state of knowledge on this topic. "
            "For each, specify: (a) the research question, (b) suggested "
            "methodology, (c) expected contribution. Format as a numbered list."
        ),
    },
    "recommendations": {
        "label": "Recommendations",
        "instruction": (
            "Provide 5–6 practical recommendations for researchers and "
            "practitioners working in this area. Address: (a) methodological "
            "best practices, (b) tooling or data needs, (c) collaboration "
            "opportunities, (d) publication or dissemination strategies. "
            "Format as a numbered list."
        ),
    },
}


class InsightGenerationAgent(BaseAgent):
    """
    Generates deep, actionable insights from a research topic or text.

    Parameters
    ----------
    name :              Agent identifier used in logs.
    model_id :          IBM Granite model override.
    generation_params : Generation parameter overrides.
    sections :          Ordered subset of section keys to generate.
                        Defaults to all four.
    """

    SYSTEM_PROMPT: str = (
        "You are a visionary research scientist and innovation strategist with "
        "expertise spanning multiple academic disciplines. Your task is to "
        "critically analyse research topics and generate deep, original, and "
        "actionable insights. Be specific, evidence-grounded, and creative. "
        "Avoid platitudes — every insight should be immediately useful to a researcher."
    )

    DEFAULT_SECTION_PARAMS: dict = {
        "max_new_tokens": 650,
        "temperature":    0.72,
        "top_p":          0.92,
    }

    def __init__(
        self,
        name:              Optional[str]       = None,
        model_id:          Optional[str]       = None,
        generation_params: Optional[dict]      = None,
        sections:          Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            name=name or "InsightGenerationAgent",
            model_id=model_id,
            generation_params=generation_params or self.DEFAULT_SECTION_PARAMS,
        )
        available = list(SECTIONS.keys())
        requested = sections or available
        invalid   = [s for s in requested if s not in SECTIONS]
        if invalid:
            raise ValueError(f"Unknown section key(s): {invalid}. Valid: {available}")
        self._sections: list[str] = requested

    # ── Abstract implementation ───────────────────────────────────────────────

    def run(self, task: str, **kwargs) -> dict:
        """
        Generate full insight report for *task*.

        Parameters
        ----------
        task :    Research topic or text to analyse.
        context : Optional RAG context string (kwarg).

        Returns
        -------
        dict — keys: ``topic``, ``sections`` (label→text), ``narrative``
        """
        context: str = kwargs.get("context", "")
        logger.info("[%s] Starting insight generation for: %.80s", self.name, task)

        sections_output: dict[str, str] = {}
        for key in self._sections:
            label = SECTIONS[key]["label"]
            logger.debug("[%s] Generating section: %s", self.name, label)
            sections_output[label] = self._generate_section(key, task, context)

        narrative = self._build_narrative(task, sections_output)
        logger.info("[%s] Insight generation complete (%d sections)", self.name, len(sections_output))

        return {"topic": task, "sections": sections_output, "narrative": narrative}

    def run_section(self, section_key: str, task: str, context: str = "") -> str:
        """Generate and return a single section by key."""
        if section_key not in SECTIONS:
            raise ValueError(f"Unknown section key: {section_key!r}. Valid: {list(SECTIONS.keys())}")
        return self._generate_section(section_key, task, context)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_section(self, section_key: str, topic: str, context: str) -> str:
        section    = SECTIONS[section_key]
        user_input = (
            f"Research topic / text: {topic}\n\n"
            f"Section: {section['label']}\n\n"
            f"Instructions: {section['instruction']}"
        )
        return self.generate(self.build_prompt(user_input, context=context))

    @staticmethod
    def _build_narrative(topic: str, sections: dict[str, str]) -> str:
        header = f"INSIGHT GENERATION REPORT\nTopic: {topic}\n{'=' * 60}\n"
        body   = "\n\n".join(
            f"{label}\n{'-' * len(label)}\n{content}"
            for label, content in sections.items()
        )
        return header + body
