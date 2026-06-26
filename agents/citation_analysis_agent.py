"""
agents/citation_analysis_agent.py
──────────────────────────────────
Analyses the citation landscape of a research topic or text passage.

Sections generated
------------------
1. Important References     — key foundational and recent citations
2. Missing Citations        — notable works absent from the discussion
3. Citation Recommendations — suggested papers to strengthen the work
4. Influential Papers       — most impactful works in the field

Inherits from BaseAgent; uses IBM Granite via WatsonxClient.
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SECTIONS: dict[str, dict] = {
    "important_references": {
        "label": "Important References",
        "instruction": (
            "Identify 6–8 important references highly relevant to this topic. "
            "For each, provide: (a) author(s) and year, (b) title or brief "
            "description, (c) why it is foundational or significant. "
            "Format as a numbered list."
        ),
    },
    "missing_citations": {
        "label": "Missing Citations",
        "instruction": (
            "Identify 4–6 notable works or authors that should be cited when "
            "discussing this topic but are often overlooked. Explain for each "
            "why the omission represents a gap. Format as a numbered list."
        ),
    },
    "citation_recommendations": {
        "label": "Citation Recommendations",
        "instruction": (
            "Recommend 5–7 specific papers or works that would strengthen "
            "research on this topic. For each, explain: (a) what it contributes, "
            "(b) how it supports or extends the current discussion, and "
            "(c) where in a paper it would best be cited. Format as a numbered list."
        ),
    },
    "influential_papers": {
        "label": "Influential Papers",
        "instruction": (
            "Describe the 5–6 most influential papers in this research area. "
            "For each, cover: (a) title and authors, (b) core contribution, "
            "(c) citation impact and why it changed the field. "
            "Format as a numbered list."
        ),
    },
}


class CitationAnalysisAgent(BaseAgent):
    """
    Analyses citations and reference patterns for a given research topic.

    Parameters
    ----------
    name :              Agent identifier used in logs.
    model_id :          IBM Granite model override.
    generation_params : Generation parameter overrides.
    sections :          Ordered subset of section keys to generate.
                        Defaults to all four.
    """

    SYSTEM_PROMPT: str = (
        "You are an expert bibliometrics analyst and academic librarian with "
        "comprehensive knowledge of research literature across disciplines. "
        "Your task is to analyse citation patterns, identify key references, "
        "and recommend impactful works. Be specific with author names, paper "
        "titles, and publication years where possible."
    )

    DEFAULT_SECTION_PARAMS: dict = {
        "max_new_tokens": 600,
        "temperature":    0.55,
        "top_p":          0.88,
    }

    def __init__(
        self,
        name:              Optional[str]       = None,
        model_id:          Optional[str]       = None,
        generation_params: Optional[dict]      = None,
        sections:          Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            name=name or "CitationAnalysisAgent",
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
        Perform citation analysis for *task*.

        Parameters
        ----------
        task :    Topic or text passage to analyse.
        context : Optional RAG context string (kwarg).

        Returns
        -------
        dict — keys: ``topic``, ``sections`` (label→text), ``narrative``
        """
        context: str = kwargs.get("context", "")
        logger.info("[%s] Starting citation analysis for: %.80s", self.name, task)

        sections_output: dict[str, str] = {}
        for key in self._sections:
            label = SECTIONS[key]["label"]
            logger.debug("[%s] Generating section: %s", self.name, label)
            sections_output[label] = self._generate_section(key, task, context)

        narrative = self._build_narrative(task, sections_output)
        logger.info("[%s] Citation analysis complete (%d sections)", self.name, len(sections_output))

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
        header = f"CITATION ANALYSIS REPORT\nTopic: {topic}\n{'=' * 60}\n"
        body   = "\n\n".join(
            f"{label}\n{'-' * len(label)}\n{content}"
            for label, content in sections.items()
        )
        return header + body
