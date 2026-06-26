"""
agents/literature_review_agent.py
──────────────────────────────────
Generates a structured, multi-section literature review for a given topic.

Sections generated
------------------
1. Introduction         — scope and importance of the topic
2. Literature Review    — synthesis of existing work, key themes, debates
3. Methodology          — common research methods used in this domain
4. Findings             — major empirical or theoretical findings
5. Limitations          — gaps and constraints in existing literature
6. Conclusion           — summary and implications

Inherits from BaseAgent; uses IBM Granite via WatsonxClient.
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SECTIONS: dict[str, dict] = {
    "introduction": {
        "label": "Introduction",
        "instruction": (
            "Write a comprehensive introduction (3–4 paragraphs) covering: "
            "(a) the significance and scope of this research topic, "
            "(b) why it matters to academia and/or industry, and "
            "(c) the key questions this literature review addresses."
        ),
    },
    "literature_review": {
        "label": "Literature Review",
        "instruction": (
            "Synthesise the existing body of literature on this topic. "
            "Organise by major themes or schools of thought. Highlight key "
            "authors, seminal works, agreements, and debates. Write 4–6 paragraphs."
        ),
    },
    "methodology": {
        "label": "Methodology",
        "instruction": (
            "Describe the dominant research methodologies and approaches used "
            "in this field. Cover both quantitative and qualitative methods, "
            "datasets commonly used, and evaluation practices. Write 2–3 paragraphs."
        ),
    },
    "findings": {
        "label": "Findings",
        "instruction": (
            "Summarise the major empirical and theoretical findings from the "
            "literature. Use numbered points for clarity. Include at least 6 "
            "distinct, substantive findings."
        ),
    },
    "limitations": {
        "label": "Limitations",
        "instruction": (
            "Critically analyse the limitations of current research on this topic. "
            "Address methodological weaknesses, sample/dataset constraints, "
            "under-explored perspectives, and reproducibility concerns. Write 2–3 paragraphs."
        ),
    },
    "conclusion": {
        "label": "Conclusion",
        "instruction": (
            "Write a concise conclusion (2–3 paragraphs) that: "
            "(a) synthesises key insights from the literature, "
            "(b) highlights the most important gaps, and "
            "(c) suggests directions for future research."
        ),
    },
}


class LiteratureReviewAgent(BaseAgent):
    """
    Produces a complete, section-by-section literature review.

    Parameters
    ----------
    name :              Agent identifier used in logs.
    model_id :          IBM Granite model override.
    generation_params : Generation parameter overrides.
    sections :          Ordered subset of section keys to generate.
                        Defaults to all six in canonical order.
    """

    SYSTEM_PROMPT: str = (
        "You are an expert academic researcher and scientific writer with deep "
        "knowledge across multiple disciplines. Your task is to write rigorous, "
        "well-structured, and evidence-informed literature reviews. "
        "Use formal academic language, cite plausible references where relevant, "
        "and maintain a neutral, scholarly tone throughout."
    )

    DEFAULT_SECTION_PARAMS: dict = {
        "max_new_tokens": 700,
        "temperature":    0.6,
        "top_p":          0.90,
    }

    def __init__(
        self,
        name:              Optional[str]       = None,
        model_id:          Optional[str]       = None,
        generation_params: Optional[dict]      = None,
        sections:          Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            name=name or "LiteratureReviewAgent",
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
        Generate the full literature review for *task*.

        Parameters
        ----------
        task :    Research topic string.
        context : Optional RAG context string (passed as kwarg).

        Returns
        -------
        dict — keys: ``topic``, ``sections`` (label→text), ``narrative``
        """
        context: str = kwargs.get("context", "")
        logger.info("[%s] Starting literature review for: %.80s", self.name, task)

        sections_output: dict[str, str] = {}
        for key in self._sections:
            label = SECTIONS[key]["label"]
            logger.debug("[%s] Generating section: %s", self.name, label)
            sections_output[label] = self._generate_section(key, task, context)

        narrative = self._build_narrative(task, sections_output)
        logger.info("[%s] Literature review complete (%d sections)", self.name, len(sections_output))

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
            f"Research topic: {topic}\n\n"
            f"Section: {section['label']}\n\n"
            f"Instructions: {section['instruction']}"
        )
        return self.generate(self.build_prompt(user_input, context=context))

    @staticmethod
    def _build_narrative(topic: str, sections: dict[str, str]) -> str:
        header = f"LITERATURE REVIEW\nTopic: {topic}\n{'=' * 60}\n"
        body   = "\n\n".join(
            f"{label}\n{'-' * len(label)}\n{content}"
            for label, content in sections.items()
        )
        return header + body
