"""
agents/trend_prediction_agent.py
─────────────────────────────────
Analyses a research topic or body of text and forecasts where the field
is heading.

Sections generated
------------------
1. Emerging Research Topics   — new sub-fields and cross-disciplinary ideas
                                 starting to appear in recent literature
2. Future Research Directions — concrete open problems and logical next steps
3. Technology Trends          — tools, methods, and platforms gaining momentum
4. Research Opportunities     — high-impact gaps where new work is most needed

Inherits from BaseAgent; uses IBM Granite via WatsonxClient.

Usage
-----
    from agents.trend_prediction_agent import TrendPredictionAgent

    agent = TrendPredictionAgent()

    # Full structured report (dict of section → text)
    result = agent.run("Large Language Models in healthcare")
    for section, content in result["sections"].items():
        print(f"\\n{'='*60}\\n{section}\\n{'='*60}\\n{content}")

    # Plain-text narrative only
    print(result["narrative"])

    # Single section
    print(agent.run_section("emerging_topics", "LLMs in healthcare"))
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# ── Section catalogue ─────────────────────────────────────────────────────────

SECTIONS: dict[str, dict] = {
    "emerging_topics": {
        "label": "Emerging Research Topics",
        "instruction": (
            "Identify 5–7 emerging research topics or sub-fields in this domain. "
            "For each topic provide: (a) a clear name, (b) why it is gaining "
            "traction now, and (c) 1–2 representative recent developments. "
            "Format each as a numbered entry."
        ),
    },
    "future_directions": {
        "label": "Future Research Directions",
        "instruction": (
            "Outline 5–6 concrete future research directions. "
            "For each direction describe: (a) the specific open problem, "
            "(b) what progress would look like, and (c) potential impact on "
            "the field. Format each as a numbered entry."
        ),
    },
    "technology_trends": {
        "label": "Technology Trends",
        "instruction": (
            "Describe 4–6 key technology trends shaping this research area — "
            "including methods, architectures, datasets, tools, or platforms. "
            "For each trend explain its current adoption level and likely "
            "trajectory over the next 3–5 years. Format each as a numbered entry."
        ),
    },
    "research_opportunities": {
        "label": "Research Opportunities",
        "instruction": (
            "Identify 4–5 high-value research opportunities where new work "
            "could have significant impact. For each opportunity specify: "
            "(a) the gap it addresses, (b) why it is under-explored, and "
            "(c) the potential scientific or practical contribution. "
            "Format each as a numbered entry."
        ),
    },
}


class TrendPredictionAgent(BaseAgent):
    """
    Forecasts research trends and opportunities for a given topic.

    Parameters
    ----------
    name:
        Agent identifier used in logs.  Defaults to ``'TrendPredictionAgent'``.
    model_id:
        IBM Granite model override.  Falls back to ``MODEL_ID`` env var.
    generation_params:
        Generation parameter overrides applied to every Granite call.
    sections:
        Ordered list of section keys to include.  Defaults to all four
        sections in canonical order.  Valid keys:
        ``'emerging_topics'``, ``'future_directions'``,
        ``'technology_trends'``, ``'research_opportunities'``.
    """

    SYSTEM_PROMPT: str = (
        "You are an expert research strategist and technology forecaster with "
        "deep knowledge of academic literature across multiple disciplines. "
        "Your task is to analyse a research domain and produce insightful, "
        "evidence-grounded predictions about where the field is heading. "
        "Be specific, forward-looking, and actionable. Avoid vague generalities."
    )

    # Trend prediction benefits from slightly more creative generation
    DEFAULT_SECTION_PARAMS: dict = {
        "max_new_tokens": 600,
        "temperature": 0.75,
        "top_p": 0.92,
    }

    def __init__(
        self,
        name: Optional[str] = None,
        model_id: Optional[str] = None,
        generation_params: Optional[dict] = None,
        sections: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            name=name or "TrendPredictionAgent",
            model_id=model_id,
            generation_params=generation_params or self.DEFAULT_SECTION_PARAMS,
        )

        # Validate and store requested sections
        available = list(SECTIONS.keys())
        requested = sections or available
        invalid = [s for s in requested if s not in SECTIONS]
        if invalid:
            raise ValueError(
                f"Unknown section key(s): {invalid}. "
                f"Valid keys: {available}"
            )
        self._sections: list[str] = requested

    # ── Abstract implementation ───────────────────────────────────────────────

    def run(self, task: str, **kwargs) -> dict:
        """
        Generate a full trend prediction report for *task*.

        Parameters
        ----------
        task:
            Research topic or body of text to analyse (e.g.
            ``"Transformer models in computer vision"``).
        context:
            Optional additional context string (e.g. an abstract or
            literature summary) passed as kwargs.

        Returns
        -------
        dict with keys:
            ``"topic"``     — the original input topic string
            ``"sections"``  — OrderedDict mapping section label → generated text
            ``"narrative"`` — all sections joined as a single plain-text string
        """
        context: str = kwargs.get("context", "")
        logger.info("[%s] Starting trend prediction for: %.80s", self.name, task)

        sections_output: dict[str, str] = {}

        for key in self._sections:
            label = SECTIONS[key]["label"]
            logger.debug("[%s] Generating section: %s", self.name, label)
            sections_output[label] = self._generate_section(key, task, context)

        narrative = self._build_narrative(task, sections_output)

        logger.info("[%s] Trend prediction complete (%d sections)", self.name, len(sections_output))

        return {
            "topic":     task,
            "sections":  sections_output,
            "narrative": narrative,
        }

    # ── Single-section API ────────────────────────────────────────────────────

    def run_section(self, section_key: str, task: str, context: str = "") -> str:
        """
        Generate and return a single section by key.

        Parameters
        ----------
        section_key:
            One of ``'emerging_topics'``, ``'future_directions'``,
            ``'technology_trends'``, ``'research_opportunities'``.
        task:
            Research topic string.
        context:
            Optional supplementary context.

        Returns
        -------
        str — generated text for that section.
        """
        if section_key not in SECTIONS:
            raise ValueError(
                f"Unknown section key: {section_key!r}. "
                f"Valid keys: {list(SECTIONS.keys())}"
            )
        return self._generate_section(section_key, task, context)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_section(
        self, section_key: str, topic: str, context: str
    ) -> str:
        """Build a focused prompt for one section and call Granite."""
        section = SECTIONS[section_key]
        user_input = (
            f"Research domain / topic: {topic}\n\n"
            f"Section to generate: {section['label']}\n\n"
            f"Instructions: {section['instruction']}"
        )
        prompt = self.build_prompt(user_input, context=context)
        return self.generate(prompt)

    @staticmethod
    def _build_narrative(topic: str, sections: dict[str, str]) -> str:
        """Concatenate all sections into a readable plain-text document."""
        header = f"TREND PREDICTION REPORT\nTopic: {topic}\n{'=' * 60}\n"
        body = "\n\n".join(
            f"{label}\n{'-' * len(label)}\n{content}"
            for label, content in sections.items()
        )
        return header + body
