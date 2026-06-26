"""
agents/knowledge_graph_agent.py
────────────────────────────────
Extracts structured knowledge-graph entities from research text.

Extracted entity types
----------------------
- authors       : person names associated with the work
- institutions  : universities, labs, companies mentioned
- concepts      : key theoretical or domain concepts
- methods       : techniques, algorithms, or approaches used
- keywords      : domain-specific index terms
- datasets      : named datasets or benchmarks referenced

Returns structured JSON so the caller can render a graph or index the data.

Inherits from BaseAgent; uses IBM Granite via WatsonxClient.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ── Empty scaffold returned when extraction fails ────────────────────────────
EMPTY_GRAPH: dict[str, Any] = {
    "authors":      [],
    "institutions": [],
    "concepts":     [],
    "methods":      [],
    "keywords":     [],
    "datasets":     [],
    "relationships": [],
}


class KnowledgeGraphAgent(BaseAgent):
    """
    Extracts knowledge-graph entities and relationships from text.

    Parameters
    ----------
    name :              Agent identifier used in logs.
    model_id :          IBM Granite model override.
    generation_params : Generation parameter overrides.
    """

    SYSTEM_PROMPT: str = (
        "You are an expert knowledge-graph engineer and information extraction "
        "specialist. Your task is to analyse research text and extract structured "
        "entities and relationships. Always return valid, well-formed JSON. "
        "Do not include any text outside the JSON object."
    )

    DEFAULT_SECTION_PARAMS: dict = {
        "max_new_tokens": 900,
        "temperature":    0.2,   # low temp → deterministic, structured output
        "top_p":          0.85,
    }

    def __init__(
        self,
        name:              Optional[str]  = None,
        model_id:          Optional[str]  = None,
        generation_params: Optional[dict] = None,
    ) -> None:
        super().__init__(
            name=name or "KnowledgeGraphAgent",
            model_id=model_id,
            generation_params=generation_params or self.DEFAULT_SECTION_PARAMS,
        )

    # ── Abstract implementation ───────────────────────────────────────────────

    def run(self, task: str, **kwargs) -> dict:
        """
        Extract knowledge-graph entities from *task* (text or topic).

        Parameters
        ----------
        task :    Research text or topic to analyse.
        context : Optional RAG context string (kwarg).

        Returns
        -------
        dict with keys:
            ``input``        — original task string
            ``graph``        — extracted entity dict (authors, institutions, …)
            ``entity_counts`` — summary count per entity type
            ``raw_response`` — raw model output (for debugging)
        """
        context: str = kwargs.get("context", "")
        logger.info("[%s] Starting knowledge graph extraction for: %.80s", self.name, task)

        prompt      = self._build_extraction_prompt(task, context)
        raw_output  = self.generate(prompt)
        graph       = self._parse_json(raw_output)

        entity_counts = {
            entity_type: len(items)
            for entity_type, items in graph.items()
            if isinstance(items, list)
        }

        logger.info(
            "[%s] Extracted entities: %s",
            self.name,
            ", ".join(f"{k}={v}" for k, v in entity_counts.items()),
        )

        return {
            "input":         task,
            "graph":         graph,
            "entity_counts": entity_counts,
            "raw_response":  raw_output,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_extraction_prompt(self, topic: str, context: str) -> str:
        """Construct the entity-extraction prompt with explicit JSON schema."""
        schema = json.dumps(
            {
                "authors":       ["<full name>"],
                "institutions":  ["<institution name>"],
                "concepts":      ["<concept name>"],
                "methods":       ["<method or technique name>"],
                "keywords":      ["<keyword>"],
                "datasets":      ["<dataset or benchmark name>"],
                "relationships": [
                    {"source": "<entity>", "relation": "<relation type>", "target": "<entity>"}
                ],
            },
            indent=2,
        )

        user_input = (
            f"Research text / topic:\n{topic}\n\n"
            "Extract all knowledge-graph entities from the text above. "
            "Return ONLY a valid JSON object that strictly follows this schema:\n\n"
            f"{schema}\n\n"
            "Rules:\n"
            "- Every field must be present even if the list is empty ([]).\n"
            "- Authors: full names only (e.g. 'Yoshua Bengio').\n"
            "- Concepts: domain-specific theoretical terms.\n"
            "- Methods: algorithms, architectures, or techniques.\n"
            "- Relationships: extract at least 3 meaningful subject-predicate-object triples.\n"
            "- Return ONLY the JSON object. No markdown, no explanation."
        )
        return self.build_prompt(user_input, context=context)

    def _parse_json(self, raw: str) -> dict[str, Any]:
        """
        Robustly parse JSON from Granite's output.

        Tries three strategies in order:
        1. Direct json.loads on the full string.
        2. Extract the first {...} block with regex.
        3. Return the empty scaffold so callers always get a consistent shape.
        """
        # Strategy 1 — clean parse
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2 — extract outermost JSON object
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3 — fallback
        logger.warning(
            "[%s] Could not parse JSON from model response — returning empty graph",
            self.name,
        )
        return {**EMPTY_GRAPH, "_parse_error": raw[:300]}
