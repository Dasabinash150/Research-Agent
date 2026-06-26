"""
agents/orchestrator.py
───────────────────────
AgentOrchestrator: coordinates the full research pipeline.

Workflow
--------
User Query
  → RAG retrieval (VectorStore)
  → LiteratureReviewAgent
  → CitationAnalysisAgent
  → TrendPredictionAgent
  → KnowledgeGraphAgent
  → InsightGenerationAgent

All results are collected and returned in a single, serialisable dict.
Each stage receives the retrieved RAG context so Granite is always
grounded in the user's uploaded documents.

Usage
-----
    from agents.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    result = orchestrator.run("Transformer models in medical imaging")
    print(result["summary"])
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Optional

from agents.literature_review_agent   import LiteratureReviewAgent
from agents.citation_analysis_agent   import CitationAnalysisAgent
from agents.trend_prediction_agent    import TrendPredictionAgent
from agents.knowledge_graph_agent     import KnowledgeGraphAgent
from agents.insight_generation_agent  import InsightGenerationAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Coordinates all research agents in a fixed, sequential pipeline.

    Parameters
    ----------
    vector_store :
        Optional pre-loaded :class:`~rag.vector_store.VectorStore`.
        When provided, each agent call is enriched with the top-k retrieved
        chunks. Omit (or pass ``None``) to run without RAG context.
    top_k :
        Number of chunks to retrieve per agent stage.
    """

    def __init__(
        self,
        vector_store=None,   # rag.VectorStore — optional, avoids circular import
        top_k: int = 5,
    ) -> None:
        self._store = vector_store
        self._top_k = top_k

        # ── Instantiate each agent once; WatsonxClient is lazy per-agent ──────
        self.literature_agent  = LiteratureReviewAgent()
        self.citation_agent    = CitationAnalysisAgent()
        self.trend_agent       = TrendPredictionAgent()
        self.kg_agent          = KnowledgeGraphAgent()
        self.insight_agent     = InsightGenerationAgent()

        logger.info("[Orchestrator] Initialised with %d agents", 5)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Execute the full pipeline for *query*.

        Parameters
        ----------
        query :
            User's research question or topic string.

        Returns
        -------
        dict with keys:
            ``query``           — original query
            ``context``         — RAG context string used by all agents
            ``literature``      — output of LiteratureReviewAgent
            ``citation``        — output of CitationAnalysisAgent
            ``trend``           — output of TrendPredictionAgent
            ``knowledge_graph`` — output of KnowledgeGraphAgent
            ``insight``         — output of InsightGenerationAgent
            ``summary``         — one-paragraph executive summary
            ``timing``          — per-stage elapsed seconds
            ``errors``          — dict of stage → error message (if any)
        """
        logger.info("[Orchestrator] Pipeline started — query: %.100s", query)
        pipeline_start = time.perf_counter()

        # ── Step 1: RAG retrieval ─────────────────────────────────────────────
        context = self._retrieve_context(query)
        logger.info("[Orchestrator] RAG context length: %d chars", len(context))

        # ── Steps 2–6: Run each agent, capture errors independently ──────────
        results = {
            "query":           query,
            "context":         context,
            "literature":      None,
            "citation":        None,
            "trend":           None,
            "knowledge_graph": None,
            "insight":         None,
            "summary":         "",
            "timing":          {},
            "errors":          {},
        }

        stages = [
            ("literature",      self.literature_agent),
            ("citation",        self.citation_agent),
            ("trend",           self.trend_agent),
            ("knowledge_graph", self.kg_agent),
            ("insight",         self.insight_agent),
        ]

        for stage_name, agent in stages:
            results[stage_name] = self._run_stage(
                stage_name, agent, query, context, results["timing"], results["errors"]
            )

        # ── Step 7: Build executive summary from insights ─────────────────────
        results["summary"] = self._build_summary(query, results)

        total = round(time.perf_counter() - pipeline_start, 2)
        results["timing"]["total"] = total
        logger.info("[Orchestrator] Pipeline complete in %.2fs", total)

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _retrieve_context(self, query: str) -> str:
        """Return top-k RAG chunks or empty string when no store is set."""
        if self._store is None:
            logger.debug("[Orchestrator] No VectorStore attached — skipping retrieval")
            return ""
        try:
            return self._store.search_text(query, top_k=self._top_k)
        except Exception as exc:                         # noqa: BLE001
            logger.warning("[Orchestrator] RAG retrieval failed: %s", exc)
            return ""

    def _run_stage(
        self,
        name:    str,
        agent,
        query:   str,
        context: str,
        timing:  dict,
        errors:  dict,
    ):
        """Run one agent stage, capturing timing and any exception."""
        t0 = time.perf_counter()
        try:
            result = agent.run(query, context=context)
            timing[name] = round(time.perf_counter() - t0, 2)
            logger.info("[Orchestrator] Stage '%s' done in %.2fs", name, timing[name])
            return result
        except Exception as exc:                         # noqa: BLE001
            timing[name] = round(time.perf_counter() - t0, 2)
            error_msg    = f"{type(exc).__name__}: {exc}"
            errors[name] = error_msg
            logger.error(
                "[Orchestrator] Stage '%s' FAILED after %.2fs — %s\n%s",
                name, timing[name], error_msg, traceback.format_exc(),
            )
            return {"error": error_msg}

    @staticmethod
    def _build_summary(query: str, results: dict) -> str:
        """Compose a short executive summary from available outputs."""
        lines = [f"Research Pipeline Summary — Query: {query}", ""]

        # Pull first insight if available
        insight = results.get("insight")
        if isinstance(insight, dict) and "sections" in insight:
            first_section = next(iter(insight["sections"].values()), "")
            if first_section:
                preview = first_section[:400].replace("\n", " ")
                lines.append(f"Key Insights: {preview}…")

        # List any errors
        if results.get("errors"):
            lines.append("")
            lines.append("Stages with errors: " + ", ".join(results["errors"].keys()))

        return "\n".join(lines)
