"""
agents/__init__.py  — public exports for the agents package
"""
# agents package
from agents.watsonx_client            import WatsonxClient, WatsonxConfigError
from agents.base_agent                import BaseAgent
from agents.literature_review_agent   import LiteratureReviewAgent
from agents.citation_analysis_agent   import CitationAnalysisAgent
from agents.trend_prediction_agent    import TrendPredictionAgent
from agents.knowledge_graph_agent     import KnowledgeGraphAgent
from agents.insight_generation_agent  import InsightGenerationAgent
from agents.orchestrator              import AgentOrchestrator

__all__ = [
    "WatsonxClient",
    "WatsonxConfigError",
    "BaseAgent",
    "LiteratureReviewAgent",
    "CitationAnalysisAgent",
    "TrendPredictionAgent",
    "KnowledgeGraphAgent",
    "InsightGenerationAgent",
    "AgentOrchestrator",
]
