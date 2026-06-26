"""
api/routes.py
─────────────
REST API blueprint — all data endpoints consumed by the dashboard.

Endpoints
---------
POST /api/upload       — ingest a PDF into the RAG vector store
POST /api/chat         — single-turn Q&A against stored documents
POST /api/summary      — literature review for a topic
POST /api/citation     — citation analysis for a topic
POST /api/trend        — trend prediction for a topic
POST /api/knowledge    — knowledge graph extraction
POST /api/insight      — insight generation for a topic
POST /api/orchestrate  — run the full 5-agent pipeline
GET  /api/dashboard    — dashboard stats / health check

Every endpoint returns JSON:
    { "ok": true,  "data": <payload> }        on success
    { "ok": false, "error": "<message>" }     on failure
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from functools import wraps
from typing import Callable

from flask import Blueprint, current_app, jsonify, request

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def success(data, status: int = 200):
    """Wrap *data* in a standard success envelope."""
    return jsonify({"ok": True, "data": data}), status


def error(message: str, status: int = 400):
    """Wrap *message* in a standard error envelope."""
    logger.warning("[API] Error response (%d): %s", status, message)
    return jsonify({"ok": False, "error": message}), status


def require_json(fn: Callable) -> Callable:
    """Decorator — reject requests without a JSON body."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return error("Request body must be JSON (Content-Type: application/json)", 415)
        return fn(*args, **kwargs)
    return wrapper


def get_query(required: bool = True) -> tuple[str, None] | tuple[None, object]:
    """Extract and validate the 'query' field from the JSON body."""
    body  = request.get_json(silent=True) or {}
    query = body.get("query", "").strip()
    if required and not query:
        return None, error("Missing required field: 'query'", 400)
    return query, None


def get_vector_store():
    """Return the app-level VectorStore from Flask application context."""
    return current_app.extensions.get("vector_store")


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/upload
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {"pdf"}

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@api_bp.route("/upload", methods=["POST"])
def upload():
    """
    Ingest a PDF into the RAG vector store.

    Request : multipart/form-data  — field name ``file``
    Response: { ok, data: { filename, chunks_added, total_vectors } }
    """
    if "file" not in request.files:
        return error("No file part in request. Use field name 'file'.", 400)

    file = request.files["file"]
    if not file.filename:
        return error("No file selected.", 400)

    if not _allowed_file(file.filename):
        return error("Only PDF files are supported.", 415)

    # ── Save upload ───────────────────────────────────────────────────────────
    upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    filepath  = os.path.join(upload_folder, safe_name)
    file.save(filepath)
    logger.info("[API /upload] Saved '%s' → %s", file.filename, filepath)

    # ── RAG ingestion pipeline ────────────────────────────────────────────────
    try:
        from rag import PDFLoader, TextChunker, VectorStore

        pages  = PDFLoader(filepath).load()
        chunks = TextChunker().split(pages)

        store  = get_vector_store()
        if store is None:
            # Create a fresh store if none is attached to the app
            store = VectorStore(
                persist_dir=current_app.config.get("VECTOR_DB_PATH", "vector_db")
            )
        store.add_chunks(chunks)

        # Attach / update on the app so other requests can use the same index
        current_app.extensions["vector_store"] = store

        return success({
            "filename":      file.filename,
            "pages_loaded":  len(pages),
            "chunks_added":  len(chunks),
            "total_vectors": store.total_vectors,
            "sources":       store.sources,
        }, 201)

    except Exception as exc:                     # noqa: BLE001
        logger.exception("[API /upload] Ingestion failed")
        return error(f"Ingestion failed: {exc}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/chat", methods=["POST"])
@require_json
def chat():
    """
    Answer a question using RAG context + IBM Granite.

    Request : { "query": "<question>" }
    Response: { ok, data: { query, answer, context_used } }
    """
    query, err = get_query()
    if err:
        return err

    try:
        from agents.base_agent import BaseAgent
        from agents.watsonx_client import WatsonxClient

        store   = get_vector_store()
        context = store.search_text(query) if store else ""

        # Build a one-shot prompt directly via WatsonxClient
        client  = WatsonxClient()
        system  = (
            "You are ResearchAgent, an expert AI research assistant. "
            "Answer the question using only the provided context. "
            "If the context does not contain enough information, say so clearly."
        )
        parts   = [system]
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Question:\n{query}")
        prompt  = "\n\n".join(parts)

        answer  = client.generate(prompt)

        return success({
            "query":        query,
            "answer":       answer,
            "context_used": bool(context),
        })

    except Exception as exc:                     # noqa: BLE001
        logger.exception("[API /chat] Generation failed")
        return error(f"Chat failed: {exc}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/summary  (Literature Review)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/summary", methods=["POST"])
@require_json
def summary():
    """
    Generate a literature review for a topic.

    Request : { "query": "<topic>" }
    Response: { ok, data: { topic, sections, narrative } }
    """
    query, err = get_query()
    if err:
        return err

    try:
        from agents.literature_review_agent import LiteratureReviewAgent

        store   = get_vector_store()
        context = store.search_text(query) if store else ""
        result  = LiteratureReviewAgent().run(query, context=context)
        return success(result)

    except Exception as exc:                     # noqa: BLE001
        logger.exception("[API /summary] Failed")
        return error(f"Literature review failed: {exc}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/citation
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/citation", methods=["POST"])
@require_json
def citation():
    """
    Perform citation analysis for a topic.

    Request : { "query": "<topic>" }
    Response: { ok, data: { topic, sections, narrative } }
    """
    query, err = get_query()
    if err:
        return err

    try:
        from agents.citation_analysis_agent import CitationAnalysisAgent

        store   = get_vector_store()
        context = store.search_text(query) if store else ""
        result  = CitationAnalysisAgent().run(query, context=context)
        return success(result)

    except Exception as exc:                     # noqa: BLE001
        logger.exception("[API /citation] Failed")
        return error(f"Citation analysis failed: {exc}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/trend
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/trend", methods=["POST"])
@require_json
def trend():
    """
    Run trend prediction for a topic.

    Request : { "query": "<topic>" }
    Response: { ok, data: { topic, sections, narrative } }
    """
    query, err = get_query()
    if err:
        return err

    try:
        from agents.trend_prediction_agent import TrendPredictionAgent

        store   = get_vector_store()
        context = store.search_text(query) if store else ""
        result  = TrendPredictionAgent().run(query, context=context)
        return success(result)

    except Exception as exc:                     # noqa: BLE001
        logger.exception("[API /trend] Failed")
        return error(f"Trend prediction failed: {exc}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/knowledge
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/knowledge", methods=["POST"])
@require_json
def knowledge():
    """
    Extract knowledge graph entities from a topic or text.

    Request : { "query": "<topic or text>" }
    Response: { ok, data: { input, graph, entity_counts, raw_response } }
    """
    query, err = get_query()
    if err:
        return err

    try:
        from agents.knowledge_graph_agent import KnowledgeGraphAgent

        store   = get_vector_store()
        context = store.search_text(query) if store else ""
        result  = KnowledgeGraphAgent().run(query, context=context)
        return success(result)

    except Exception as exc:                     # noqa: BLE001
        logger.exception("[API /knowledge] Failed")
        return error(f"Knowledge graph extraction failed: {exc}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/insight
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/insight", methods=["POST"])
@require_json
def insight():
    """
    Generate research insights for a topic.

    Request : { "query": "<topic>" }
    Response: { ok, data: { topic, sections, narrative } }
    """
    query, err = get_query()
    if err:
        return err

    try:
        from agents.insight_generation_agent import InsightGenerationAgent

        store   = get_vector_store()
        context = store.search_text(query) if store else ""
        result  = InsightGenerationAgent().run(query, context=context)
        return success(result)

    except Exception as exc:                     # noqa: BLE001
        logger.exception("[API /insight] Failed")
        return error(f"Insight generation failed: {exc}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/orchestrate  — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/orchestrate", methods=["POST"])
@require_json
def orchestrate():
    """
    Run the complete 5-agent research pipeline.

    Request : { "query": "<topic>" }
    Response: { ok, data: { query, literature, citation, trend,
                             knowledge_graph, insight, summary, timing, errors } }
    """
    query, err = get_query()
    if err:
        return err

    try:
        from agents.orchestrator import AgentOrchestrator

        store  = get_vector_store()
        orch   = AgentOrchestrator(vector_store=store)
        result = orch.run(query)
        return success(result)

    except Exception as exc:                     # noqa: BLE001
        logger.exception("[API /orchestrate] Failed")
        return error(f"Orchestration failed: {exc}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/dashboard  — stats & health
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/dashboard", methods=["GET"])
def dashboard_stats():
    """
    Return dashboard statistics and health check payload.

    Response: { ok, data: { status, vector_store, upload_folder, model_id } }
    """
    store        = get_vector_store()
    upload_dir   = current_app.config.get("UPLOAD_FOLDER", "uploads")
    uploaded_pdfs = []

    if os.path.isdir(upload_dir):
        uploaded_pdfs = [
            f for f in os.listdir(upload_dir)
            if f.lower().endswith(".pdf")
        ]

    return success({
        "status":         "ok",
        "model_id":       current_app.config.get("MODEL_ID", ""),
        "vector_store": {
            "total_vectors": store.total_vectors if store else 0,
            "sources":       store.sources       if store else [],
        },
        "uploaded_pdfs":  len(uploaded_pdfs),
        "upload_folder":  upload_dir,
    })
