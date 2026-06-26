"""
app.py — Research Agent application factory & entry point
──────────────────────────────────────────────────────────

Architecture
------------
Flask
 ├── main blueprint   (public pages: /, /dashboard)
 ├── api  blueprint   (REST JSON: /api/*)
 └── VectorStore      (attached to app.extensions for request sharing)

Logging
-------
Structured logging is configured at startup.  All modules use
``logging.getLogger(__name__)`` so the hierarchy is:
    research_agent
    ├── agents.*
    ├── rag.*
    └── api.*

Run
---
    python app.py
    # → http://127.0.0.1:5000
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, render_template
from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup (called before create_app so imports that log at module level
# already have a handler)
# ─────────────────────────────────────────────────────────────────────────────

def configure_logging(log_level: str = "INFO") -> None:
    """
    Set up root logger with:
    - coloured StreamHandler to stdout
    - RotatingFileHandler → logs/research_agent.log (max 5 MB × 3 backups)
    """
    os.makedirs("logs", exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        "logs/research_agent.log",
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "ibm_watsonx_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(config_class=Config) -> Flask:
    """
    Create and configure the Flask application.

    Parameters
    ----------
    config_class :
        Configuration class to use.  Defaults to :class:`config.Config`.

    Returns
    -------
    Flask — fully configured application instance.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    logger = logging.getLogger(__name__)
    logger.info("Creating Research Agent application…")

    # ── Ensure required directories exist ────────────────────────────────────
    for folder_key in ("UPLOAD_FOLDER", "VECTOR_DB_PATH", "REPORTS_FOLDER"):
        path = app.config.get(folder_key)
        if path:
            os.makedirs(path, exist_ok=True)
            logger.debug("Directory ensured: %s", path)

    # ── Initialise persistent VectorStore (lazy — loads from disk if present) ─
    _init_vector_store(app)

    # ── Register blueprints ───────────────────────────────────────────────────
    _register_blueprints(app)

    # ── Register global error handlers ───────────────────────────────────────
    _register_error_handlers(app)

    logger.info("Application ready — debug=%s", app.debug)
    return app


def _init_vector_store(app: Flask) -> None:
    """Attach a VectorStore to app.extensions so all requests share one index."""
    try:
        from rag.vector_store import VectorStore

        store = VectorStore(
            persist_dir=app.config.get("VECTOR_DB_PATH", "vector_db")
        )
        app.extensions["vector_store"] = store
        logging.getLogger(__name__).info(
            "VectorStore ready — %d vectors from %d source(s)",
            store.total_vectors,
            len(store.sources),
        )
    except Exception as exc:                    # noqa: BLE001
        # Non-fatal: app works without a pre-loaded index
        logging.getLogger(__name__).warning(
            "VectorStore could not be initialised: %s — uploads will create a fresh index",
            exc,
        )
        app.extensions["vector_store"] = None


def _register_blueprints(app: Flask) -> None:
    """Import and register all blueprints."""
    logger = logging.getLogger(__name__)

    # ── Main (public pages) ───────────────────────────────────────────────────
    from main.routes import main_bp
    app.register_blueprint(main_bp)
    logger.debug("Registered blueprint: main")

    # ── Dashboard page ────────────────────────────────────────────────────────
    from flask import Blueprint
    dashboard_bp = Blueprint("dashboard", __name__)

    @dashboard_bp.route("/dashboard")
    def dashboard():
        """Render the research dashboard SPA."""
        return render_template("dashboard.html")

    app.register_blueprint(dashboard_bp)
    logger.debug("Registered blueprint: dashboard")

    # ── REST API ──────────────────────────────────────────────────────────────
    from api.routes import api_bp
    app.register_blueprint(api_bp)
    logger.debug("Registered blueprint: api  (prefix=/api)")


def _register_error_handlers(app: Flask) -> None:
    """Register JSON-friendly error handlers for common HTTP errors."""
    logger = logging.getLogger(__name__)

    @app.errorhandler(400)
    def bad_request(exc):
        logger.warning("[HTTP 400] %s", exc)
        return jsonify({"ok": False, "error": "Bad request", "detail": str(exc)}), 400

    @app.errorhandler(404)
    def not_found(exc):
        logger.info("[HTTP 404] %s", exc)
        return jsonify({"ok": False, "error": "Resource not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return jsonify({"ok": False, "error": "Method not allowed"}), 405

    @app.errorhandler(413)
    def payload_too_large(exc):
        return jsonify({"ok": False, "error": "File too large. Maximum size is 16 MB."}), 413

    @app.errorhandler(500)
    def internal_error(exc):
        logger.exception("[HTTP 500] Unhandled exception")
        return jsonify({"ok": False, "error": "Internal server error"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Configure logging before creating the app so all startup messages appear
    configure_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))

    app = create_app()

    host  = os.getenv("FLASK_HOST",  "0.0.0.0")
    port  = int(os.getenv("FLASK_PORT", 5000))
    debug = app.config.get("DEBUG", True)

    logging.getLogger(__name__).info(
        "Starting server — http://%s:%s  (debug=%s)", host, port, debug
    )
    app.run(host=host, port=port, debug=debug)
