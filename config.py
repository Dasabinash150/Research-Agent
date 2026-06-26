import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # ── Flask ──────────────────────────────────────────────────────────────────
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    # ── File uploads ───────────────────────────────────────────────────────────
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    # ── Vector database ────────────────────────────────────────────────────────
    VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

    # ── Reports ────────────────────────────────────────────────────────────────
    REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")

    # ── IBM watsonx.ai ─────────────────────────────────────────────────────────
    IBM_API_KEY    = os.getenv("IBM_API_KEY", "")
    IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "")
    IBM_URL        = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")
    MODEL_ID       = os.getenv("MODEL_ID", "ibm/granite-4-h-small")

    # ── Other LLM / API keys (optional) ───────────────────────────────────────
    OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
