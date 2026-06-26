"""
rag/embedder.py
───────────────
Generate dense vector embeddings for text chunks.

Uses the ``ibm-watsonx-ai`` text-embeddings endpoint (ibm/slate models) when
IBM credentials are available; falls back to a lightweight sentence-transformers
model (``all-MiniLM-L6-v2``, 384-dim) for offline / no-credential use.

The returned vectors are always ``numpy.ndarray`` of shape ``(N, D)`` in
float32 — compatible with FAISS directly.
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# ── IBM embedding model ───────────────────────────────────────────────────────
IBM_EMBEDDING_MODEL = "ibm/slate-125m-english-rtrvr"
EMBEDDING_DIM_IBM   = 768

# ── Fallback local model ──────────────────────────────────────────────────────
LOCAL_MODEL_NAME  = "all-MiniLM-L6-v2"
EMBEDDING_DIM_LOCAL = 384


class Embedder:
    """
    Embed a list of text strings into a float32 numpy matrix.

    Automatically selects IBM watsonx embeddings when ``IBM_API_KEY`` and
    ``IBM_PROJECT_ID`` are set; otherwise falls back to the local
    sentence-transformers model.

    Parameters
    ----------
    force_local :
        Skip IBM and always use the local sentence-transformers model.
    batch_size :
        Number of texts to embed per API call (IBM) or inference pass (local).
    """

    def __init__(
        self,
        force_local: bool = False,
        batch_size:  int  = 32,
    ) -> None:
        self.batch_size = batch_size
        self._model     = None          # lazy-loaded

        # Decide backend
        has_ibm = bool(
            os.getenv("IBM_API_KEY") and os.getenv("IBM_PROJECT_ID")
        )
        self._use_ibm = has_ibm and not force_local

        if self._use_ibm:
            self.dim = EMBEDDING_DIM_IBM
            logger.info("[Embedder] Backend: IBM watsonx (%s)", IBM_EMBEDDING_MODEL)
        else:
            self.dim = EMBEDDING_DIM_LOCAL
            logger.info("[Embedder] Backend: local sentence-transformers (%s)", LOCAL_MODEL_NAME)

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed *texts* and return an ``(N, D)`` float32 matrix.

        Parameters
        ----------
        texts :
            List of strings to embed.  Empty strings are replaced with a
            single space to avoid API errors.

        Returns
        -------
        np.ndarray — shape ``(len(texts), self.dim)``, dtype float32.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        # Sanitise — replace blanks with a space
        clean = [t.strip() or " " for t in texts]

        if self._use_ibm:
            return self._embed_ibm(clean)
        return self._embed_local(clean)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string and return a 1-D float32 array of length D."""
        return self.embed([text])[0]

    # ── IBM backend ───────────────────────────────────────────────────────────

    def _embed_ibm(self, texts: List[str]) -> np.ndarray:
        """Call the IBM watsonx embeddings endpoint in batches."""
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import Embeddings

        if self._model is None:
            creds = Credentials(
                url=os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
                api_key=os.getenv("IBM_API_KEY"),
            )
            client = APIClient(
                credentials=creds,
                project_id=os.getenv("IBM_PROJECT_ID"),
            )
            self._model = Embeddings(
                model_id=IBM_EMBEDDING_MODEL,
                api_client=client,
            )

        all_vectors: List[np.ndarray] = []

        # Batch to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            response = self._model.embed_documents(batch)
            # response is a list of float lists
            all_vectors.extend(response)
            logger.debug(
                "[Embedder/IBM] Embedded batch %d-%d", i, i + len(batch)
            )

        matrix = np.array(all_vectors, dtype=np.float32)
        return matrix

    # ── Local fallback backend ────────────────────────────────────────────────

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Use sentence-transformers locally (no network required)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(LOCAL_MODEL_NAME)
                logger.info(
                    "[Embedder/local] Loaded SentenceTransformer '%s'",
                    LOCAL_MODEL_NAME,
                )
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install it with:  pip install sentence-transformers"
                ) from exc

        all_vectors: List[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            vecs  = self._model.encode(batch, show_progress_bar=False)
            all_vectors.append(vecs)
            logger.debug(
                "[Embedder/local] Encoded batch %d-%d", i, i + len(batch)
            )

        return np.vstack(all_vectors).astype(np.float32)
