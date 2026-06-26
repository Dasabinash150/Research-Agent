"""
rag/vector_store.py
───────────────────
FAISS-backed vector store for storing and searching text chunk embeddings.

Responsibilities
----------------
* Add chunks + their embeddings to an in-memory FAISS index.
* Persist the index + metadata to ``vector_db/`` so it survives restarts.
* Retrieve the top-k most relevant :class:`~rag.chunker.TextChunk` objects
  for any query string.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import asdict
from typing import List, Optional, Tuple

import faiss
import numpy as np

from rag.chunker import TextChunk
from rag.embedder import Embedder

logger = logging.getLogger(__name__)

# ── File names written inside VECTOR_DB_PATH ─────────────────────────────────
INDEX_FILE    = "faiss.index"
METADATA_FILE = "metadata.pkl"


class VectorStore:
    """
    FAISS flat-L2 index with persistent metadata store.

    Parameters
    ----------
    persist_dir :
        Directory where the FAISS index and chunk metadata are saved.
        Defaults to the ``VECTOR_DB_PATH`` config value or ``./vector_db``.
    embedder :
        :class:`~rag.embedder.Embedder` instance used to vectorise query
        strings at search time.  If omitted, a default is created.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedder:    Optional[Embedder] = None,
    ) -> None:
        self.persist_dir = persist_dir or os.getenv("VECTOR_DB_PATH", "vector_db")
        os.makedirs(self.persist_dir, exist_ok=True)

        self._embedder: Embedder = embedder or Embedder()
        self._index:    Optional[faiss.Index] = None
        self._chunks:   List[TextChunk] = []   # parallel list — index i ↔ chunk i

        # Load persisted index if one exists
        if self._index_exists():
            self._load()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Embed *chunks* and add them to the FAISS index.

        Parameters
        ----------
        chunks :
            Output of :meth:`~rag.chunker.TextChunker.split`.
        """
        if not chunks:
            logger.warning("[VectorStore] add_chunks called with empty list — skipped")
            return

        texts = [c.text for c in chunks]
        logger.info("[VectorStore] Embedding %d chunks…", len(texts))
        vectors = self._embedder.embed(texts)   # (N, D) float32

        # Initialise FAISS index on first add
        if self._index is None:
            dim = vectors.shape[1]
            self._index = faiss.IndexFlatL2(dim)
            logger.info("[VectorStore] Created FAISS IndexFlatL2(dim=%d)", dim)

        self._index.add(vectors)
        self._chunks.extend(chunks)

        logger.info(
            "[VectorStore] Index now contains %d vectors", self._index.ntotal
        )
        self._save()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[TextChunk]:
        """
        Return the *top_k* most relevant chunks for *query*.

        Parameters
        ----------
        query :
            Raw user query string.
        top_k :
            Number of chunks to return.

        Returns
        -------
        List[TextChunk] — ranked by L2 distance (closest first).
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("[VectorStore] Index is empty — returning no results")
            return []

        q_vec = self._embedder.embed_one(query).reshape(1, -1)   # (1, D)
        k     = min(top_k, self._index.ntotal)

        distances, indices = self._index.search(q_vec, k)

        results: List[TextChunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:       # FAISS sentinel for "not enough results"
                continue
            chunk = self._chunks[idx]
            logger.debug(
                "[VectorStore] Hit chunk_id=%d  dist=%.4f  page=%d",
                chunk.chunk_id, dist, chunk.page_number,
            )
            results.append(chunk)

        logger.info(
            "[VectorStore] Query '%s…' → %d results", query[:50], len(results)
        )
        return results

    def search_text(self, query: str, top_k: int = 5) -> str:
        """
        Convenience wrapper — returns retrieved chunks joined as a single string.

        Useful for feeding directly into :meth:`BaseAgent.build_prompt`.
        """
        chunks = self.search(query, top_k=top_k)
        if not chunks:
            return ""
        return "\n\n---\n\n".join(
            f"[Source: {c.source}, Page {c.page_number}]\n{c.text}"
            for c in chunks
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal if self._index else 0

    @property
    def sources(self) -> List[str]:
        """Unique source filenames indexed so far."""
        return sorted({c.source for c in self._chunks})

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        idx_path  = os.path.join(self.persist_dir, INDEX_FILE)
        meta_path = os.path.join(self.persist_dir, METADATA_FILE)

        faiss.write_index(self._index, idx_path)

        with open(meta_path, "wb") as fh:
            pickle.dump(self._chunks, fh)

        logger.info(
            "[VectorStore] Saved index (%d vectors) → %s",
            self._index.ntotal, self.persist_dir,
        )

    def _load(self) -> None:
        """Load a persisted FAISS index from disk."""
        idx_path  = os.path.join(self.persist_dir, INDEX_FILE)
        meta_path = os.path.join(self.persist_dir, METADATA_FILE)

        self._index = faiss.read_index(idx_path)

        with open(meta_path, "rb") as fh:
            self._chunks = pickle.load(fh)

        logger.info(
            "[VectorStore] Loaded index (%d vectors, %d chunks) from %s",
            self._index.ntotal, len(self._chunks), self.persist_dir,
        )

    def _index_exists(self) -> bool:
        idx_path  = os.path.join(self.persist_dir, INDEX_FILE)
        meta_path = os.path.join(self.persist_dir, METADATA_FILE)
        return os.path.exists(idx_path) and os.path.exists(meta_path)

    def clear(self) -> None:
        """
        Wipe the in-memory index and delete persisted files.
        Use with care — irreversible.
        """
        self._index  = None
        self._chunks = []

        for fname in (INDEX_FILE, METADATA_FILE):
            path = os.path.join(self.persist_dir, fname)
            if os.path.exists(path):
                os.remove(path)

        logger.warning("[VectorStore] Index cleared and files deleted")
