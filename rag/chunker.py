"""
rag/chunker.py
──────────────
Split :class:`~rag.pdf_loader.PageDocument` text into overlapping chunks
suitable for embedding and vector search.

Strategy: fixed-size character window with configurable overlap.
Each output chunk carries full provenance metadata (source file,
page number, chunk index within the document).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from rag.pdf_loader import PageDocument

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 1000   # characters per chunk
DEFAULT_CHUNK_OVERLAP = 200    # characters shared between consecutive chunks


@dataclass
class TextChunk:
    """A single text window ready to be embedded."""
    chunk_id:    int           # global index across the whole document
    text:        str           # chunk content
    source:      str           # originating filename
    page_number: int           # page where the chunk starts
    start_char:  int           # character offset within that page's text
    metadata:    dict = field(default_factory=dict)


class TextChunker:
    """
    Split a list of :class:`PageDocument` objects into :class:`TextChunk` objects.

    Parameters
    ----------
    chunk_size :
        Maximum number of characters per chunk.
    chunk_overlap :
        Number of characters to repeat at the start of the next chunk.
        Must be less than *chunk_size*.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def split(self, pages: List[PageDocument]) -> List[TextChunk]:
        """
        Chunk every page and return a flat, ordered list of :class:`TextChunk`.

        Parameters
        ----------
        pages :
            Output of :meth:`~rag.pdf_loader.PDFLoader.load`.

        Returns
        -------
        List[TextChunk] — chunks in document order with global chunk IDs.
        """
        all_chunks: List[TextChunk] = []
        global_id = 0

        for page in pages:
            page_chunks = self._split_text(
                text=page.text,
                source=page.source,
                page_number=page.page_number,
                start_global_id=global_id,
                page_metadata=page.metadata,
            )
            all_chunks.extend(page_chunks)
            global_id += len(page_chunks)

        logger.info(
            "[TextChunker] %d pages → %d chunks "
            "(size=%d, overlap=%d)",
            len(pages), len(all_chunks),
            self.chunk_size, self.chunk_overlap,
        )
        return all_chunks

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _split_text(
        self,
        text: str,
        source: str,
        page_number: int,
        start_global_id: int,
        page_metadata: dict,
    ) -> List[TextChunk]:
        """Slide a window over *text* and produce chunks."""
        chunks: List[TextChunk] = []
        step   = self.chunk_size - self.chunk_overlap
        pos    = 0
        local_id = 0

        while pos < len(text):
            window = text[pos: pos + self.chunk_size]
            if window.strip():          # skip whitespace-only windows
                chunks.append(
                    TextChunk(
                        chunk_id=start_global_id + local_id,
                        text=window,
                        source=source,
                        page_number=page_number,
                        start_char=pos,
                        metadata={**page_metadata, "chunk_local_id": local_id},
                    )
                )
                local_id += 1
            pos += step

        return chunks
