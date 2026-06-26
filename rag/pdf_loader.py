"""
rag/pdf_loader.py
─────────────────
Extract plain text from uploaded PDF files using PyMuPDF (fitz).

Each page is extracted individually so page numbers can be tracked and
attached as metadata alongside every text chunk downstream.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageDocument:
    """Holds the raw text and metadata for a single PDF page."""
    page_number: int          # 1-based
    text: str                 # raw extracted text
    source: str               # original filename
    metadata: dict = field(default_factory=dict)


class PDFLoader:
    """
    Load and extract text from a PDF file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the PDF file.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file is not a PDF (checked by extension and fitz open).
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self._validate()

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> List[PageDocument]:
        """
        Open the PDF and return one :class:`PageDocument` per page.

        Empty pages (whitespace only) are skipped with a DEBUG log entry.

        Returns
        -------
        List[PageDocument] — one entry per non-empty page, in page order.
        """
        logger.info("[PDFLoader] Loading '%s'", self.filename)
        pages: List[PageDocument] = []

        with fitz.open(self.filepath) as doc:
            total = len(doc)
            logger.info("[PDFLoader] %d pages found in '%s'", total, self.filename)

            for idx, page in enumerate(doc):
                # extract_text uses the default "text" mode — preserves reading order
                text = page.get_text("text").strip()

                if not text:
                    logger.debug("[PDFLoader] Page %d is empty — skipped", idx + 1)
                    continue

                pages.append(
                    PageDocument(
                        page_number=idx + 1,
                        text=text,
                        source=self.filename,
                        metadata={
                            "total_pages": total,
                            "filepath": self.filepath,
                        },
                    )
                )

        logger.info(
            "[PDFLoader] Extracted %d non-empty pages from '%s'",
            len(pages),
            self.filename,
        )
        return pages

    def load_text(self) -> str:
        """Return all page text joined by newlines (convenience method)."""
        return "\n".join(p.text for p in self.load())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate(self) -> None:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"PDF not found: {self.filepath}")

        ext = os.path.splitext(self.filepath)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"PDFLoader only accepts: {self.SUPPORTED_EXTENSIONS}"
            )
