"""
Document Parser Module
Extracts text content from PDF files page-by-page using PyPDF2.
Produces structured document representations for downstream processing.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF (fitz) is required. Install via: pip install pymupdf")


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int          # 1-indexed
    raw_text: str             # Full extracted text
    char_count: int = 0       # Number of characters on this page

    def __post_init__(self):
        self.char_count = len(self.raw_text)


@dataclass
class ParsedDocument:
    """Represents a fully parsed PDF document."""
    paper_id: str                            # Unique identifier (filename stem)
    file_path: str                           # Absolute path to the source PDF
    title: str = ""                          # Extracted or inferred title
    total_pages: int = 0                     # Total number of pages
    pages: List[PageContent] = field(default_factory=list)
    full_text: str = ""                      # Concatenated text from all pages
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.full_text.strip()) == 0


class DocumentParser:
    """Parses PDF files and extracts structured text content."""

    def __init__(self):
        """Initialize the document parser."""
        pass

    def parse(self, pdf_path: str, paper_id: Optional[str] = None) -> ParsedDocument:
        """
        Parse a single PDF file and extract all text content.

        Args:
            pdf_path: Absolute path to the PDF file.
            paper_id: Optional identifier for the paper. If None, uses filename.

        Returns:
            ParsedDocument with extracted text and metadata.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file cannot be parsed.
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if paper_id is None:
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]

        try:
            doc_handle = fitz.open(pdf_path)
        except Exception as e:
            raise ValueError(f"Failed to read PDF '{pdf_path}': {e}")

        pages: List[PageContent] = []
        all_text_parts: List[str] = []

        for i in range(len(doc_handle)):
            try:
                page = doc_handle.load_page(i)
                text = page.get_text() or ""
            except Exception:
                text = ""

            page_content = PageContent(
                page_number=i + 1,
                raw_text=text
            )
            pages.append(page_content)
            all_text_parts.append(text)

        full_text = "\n\n".join(all_text_parts)

        # Extract PDF metadata if available
        pdf_metadata = {}
        if doc_handle.metadata:
            pdf_metadata = doc_handle.metadata

        # Attempt to infer title from first page or metadata
        title = pdf_metadata.get("title", "")
        if not title and pages:
            title = self._infer_title(pages[0].raw_text)

        doc = ParsedDocument(
            paper_id=paper_id,
            file_path=pdf_path,
            title=title,
            total_pages=len(pages),
            pages=pages,
            full_text=full_text,
            metadata=pdf_metadata
        )

        doc_handle.close()
        return doc

    def parse_multiple(self, pdf_paths: List[str]) -> List[ParsedDocument]:
        """
        Parse multiple PDF files.

        Args:
            pdf_paths: List of paths to PDF files.

        Returns:
            List of ParsedDocument objects.
        """
        documents = []
        for path in pdf_paths:
            try:
                doc = self.parse(path)
                if doc.is_empty:
                    print(f"[DocumentParser] Warning: No text extracted from '{path}'")
                else:
                    print(f"[DocumentParser] Parsed '{doc.paper_id}': "
                          f"{doc.total_pages} pages, {len(doc.full_text)} chars")
                documents.append(doc)
            except Exception as e:
                print(f"[DocumentParser] Error parsing '{path}': {e}")

        return documents

    def _infer_title(self, first_page_text: str) -> str:
        """
        Attempt to infer the paper title from the first page text.
        Heuristic: the title is typically the first non-empty line.

        Args:
            first_page_text: Raw text from the first page.

        Returns:
            Inferred title string.
        """
        lines = first_page_text.strip().split("\n")
        for line in lines:
            cleaned = line.strip()
            # Skip very short lines (likely page numbers or headers)
            if len(cleaned) > 10:
                # Title is unlikely to be longer than 200 characters
                return cleaned[:200]
        return "Untitled"
