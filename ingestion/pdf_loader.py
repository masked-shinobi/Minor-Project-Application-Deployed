"""
PDF Loader Module
Discovers and loads PDF files from the data/papers directory.
"""

import os
import glob
from typing import List, Optional
from pathlib import Path


class PDFLoader:
    """Loads PDF file paths from a specified directory."""

    def __init__(self, papers_dir: Optional[str] = None):
        """
        Initialize the PDF loader.

        Args:
            papers_dir: Path to the directory containing PDF files.
                       Defaults to rag_system/data/papers/ relative to project root.
        """
        if papers_dir is None:
            # Default: rag_system/data/papers relative to this file's location
            self.papers_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "papers"
            )
        else:
            self.papers_dir = os.path.abspath(papers_dir)

    def discover_pdfs(self) -> List[str]:
        """
        Discover all PDF files in the papers directory.

        Returns:
            List of absolute paths to PDF files.

        Raises:
            FileNotFoundError: If the papers directory does not exist.
        """
        if not os.path.isdir(self.papers_dir):
            raise FileNotFoundError(
                f"Papers directory not found: {self.papers_dir}"
            )

        pdf_paths = sorted(glob.glob(os.path.join(self.papers_dir, "*.pdf")))

        if not pdf_paths:
            print(f"[PDFLoader] Warning: No PDF files found in {self.papers_dir}")

        return pdf_paths

    def get_paper_name(self, pdf_path: str) -> str:
        """
        Extract a clean paper name from the file path.

        Args:
            pdf_path: Full path to the PDF file.

        Returns:
            The filename without extension, used as paper identifier.
        """
        return Path(pdf_path).stem

    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Basic validation that a file exists and has .pdf extension.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            True if the file exists and has .pdf extension.
        """
        return os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf")
