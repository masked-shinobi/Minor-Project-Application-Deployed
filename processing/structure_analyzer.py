"""
Structure Analyzer Module
Identifies the logical structure of a research paper (abstract, introduction,
methodology, results, conclusion, references) from extracted text.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# Common section headings found in academic papers (case-insensitive patterns)
STANDARD_SECTIONS = [
    "abstract",
    "introduction",
    "related work",
    "literature review",
    "background",
    "methodology",
    "methods",
    "materials and methods",
    "proposed method",
    "proposed approach",
    "system design",
    "architecture",
    "implementation",
    "experimental setup",
    "experiments",
    "results",
    "results and discussion",
    "discussion",
    "analysis",
    "evaluation",
    "conclusion",
    "conclusions",
    "future work",
    "acknowledgements",
    "acknowledgments",
    "references",
    "bibliography",
    "appendix",
]


@dataclass
class Section:
    """Represents a detected section of a research paper."""
    section_id: str                  # e.g., "sec_0", "sec_1"
    heading: str                     # Detected heading text
    normalized_heading: str          # Lowercased, stripped heading
    content: str                     # Full text of the section
    start_char: int                  # Character offset where section starts in full_text
    end_char: int                    # Character offset where section ends
    page_numbers: List[int] = field(default_factory=list)  # Pages this section spans


class StructureAnalyzer:
    """Analyzes the structure of a parsed research paper document."""

    def __init__(self, custom_sections: Optional[List[str]] = None):
        """
        Initialize the structure analyzer.

        Args:
            custom_sections: Optional additional section heading patterns.
        """
        self.known_sections = STANDARD_SECTIONS.copy()
        if custom_sections:
            self.known_sections.extend([s.lower() for s in custom_sections])

        # Build a regex pattern for detecting section headings
        # Matches lines that look like: "1. Introduction", "II. Methods", "Abstract", etc.
        self._heading_pattern = re.compile(
            r"^"
            r"(?:(?:\d+\.?\s*)|(?:[IVXivx]+\.?\s*))?"  # Optional numbering
            r"([A-Z][A-Za-z\s&,:-]{2,60})"               # Heading text
            r"\s*$",
            re.MULTILINE
        )

    def analyze(self, full_text: str) -> List[Section]:
        """
        Analyze the full text of a paper and identify its sections.

        Args:
            full_text: The complete text of the paper.

        Returns:
            List of Section objects representing the paper's structure.
        """
        # Find all potential heading positions
        heading_positions = self._find_headings(full_text)

        if not heading_positions:
            # If no headings detected, treat entire text as one section
            return [Section(
                section_id="sec_0",
                heading="Full Document",
                normalized_heading="full document",
                content=full_text,
                start_char=0,
                end_char=len(full_text)
            )]

        sections = []
        for i, (start, heading_text) in enumerate(heading_positions):
            # Determine section end
            if i + 1 < len(heading_positions):
                end = heading_positions[i + 1][0]
            else:
                end = len(full_text)

            # Section content starts after the heading line
            heading_end = start + len(heading_text)
            content = full_text[heading_end:end].strip()

            normalized = heading_text.strip().lower()
            normalized = re.sub(r"^[\d\.ivx]+\s*", "", normalized).strip()

            sections.append(Section(
                section_id=f"sec_{i}",
                heading=heading_text.strip(),
                normalized_heading=normalized,
                content=content,
                start_char=start,
                end_char=end
            ))

        # If there's text before the first heading, capture it as a preamble
        first_heading_start = heading_positions[0][0]
        if first_heading_start > 0:
            preamble_text = full_text[:first_heading_start].strip()
            if preamble_text:
                preamble = Section(
                    section_id="sec_preamble",
                    heading="Preamble",
                    normalized_heading="preamble",
                    content=preamble_text,
                    start_char=0,
                    end_char=first_heading_start
                )
                sections.insert(0, preamble)

        return sections

    def _find_headings(self, text: str) -> List[tuple]:
        """
        Find heading positions in the text.

        Returns:
            List of (char_offset, heading_text) tuples.
        """
        headings = []
        for match in self._heading_pattern.finditer(text):
            heading_text = match.group(0).strip()
            normalized = re.sub(r"^[\d\.ivx]+\s*", "", heading_text.lower()).strip()

            # Check if this looks like a known section heading
            if self._is_known_heading(normalized):
                headings.append((match.start(), heading_text))

        return headings

    def _is_known_heading(self, normalized_text: str) -> bool:
        """
        Check if the text matches a known section heading.

        Args:
            normalized_text: Lowercased, stripped heading text.

        Returns:
            True if it matches a known heading pattern.
        """
        for known in self.known_sections:
            if known in normalized_text or normalized_text in known:
                return True
        return False

    def get_section_by_name(self, sections: List[Section], name: str) -> Optional[Section]:
        """
        Find a section by its normalized heading name.

        Args:
            sections: List of analyzed sections.
            name: Section name to search for (case-insensitive).

        Returns:
            The matching Section, or None if not found.
        """
        name_lower = name.lower().strip()
        for section in sections:
            if name_lower in section.normalized_heading or section.normalized_heading in name_lower:
                return section
        return None
