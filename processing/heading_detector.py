"""
Heading Detector Module
Identifies headings and sub-headings in extracted text using heuristic rules.
"""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class DetectedHeading:
    """Represents a detected heading in the text."""
    text: str                   # The heading text
    level: int                  # Heading level (1 = main, 2 = sub, 3 = subsub)
    line_number: int            # Line index in the source text
    char_offset: int            # Character offset from start of text


class HeadingDetector:
    """
    Detects headings and sub-headings in academic paper text using heuristics:
    - Lines that are ALL CAPS or Title Case
    - Lines preceded by numbering (1., 1.1., etc.)
    - Short lines followed by longer paragraph text
    """

    # Pattern for numbered headings: "1.", "1.1", "1.1.1", "A.", "I.", etc.
    NUMBERED_HEADING = re.compile(
        r"^(\d+(?:\.\d+)*\.?\s+|[A-Z]\.?\s+|[IVXivx]+\.?\s+)"
        r"([A-Z][A-Za-z\s&,:-]{2,80})\s*$"
    )

    # Pattern for ALL CAPS lines (likely headings)
    ALL_CAPS = re.compile(r"^[A-Z\s\d:&,-]{5,80}$")

    def detect(self, text: str) -> List[DetectedHeading]:
        """
        Detect headings in the given text.

        Args:
            text: Full text to analyze.

        Returns:
            List of DetectedHeading objects.
        """
        lines = text.split("\n")
        headings = []
        char_offset = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                char_offset += len(line) + 1
                continue

            heading = self._classify_heading(stripped, i, char_offset, lines)
            if heading:
                headings.append(heading)

            char_offset += len(line) + 1

        return headings

    def _classify_heading(
        self, line: str, line_idx: int, char_offset: int, all_lines: List[str]
    ) -> DetectedHeading | None:
        """
        Classify a line as a heading and determine its level.

        Returns:
            DetectedHeading if the line is a heading, None otherwise.
        """
        # Check for numbered headings (e.g., "1. Introduction", "3.2 Results")
        numbered_match = self.NUMBERED_HEADING.match(line)
        if numbered_match:
            numbering = numbered_match.group(1).strip().rstrip(".")
            # Determine level from numbering depth
            if "." in numbering:
                level = min(numbering.count(".") + 1, 3)
            else:
                level = 1
            return DetectedHeading(
                text=line, level=level,
                line_number=line_idx, char_offset=char_offset
            )

        # Check for ALL CAPS lines (level 1)
        if self.ALL_CAPS.match(line) and len(line) > 4:
            return DetectedHeading(
                text=line, level=1,
                line_number=line_idx, char_offset=char_offset
            )

        # Check for short Title Case lines that look like headings
        if (
            len(line) < 80
            and line == line.title()
            and not line.endswith(".")
            and self._is_followed_by_paragraph(line_idx, all_lines)
        ):
            return DetectedHeading(
                text=line, level=2,
                line_number=line_idx, char_offset=char_offset
            )

        return None

    def _is_followed_by_paragraph(self, line_idx: int, lines: List[str]) -> bool:
        """Check if the line is followed by a longer paragraph (heuristic for heading)."""
        for j in range(line_idx + 1, min(line_idx + 3, len(lines))):
            next_line = lines[j].strip()
            if len(next_line) > 100:
                return True
        return False
