"""
Table Parser Module
Detects and preserves tabular data from extracted PDF text.
"""

import re
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class DetectedTable:
    """Represents a detected table in the text."""
    table_id: str
    raw_text: str                           # Original text representation
    rows: List[List[str]] = field(default_factory=list)  # Parsed rows
    caption: str = ""                       # Table caption if found
    section_heading: str = ""               # Section where the table was found
    page_number: Optional[int] = None

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        if self.rows:
            return max(len(r) for r in self.rows)
        return 0


class TableParser:
    """
    Detects and parses tables from extracted PDF text.

    Note: PDF text extraction often loses table formatting. This parser uses
    heuristics to detect tabular patterns:
    - Lines with consistent column separators (multiple spaces, tabs)
    - Lines with consistent numeric patterns in columns
    - "Table N:" captions
    """

    # Pattern for table captions
    TABLE_CAPTION = re.compile(
        r"(?:Table|TABLE)\s+(\d+)[.:]\s*(.*)",
        re.IGNORECASE
    )

    def detect_tables(self, text: str, paper_id: str = "") -> List[DetectedTable]:
        """
        Detect tables in the text.

        Args:
            text: Full text to scan for tables.
            paper_id: Paper identifier for table IDs.

        Returns:
            List of DetectedTable objects.
        """
        tables = []
        lines = text.split("\n")
        i = 0
        table_count = 0

        while i < len(lines):
            # Check for table caption
            caption_match = self.TABLE_CAPTION.match(lines[i].strip())
            if caption_match:
                caption = caption_match.group(0)
                # Look for tabular data following the caption
                table_lines, end_idx = self._extract_table_lines(lines, i + 1)
                if table_lines:
                    rows = self._parse_rows(table_lines)
                    table = DetectedTable(
                        table_id=f"{paper_id}_table_{table_count}",
                        raw_text="\n".join(table_lines),
                        rows=rows,
                        caption=caption
                    )
                    tables.append(table)
                    table_count += 1
                    i = end_idx
                    continue

            # Check for tabular patterns even without caption
            if self._looks_tabular(lines[i]) and i + 1 < len(lines):
                table_lines, end_idx = self._extract_table_lines(lines, i)
                if len(table_lines) >= 2:  # At least 2 rows to be a table
                    rows = self._parse_rows(table_lines)
                    if rows and len(rows[0]) >= 2:  # At least 2 columns
                        table = DetectedTable(
                            table_id=f"{paper_id}_table_{table_count}",
                            raw_text="\n".join(table_lines),
                            rows=rows
                        )
                        tables.append(table)
                        table_count += 1
                        i = end_idx
                        continue

            i += 1

        return tables

    def _looks_tabular(self, line: str) -> bool:
        """Check if a line looks like it could be part of a table."""
        stripped = line.strip()
        if not stripped:
            return False

        # Multiple segments separated by 2+ spaces
        segments = re.split(r"\s{2,}", stripped)
        if len(segments) >= 3:
            return True

        # Tab-separated
        if "\t" in stripped and stripped.count("\t") >= 2:
            return True

        return False

    def _extract_table_lines(self, lines: List[str], start: int) -> tuple:
        """
        Extract consecutive lines that look tabular starting from a position.

        Returns:
            (table_lines, end_index)
        """
        table_lines = []
        i = start

        while i < len(lines):
            stripped = lines[i].strip()

            # Empty line may end the table
            if not stripped:
                if table_lines:
                    break
                i += 1
                continue

            # Check if this line is tabular
            if self._looks_tabular(lines[i]) or self._is_separator_line(stripped):
                table_lines.append(stripped)
            elif table_lines:
                # Non-tabular line after table lines — table ends
                break

            i += 1

        return table_lines, i

    def _is_separator_line(self, line: str) -> bool:
        """Check if a line is a table separator (dashes, equals, etc.)."""
        cleaned = line.replace(" ", "").replace("+", "").replace("|", "")
        return len(cleaned) > 3 and all(c in "-=_" for c in cleaned)

    def _parse_rows(self, table_lines: List[str]) -> List[List[str]]:
        """
        Parse table lines into rows of cells.

        Uses multi-space splitting as the primary method.
        """
        rows = []
        for line in table_lines:
            if self._is_separator_line(line):
                continue
            # Split by 2+ spaces or tabs
            cells = re.split(r"\s{2,}|\t+", line.strip())
            cells = [c.strip() for c in cells if c.strip()]
            if cells:
                rows.append(cells)
        return rows

    def table_to_text(self, table: DetectedTable) -> str:
        """
        Convert a DetectedTable to a readable text representation
        suitable for embedding.

        Args:
            table: The detected table.

        Returns:
            A text representation of the table.
        """
        parts = []
        if table.caption:
            parts.append(table.caption)

        if table.rows:
            for row in table.rows:
                parts.append(" | ".join(row))
        else:
            parts.append(table.raw_text)

        return "\n".join(parts)
