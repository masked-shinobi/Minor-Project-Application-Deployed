"""
Metadata Database Module
SQLite-based storage for chunk metadata, paper information, and keywords.
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChunkMetadata:
    """Metadata record for a text chunk."""
    chunk_id: str
    paper_id: str
    paper_title: str
    section_heading: str
    content: str
    summary: str
    keywords: str           # Comma-separated keywords
    page_numbers: str       # Comma-separated page numbers
    char_start: int
    char_end: int


class MetadataDB:
    """
    SQLite database for storing and querying chunk metadata.
    Provides structured search to complement FAISS vector search.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the metadata database.

        Args:
            db_path: Path to the SQLite database file.
                    Defaults to rag_system/data/metadata.db
        """
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "metadata.db"
            )

        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create database tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    file_path TEXT,
                    total_pages INTEGER,
                    total_chunks INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}'
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    section_heading TEXT,
                    content TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    keywords TEXT DEFAULT '',
                    page_numbers TEXT DEFAULT '',
                    char_start INTEGER DEFAULT 0,
                    char_end INTEGER DEFAULT 0,
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_paper 
                ON chunks(paper_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_section 
                ON chunks(section_heading)
            """)

            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Create a database connection."""
        return sqlite3.connect(self.db_path)

    def add_paper(
        self,
        paper_id: str,
        title: str,
        file_path: str,
        total_pages: int,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add or update a paper record.

        Args:
            paper_id: Unique paper identifier.
            title: Paper title.
            file_path: Path to the source PDF.
            total_pages: Number of pages.
            metadata: Optional additional metadata dict.
        """
        import json
        meta_json = json.dumps(metadata or {})

        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO papers 
                (paper_id, title, file_path, total_pages, metadata_json)
                VALUES (?, ?, ?, ?, ?)
            """, (paper_id, title, file_path, total_pages, meta_json))
            conn.commit()

    def add_chunk(self, metadata: ChunkMetadata) -> None:
        """Add a single chunk metadata record."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, paper_id, section_heading, content, summary, 
                 keywords, page_numbers, char_start, char_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.chunk_id, metadata.paper_id,
                metadata.section_heading, metadata.content,
                metadata.summary, metadata.keywords,
                metadata.page_numbers, metadata.char_start, metadata.char_end
            ))
            conn.commit()

    def add_chunks_batch(self, metadata_list: List[ChunkMetadata]) -> None:
        """Add multiple chunk metadata records in a batch."""
        with self._connect() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, paper_id, section_heading, content, summary,
                 keywords, page_numbers, char_start, char_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (m.chunk_id, m.paper_id, m.section_heading, m.content,
                 m.summary, m.keywords, m.page_numbers, m.char_start, m.char_end)
                for m in metadata_list
            ])
            conn.commit()

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk by its ID.

        Returns:
            Dict with chunk data, or None if not found.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()

            if row:
                return dict(row)
        return None

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs.

        Returns:
            List of chunk dicts.
        """
        if not chunk_ids:
            return []

        placeholders = ",".join(["?"] * len(chunk_ids))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids
            ).fetchall()
            return [dict(r) for r in rows]

    def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search chunks by keywords (substring match in keywords field).

        Args:
            keywords: List of keywords to search for.
            limit: Maximum number of results.

        Returns:
            List of matching chunk dicts.
        """
        conditions = " OR ".join(["keywords LIKE ?"] * len(keywords))
        params = [f"%{kw}%" for kw in keywords]

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE {conditions} LIMIT ?",
                params + [limit]
            ).fetchall()
            return [dict(r) for r in rows]

    def get_paper_chunks(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chunks WHERE paper_id = ? ORDER BY char_start",
                (paper_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def list_papers(self) -> List[Dict[str, Any]]:
        """List all papers in the database."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM papers").fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        with self._connect() as conn:
            paper_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            return {
                "total_papers": paper_count,
                "total_chunks": chunk_count
            }
