import sqlite3
import os

db_path = "data/metadata.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("--- Inspecting Middle Paper Content (TEST_paper) ---")
# 12 is roughly the middle of 28 chunks
cursor.execute("SELECT section_heading, content FROM chunks WHERE paper_id = 'TEST_paper' ORDER BY chunk_id LIMIT 3 OFFSET 12")
chunks = cursor.fetchall()

for i, c in enumerate(chunks):
    print(f"\nCHUNK {i + 13} | SECTION: {c['section_heading']}")
    print("-" * 50)
    # Print the first 300 chars
    print(c['content'][:300] + "...")
    print("-" * 50)

conn.close()
