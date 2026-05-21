"""
Blind-Sight RAG — Database Inspection Utility
==============================================
Quick diagnostic tool to inspect the SQLite metadata database and FAISS index.

Usage:
    python check_db.py

Output:
    - Database schema (tables and columns)
    - Row count in hazards table
    - Sample entries from the database
"""

import sqlite3

def check_database() -> None:
    """Inspect SQLite database schema and contents."""
    conn = sqlite3.connect('metadata.db')
    cur = conn.cursor()
    
    # Print schema
    print("=" * 60)
    print("DATABASE SCHEMA")
    print("=" * 60)
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    for table_sql in tables:
        print(table_sql[0])
    
    print("\n" + "=" * 60)
    print("TABLE CONTENTS")
    print("=" * 60)
    
    # Print row count
    cur.execute("SELECT COUNT(*) FROM hazards")
    count = cur.fetchone()[0]
    print(f"Total hazard entries: {count}")
    
    # Print sample entries
    print(f"\nFirst 2 entries:")
    cur.execute("SELECT * FROM hazards LIMIT 2")
    rows = cur.fetchall()
    for row in rows:
        print(f"  Row: {row}")
    
    conn.close()
    print("=" * 60)

if __name__ == "__main__":
    check_database()