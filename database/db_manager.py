"""SQLite persistence layer for papers, chunks, vector mappings and evaluations."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from config import path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

PAPER_COLUMNS = {
    "paper_id", "title", "abstract", "authors", "publication_year", "conference", "doi",
    "source_url", "pdf_url", "pdf_path", "source_name", "venue_verification", "file_sha256",
    "page_count", "full_text_available", "download_status", "extraction_status",
    "status_detail", "corpus",
}


class DatabaseManager:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else path("database_path")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ papers
    @staticmethod
    def _clean(data: dict[str, Any]) -> dict[str, Any]:
        unknown = set(data) - PAPER_COLUMNS
        if unknown:
            raise ValueError(f"Unknown paper columns: {sorted(unknown)}")
        out = dict(data)
        if isinstance(out.get("authors"), (list, tuple)):
            out["authors"] = json.dumps(list(out["authors"]), ensure_ascii=False)
        return out

    def insert_paper(self, paper: dict[str, Any]) -> bool:
        """Insert a paper; returns False if it already exists (by id or source URL)."""
        data = self._clean(paper)
        cols = ", ".join(data)
        marks = ", ".join("?" for _ in data)
        with self.transaction() as c:
            cur = c.execute(f"INSERT OR IGNORE INTO papers ({cols}) VALUES ({marks})", list(data.values()))
            return cur.rowcount == 1

    def update_paper(self, paper_id: str, **fields: Any) -> None:
        if not fields:
            return
        data = self._clean(fields)
        sets = ", ".join(f"{k} = ?" for k in data)
        with self.transaction() as c:
            c.execute(f"UPDATE papers SET {sets} WHERE paper_id = ?", [*data.values(), paper_id])

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return self._paper_row(row)

    def get_paper_by_source(self, source_url: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM papers WHERE source_url = ?", (source_url,)).fetchone()
        return self._paper_row(row)

    def find_by_sha(self, sha256: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM papers WHERE file_sha256 = ?", (sha256,)).fetchone()
        return self._paper_row(row)

    @staticmethod
    def _paper_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        if d.get("authors"):
            try:
                d["authors"] = json.loads(d["authors"])
            except json.JSONDecodeError:
                pass
        return d

    def list_papers(self, conference: str | None = None, download_status: str | None = None,
                    corpus: str = "reference") -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM papers WHERE corpus = ?", [corpus]
        if conference:
            sql += " AND conference = ?"
            args.append(conference)
        if download_status:
            sql += " AND download_status = ?"
            args.append(download_status)
        sql += " ORDER BY created_at, paper_id"
        return [self._paper_row(r) for r in self.conn.execute(sql, args)]

    def search_metadata(self, text: str, conference: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        like = f"%{text}%"
        sql = "SELECT * FROM papers WHERE (title LIKE ? OR abstract LIKE ? OR authors LIKE ?)"
        args: list[Any] = [like, like, like]
        if conference:
            sql += " AND conference = ?"
            args.append(conference)
        sql += " LIMIT ?"
        args.append(limit)
        return [self._paper_row(r) for r in self.conn.execute(sql, args)]

    def count_valid(self, conference: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE corpus='reference' AND conference=? AND download_status='valid'",
            (conference,),
        ).fetchone()[0]

    def status_counts(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT conference, download_status, COUNT(*) AS n FROM papers
               WHERE corpus='reference' GROUP BY conference, download_status"""
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ chunks
    def insert_chunks(self, chunks: Iterable[dict[str, Any]]) -> int:
        cols = ("chunk_id", "paper_id", "conference", "section_name", "page_number",
                "chunk_index", "chunk_text", "token_count")
        rows = [tuple(ch.get(k) for k in cols) for ch in chunks]
        with self.transaction() as c:
            before = c.total_changes
            c.executemany(
                f"INSERT OR IGNORE INTO chunks ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", rows
            )
            return c.total_changes - before

    def get_chunks(self, paper_id: str | None = None, embedding_status: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM chunks WHERE 1=1", []
        if paper_id:
            sql += " AND paper_id = ?"
            args.append(paper_id)
        if embedding_status:
            sql += " AND embedding_status = ?"
            args.append(embedding_status)
        sql += " ORDER BY paper_id, chunk_index"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"""SELECT c.*, p.title, p.authors, p.publication_year, p.doi, p.source_url
                FROM chunks c JOIN papers p ON p.paper_id = c.paper_id
                WHERE c.chunk_id IN ({marks})""", chunk_ids)
        by_id = {r["chunk_id"]: dict(r) for r in rows}
        return [by_id[i] for i in chunk_ids if i in by_id]

    # -------------------------------------------------------------- vector map
    def add_vector_mappings(self, model: str, pairs: list[tuple[int, str]]) -> None:
        with self.transaction() as c:
            c.executemany("INSERT INTO vector_map (vector_id, embedding_model, chunk_id) VALUES (?, ?, ?)",
                          [(vid, model, cid) for vid, cid in pairs])
            c.executemany("UPDATE chunks SET embedding_status='embedded', embedding_model=? WHERE chunk_id=?",
                          [(model, cid) for _, cid in pairs])

    def chunk_ids_for_vectors(self, model: str, vector_ids: list[int]) -> dict[int, str]:
        if not vector_ids:
            return {}
        marks = ",".join("?" * len(vector_ids))
        rows = self.conn.execute(
            f"SELECT vector_id, chunk_id FROM vector_map WHERE embedding_model=? AND vector_id IN ({marks})",
            [model, *vector_ids])
        return {r["vector_id"]: r["chunk_id"] for r in rows}

    # ------------------------------------------------- queries / logs / evals
    def insert_query(self, uploaded_paper_id: str, query: str, category: str | None,
                     reason: str | None, source: str | None) -> int:
        with self.transaction() as c:
            cur = c.execute(
                "INSERT INTO queries (uploaded_paper_id, generated_query, query_category, query_reason, search_source)"
                " VALUES (?, ?, ?, ?, ?)", (uploaded_paper_id, query, category, reason, source))
            return cur.lastrowid

    def insert_retrieval_logs(self, rows: list[dict[str, Any]]) -> None:
        cols = ("uploaded_paper_id", "query_id", "chunk_id", "similarity_score",
                "reranker_score", "initial_rank", "final_rank")
        with self.transaction() as c:
            c.executemany(f"INSERT INTO retrieval_logs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                          [tuple(r.get(k) for k in cols) for r in rows])

    def get_retrieval_logs(self, uploaded_paper_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM retrieval_logs WHERE uploaded_paper_id=? ORDER BY query_id, final_rank",
            (uploaded_paper_id,))
        return [dict(r) for r in rows]

    def insert_evaluation(self, ev: dict[str, Any]) -> int:
        cols = ("uploaded_paper_id", "novelty_score", "methodology_score", "experimental_score",
                "literature_score", "results_score", "reproducibility_score",
                "publishability_decision", "decision_rationale", "confidence")
        with self.transaction() as c:
            cur = c.execute(f"INSERT INTO evaluations ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                            tuple(ev.get(k) for k in cols))
            return cur.lastrowid

    def upsert_conference(self, acronym: str, name: str, description: str | None = None,
                          topics: str | None = None, profile: str | None = None) -> int:
        with self.transaction() as c:
            c.execute(
                """INSERT INTO conferences (conference_acronym, conference_name, description, topics, reference_profile)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(conference_acronym) DO UPDATE SET
                     conference_name=excluded.conference_name,
                     description=COALESCE(excluded.description, description),
                     topics=COALESCE(excluded.topics, topics),
                     reference_profile=COALESCE(excluded.reference_profile, reference_profile)""",
                (acronym, name, description, topics, profile))
            return c.execute("SELECT conference_id FROM conferences WHERE conference_acronym=?",
                             (acronym,)).fetchone()[0]

    def store_recommendation(self, evaluation_id: int, conference_id: int, score: float, rationale: str) -> int:
        with self.transaction() as c:
            cur = c.execute(
                "INSERT INTO conference_recommendations (evaluation_id, conference_id, recommendation_score, rationale)"
                " VALUES (?, ?, ?, ?)", (evaluation_id, conference_id, score, rationale))
            return cur.lastrowid

    # --------------------------------------------------------------- settings
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.transaction() as c:
            c.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, value))

    def delete_chunks(self, paper_id: str) -> None:
        with self.transaction() as c:
            c.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))

    # ---------------------------------------------------------- search cache
    def get_cached_search(self, source: str, query: str) -> list[dict] | None:
        row = self.conn.execute("SELECT response FROM external_search_cache WHERE source=? AND query=?",
                                (source, query)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_search(self, source: str, query: str, results: list[dict]) -> None:
        with self.transaction() as c:
            c.execute("INSERT OR REPLACE INTO external_search_cache(source, query, response) VALUES (?, ?, ?)",
                      (source, query, json.dumps(results, ensure_ascii=False)))
