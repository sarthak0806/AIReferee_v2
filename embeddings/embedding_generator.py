"""Incremental chunk embedding into FAISS.

Only chunks without a vector for the given model are embedded. If the index on
disk disagrees with SQLite (crash, deleted file, model/dimension change), the
index for that model is rebuilt from the chunks in SQLite.
"""
from __future__ import annotations

import time

from config import CONFIG
from database.db_manager import DatabaseManager
from embeddings.embedding_model import get_embedding_model
from retrieval.faiss_manager import FaissManager
from utils.logging_config import get_logger

log = get_logger("embedding_generator")


def pending_chunks(db: DatabaseManager, model_name: str, conferences: list[str] | None = None) -> list[dict]:
    sql = ("SELECT c.chunk_id, c.chunk_text FROM chunks c JOIN papers p ON p.paper_id=c.paper_id "
           "LEFT JOIN vector_map v ON v.chunk_id=c.chunk_id AND v.embedding_model=? "
           "WHERE v.chunk_id IS NULL AND p.corpus='reference'")
    args: list = [model_name]
    if conferences:
        sql += f" AND c.conference IN ({','.join('?' * len(conferences))})"
        args += conferences
    sql += " ORDER BY c.paper_id, c.chunk_index"
    return [dict(r) for r in db.conn.execute(sql, args)]


def open_index(db: DatabaseManager, model_name: str | None = None) -> tuple[FaissManager, object]:
    model = get_embedding_model(model_name)
    fm = FaissManager(model.dim, model.name, db)
    try:
        fm.load_or_create()
    except ValueError as exc:
        log.warning("%s; rebuilding", exc)
        fm.reset()
    return fm, model


def embed_pending(db: DatabaseManager, model_name: str | None = None, rebuild: bool = False) -> dict:
    fm, model = open_index(db, model_name)
    check = fm.validate()
    if not check["ok"] and not rebuild:
        removed = fm.remove_stale()          # cheap repair: chunks were replaced
        if removed:
            log.info("Removed %d stale vectors for re-chunked papers", removed)
            fm.save()
        check = fm.validate()
    if rebuild or not check["ok"]:
        if not check["ok"]:
            log.warning("Index inconsistent (%s); rebuilding", "; ".join(check["problems"]))
        fm.reset()
    todo = pending_chunks(db, model.name)
    log.info("[%s] %d chunks to embed (index has %d)", model.name, len(todo), fm.ntotal)
    t0 = time.perf_counter()
    batch = CONFIG["embedding_batch_size"] * 8   # persist progress every few batches
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        vecs = model.encode_documents([c["chunk_text"] for c in part])
        fm.add(vecs, [c["chunk_id"] for c in part])
        fm.save()
        log.info("[%s] embedded %d/%d", model.name, min(i + batch, len(todo)), len(todo))
    elapsed = time.perf_counter() - t0
    result = fm.validate()
    result.update({"model": model.version, "new_vectors": len(todo), "seconds": round(elapsed, 1),
                   "index_file": str(fm.index_file)})
    return result
