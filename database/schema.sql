PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers (
    paper_id            TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    abstract            TEXT,
    authors             TEXT,               -- JSON list
    publication_year    INTEGER,
    conference          TEXT NOT NULL,
    doi                 TEXT,
    source_url          TEXT NOT NULL UNIQUE,
    pdf_url             TEXT,
    pdf_path            TEXT,
    source_name         TEXT NOT NULL,      -- cvf_open_access | neurips_proceedings | acl_anthology | arxiv
    venue_verification  TEXT,               -- how conference membership was verified
    file_sha256         TEXT,
    page_count          INTEGER,
    full_text_available INTEGER NOT NULL DEFAULT 0,
    download_status     TEXT NOT NULL DEFAULT 'pending',   -- pending|downloaded|valid|invalid|failed|metadata_only
    extraction_status   TEXT NOT NULL DEFAULT 'pending',   -- pending|extracted|failed|needs_ocr
    status_detail       TEXT,
    corpus              TEXT NOT NULL DEFAULT 'reference',  -- reference | evaluation | uploaded
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_papers_conf_status ON papers(conference, download_status);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_sha ON papers(file_sha256) WHERE file_sha256 IS NOT NULL;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,      -- stable: <paper_id>::<seq>
    paper_id         TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    conference       TEXT,
    section_name     TEXT,
    page_number      INTEGER,
    chunk_index      INTEGER NOT NULL,
    chunk_text       TEXT NOT NULL,
    token_count      INTEGER,
    embedding_model  TEXT,
    embedding_status TEXT NOT NULL DEFAULT 'pending',      -- pending|embedded
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (paper_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embed ON chunks(embedding_status, embedding_model);

-- Explicit FAISS vector-id -> SQLite chunk-id mapping.
CREATE TABLE IF NOT EXISTS vector_map (
    vector_id       INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    chunk_id        TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (embedding_model, vector_id),
    UNIQUE (embedding_model, chunk_id)
);

CREATE TABLE IF NOT EXISTS queries (
    query_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_paper_id TEXT NOT NULL,
    generated_query   TEXT NOT NULL,
    query_category    TEXT,
    query_reason      TEXT,
    search_source     TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_queries_paper ON queries(uploaded_paper_id);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    retrieval_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_paper_id TEXT NOT NULL,
    query_id          INTEGER REFERENCES queries(query_id) ON DELETE CASCADE,
    chunk_id          TEXT REFERENCES chunks(chunk_id),
    similarity_score  REAL,
    reranker_score    REAL,
    initial_rank      INTEGER,
    final_rank        INTEGER,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_retrieval_paper ON retrieval_logs(uploaded_paper_id);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_paper_id      TEXT NOT NULL,
    novelty_score          REAL,
    methodology_score      REAL,
    experimental_score     REAL,
    literature_score       REAL,
    results_score          REAL,
    reproducibility_score  REAL,
    publishability_decision TEXT NOT NULL,  -- PUBLISHABLE | NON-PUBLISHABLE | HUMAN_REVIEW
    decision_rationale     TEXT,
    confidence             REAL,
    created_at             TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_eval_paper ON evaluations(uploaded_paper_id);

CREATE TABLE IF NOT EXISTS conferences (
    conference_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    conference_name    TEXT NOT NULL,
    conference_acronym TEXT NOT NULL UNIQUE,
    description        TEXT,
    topics             TEXT,
    reference_profile  TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conference_recommendations (
    recommendation_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id        INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
    conference_id        INTEGER NOT NULL REFERENCES conferences(conference_id),
    recommendation_score REAL,
    rationale            TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Key/value settings (e.g. the chunking signature used for the stored chunks).
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Cache of external search responses (saves SerpAPI quota and arXiv calls).
CREATE TABLE IF NOT EXISTS external_search_cache (
    source     TEXT NOT NULL,
    query      TEXT NOT NULL,
    response   TEXT NOT NULL,           -- JSON list of normalised results
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (source, query)
);
