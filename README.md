# AIReferee

AIReferee is an AI-assisted research-paper evaluation system built for the Kharagpur Data Science Hackathon. It does two jobs:

1. **Publishability.** It classifies a paper as Publishable or Non-Publishable. An internal HUMAN_REVIEW state is used when the evidence is too weak for a confident decision.
2. **Conference selection.** For a publishable paper, it picks one of CVPR, NeurIPS, EMNLP, TMLR or KDD and writes a rationale of at most 100 words. DAA is kept in the output schema but is otherwise skipped (see Known limitations).

## Workflow

```
OFFLINE (run once, then incrementally)
  conference sources -> verified PDFs -> extraction (PyMuPDF, pypdf fallback) -> section-aware chunking
  -> SQLite (papers, chunks) -> MiniLM embeddings -> FAISS IndexFlatIP + vector_map (SQLite)
  -> conference profiles (centroids, TF-IDF keywords, LLM summary)

ONLINE (per uploaded paper)
  PDF -> structured profile (Groq, Pydantic) -> 5-8 queries (Groq)
  -> FAISS top-30 per query (bi-encoder) + arXiv API + Google Scholar (SerpAPI)
  -> merge and deduplicate -> Cross-Encoder rerank -> evidence selection (diverse, deduplicated)
  -> 6-dimension assessment (Groq) -> weighted score + critical-failure rules -> decision
  -> if publishable: conference matching (centroid + neighbours + keywords + LLM fit) -> rationale (<= 100 words)
  -> results.csv, PDF report, Streamlit UI, full audit trail in SQLite and JSON
```

## Setup

```bash
# venv at e:/AI_refree/.venv; run all commands from AIReferee/
../.venv/Scripts/pip install -r requirements.txt
cp .env.example .env      # GROQ_API_KEY=..., SERPAPI_API_KEY=...
```

The default LLM is `openai/gpt-oss-120b` on Groq; Groq no longer serves `llama-3.3-70b`. Without a Groq key, every LLM step falls back to labelled heuristics, and outputs record which path was used (`groq:<model>`, `heuristic-fallback` or `template-fallback`). Heuristic assessments carry confidence 0.3, which sends the paper to HUMAN_REVIEW.

## Commands

| Task | Command |
|---|---|
| Build the whole corpus (collect, extract and chunk, embed and index) | `python -m ingestion.ingestion_pipeline build` |
| Individual stages | `... collect`, `... process [--force]`, `... index [--rebuild] [--model M]`, `... status` |
| Conference profiles | `python -m evaluation.conference_profiler` |
| Evaluate the hackathon set | put PDFs in `data/evaluation_dataset/papers/` (e.g. `P001.pdf`), optionally `labels.csv`, then run `python -m scripts.run_evaluation [--labelled-only] [--no-external]` |
| UI | `streamlit run app.py` |
| Benchmarks | `python -m embeddings.benchmark retrieval --models all`, `... rerank`, `... conference` |
| Tests | `python -m pytest` (unit), `-m corpus` (real corpus), `-m e2e` (live pipeline) |

`run_evaluation` writes `data/reports/results.csv` and `evaluation_summary.json`, plus `data/reports/<id>/result.json` and `report.pdf` for each paper. When `labels.csv` is present, the summary also includes accuracy, precision, recall, F1 and the confusion matrix. Labels are read only after prediction.

## Scaling

The only setting you need to change is `papers_per_conference` in `config.py` (20 → 50 → 100 → 200 → 334). Each stage is incremental:

- Valid PDFs are never downloaded again.
- Extracted papers are skipped unless `--force` is given.
- Only chunks without a vector for the model are embedded.
- The FAISS index is loaded from disk, appended to and validated against SQLite. If they disagree, it is rebuilt automatically.
- Changing `chunk_size`, `chunk_overlap` or the embedding model changes a stored signature, which forces the corpus to be re-chunked.

Above roughly 4,000 chunks you can switch `faiss_index_type` to `ivf_flat`.

To go past one proceedings volume per venue, add years or volumes to `CONFERENCE_SOURCES`. The pipeline code stays the same.

## Reference corpus (current)

There are 100 verified PDFs: 20 each for CVPR, NeurIPS, EMNLP, TMLR and KDD.

| Venue | Source | Venue verification |
|---|---|---|
| CVPR | CVF Open Access 2024 | official proceedings |
| NeurIPS | proceedings.neurips.cc 2023 (main track) | official proceedings |
| EMNLP | ACL Anthology `2023.emnlp-main` | official proceedings |
| TMLR | arXiv API | the author's comment or journal_ref says the paper was published in TMLR |
| KDD | arXiv API | the author's comment or journal_ref says it was accepted at KDD 2024 or 2023 |

OpenReview, DBLP and the ACM DL serve bot challenges, and the collector does not bypass them. The arXiv versions used for TMLR and KDD are author versions.

During extraction, venue boilerplate is removed so it can't leak the label into retrieval or conference matching. This covers the CVF watermark, the ACM copyright, CCS and reference-format blocks, the NeurIPS and EMNLP footers and the TMLR header. References, front matter and acknowledgements are not embedded.

## Decision policy

| Setting | Value |
|---|---|
| Dimension weights | novelty .25, methodology .25, experimental .20, literature .10, results .10, reproducibility .10 |
| Publishable threshold | weighted score ≥ 6.0 (on a 0–10 scale) |
| Critical-failure rules | methodology, experimental or results score < 3, or an LLM-flagged critical flaw with confidence ≥ 0.5, gives NON-PUBLISHABLE regardless of the other scores |
| HUMAN_REVIEW | mean confidence < 0.4, score within ±0.5 of the threshold, or insufficient extracted text |
| HUMAN_REVIEW in the CSV | `human_review_csv_policy = "score_threshold"`: the side of the threshold the score falls on. The alternative is `"non_publishable"`. The internal report always keeps HUMAN_REVIEW |
| CSV rules | Non-Publishable rows get `na` / `na`; Publishable rows get a venue, never DAA, and a rationale of at most 100 words (validated before writing) |

These weights and thresholds are starting assumptions. They are not validated standards, and should be calibrated on the labelled hackathon papers.

## Data and outputs

- `data/aireferee.db` holds these tables: papers, chunks, vector_map, queries, retrieval_logs, evaluations, conferences, conference_recommendations, settings, external_search_cache.
- `data/faiss/reference.index` is the MiniLM index. Benchmark models get `reference__<model>.index`. Conference centroids are in `conference_centroids.npz`.
- `data/extracted/papers/<id>.json` caches the section-structured extraction of each paper.
- `data/reports/` holds the benchmark JSON files, `results.csv` and the per-paper reports.

## Known limitations

- **DAA is skipped** because the acronym is unverified. It is never collected and never recommended.
- **The Google Scholar free plan allows 250 searches a month.** AIReferee makes at most 2 Scholar searches per paper and caches every response. Use `--no-external` for large batches.
- **OCR is not bundled.** Scanned PDFs are marked `needs_ocr`.
- **Some TMLR PDFs drop spaces between words** in a few places, which slightly degrades their embeddings.
- **The retrieval benchmark is a known-item proxy,** not expert relevance labels. See `embeddings/benchmark.py`.
- **The conference leave-one-out test excludes the LLM signal,** to save API calls.
- **arXiv and Scholar results are metadata only.** They inform literature coverage and novelty, but they are never cited as full-text passages.
