JoyBookers — dataset layout (university project scope)
======================================================

Each dataset has ONE job. Do not merge millions of Goodreads rows.

  DS1  data/raw/ds1_goodreads_2m/   → Collaborative Filtering (SVD) + user K-Means
  DS2  data/raw/ds2_goodreads_100k/   → Content-based recommendation (primary)
  DS3  data/raw/ds3_goodreads_best/   → Enrichment (tags, characters → merged into DS2)
  DS4  data/raw/ds4_amazon_reviews/   → NLP / sentiment (independent, no Goodreads link)

Legacy DS1 flat layout still works: data/raw/book*.csv + user_rating_*.csv

Performance constraints (8–16 GB RAM)
-------------------------------------
  - Do NOT build TF-IDF/BoW for the full DS1 catalog (1.5M+ books)
  - Content vectors: DS2 + DS3 only (~100k–150k rows, scipy.sparse)
  - Amazon reviews: use --ds4-sample for large corpora

Run pipeline
------------
  pip install -r requirements.txt
  python scripts/run_data_pipeline.py --stages all

  python scripts/run_data_pipeline.py --stages preprocess features splits
  python scripts/run_data_pipeline.py --ds4-sample 100000

Stages
------
  analyze     → data/processed/analysis/analyze_all.json
  preprocess  → data/processed/ds1..ds4/
  features    → interactions (DS1), clustering/users (DS1), content (DS2+DS3), nlp (DS4)
  splits      → cf_train/test (DS1), nlp_train/val/test (DS4)

  "resolve" is deprecated (no-op). DS3→DS2 merge happens inside content features.

ML modules (target architecture)
--------------------------------
  Dataset A (DS1)  → user-item matrix → SVD → CF recommendations
                   → user features → K-Means → user segments

  Dataset B+C      → genres/tags/authors → sparse BoW → cosine → similar books

  Dataset D (DS4)  → review text → TF-IDF → Logistic Regression → sentiment
