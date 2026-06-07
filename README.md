# JoyBookers — Book Recommendation System

University project: collaborative filtering, content-based recommendations, sentiment analysis, and user clustering over four Goodreads / Amazon datasets.

## Documentation

| Language | Document |
|----------|----------|
| **Polski** (etap danych i ML, wyniki) | [ETAP_DANYCH_I_ML.md](ETAP_DANYCH_I_ML.md) |
| Reports (JSON metrics in git) | [reports/README.md](reports/README.md) |
| Raw data layout | [data/raw/README.txt](data/raw/README.txt) |

## Quick start

Full setup (data → ML → hybrid artifacts → reports → SQLite):

```powershell
pip install -r requirements.txt
python scripts/setup_all.py
uvicorn app.main:app --reload
```

Skip stages when artifacts already exist:

```powershell
python scripts/setup_all.py --skip-data --skip-ml
```

Manual steps (equivalent to `setup_all.py`):

```powershell
python scripts/run_data_pipeline.py --stages all
python scripts/ml/run_ml_pipeline.py --stages all
python scripts/build_cluster_affinity.py
python scripts/build_genre_priors.py
python scripts/train_hybrid_weights.py
python scripts/evaluate_hybrid_baselines.py
python scripts/export_reports.py
python scripts/load_db.py --books-limit 20000 --ratings-limit 50000
uvicorn app.main:app --reload
```
