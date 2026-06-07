# JoyBookers — Book Recommendation System

University project: collaborative filtering, content-based recommendations, sentiment analysis, and user clustering over four Goodreads / Amazon datasets.

## Documentation

| Language | Document |
|----------|----------|
| **Polski** (etap danych i ML, wyniki) | [ETAP_DANYCH_I_ML.md](ETAP_DANYCH_I_ML.md) |
| Reports (JSON metrics in git) | [reports/README.md](reports/README.md) |
| Raw data layout | [data/raw/README.txt](data/raw/README.txt) |

## Quick start

```powershell
pip install -r requirements.txt
python scripts/run_data_pipeline.py --stages all
python scripts/ml/run_ml_pipeline.py --stages all
python scripts/export_reports.py
uvicorn app.main:app --reload
```
