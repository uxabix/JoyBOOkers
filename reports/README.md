# Reports (version-controlled)

JSON reports and EDA plots exported from the data and ML pipelines. **Binary artifacts** (`.pkl`, `.parquet`, `.npz`) stay in `data/processed/` and are **not** committed to git.

## Key files for analysis without re-running code

| File | Contents |
|------|----------|
| **`ml/evaluation/evaluate_all.json`** | Aggregated metrics for all 4 modules (CF, content, sentiment, clustering) |
| **`ml/evaluation/train_all.json`** | Aggregated training summaries for all modules |
| `ml/evaluation/ml_pipeline_summary.json` | Last ML pipeline run (may cover a single module only) |
| `features/features_summary.json` | Data-features stage rollup (interactions, clustering, content, NLP) |
| `data_pipeline/preprocess_summary.json` | DS1–DS4 cleaning statistics |
| `data_pipeline/pipeline_summary.json` | Full data-pipeline run summary |
| `manifest.json` | List of all exported files |

## Directory layout

| Folder | Contents |
|--------|----------|
| `data_pipeline/` | analyze, preprocess, pipeline summaries |
| `features/` | Feature reports (interactions, clustering, content, NLP) |
| `splits/` | CF / NLP split metadata |
| `ml/collaborative/` | SVD train / preprocess |
| `ml/clustering/` | K-Means train / preprocess |
| `ml/content/` | TF-IDF train / preprocess |
| `ml/sentiment/` | Sentiment train / preprocess |
| `ml/evaluation/` | **`evaluate_all.json`** + per-module evaluate reports |
| `eda/` | DS1 exploratory PNG plots |

## Refreshing reports

Reports are updated **automatically** at the end of:

- `scripts/run_data_pipeline.py`
- `scripts/ml/run_ml_pipeline.py`

Manual refresh (recommended after any partial run):

```powershell
python scripts/export_reports.py
```

`export_reports` rebuilds `evaluate_all.json` and `train_all.json` from the latest `*/evaluate_report.json` and `*/train_report.json` files on disk.

## Git

The `reports/` folder is **not** in `.gitignore`. To version results for analysis on another machine:

```powershell
git add reports/
git commit -m "Add pipeline and ML evaluation reports"
```

Without a commit, files exist only locally.

See also: [`ETAP_DANYCH_I_ML.md`](../ETAP_DANYCH_I_ML.md) §9 — full catalog of every analysis file and what each contains.
