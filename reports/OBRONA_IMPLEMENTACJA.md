# Raport implementacji — przygotowanie do obrony

Data: 2025-06-07  
Projekt: JoyBookers

## Cel

Uzupełnienie brakujących elementów **w kodzie** (nie slajdów), aby punkty rubryki 50 pkt były widoczne w aplikacji web i pipeline był uruchamialny jednym poleceniem.

---

## 1. `scripts/setup_all.py` — pełny pipeline

**Plik:** `scripts/setup_all.py`  
**README:** zaktualizowany Quick start

**Co robi:** Uruchamia kolejno:
1. `run_data_pipeline.py --stages all`
2. `ml/run_ml_pipeline.py --stages all`
3. `build_cluster_affinity.py`
4. `build_genre_priors.py`
5. `train_hybrid_weights.py`
6. `evaluate_hybrid_baselines.py` (opcjonalnie `--skip-hybrid-eval`)
7. `export_reports.py`
8. `load_db.py`

**Flagi:** `--skip-data`, `--skip-ml`, `--books-limit`, `--ratings-limit`

**Użycie:**
```powershell
python scripts/setup_all.py
uvicorn app.main:app --reload
```

---

## 2. `/analytics` — selekcja cech i wartości odstające

**Backend:** `app/services/reports_service.py`
- `_build_feature_selection_rows()` — tabela modułów ML i wybranych cech
- `_build_outlier_summary()` — outliery z `preprocess_summary.json` + `analyze_all.json`
- Metryki CF: `precision_at_k`, `recall_at_k`
- Metryki hybrid Ridge z `ml/evaluation/hybrid/baseline_comparison.json`

**Szablony:**
- `app/templates/analytics/_feature_selection.html`
- `app/templates/analytics/_outliers.html`
- `app/templates/analytics/dashboard.html` — sekcje + Precision@10 + Hybrid Ridge

**Rubryka:** selekcja atrybutów (2 pkt), identyfikacja outlierów (2 pkt), ocena algorytmów (3 pkt)

---

## 3. `/clustering` — formularz nowego profilu

**Endpoint:** `POST /clustering/predict` w `app/routers/web/pages.py`  
**Szablony:**
- `app/templates/clustering/_predict_form.html`
- `app/templates/clustering/_predict_result.html`

**Logika:** 1–5 ocen → `UserClusteringEngine.predict_cluster()` → etykieta klastra (ten sam model co `/me`).

**Rubryka:** przyporządkowanie nowego profilu do grupy (1 pkt)

---

## 4. Eksport metryk hybrid

**Plik:** `bookrec/reports_export.py`  
Dodano kopię `data/processed/models/evaluation/hybrid/` → `reports/ml/evaluation/hybrid/`

---

## 5. Polskie nagłówki UI

**Pliki:**
- `app/templates/base.html` — nawigacja, `lang="pl"`
- `app/templates/analytics/dashboard.html`
- `app/templates/clustering/dashboard.html`
- `app/templates/recommendations/user.html`
- `app/templates/sentiment/index.html`

---

## 6. Testy

**Plik:** `tests/test_api_skeleton.py`
- `test_clustering_predict` — POST z ocenami
- `test_clustering_predict_invalid` — walidacja 1–5
- `test_analytics_rubric_sections` — sekcje rubryki na `/analytics`

---

## Mapowanie rubryki → URL / pliki

| Element rubryki | Gdzie w projekcie |
|-----------------|-------------------|
| Przygotowanie danych | `scripts/run_data_pipeline.py`, `/analytics` (wolumeny) |
| Wizualizacja | `/analytics` → EDA PNG |
| Selekcja cech | `/analytics` → tabela selekcji |
| Wartości odstające | `/analytics` → tabela outlierów |
| Miara podobieństwa | content: cosine; opis w tabeli selekcji |
| Ocena algorytmów | `/analytics` → RMSE, Precision@10, F1, silhouette, hybrid |
| Rekomendacje | `/recommendations` |
| Cold-start | `/me`, genre priors w hybrid |
| Klasyfikacja komentarza | `/sentiment` |
| Nowy profil → grupa | `/clustering` → formularz 5 ocen |
| ≥3 kategorie algorytmów | K-Means, LR, SVD, Ridge — `/analytics` |
| NLP | `/sentiment` + content TF-IDF |
| Aplikacja web + input | FastAPI + formularze HTMX |

---

## Weryfikacja po wdrożeniu

```powershell
pytest tests/test_api_skeleton.py -v
python scripts/export_reports.py   # jeśli hybrid eval już był
uvicorn app.main:app --reload
```

Sprawdź w przeglądarce:
- `/analytics` — tabele selekcji i outlierów, Precision@10
- `/clustering` — formularz → wynik klastra
- Nawigacja po polsku
